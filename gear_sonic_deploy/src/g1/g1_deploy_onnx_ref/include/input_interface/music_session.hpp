/**
 * @file music_session.hpp
 * @brief G1 仿真音乐会话：可靠接收 SMPL 参考，并按统一单调时钟提供连续未来窗口。
 *
 * 数据面沿用 SONIC v3 的 pose 前缀、1280 字节 JSON 头和小端数组；会话控制使用
 * ZMQ REP 的 JSON 请求及可选二进制附件。接收线程完成完整校验后才发布不可变快照，
 * 控制线程不会等待网络。每个会话只允许追加连续帧，重复请求返回原应答，冲突请求
 * 不得重放动作。该接口只允许绑定本机，部署入口还必须验证 lo、CRC 仿真模式与 mode 2。
 *
 * 时间线包含起舞前缀和收尾。frame_index=0 对应 epoch_ns，而 audio_start_frame
 * 对应音乐第一个采样。缓冲容量为 15 秒，保留已消费的十帧历史；播放过程不会因为
 * 新包到达而重置帧号或朝向。欠载、心跳丢失及非法有效会话数据会锁存故障，交给
 * 协调器同时停止音频和 MuJoCo，只有新的 prepare 才能开始下一次会话。
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <nlohmann/json.hpp>
#include <zmq.hpp>
#include "../motion_data_reader.hpp"

class MusicSession {
 public:
  using Json = nlohmann::json;
  struct Frame {
    std::array<double, 29> joint_pos{}, joint_vel{};
    std::array<double, 4> quat{};
    std::array<MotionSequence::Point, 24> joints{};
    std::array<MotionSequence::Point, 21> pose{};
  };
  struct View {
    std::shared_ptr<const MotionSequence> motion;
    int local_frame = 0;
    int64_t global_frame = 0;
    bool play = false, fault = false;
    std::string session_id;
  };
  static int64_t NowNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
  }

  explicit MusicSession(const std::string& endpoint, bool start_server = true)
      : endpoint_(endpoint) {
    if (endpoint.rfind("tcp://127.0.0.1:", 0) != 0)
      throw std::runtime_error("音乐接口只能绑定 tcp://127.0.0.1");
    if (start_server) {
      thread_ = std::thread([this] { Serve(); });
      std::unique_lock<std::mutex> lock(start_mutex_);
      started_.wait(lock, [this] { return server_ready_; });
      if (!server_error_.empty()) {
        lock.unlock();
        thread_.join();
        throw std::runtime_error(server_error_);
      }
    }
  }
  ~MusicSession() { running_ = false; if (thread_.joinable()) thread_.join(); }

  // 纯内存请求入口同样用于协议测试，所有状态更新必须持有本锁。
  Json Request(const Json& request, const std::string& binary = "") {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string op, sid;
    int64_t seq=-1;
    const std::string identity = request.dump() + binary;
    try {
      sid=request.value("session_id", ""); op=request.value("op", "");
      if (!request.contains("seq") || !request["seq"].is_number_integer())
        throw std::runtime_error("请求 seq 必须是整数");
      seq=request.at("seq").get<int64_t>();
      if (seq < 0) throw std::runtime_error("请求必须具有非负整数 seq");
      if (sid == session_id_) {
        auto found = replies_.find(seq);
        if (found != replies_.end()) {
          if (found->second.first != identity) throw std::runtime_error("相同 seq 的请求内容冲突");
          heartbeat_ns_ = NowNs();
          return found->second.second;
        }
      }
      if (op == "prepare") {
        if (sid.empty() || sid == session_id_) throw std::runtime_error("prepare 必须使用新的 session_id");
        if (state_ == "playing" || state_ == "armed" || state_ == "prepared") throw std::runtime_error("现有会话仍在活动");
        const int64_t audio_frames = request.at("audio_frames").get<int64_t>();
        const int64_t prefix = request.value("audio_start_frame", int64_t(100));
        if (audio_frames < 1 || audio_frames > 3600000 || prefix < 50 || prefix > 250)
          throw std::runtime_error("音频帧数或起舞前缀越界");
        session_id_ = sid; audio_frames_ = audio_frames; prefix_ = prefix;
        frames_.clear(); motion_.reset(); replies_.clear(); reply_order_.clear();
        base_ = 0; received_ = -1; consumed_ = used_frame_ = 0; epoch_ns_ = control_ns_ = 0; complete_ = false;
        state_ = "prepared"; error_.clear(); last_seq_ = -1; control_ready_ = false;
        ticks_.clear(); audit_=Json::object();
      } else if (op != "status" || !sid.empty()) {
        if (sid.empty() || sid != session_id_) throw std::runtime_error("session_id 不匹配");
        if (seq <= last_seq_) throw std::runtime_error("seq 已过期或顺序错误");
        if (op == "append") {
          if (complete_ || state_ == "fault" || state_ == "stopped") throw std::runtime_error("当前会话不可追加");
          auto incoming = Decode(binary);
          if (request.at("start_frame").get<int64_t>() != received_ + 1 ||
              request.at("end_frame").get<int64_t>() != received_ + int64_t(incoming.size()))
            throw std::runtime_error("追加帧存在缺口或重叠");
          for (size_t i = 0; i < incoming.size(); ++i)
            if (decoded_indices_[i] != received_ + 1 + int64_t(i))
              throw std::runtime_error("附件 frame_index 与会话帧号不一致");
          while (base_ < consumed_ - 10 && !frames_.empty()) { frames_.pop_front(); ++base_; }
          // 750 帧是未消费未来容量，另外保留当前帧和十帧历史作为只读上下文。
          if (received_+int64_t(incoming.size())-consumed_ > 750)
            throw std::runtime_error("动作缓冲超过 15 秒容量");
          frames_.insert(frames_.end(), incoming.begin(), incoming.end());
          received_ += int64_t(incoming.size());
          PublishSnapshot();
        } else if (op == "start") {
          const int64_t epoch = request.at("epoch_ns").get<int64_t>();
          if (state_ != "prepared" || !control_ready_ || received_ + 1 < prefix_ + std::min<int64_t>(audio_frames_, 345))
            throw std::runtime_error("控制器或起播预缓冲尚未准备完成");
          if (epoch < NowNs() + 100000000LL || epoch > NowNs() + 10000000000LL)
            throw std::runtime_error("起播时间必须位于未来 0.1 至 10 秒");
          epoch_ns_ = epoch; state_ = "armed";
        } else if (op == "finish") {
          if (state_ == "fault" || state_ == "stopped") throw std::runtime_error("已终止会话不能完成");
          if (received_ < prefix_ + audio_frames_ + 59)
            throw std::runtime_error("收尾必须包含 50 帧过渡及 10 帧静止保护");
          complete_ = true;
        } else if (op == "stop") {
          if (request.value("graceful", false)) {
            // 用户明确停止时才替换尚未播放的未来段，网络重试仍由原请求编号去重。
            const int64_t cut = request.at("start_frame").get<int64_t>();
            auto tail = Decode(binary);
            if (state_ != "playing" || cut < consumed_+10 || cut > received_-9 || tail.size() != 60)
              throw std::runtime_error("平滑停止的切换位置或收尾长度无效");
            for (size_t i=0; i<tail.size(); ++i)
              if (decoded_indices_[i] != cut+int64_t(i)) throw std::runtime_error("收尾帧号不连续");
            while (base_+int64_t(frames_.size()) > cut) frames_.pop_back();
            frames_.insert(frames_.end(), tail.begin(), tail.end());
            received_=cut+59; complete_=true; PublishSnapshot();
          } else {
            state_ = request.value("fault", false) ? "fault" : "stopped";
            error_ = request.value("reason", "用户停止");
          }
        } else if (op != "status") throw std::runtime_error("未知音乐会话操作");
      }
      if (sid == session_id_) heartbeat_ns_ = NowNs();
      Json reply = Status(); reply["ok"] = true; reply["seq"] = seq;
      if (op == "status" && request.value("audit",false)) reply["observation_audit"]=audit_;
      if (op == "status" && request.value("metrics",false)) {
        // 完整控制周期统计只在验收收尾时计算，普通心跳保持常数开销。
        std::vector<double> samples;
        int64_t first_ns=0;
        for (const auto& tick:ticks_) if (tick.first >= epoch_ns_ && epoch_ns_>0) {
          if (!first_ns) first_ns=tick.first;
          samples.push_back(tick.second);
        }
        if (!samples.empty()) {
          double sum=0; for(double v:samples) sum+=v;
          std::sort(samples.begin(),samples.end());
          reply["control_metrics"]={{"count",samples.size()},{"mean_ms",sum/samples.size()},
            {"p99_ms",samples[size_t(std::ceil(0.99*samples.size()))-1]}, {"max_ms",samples.back()},
            {"wall_hz",double(samples.size()-1)*1e9/std::max<int64_t>(1,control_ns_-first_ns)}};
        }
      }
      if (sid == session_id_) {
        last_seq_ = std::max(last_seq_, seq);
        replies_[seq] = {identity, reply}; reply_order_.push_back(seq);
        while (reply_order_.size() > 16) { replies_.erase(reply_order_.front()); reply_order_.pop_front(); }
      }
      return reply;
    } catch (const std::exception& e) {
      if (sid == session_id_ && op != "status" && op != "prepare") {
        state_ = "fault"; error_ = e.what();
      }
      return Json{{"ok", false}, {"error", e.what()}, {"state", state_}, {"seq", seq}};
    }
  }

  View Read(int64_t now = NowNs()) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool active = state_ == "armed" || state_ == "playing";
    if (active && now - heartbeat_ns_ > 1500000000LL) { state_ = "fault"; error_ = "协调器心跳超时"; }
    bool play = active && now >= epoch_ns_ && state_ != "fault";
    int64_t frame = play ? (now - epoch_ns_) / 20000000LL : consumed_;
    if (play) {
      state_ = "playing";
      if (complete_ && frame >= received_ - 9) { frame = received_ - 9; state_ = "finished"; play = false; }
      else if ((!complete_ && received_ - frame < 25) || frame + 9 > received_) {
        state_ = "fault"; error_ = "未来参考缓冲不足"; play = false;
      }
    }
    if (state_ != "fault") consumed_ = frame;
    View result{motion_, int(std::max<int64_t>(0, consumed_ - base_)), consumed_, play,
                state_ == "fault" || state_ == "stopped", session_id_};
    return result;
  }
  void MarkControl(bool ready, int64_t used_frame, double compute_ms = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    control_ready_ = ready; used_frame_ = used_frame; control_ns_ = NowNs(); compute_ms_ = compute_ms;
    ticks_.emplace_back(control_ns_,compute_ms);
  }
  void MarkObservation(const std::vector<double>& observation, const std::array<double,4>& base,
                       const std::array<double,4>& heading, int64_t frame, bool play) {
    std::lock_guard<std::mutex> lock(mutex_);
    audit_={{"frame",frame},{"play",play},{"encoder",observation},{"robot_quat",base},{"heading",heading}};
  }
  void Fault(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_); state_ = "fault"; error_ = reason;
  }

 private:
  Json Status() const {
    return Json{{"session_id", session_id_}, {"state", state_}, {"error", error_},
                {"received_frame", received_}, {"used_frame", used_frame_},
                {"buffer_seconds", std::max<int64_t>(0, received_ - consumed_) / 50.0},
                {"encoder_mode", 2}, {"control_ready", control_ready_},
                {"control_ns", control_ns_}, {"compute_ms", compute_ms_},
                {"epoch_ns", epoch_ns_}, {"monotonic_ns", NowNs()}};
  }
  std::vector<Frame> Decode(const std::string& data) {
    if (data.size() < 1284 || data.compare(0, 4, "pose") != 0) throw std::runtime_error("缺少 v3 pose 数据头");
    auto header = Json::parse(data.substr(4, 1280).c_str());
    if (header.at("v") != 3 || header.at("endian") != "le") throw std::runtime_error("只接受小端 Protocol v3");
    std::map<std::string, std::pair<Json, size_t>> fields;
    size_t offset = 1284; int n = -1;
    const std::map<std::string, std::vector<int>> tails = {
      {"smpl_pose", {21,3}}, {"smpl_joints", {24,3}}, {"body_quat", {4}},
      {"joint_pos", {29}}, {"joint_vel", {29}}, {"frame_index", {}}};
    for (const auto& field : header.at("fields")) {
      const std::string name = field.at("name");
      if (!tails.count(name) || fields.count(name)) throw std::runtime_error("未知或重复的动作字段");
      const auto shape = field.at("shape").get<std::vector<int>>();
      if (shape.empty() || shape[0] < 1 || shape[0] > 750) throw std::runtime_error("动作块帧数非法");
      if (n < 0) n = shape[0];
      if (n != shape[0] || std::vector<int>(shape.begin()+1, shape.end()) != tails.at(name))
        throw std::runtime_error("动作字段形状不匹配");
      const std::string dtype = field.at("dtype");
      if (dtype != (name == "frame_index" ? "i64" : "f32")) throw std::runtime_error("动作字段 dtype 不匹配");
      size_t count = 1; for (int dim : shape) count *= size_t(dim);
      const size_t bytes = count * (name == "frame_index" ? 8 : 4);
      if (offset + bytes > data.size()) throw std::runtime_error("动作附件被截断");
      fields[name] = {field, offset}; offset += bytes;
    }
    if (fields.size() != 6 || offset != data.size()) throw std::runtime_error("动作附件字段不完整或存在尾部数据");
    std::vector<Frame> result(n); decoded_indices_.resize(n);
    auto read_float = [&](const std::string& name, size_t index) {
      const char* pointer=data.data()+fields.at(name).second+index*4;
      uint32_t bits; std::memcpy(&bits,pointer,4);
      // Release 启用 fast-math，必须按 IEEE754 位模式检查，避免 isfinite 被编译器消除。
      if ((bits & 0x7f800000U) == 0x7f800000U) throw std::runtime_error("动作包含 NaN 或 Inf");
      float value; std::memcpy(&value,pointer,4);
      return double(value);
    };
    for (int i = 0; i < n; ++i) {
      std::memcpy(&decoded_indices_[i], data.data()+fields.at("frame_index").second+i*8, 8);
      for (int j=0;j<29;++j) { result[i].joint_pos[j]=read_float("joint_pos",i*29+j); result[i].joint_vel[j]=read_float("joint_vel",i*29+j); }
      double norm=0; for(int j=0;j<4;++j) { result[i].quat[j]=read_float("body_quat",i*4+j); norm+=result[i].quat[j]*result[i].quat[j]; }
      if (std::abs(norm-1)>0.001) throw std::runtime_error("根四元数不是单位四元数");
      for(int j=0;j<24;++j) for(int k=0;k<3;++k) result[i].joints[j][k]=read_float("smpl_joints",i*72+j*3+k);
      for(int j=0;j<21;++j) for(int k=0;k<3;++k) result[i].pose[j][k]=read_float("smpl_pose",i*63+j*3+k);
    }
    return result;
  }
  void PublishSnapshot() {
    auto out = std::make_shared<MotionSequence>();
    out->name = "music"; out->ReserveCapacity(int(frames_.size()),29,1,1,24,21);
    out->timesteps = int(frames_.size()); out->SetEncodeMode(2); out->SetBodyPartIndexes({0});
    int i=0; for (const auto& frame : frames_) {
      std::copy(frame.joint_pos.begin(),frame.joint_pos.end(),out->JointPositions(i));
      std::copy(frame.joint_vel.begin(),frame.joint_vel.end(),out->JointVelocities(i));
      out->BodyQuaternions(i)[0]=frame.quat;
      std::copy(frame.joints.begin(),frame.joints.end(),out->SmplJoints(i));
      std::copy(frame.pose.begin(),frame.pose.end(),out->SmplPoses(i)); ++i;
    }
    motion_ = out;
  }
  void Serve() {
    try {
      zmq::context_t context(1); zmq::socket_t socket(context,zmq::socket_type::rep);
      socket.set(zmq::sockopt::linger,0); socket.set(zmq::sockopt::rcvtimeo,100);
      socket.set(zmq::sockopt::sndtimeo,100); socket.set(zmq::sockopt::maxmsgsize,int64_t(2*1024*1024));
      socket.bind(endpoint_);
      { std::lock_guard<std::mutex> lock(start_mutex_); server_ready_=true; } started_.notify_one();
      while (running_) {
        zmq::message_t message;
        if (!socket.recv(message,zmq::recv_flags::none)) continue;
        std::vector<std::string> parts{message.to_string()};
        while(socket.get(zmq::sockopt::rcvmore)) {
          zmq::message_t part;
          if (!socket.recv(part,zmq::recv_flags::none)) break;
          parts.push_back(part.to_string());
        }
        Json reply;
        try {
          if(parts.size()>2) throw std::runtime_error("会话请求最多包含两个消息段");
          reply=Request(Json::parse(parts[0]),parts.size()==2?parts[1]:"");
        } catch(const std::exception& e) { reply={{"ok",false},{"error",e.what()}}; }
        auto encoded=reply.dump(); socket.send(zmq::buffer(encoded),zmq::send_flags::none);
      }
    } catch(const std::exception& e) {
      { std::lock_guard<std::mutex> lock(start_mutex_); server_error_=e.what(); server_ready_=true; } started_.notify_one();
      Fault(e.what());
    }
  }
  std::string endpoint_, session_id_, state_="idle", error_;
  std::mutex mutex_, start_mutex_;
  std::condition_variable started_;
  bool server_ready_=false, complete_=false, control_ready_=false;
  std::string server_error_;
  std::atomic<bool> running_{true}; std::thread thread_;
  std::deque<Frame> frames_; std::shared_ptr<const MotionSequence> motion_;
  std::vector<int64_t> decoded_indices_;
  int64_t base_=0, received_=-1, consumed_=0, used_frame_=0, audio_frames_=0, prefix_=100;
  int64_t epoch_ns_=0, heartbeat_ns_=0, control_ns_=0, last_seq_=-1;
  double compute_ms_=0;
  std::vector<std::pair<int64_t,double>> ticks_;
  Json audit_=Json::object();
  std::map<int64_t,std::pair<std::string,Json>> replies_; std::deque<int64_t> reply_order_;
};
