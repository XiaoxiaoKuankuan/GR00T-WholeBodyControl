/**
 * @file test_music_session.cpp
 * @brief 音乐会话状态机和二进制协议的独立测试，不连接 DDS、GPU 或机器人。
 *
 * 使用真实 v3 附件验证连续追加、快照不变性、重试幂等、缺帧和非法数值拒绝、
 * 未来窗口欠载、心跳丢失及新会话隔离。时间由 Read 的显式参数推进，测试无需
 * 墙钟等待；所有示例仅使用本机未启动的 REP 接口实例。
 */
#include <gtest/gtest.h>
#include "input_interface/music_session.hpp"

namespace {
using Json = nlohmann::json;
std::string Packet(int64_t start, int n, bool nan = false) {
  Json fields = Json::array();
  std::string payload;
  const std::vector<std::pair<std::string, std::vector<int>>> spec = {
    {"smpl_pose", {n,21,3}}, {"smpl_joints", {n,24,3}}, {"body_quat", {n,4}},
    {"joint_pos", {n,29}}, {"joint_vel", {n,29}}, {"frame_index", {n}}};
  for (const auto& [name, shape] : spec) {
    fields.push_back({{"name",name},{"dtype",name == "frame_index" ? "i64":"f32"},{"shape",shape}});
    int count=1; for (int d:shape) count*=d;
    for(int i=0;i<count;++i) {
      if(name == "frame_index") {
        int64_t value=start+i; payload.append(reinterpret_cast<char*>(&value),8);
      } else {
        float value=name == "body_quat" && i%4 == 0 ? 1.f : 0.f;
        if(nan && name == "smpl_pose" && i == 0) value=std::numeric_limits<float>::quiet_NaN();
        payload.append(reinterpret_cast<char*>(&value),4);
      }
    }
  }
  std::string header=Json{{"v",3},{"endian","le"},{"frame_count",1},{"fields",fields}}.dump();
  header.resize(1280,' ');
  return "pose"+header+payload;
}
Json Req(std::string op, int seq, std::string session="test") {
  return Json{{"op",op},{"session_id",session},{"seq",seq}};
}
void Prepare(MusicSession& s, int audio=600) {
  auto r=Req("prepare",0); r["audio_frames"]=audio;
  ASSERT_TRUE(s.Request(r)["ok"]);
}
Json Append(MusicSession& s,int seq,int64_t start,int n,bool nan=false) {
  auto r=Req("append",seq); r["start_frame"]=start; r["end_frame"]=start+n-1;
  return s.Request(r,Packet(start,n,nan));
}
}

TEST(MusicSession, AppendRetryAndImmutableSnapshot) {
  MusicSession s("tcp://127.0.0.1:5560",false); Prepare(s);
  ASSERT_TRUE(Append(s,1,0,300)["ok"]);
  auto before=s.Read();
  ASSERT_TRUE(Append(s,1,0,300)["ok"]);
  ASSERT_TRUE(Append(s,2,300,150)["ok"]);
  EXPECT_EQ(before.motion->timesteps,300);
  EXPECT_EQ(s.Read().motion->timesteps,450);
  EXPECT_EQ(s.Read().motion->GetEncodeMode(),2);
}

TEST(MusicSession, RejectGapConflictAndNonFinite) {
  for(int kind=0;kind<3;++kind) {
    MusicSession s("tcp://127.0.0.1:5560",false); Prepare(s);
    ASSERT_TRUE(Append(s,1,0,100)["ok"]);
    auto bad=kind == 0 ? Append(s,2,101,100) :
             kind == 1 ? Append(s,1,100,100) : Append(s,2,100,100,true);
    EXPECT_FALSE(bad["ok"].get<bool>());
    EXPECT_TRUE(s.Read().fault);
  }
}

TEST(MusicSession, AbsoluteClockAndUnderflow) {
  MusicSession s("tcp://127.0.0.1:5560",false); Prepare(s);
  ASSERT_TRUE(Append(s,1,0,450)["ok"]); s.MarkControl(true,0);
  auto start=Req("start",2); const int64_t epoch=MusicSession::NowNs()+200000000;
  start["epoch_ns"]=epoch; ASSERT_TRUE(s.Request(start)["ok"]);
  EXPECT_EQ(s.Read(epoch+400000000).global_frame,20);
  ASSERT_TRUE(Append(s,3,450,150)["ok"]);
  EXPECT_EQ(s.Read(epoch+420000000).global_frame,21);
  EXPECT_TRUE(s.Read(epoch+2000000000LL).fault);
}

TEST(MusicSession, ShortEndingAndNewSessionIsolation) {
  MusicSession s("tcp://127.0.0.1:5560",false); Prepare(s,10);
  ASSERT_TRUE(Append(s,1,0,170)["ok"]);
  ASSERT_TRUE(s.Request(Req("finish",2))["ok"]);
  s.MarkControl(true,0);
  auto start=Req("start",3); auto epoch=MusicSession::NowNs()+200000000;
  start["epoch_ns"]=epoch; ASSERT_TRUE(s.Request(start)["ok"]);
  EXPECT_EQ(s.Read(epoch+400000000).global_frame,20);
  auto stop=Req("stop",4); stop["fault"]=true; ASSERT_TRUE(s.Request(stop)["ok"]);
  auto prepare=Req("prepare",0,"new"); prepare["audio_frames"]=100;
  EXPECT_TRUE(s.Request(prepare)["ok"]);
  EXPECT_FALSE(s.Read().motion);
  EXPECT_FALSE(s.Request(Req("status",5))["ok"].get<bool>());
}

TEST(MusicSession, MissingFutureBufferFailsBeforeHeartbeatTimeout) {
  MusicSession s("tcp://127.0.0.1:5560",false);
  auto prepare=Req("prepare",0); prepare["audio_frames"]=1; prepare["audio_start_frame"]=50;
  ASSERT_TRUE(s.Request(prepare)["ok"]);
  ASSERT_TRUE(Append(s,1,0,51)["ok"]); s.MarkControl(true,0);
  auto start=Req("start",2); auto epoch=MusicSession::NowNs()+200000000;
  start["epoch_ns"]=epoch; ASSERT_TRUE(s.Request(start)["ok"]);
  EXPECT_TRUE(s.Read(epoch+550000000).fault);
  EXPECT_EQ(s.Request(Req("status",3))["error"],"未来参考缓冲不足");
}

TEST(MusicSession, GracefulStopPreservesPastAndReplacesOnlyFuture) {
  MusicSession s("tcp://127.0.0.1:5560",false); Prepare(s);
  ASSERT_TRUE(Append(s,1,0,450)["ok"]); s.MarkControl(true,0);
  auto start=Req("start",2); auto epoch=MusicSession::NowNs()+200000000;
  start["epoch_ns"]=epoch; ASSERT_TRUE(s.Request(start)["ok"]);
  auto before=s.Read(epoch+400000000);
  auto stop=Req("stop",3); stop["graceful"]=true; stop["start_frame"]=45;
  EXPECT_TRUE(s.Request(stop,Packet(45,60))["ok"]);
  EXPECT_EQ(before.motion->timesteps,450);
  EXPECT_EQ(s.Read(epoch+420000000).global_frame,21);
  EXPECT_EQ(s.Request(Req("status",4))["received_frame"],104);
}

TEST(MusicSession, InvalidSeqAndActivePrepareAreRejected) {
  MusicSession s("tcp://127.0.0.1:5560",false); Prepare(s);
  auto second=Req("prepare",0,"other"); second["audio_frames"]=100;
  EXPECT_FALSE(s.Request(second)["ok"].get<bool>());
  auto bad=Req("append",1); bad["seq"]=1.5;
  EXPECT_FALSE(s.Request(bad)["ok"].get<bool>());
  EXPECT_TRUE(s.Read().fault);
}
