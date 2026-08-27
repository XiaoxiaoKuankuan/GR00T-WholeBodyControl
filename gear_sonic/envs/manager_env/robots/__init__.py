# Re-export robot configs for backward compatibility.
# Import from gear_sonic.envs.manager_env.robots.g1, .h2 or .bumi3 directly for new code.
from gear_sonic.envs.manager_env.robots.bumi3 import *  # noqa: F401,F403
from gear_sonic.envs.manager_env.robots.g1 import *  # noqa: F401,F403
from gear_sonic.envs.manager_env.robots.h2 import *  # noqa: F401,F403
