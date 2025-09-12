# NOTE: Env must be imported here in order to trigger metaclass registering
from .envs.rag_engine.rag_engine import RAGEngineEnv
from .envs.rag_engine.rag_engine_v2 import RAGEngineEnvV2
from .envs.visual_agent.vl_agent_v1 import VLAgentEnvV1
from .envs.visual_agent.vl_agent_v2 import VLAgentEnvV2
from .envs.mm_process_engine.visual_toolbox import VisualToolBox
from .envs.mm_process_engine.visual_toolbox_v2 import VisualToolBoxV2
from .envs.mm_process_engine.visual_toolbox_v3 import VisualToolBoxV3
from .envs.mm_process_engine.visual_toolbox_v4 import VisualToolBoxV4
from .envs.mm_process_engine.visual_toolbox_v5 import VisualToolBoxV5
from .envs.visual_agent.vl_agent_v2 import VLAgentEnvV2
from .envs.visual_agent.vl_agent_v3 import VLAgentEnvV3

try:
    from .envs.visual_agent.mm_search_engine import MMSearchEngine
except Exception as err:
    print(f' [ERROR] Failed to register MMSearchEngine : {err=}')

try:
    from .envs.frozenlake.frozenlake import FrozenLakeTool
except Exception as err:
    print(f' [ERROR] Failed to register FrozenLakeTool : {err=}')

from .parallel_env import agent_rollout_loop
