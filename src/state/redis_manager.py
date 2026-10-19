import json
from typing import Optional
import redis
from src.config import settings
from src.logger import logger
from src.models.state import AgentState


class RedisStateManager:
    """Manages persistent serialization and deserialization of AgentState using Redis."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, db: Optional[int] = None):
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self.db = db if db is not None else settings.redis_db
        self._client: Optional[redis.Redis] = None
        self._fallback_memory: dict[str, str] = {}

    @property
    def client(self) -> Optional[redis.Redis]:
        if self._client is None:
            try:
                client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=5,
                )
                client.ping()
                self._client = client
                logger.info(f"State Manager: Successfully connected to Redis at {self.host}:{self.port}")
            except Exception as e:
                logger.warning(
                    f"State Manager: Could not connect to Redis at {self.host}:{self.port} ({e}). "
                    "Operating with memory fallback for offline resilience."
                )
                self._client = None
        return self._client

    def _state_key(self, session_id: str) -> str:
        return f"literaquest:session:{session_id}:state"

    def _step_key(self, session_id: str, step: str) -> str:
        return f"literaquest:session:{session_id}:step:{step}"

    def save_state(self, state: AgentState) -> None:
        """Serializes and saves the AgentState to Redis and records the step snapshot."""
        key = self._state_key(state.session_id)
        step_key = self._step_key(state.session_id, state.current_step)
        data = state.model_dump_json()

        try:
            if self.client:
                # 24 hour TTL for session states
                self.client.set(key, data, ex=86400)
                self.client.set(step_key, data, ex=86400)
                self.client.set("literaquest:latest_session", state.session_id, ex=86400)
                logger.info(
                    f"State Manager: Saved agent state to Redis under key '{key}' "
                    f"[Step: '{state.current_step}', Papers: {len(state.papers)}]"
                )
                return
        except Exception as e:
            logger.error(f"State Manager: Failed writing to Redis: {e}")

        # In-memory fallback
        self._fallback_memory[key] = data
        self._fallback_memory[step_key] = data
        self._fallback_memory["literaquest:latest_session"] = state.session_id
        logger.info(f"State Manager: Saved agent state in memory fallback [Step: '{state.current_step}']")

    def load_state(self, session_id: str) -> Optional[AgentState]:
        """Loads and deserializes the AgentState from Redis."""
        key = self._state_key(session_id)
        raw_data: Optional[str] = None

        try:
            if self.client:
                raw_data = self.client.get(key)
                if raw_data:
                    logger.info(f"State Manager: Loaded agent state from Redis for session '{session_id}'")
        except Exception as e:
            logger.error(f"State Manager: Failed reading from Redis: {e}")

        if not raw_data:
            raw_data = self._fallback_memory.get(key)
            if raw_data:
                logger.info(f"State Manager: Loaded agent state from memory fallback for session '{session_id}'")

        if raw_data:
            try:
                return AgentState.model_validate_json(raw_data)
            except Exception as e:
                logger.error(f"State Manager: Deserialization failed for session '{session_id}': {e}")
                return None

        logger.warning(f"State Manager: No state found for session '{session_id}'")
        return None

    def get_latest_session_id(self) -> Optional[str]:
        """Retrieves the most recent session ID recorded in Redis or memory."""
        try:
            if self.client:
                return self.client.get("literaquest:latest_session")
        except Exception:
            pass
        return self._fallback_memory.get("literaquest:latest_session")


redis_manager = RedisStateManager()
