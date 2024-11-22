import logging
from typing import List, Dict, Any, Mapping

LOGGER = logging.getLogger("MINT")


class BaseLM:
    """Base class for an agent."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        LOGGER.info(f"Initialized {self.__class__.__name__} with config: {config}")
        self.stop_words = ["\n---"]

    def add_system_message(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        # Prepend the prompt with the system message
        first_msg = messages[0]
        assert first_msg["role"] == "user"
        system, examples, task = first_msg["content"].split("\n---\n")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": examples + "\n---\n" + task},
        ] + messages[1:]
        return messages