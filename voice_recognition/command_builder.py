import re
from typing import List, Optional


class CommandBuilder:
    def __init__(self, start_kw: str, end_kw: str) -> None:
        self._start_kw = start_kw
        self._end_kw = end_kw

    def get_commands_in_texts(self, texts: List[str]) -> List[str]:
        """Get all commands in texts.

        A "command" is the content between the start_kw and the end_kw.
        All commands found will be returned in a list.

        Args:
            texts: Texts to build command from

        Returns:
            List[str]: All commands in texts
        """

        full_text = " ".join(texts)

        # find commands start and end
        full_text = full_text.replace(self._start_kw, "<<<")
        full_text = full_text.replace(self._end_kw, ">>>")

        # find all commands using regex
        regex = re.compile("<<<(.*)>>>")

        commands = regex.findall(full_text)

        commands = [self._process_cmd(cmd) for cmd in commands]

        return commands

    def _process_cmd(self, cmd: str) -> str:
        """Process command string.

        Args:
            cmd: Command string to process

        Returns:
            str: Processed command
        """

        return cmd.strip()
