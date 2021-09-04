from voice_recognition.command_builder import CommandBuilder


def test_build_command():
    texts = ["start command", "random test text", "that has a command", "end command"]

    expected_command = "random test text that has a command"

    cb = CommandBuilder("start command", "end command")
    command = cb.get_commands_in_texts(texts)

    assert command == [expected_command]
