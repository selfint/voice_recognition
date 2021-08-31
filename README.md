# voice interface

## flow

1. sox generates files of a set duration (1 or 2 seconds)

2. using inotify, python receives the newest file

3. the deepspeech model is then run on the file, in a separate thread

4. the interpreted text from the audio file is pushed to a PriorityQueue

5. another python script reads from the PriorityQueue

The result is a queue the outputs the words the user is saying, in real
time