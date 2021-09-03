from typing import List
import numpy as np
from pydub.audio_segment import AudioSegment
from pydub.silence import detect_nonsilent


class SoundDetector:
    def __init__(self, silence_threshold: int, min_silence_length: int) -> None:
        self._silence_threshold = silence_threshold
        self._min_silence_length = min_silence_length
        self._min_silence_length_bytes = self.ms_to_byte_index(
            min_silence_length,
            16000,
            16,
        )

    def detect_sound(
        self,
        audio: np.ndarray,
        keep_silence: bool = True
    ) -> List[np.ndarray]:
        """Get all parts in audio that are surrounded by silence.

        Detect silence when a series longer than the ``min_silence_length`` of
        signals is below the ``silence_threshold``.

        Args:
            audio: Raw audio to split by silence
            keep_silence: Whether to keep silence around audio or not

        Returns:
            List[np.ndarray]: Audio slices split by silence
        """

        segment = AudioSegment(
            audio.tobytes(),
            frame_rate=16000,
            channels=1,
            sample_width=2,
        )
        nonsilences: List[List[int]] = detect_nonsilent(
            audio_segment=segment,
            min_silence_len=self._min_silence_length,
            silence_thresh=self._silence_threshold,
        )

        parts = []
        for nonsilence_start, nonsilence_end in nonsilences:
            byte_start = self.ms_to_byte_index(nonsilence_start, 16000, 16)
            byte_end = self.ms_to_byte_index(nonsilence_end, 16000, 16)

            if keep_silence:
                byte_start = max(0, byte_start - self._min_silence_length_bytes)
                byte_end = min(
                    len(audio),
                    byte_end + self._min_silence_length_bytes
                )

            parts.append((byte_start, byte_end))

        return parts

    @staticmethod
    def ms_to_byte_index(ms: int, sample_rate: int, bit_depth: int) -> int:
        """Convert from millisecond to byte index in a (mono channel) audio buffer.

        The sample rate is the amount of audio samples taken per second.
        The bit depth is the bit size of each audio sample.

        So to convert from a millisecond to a byte index, we do the following:

            byte_index = ms * samples_per_ms * bytes_per_sample

        For example, if the sample rate is 16k, and the bit-depth is 16:
        - Each sample represents 1/16000 of a second. Or in other words,
          there are 16000 samples per second. 
        - Each sample is 16 / 8 = 2 bytes.

        So to convert from a millisecond to a byte_index we do the following:
        byte_index = ms * samples_per_ms * bytes_per_sample
        V
        byte_index = ms * 16 * 2

        So the first second's (1000 ms) byte index is:
        byte_index = 1000 * 16 * 2 = 32000

        Args:
            ms: Millisecond of bit index
            sample_rate: Sample rate of audio
            bit_depth: Bit depth of audio

        Returns:
            int: Bit index of millisecond
        """

        sample_rate_per_ms = sample_rate / 1000
        bytes_per_sample = int(bit_depth / 8)
        byte_index = ms * sample_rate_per_ms * bytes_per_sample

        # TODO: do we need to use int?
        # TODO: why do we need to divide by 8?
        byte_index = int(ms * sample_rate_per_ms * bytes_per_sample / 8)

        return byte_index


