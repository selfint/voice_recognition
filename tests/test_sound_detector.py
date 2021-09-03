import numpy as np
from pydub.utils import ratio_to_db

from voice_recognition.sound_detector import SoundDetector


def test_detect_silence():
    # we are using a bit-depth of 16
    max_audio_strength = 2 ** 15
    one_second_bytes = SoundDetector.ms_to_byte_index(1000, 16000, 16)
    quarter_second_bytes = SoundDetector.ms_to_byte_index(250, 16000, 16)
    half_second_bytes = SoundDetector.ms_to_byte_index(500, 16000, 16)

    sound = ratio_to_db(0.5) * max_audio_strength
    silence = ratio_to_db(0.0) * max_audio_strength
    audio = np.array(
          [sound]   * one_second_bytes      # 0.00 -- 1.00
        + [silence] * quarter_second_bytes  # 1.00 -- 1.25
        + [sound]   * one_second_bytes      # 1.25 -- 2.25
        + [silence] * half_second_bytes     # 2.25 -- 2.75
        + [sound]   * one_second_bytes      # 2.75 -- 3.75
    )

    sd = SoundDetector(
        silence_threshold=-16,
        min_silence_length=250
    )

    extracted_parts = sd.detect_sound(audio, keep_silence=True)
    assert len(extracted_parts) == 3

    extracted_start_0 = extracted_parts[0][0]
    extracted_end_0 = extracted_parts[0][1]
    assert extracted_start_0 == 0
    assert one_second_bytes < extracted_end_0 < one_second_bytes * 1.25

    extracted_start_1 = extracted_parts[1][0]
    extracted_end_1 = extracted_parts[1][1]
    assert one_second_bytes < extracted_start_1 < one_second_bytes * 2.25
    assert one_second_bytes * 2.25 < extracted_end_1 < one_second_bytes * 2.75

    extracted_start_2 = extracted_parts[2][0]
    extracted_end_2 = extracted_parts[2][1]
    assert one_second_bytes * 2.25 < extracted_start_2 < one_second_bytes * 2.75
    assert extracted_end_2 == one_second_bytes * 3.75
