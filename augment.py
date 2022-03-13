import os, sys, re
import glob
import argparse

import librosa
import spec_augment_tensorflow
import soundfile as sf

parser = argparse.ArgumentParser(description='Spec Augment')
parser.add_argument('--time-warp-para', default=80,
                    help='time warp parameter W')
parser.add_argument('--frequency-mask-para', default=100,
                    help='frequency mask parameter F')
parser.add_argument('--time-mask-para', default=27,
                    help='time mask parameter T')
parser.add_argument('--masking-line-number', default=1,
                    help='masking line number')

args = parser.parse_args()
time_warping_para = args.time_warp_para
time_masking_para = args.frequency_mask_para
frequency_masking_para = args.time_mask_para
masking_line_number = args.masking_line_number

output_dir = 'augment'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if __name__ == "__main__":

    for audio_path in sys.stdin:

        audio_path = audio_path.rstrip()

        if not audio_path.endswith('.flac'):
            continue
        if not os.path.exists(audio_path):
            continue
        
        print(audio_path, flush=True)

        save_path = os.path.basename(audio_path)
        save_path = os.path.join(output_dir, save_path)

        # Step 0 : load audio file, extract mel spectrogram

        audio, sampling_rate = librosa.load(audio_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=256, hop_length=128, fmax=8000)

        # reshape spectrogram shape to [batch_size, time, frequency, 1]
        #shape = mel_spectrogram.shape
        #mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))

        warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=mel_spectrogram, time_warping_para=80, frequency_masking_para=27, time_masking_para=100, frequency_mask_num=1, time_mask_num=1)
        data=librosa.feature.inverse.mel_to_audio(warped_masked_spectrogram, sr=sampling_rate, hop_length=128)
        #sampling_rate = 16000 
        sf.write(save_path, data, sampling_rate, subtype='PCM_16')
