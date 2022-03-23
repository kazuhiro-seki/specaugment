import os, sys, re
import glob
import argparse
from pathlib import Path

import librosa
import spec_augment_tensorflow
import soundfile as sf

from multiprocessing import Pool
import multiprocessing as multi

parser = argparse.ArgumentParser(description='SpecAugment')
parser.add_argument('--output', default='augment',
                    help='output dir')
parser.add_argument('--time-warp-para', default=10,
                    help='time warp parameter W')
parser.add_argument('--frequency-mask-para', default=10,
                    help='frequency mask parameter F')
parser.add_argument('--time-mask-para', default=10,
                    help='time mask parameter T')
parser.add_argument('--masking-line-number', default=1,
                    help='masking line number')
parser.add_argument('--cpu-count', default=10,
                    help='cpu count (set -1 for all available cpu)')
parser.add_argument('--force', default=False, 
                    action='store_true',
                    help='overwrite converted files even if existed')

args = parser.parse_args()
time_warping_para = args.time_warp_para
time_masking_para = args.frequency_mask_para
frequency_masking_para = args.time_mask_para
masking_line_number = args.masking_line_number
output_dir = args.output

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def convert(audio_path):

    save_path = os.path.basename(audio_path)
    save_path = os.path.join(output_dir, save_path)
    started = save_path + '.started'

    Path(started).touch()

    print(f'Processing {audio_path}...', flush=True)

    # Step 0 : load audio file, extract mel spectrogram

    audio, sampling_rate = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=256, hop_length=128, fmax=8000)

    warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=mel_spectrogram, time_warping_para=time_warping_para, frequency_masking_para=frequency_masking_para, time_masking_para=time_masking_para, frequency_mask_num=masking_line_number, time_mask_num=masking_line_number)
    sampling_rate = 16000 
    data=librosa.feature.inverse.mel_to_audio(warped_masked_spectrogram, sr=sampling_rate, hop_length=128)
    sf.write(save_path, data, sampling_rate, subtype='PCM_16')

    # delete lock file
    os.remove(started)

if __name__ == "__main__":

    paths = []
    for audio_path in sys.stdin:

        audio_path = audio_path.rstrip()

        if not audio_path.endswith('.flac'):
            continue
        if not os.path.exists(audio_path):
            continue

        if not args.force:

            save_path = os.path.basename(audio_path)
            save_path = os.path.join(output_dir, save_path)
            started = save_path + '.started'

            if os.path.exists(started) or os.path.exists(save_path):
                continue

        paths.append(audio_path)

    #print(paths)

    # number of cpu cores
    if args.cpu_count == -1:
        p = Pool(multi.cpu_count())
    else:
        p = Pool(args.cpu_count)

    # convert in parallel
    p.map(convert, paths)
    p.close()
