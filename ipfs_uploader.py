"""
A script which splits the video according to the max video size
limit and uploads the files to the IPFS.
"""
from math import ceil
from os import listdir, remove, makedirs, path
from os.path import join, getsize, exists
from shutil import move
from time import sleep

import ipfshttpclient
from moviepy.video.io.VideoFileClip import VideoFileClip

from config import Config


def get_next_interval(duration, num_splits):
    """
    :param duration: length of the video
    :param num_splits: number of segments to split the video into
    :return: the generator returns a pair of floats marking
             the beginning and the end of the video segment
    """
    segment_duration = duration / num_splits
    for i in range(num_splits):
        yield i * segment_duration, (i + 1) * segment_duration


if __name__ == '__main__':
    # Load configuration
    conf = Config()
    ipfs_client = ipfshttpclient.connect(conf.IPFS_DAEMON_ADDRESS)

    # 1) Make sure the dir to which the uploaded videos will be moved exists
    uploaded_dir = join(conf.VIDEO_DIR, 'uploaded')
    if not exists(uploaded_dir):
        makedirs(uploaded_dir)

    while True:
        try:
            # 2) Iterate through files
            for file_ in listdir(conf.VIDEO_DIR):
                if file_.endswith('mp4'):
                    files_to_upload = []

                    # 3) Get video size
                    video_path = join(conf.VIDEO_DIR, file_)
                    size_in_bytes = getsize(video_path)
                    if size_in_bytes > conf.MAX_TRANSACTION_SIZE:
                        # 4) If the size exceeds the transaction limit split it into chunks
                        num_splits = ceil(size_in_bytes / conf.MAX_TRANSACTION_SIZE)
                        video = VideoFileClip(video_path)
                        for i, (t_start, t_end) in enumerate(get_next_interval(video.duration, num_splits)):
                            video_clip_path = join(conf.VIDEO_DIR, f'{file_[0:-4]}_part{i}.mp4')
                            video_clip = video.subclip(t_start, t_end)
                            video_clip.write_videofile(video_clip_path, codec='libx265')
                            files_to_upload.append(video_clip_path)
                    else:
                        files_to_upload.append(video_path)

                    # 4) Upload the video and after doing so move it to the uploaded folder
                    for file_to_upload in files_to_upload:

                        print(f'Uploading {path.basename(file_to_upload)}...')

                        # upload video to IPFS
                        res = ipfs_client.add(file_to_upload)

                        # outputs IPFS hash (=hash of file content)
                        ipfs_hash = res['Hash']
                        print(f'Hash: {ipfs_hash}\n')

                        with open(path.join(conf.VIDEO_DIR, "uploaded_hashes.txt"), "a") as myfile:
                            myfile.write(path.basename(file_to_upload) + ', ' + ipfs_hash + '\n')

                        if 'part' in file_to_upload:
                            print(f'Deleting video part {file_to_upload}.')
                            remove(file_to_upload)
                            to_move = f'{file_to_upload[0:-10]}.mp4'
                            if exists(to_move):
                                move(to_move, uploaded_dir)
                        else:
                            move(file_to_upload, uploaded_dir)

            print(f'Going to sleep for {conf.SLEEP_INTERVAL} seconds')
            sleep(conf.SLEEP_INTERVAL)
        except KeyboardInterrupt:
            print('Exiting')
            break
