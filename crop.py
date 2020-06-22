import sys
import glob
import subprocess
import os
import time
import random
import sys
# sys.path.append('..')
from yoloface import yoloface


# first, we need to somehow create screenshots.
def create_snapshots(video, out_paths):
    start = time.time()
    subprocess.run(
        # ['ionice', '-c2', '-n2',
        ['ffmpeg', "-y", "-loglevel", "quiet",
         '-i', video, '-vf', "select='not(mod(n\,500))'",
         '-vsync', 'vfr',
         #os.path.join(out_dir, '%dselect.png')],
         out_paths],
         check=True)
    print('Method 2 took {} time'.format(time.time() - start))

# then, we need to get the crop coordinates :)
def get_crop_coordinates(path):
    """
    :param path: Path to directory with snapshots
    :return: Crop coordinates per snapshot
    """
    faces = yoloface.get_face_coordinates(['--image', path])
    for face in faces:
        print(face)


if __name__ == '__main__':
    video = '/nas/ecog_project/video/cb46fd46/cb46fd46_5/cb46fd46_5_0123.avi'
    output = '/home/emil/CropFinder/outputs/%1d.png'
    if os.path.exists(video):
        create_snapshots(video, output)
    else:
        print('lel')

    # next, get crop coordinates
    coords = get_crop_coordinates(output)
