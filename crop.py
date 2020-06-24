import subprocess
import os
import glob
import time
import pandas as pd
import statistics
import argparse
import yoloface
from multiprocessing import Pool


# first, we need to somehow create screenshots.
def create_snapshots(video, out_paths):
    # start = time.time()
    subprocess.run(
        # ['ionice', '-c2', '-n2',
        ['ffmpeg', "-y", "-loglevel", "quiet",
         '-i', video, '-vf', "select='not(mod(n\,500))'",
         '-vsync', 'vfr',
         # os.path.join(out_dir, '%dselect.png')],
         out_paths],
        check=True)
    # print('Creating snapshots took {} seconds'.format(time.time() - start))


def find_videos(parent_dir):
    # find all videos for that patient
    paths = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith(".avi") and 'out' not in file and 'au' not in file:
                paths.append(os.path.join(root, file))
    return paths


# then, we need to get the crop coordinates :)
def get_crop_coordinates(path):
    """
    :param path: Path to directory with snapshots
    :return: Crop coordinates union (largest superset)
    """
    # first, get coordinates for all snapshots
    # crop coordinates in [left, top, width, height] - typical cv syntax.
    faces = yoloface.get_face_coordinates(path)
    # next, get correct coordinates. We use median in case the face of another person was detected in one of the frames
    min_x = statistics.median([f[0] - 40 for f in faces])  # most left, extra margin
    min_y = statistics.median([f[1] - 40 for f in faces])  # most top, extra margin
    max_w = statistics.median([f[2] + 80 for f in faces])  # max width, with some extra margin
    max_h = statistics.median([f[3] + 80 for f in faces])  # mac height, with some extra margin
    return [min_x, min_y, max_w, max_h]


def delete_snapshots(path):
    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png"):
                paths.append(os.path.join(root, file))
    [os.remove(p) for p in paths]


def main(input_path, output_path):
    # # first, create appropriate folders
    base = os.path.basename(input_path)  # get filename only
    patient, session, vid = os.path.splitext(base)[0].split('_')  # get rid of .avi file extension

    # next, create screenshots. How do we name them?
    screen_link = os.path.join(output_path, '_'.join([patient, session, vid]) + '_%1d.png')
    create_snapshots(input_path, screen_link)
    coords = get_crop_coordinates(screen_link)
    # now, return
    return [vid, coords]


def test_crop(vid, crop_coords):
    resize_factor = 5
    x_min, y_min, width, height = crop_coords
    subprocess.run(
        ['ionice', '-c2', '-n2',
         'ffmpeg', "-y",  # "-loglevel", "quiet",
         '-i', vid, '-vf', "crop={0}:{1}:{2}:{3}, scale={4}*iw:{4}*ih".format(
            str(width), str(height), str(x_min), str(y_min), str(resize_factor)),
         '-c:a', 'copy',
         '/home/emil/CropFinder/outputs/test_crop.mp4'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--patient', help='path to config file')
    parser.add_argument('--output-path', type=str,
                        default='/home/emil/new_crop_coords/',
                        help='path to weights of model')
    # parser.add_argument('--input-path', help='path to input videos')
    args = parser.parse_args()
    patients = ['b541ad49', 'aa9da7b2', 'd49ed324', 'da3971ee', 'd7d5f068', 'abdb496b', 'aa97abcd',
                'be66b17c']  # will be more at some point
    video_root = '/nas/ecog_project/video/'
    for p in patients:
        # create output dir for this patient
        out_dir = os.path.join(args.output_path, p)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # find all sessions
        paths_to_sess = [g for g in glob.glob(os.path.join(video_root, p, '*')) if p in os.path.basename(g)]
        # for each sess, get all cropping coordinates (one set of coords per vid)
        for s in paths_to_sess:
            # check if hdf file already exists.
            fname = os.path.basename(s) + '.hdf'
            hdf_path = os.path.join(out_dir, fname)
            if os.path.isfile(hdf_path):
                continue
            start = time.time()
            # find .avi files corresponding to current sess
            paths = find_videos(s)
            pool = Pool(8)
            # Pool technicalities
            input = zip(paths, len(paths) * [out_dir])
            ret = pool.starmap(main, input)
            # create csv/hdf file:
            if len(ret) > 0:
                df = pd.DataFrame(data=ret, columns=['video', 'coordinates'])
                df.to_hdf(hdf_path, key='df')
            delete_snapshots(out_dir)
            print('current session took {} seconds'.format(time.time()-start))

            # test cut
            # test_crop(paths[0], df.loc[0, 'coordinates'])
