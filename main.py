from vidaug import augmentors as va
import cv2 as cv
from PIL import Image, ImageSequence
import os
import argparse
from pathlib import Path
import random

parser = argparse.ArgumentParser()
parser.add_argument('--frames_dir_path', default=None, type=Path, help='Directory path of the video frames produced by '
                                                                       'the generate_video_jpgs.py script')

parser.add_argument('--output_augmented_path', default=None, type=Path,
                    help='Directory path of the augmented video frames')

parser.add_argument('--seed', default=5, type=int,
                    help='Seed for "random" filters')


def load_frames(path):
    frames = []
    for frame_image in os.listdir(path):
        img = cv.imread(os.path.join(path, frame_image))
        frames.append(img)

    return frames


def load_video(video_path):
    vidcap = cv.VideoCapture(video_path)
    frames = []

    success, image = vidcap.read()
    while success:
        success, image = vidcap.read()
        frames.append(image)

    return frames


def gif_loader(path, modality="RGB"):
    frames = []
    with open(path, 'rb') as f:
        with Image.open(f) as video:
            index = 1
            for frame in ImageSequence.Iterator(video):
                frames.append(frame.convert(modality))
                index += 1
        return frames


if __name__ == '__main__':
    args = parser.parse_args()

    random.seed(args.seed)

    # Filters that makes sense
    allFilters = va.AllOf(
        [
            va.RandomResize(scaling_factor=1.1),
            va.HorizontalFlip(),
            va.GaussianBlur(sigma=1),
            va.ElasticTransformation(),
            va.Add(value=10),
            va.Multiply(value=2),
            va.Salt(),
            va.Pepper()
        ]
    )

    frames_dir_path = args.frames_dir_path
    output_augmented_path = args.output_augmented_path

    target_classes = os.listdir(frames_dir_path)

    for target_class in target_classes:
        target_class_path = os.path.join(frames_dir_path, target_class)
        video_names = os.listdir(target_class_path)

        for video_name in video_names:
            video_path = os.path.join(target_class_path, video_name)

            filters_to_actually_calculate = []
            for filterIdx, augFilter in enumerate(allFilters.transforms):
                output_path = os.path.join(output_augmented_path, target_class, video_name, str(augFilter))

                if os.path.exists(output_path):
                    print("Path {} already exists. Skipping...".format(output_path))
                else:
                    filters_to_actually_calculate.append(augFilter)

            localFilters = va.AllOf(filters_to_actually_calculate)

            if not localFilters.transforms:
                print("No filters to apply")
            else:
                frames = load_frames(video_path)
                augmented_frames = localFilters(frames)

                idx = 0
                while idx < len(augmented_frames):
                    filter_name = str(localFilters.transforms[idx])
                    augmented_frames_for_filter = augmented_frames[idx]

                    output_path = os.path.join(output_augmented_path, target_class, video_name, filter_name)
                    if os.path.exists(output_path):
                        print("Path {} already exists. Skipping...".format(output_path))
                    else:
                        os.makedirs(output_path, 0o755)

                        print("Writing augmented frames in directory {}".format(output_path))

                        for frame_idx, augmented_frame in enumerate(augmented_frames_for_filter):
                            output_image = Image.fromarray(augmented_frame)
                            output_image.save(os.path.join(output_path, "image_{}.jpg".format(str(frame_idx).zfill(5))))

                    idx += 1


    # loaded_video = load_frames("/mnt/external-drive/datasets/hollywood2/Hollywood2/frames/running/actioncliptest00005/")
    # augmented_frames = seq(loaded_video)
    #
    # for augmented_frame in augmented_frames:
    #     output_image = Image.fromarray(augmented_frame)
    #     output_image.save("image_%05d.jpg")
    # augmented_frames[0].save("/home/eugenio/Desktop/relazione_network/out.gif", save_all=True,
    #                          append_images=augmented_frames[1:], duration=100, loop=0)
