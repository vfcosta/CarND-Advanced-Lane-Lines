from moviepy.editor import VideoFileClip
from lanelines.pipeline import Pipeline
import os.path

base_dir = os.path.dirname(__file__)
pipeline = Pipeline()


def process_frame(image):
    lines_image, line_left, line_right, lines_original = pipeline.process_image(image)
    return lines_original


def process_video(filename):
    white_output = os.path.join(base_dir, '..', filename+'_processed.mp4')
    clip1 = VideoFileClip(os.path.join(base_dir, '..', filename+'.mp4'))
    white_clip = clip1.fl_image(process_frame)
    white_clip.write_videofile(white_output, audio=False)


if __name__ == '__main__':
    process_video('project_video')
    process_video('challenge_video')
    process_video('harder_challenge_video')
