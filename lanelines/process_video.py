from moviepy.editor import VideoFileClip
import os.path
import cv2
from lanelines.pipeline import Pipeline

base_dir = os.path.dirname(__file__)
pipeline = Pipeline()


def process_frame(image):
    """Use pipeline to process a single image frame and return an image with lane lines drawn on top"""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert from RGB to BGR
    lines_image, line_left, line_right, lines_original = pipeline.process_image(image)
    return cv2.cvtColor(lines_original, cv2.COLOR_BGR2RGB)


def process_video(filename):
    """Process a video using the lane finding pipeline."""
    white_output = os.path.join(base_dir, '..', filename+'_processed.mp4')
    clip1 = VideoFileClip(os.path.join(base_dir, '..', filename+'.mp4'))
    white_clip = clip1.fl_image(process_frame)
    white_clip.write_videofile(white_output, audio=False)


if __name__ == '__main__':
    process_video('project_video')
    process_video('challenge_video')
    process_video('harder_challenge_video')
