from moviepy.editor import VideoFileClip
from lanelines.camera import Camera
from lanelines.detector import Detector
from lanelines.binarizer import Binarizer


camera = Camera.load()
binarizer = Binarizer()
detector = Detector(camera)


def process_image(image):
    lines_image, line_left, line_right, offset, lines_original = detector.detect(image)
    return lines_original


def process_video(filename):
    white_output = '../'+filename+'_processed.mp4'
    clip1 = VideoFileClip('../'+filename+'.mp4')
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)


def execute():
    process_video('project_video')

if __name__ == '__main__':
    execute()
