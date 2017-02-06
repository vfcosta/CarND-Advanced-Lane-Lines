from moviepy.editor import VideoFileClip
import lanelines.pipeline as pipeline


def process_frame(image):
    lines_image, line_left, line_right, offset, lines_original, top_down = pipeline.process_image(image)
    return lines_original


def process_video(filename):
    white_output = '../'+filename+'_processed.mp4'
    clip1 = VideoFileClip('../'+filename+'.mp4')
    white_clip = clip1.fl_image(process_frame)
    white_clip.write_videofile(white_output, audio=False)


def execute():
    process_video('project_video')

if __name__ == '__main__':
    execute()
