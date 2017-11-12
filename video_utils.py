import numpy as np
import cv2
import camera_calibration
import image_processing
import lane_detection
import curvature_estimatio
import time
import pipeline
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip


input_video_name = 'project_video.mp4'
output_video_name = 'output.mp4'
def process_video(input_video_name, output_video_name):
    print(output_video_name)
    # input_video = VideoClip(input_video_name)
    input_video = VideoFileClip(input_video_name)
    output_video = input_video.fl_image(pipeline.video_pipeline)
    output_video.write_videofile(output_video_name, audio=False)

process_video(input_video_name, output_video_name)