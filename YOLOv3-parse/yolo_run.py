from yolo import YOLO, detect_video, detect_img


if __name__ == '__main__':
    # 检测图片
    detect_img(YOLO(), 'test/person.jpg')

    # 检测视频
    # detect_video(YOLO(), 'test/test_video.mp4', 'test/test_video_out.mp4')

    # 检测camera
    # detect_video(YOLO(), video_path=0, output_path='test/test_video_out.mp4')
