import os
import cv2 as cv
import numpy as np
import argparse
import sys
import tensorflow as tf
from imutils.video import FPS
from pathlib import Path
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import tf_ops_assist as utils_ops


file_path = Path.cwd()
out_file_path = Path(file_path / "test_out/")
detection_graph = tf.Graph()


def choose_run_mode():
    """
    choose image or video or webcam as input
    """
    global out_file_path
    if args.image:
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.image)

        out_file_path = str(out_file_path / (args.image[:-4] + '_out.jpg'))
    elif args.video:
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        out_file_path = str(out_file_path / (args.video[:-4] + '_out.mp4'))
    else:
        # Webcam input
        cap = cv.VideoCapture(0)
        out_file_path = str(out_file_path / 'webcam_out.mp4')
    return cap


def get_category():
    """
    Load names of coco classes
    """
    # List of the strings that is used to add correct label for each box.
    label_file = str(file_path / 'model/mscoco_label_map.pbtxt')
    num_classes = 90

    label_map = label_map_util.load_labelmap(label_file)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def load_pretrain_model():
    # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    # 可在上述网址下载其它tensorflow预训练模型
    graph_file = str(file_path / 'model/mask_rcnn_inception_v2_coco.pb')
    # graph_file = str(file_path / 'model/ssd_resnet_50_fpn_coco.pb')
    # graph_file = str(file_path / 'model/ssd_mobilenet_v2_coco.pb')
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


def set_video_writer(cap, write_fps=25):
    """
    Get the video writer initialized to save the output video
    """
    vid_writer = cv.VideoWriter(out_file_path, cv.VideoWriter_fourcc(*'mp4v'), write_fps,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                 round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    return vid_writer


def show_fps(frame):
    """
    show the real-time FPS
    """
    if not args.image:
        fps.update()
        fps.stop()
        fps_label = "FPS: {:.2f}".format(fps.fps())
        cv.putText(frame, fps_label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def run_inference_for_single_image(image, sess):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    # 假如读取的是带有mask输出的模型
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


if __name__ == '__main__':
    # Usage example:  python object_detection_tf_API.py --video=run.mp4
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--video', help='Path to video file.')
    args = parser.parse_args()

    cap = choose_run_mode()
    classes = get_category()
    load_pretrain_model()
    category_idx = get_category()

    fps = FPS().start()
    video_writer = set_video_writer(cap)

    with tf.Session(graph=detection_graph) as sess:
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print("Output file is stored as ", out_file_path)
                cv.waitKey(3000)
                break

            # Actual detection.
            output_dict = run_inference_for_single_image(frame, sess)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_idx,
                instance_masks=output_dict.get('detection_masks'),
                min_score_thresh=0.5,
                use_normalized_coordinates=True,
                line_thickness=4)
            show_fps(frame)
            # Write the frame with the detection boxes
            if args.image:
                cv.imwrite(out_file_path, frame)
            else:
                video_writer.write(frame)
            cv.imshow('Tensorflow API object detection', frame)
        if not args.image:
            video_writer.release()
            cap.release()
        cv.destroyAllWindows()