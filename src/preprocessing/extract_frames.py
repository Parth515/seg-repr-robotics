import rclpy
from cv_bridge import CvBridge
import cv2
import os
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def extract_images_from_bag(bag_path, topic_name='/camera/image_raw', output_dir='data/interim/robot_frames'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize reader and storage options (supports mcap and sqlite3)
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    # Map topics to types for deserialization
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    bridge = CvBridge()
    count = 0

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == topic_name:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            
            # Convert ROS Image to OpenCV BGR image
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Save image
            filename = os.path.join(output_dir, f"frame_{count:05d}.png")
            cv2.imwrite(filename, cv_img)
            count += 1
    
    print(f"Extraction complete. {count} images saved to: {output_dir}")

def extract_from_video(video_path, output_dir='video_frames'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every frame as a JPEG
        filename = os.path.join(output_dir, f"frame_{count:05d}.png")
        cv2.imwrite(filename, frame)
        count += 1
    
    cap.release()
    print(f"Done. Extracted {count} frames to: {output_dir}")


if __name__ == '__main__':
    # REPLACE with your bag folder path and topic name
    extract_images_from_bag('rosbag2_2025_11_14-10_52_36', topic_name='/camera/camera/color/image_raw')

    # REPLACE with your video file name
    # extract_from_video('input_video.mp4')

