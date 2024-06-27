import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from tqdm import tqdm

blue = (26, 128, 187)
orange = (234, 128, 28)

EDGE_COLORS = {
    (0, 1): blue,
    (0, 2): orange,
    (1, 3): blue,
    (2, 4): orange,
    (0, 5): blue,
    (0, 6): orange,
    (5, 7): blue,
    (7, 9): blue,
    (6, 8): orange,
    (8, 10): orange,
    (5, 6): blue,
    (5, 11): blue,
    (6, 12): orange,
    (11, 12): orange,
    (11, 13): blue,
    (13, 15): blue,
    (12, 14): orange,
    (14, 16): orange
}


def movenet(input_image, interpreter):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Output:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def draw_keypoints(frame, keypoints, threshold):
    """Draws the keypoints on a image frame
    
    Args:
    frame: np.array, image frame.
    keypoints: np.array, keypoints detected by the model.
    threshold: float, threshold for the keypoints.

    Output: frame with the keypoints drawn.

    """
    # Get the shape of the frame
    x, y, _ = frame.shape
    # Denormalize the coordinates of the keypoints to fit orginal frame
    denormalized_coordinates = np.squeeze(np.multiply(keypoints, [x,y,1]))
    # Iterate over the keypoints
    for keypoint in denormalized_coordinates:
        # Unpack the keypoint values
        keypoint_y, keypoint_x, keypoint_confidence = keypoint
        if keypoint_confidence > threshold:
            # Draw the keypoints
            cv2.circle(
                img=frame, 
                center=(int(keypoint_x), int(keypoint_y)), 
                radius=2, 
                color=(255,0,0),
                thickness=-1
            )
    return denormalized_coordinates

def draw_edges(denormalized_coordinates, frame, edges_colors, threshold):
    """Draws the edges between the keypoints

    Outpit: frame with the edges drawn.

    Args:
    denormalized_coordinates: np.array, coordinates of the keypoints.
    frame: np.array, image frame.
    edges_colors: dict, colors of the edges.
    threshold: float, threshold for the keypoints.

    """
    # Iterate over the edges
    for edge, color in edges_colors.items():
        # Get the dict value associated to the actual edge
        p1, p2 = edge
        # Get the points
        y1, x1, confidence_1 = denormalized_coordinates[p1]
        y2, x2, confidence_2 = denormalized_coordinates[p2]
        # Draw the line from point 1 to point 2, the confidence > threshold
        if (confidence_1 > threshold) & (confidence_2 > threshold):      
            cv2.line(
                img=frame, 
                pt1=(int(x1), int(y1)),
                pt2=(int(x2), int(y2)), 
                color=color, 
                thickness=1, 
                lineType=cv2.LINE_AA
            )

### ----------------------------------------------------------------------------- ###

## Run inference on a video
def run_inference(video_path, model_path, input_size, threshold, output_path):
    """
    Run inference of 17 joints of a person on a video file using a MoveNet model.
    Saves the output video with the keypoints and edges drawn.

    Args:
    video_path: str, path to the video file.
    model_path: str, path to the tflite model.
    input_size: int, size of the input image.
    threshold: float, threshold for the keypoints.
    output_path: str, path to save the output video.
    """
    # Create a list to store the output frames and the initial shape of the video
    output_frames = []
    initial_shape = []

    # Capture the video
    capture = cv2.VideoCapture(video_path)
    
    # Get video parameters
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    initial_shape.append(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    initial_shape.append(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Display parameter
    print(f"Frame count: {frame_count}")    
    print(f"Width: {initial_shape[0]}, Height: {initial_shape[1]}")
    bar = tqdm(total=frame_count, desc = "Processing frames", unit = "frames")

    # Load the model
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # Iterate over the frames
    while True:
        ret, frame = capture.read()
        if frame is None: 
            break        

        # Copy the frame and convert it to RGB
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image
        image = cv2.resize(image, (input_size,input_size))
        # Resize to the target shape and cast to an int32 vector
        input_image = tf.cast(tf.image.resize_with_pad(image, input_size, input_size), dtype=tf.int32)
        # Create a batch (input tensor)
        input_image = tf.expand_dims(input_image, axis=0)
        
        # Perform inference
        keypoints = movenet(input_image, interpreter)
        
        # Iterate over the keypoints
        for instance in keypoints: 
            # Draw the keypoints
            denormalized_coordinates = draw_keypoints(frame, instance, threshold)
            # Draw the edges
            draw_edges(denormalized_coordinates, frame, EDGE_COLORS, threshold)

        # Save the output frames as a video
        output_frames.append(frame)
        # Update the progress bar
        bar.update(1)
        
    # Save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (initial_shape[0], initial_shape[1]))
    for frame_p in output_frames:
        out.write(frame_p)

    print(f"Output video saved at {output_path}")
    capture.release()
    out.release()
    bar.close()



model_folder_path = 'MoveNet/Models/tflite'
models = [
    "movenet_lightning_f16.tflite",
    "movenet_lightning_int8.tflite",
    "movenet_thunder_f16.tflite",
    "movenet_thunder_int8.tflite",
]

output_frames = run_inference(
    video_path='./EMU_videos/p123-19.mp4',
    model_path=f"{model_folder_path}/{models[0]}",
    input_size = 192,
    threshold=0.11,
    output_path='testmp4.mp4'
)