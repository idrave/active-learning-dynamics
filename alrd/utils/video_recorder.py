from __future__ import annotations
import cv2
from threading import Event, Thread
from pathlib import Path
import time

class VideoRecorder:
    def __init__(self, output_dir: str | Path, webcam_index: int = 0):
        self.output_dir = Path(output_dir)
        self.webcam_index = webcam_index
        self.stop_event = Event()
        self.thread = None
        self.count = 0
        self.output_file = None
        self.camera = None
    
    def open(self):
        start = time.time()
        self.camera = cv2.VideoCapture(self.webcam_index)
        print('init camera took', time.time() - start)
    
    def close(self):
        self.camera.release()
        self.camera = None

    def record_video(self):
        print('recording video', self.output_file)
        video_writer = None
        while not self.stop_event.is_set():
            ret, frame = self.camera.read()
            if ret:
                # Initialize the video writer when the first frame is captured
                if video_writer is None:
                    print('video writer initializing')
                    frame_height, frame_width, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(self.output_file, fourcc, 20.0, (frame_width, frame_height))

                # Write the frame to the video file
                video_writer.write(frame)

        # Release the video capture and writer resources
        if video_writer is not None:
            video_writer.release()

    def start_recording(self):
        self.output_file = str(self.output_dir / f"{self.count:03d}.mp4")
        self.thread = Thread(target=self.record_video)
        self.thread.start()
        return self.output_file

    def stop_recording(self):
        self.stop_event.set()
        self.thread.join()
        self.thread = None
        self.stop_event.clear()
        self.count += 1

    def __enter__(self):
        self.open()
        self.start_recording()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()
        self.close()

# Example usage
if __name__ == '__main__':
    output = 'video'
    Path(output).mkdir(exist_ok=True)
    # Using the VideoRecorder as a context manager
    with VideoRecorder(output, 1) as video_recorder:
        # Wait for some time (e.g., 10 seconds)
        import time
        print('hi')
        time.sleep(5)
        print('ended')

    # Recording automatically stops upon exiting the context
