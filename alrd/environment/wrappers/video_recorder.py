from __future__ import annotations
from alrd.utils.video_recorder import VideoRecorder
from gym.core import Env, Wrapper
from pathlib import Path
import time

_DEFAULT_SLEEP = 1.5

class VideoRecordingWrapper(Wrapper):
    def __init__(self, env: Env, output_dir: str | Path, webcam_index: int = 0, sleep: float = _DEFAULT_SLEEP):
        super().__init__(env)
        self.__video_recorder = VideoRecorder(output_dir, webcam_index)
        self.__video_recorder.open()
        self.__started = False
        self.__sleep = sleep
    
    def __start_recording(self):
        self.__video_recorder.start_recording()
        self.__started = True

    def __stop_recording(self):
        time.sleep(self.__sleep)
        self.__video_recorder.stop_recording()
        self.__started = False

    def reset(self, **kwargs):
        if self.__started:
            self.__stop_recording()
        obs, info = super().reset(**kwargs)
        self.__start_recording()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if (terminated or truncated) and self.__started:
            self.__stop_recording()
        return obs, reward, terminated, truncated, info
    
    def close(self):
        if self.__started:
            self.__stop_recording()
        super().close()
        self.__video_recorder.close()