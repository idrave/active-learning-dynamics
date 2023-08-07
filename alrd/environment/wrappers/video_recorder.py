from __future__ import annotations
from alrd.utils.video_recorder import VideoRecorder
from gym.core import Env, Wrapper
from pathlib import Path

class VideoRecordingWrapper(Wrapper):
    def __init__(self, env: Env, output_dir: str | Path, webcam_index: int = 0):
        super().__init__(env)
        self.__video_recorder = VideoRecorder(output_dir, webcam_index)
        self.__video_recorder.open()
        self.__started = False
    
    def reset(self, **kwargs):
        if self.__started:
            self.__video_recorder.stop_recording()
        obs, info = super().reset(**kwargs)
        self.__video_recorder.start_recording()
        self.__started = True
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            self.__video_recorder.stop_recording()
            self.__started = False
        return obs, reward, terminated, truncated, info
    
    def close(self):
        if self.__started:
            self.__video_recorder.stop_recording()
            self.__started = False
        super().close()
        self.__video_recorder.close()