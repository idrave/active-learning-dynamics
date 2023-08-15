from __future__ import annotations
from alrd.utils.video_recorder import VideoRecorder
from gym.core import Env, Wrapper
from pathlib import Path
import time

_DEFAULT_SLEEP = 1.5

class VideoRecordingWrapper(Wrapper):
    def __init__(self, env: Env, output_dir: str | Path, webcam_index: int = 0, sleep: float = _DEFAULT_SLEEP):
        super().__init__(env)
        self.__output_dir = Path(output_dir)
        self.__video_recorder = VideoRecorder(webcam_index)
        self.__video_recorder.open()
        self.__started = False
        self.__sleep = sleep
        self.__counter = 0
    
    def __start_recording(self, tag):
        name = f'{self.__counter:03d}.mp4'
        if tag is not None:
            name = f'{tag}-{name}'
        self.__video_recorder.start_recording(self.__output_dir / name)
        self.__started = True

    def __stop_recording(self):
        time.sleep(self.__sleep)
        self.__video_recorder.stop_recording()
        self.__started = False
        self.__counter += 1

    def reset(self, options: dict | None = None, **kwargs):
        if self.__started:
            self.__stop_recording()
        tag = None
        if options is not None:
            tag = options.get('tag', None)
        obs, info = super().reset(options=options, **kwargs)
        self.__start_recording(tag)
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