from alrd.utils.video_recorder import VideoRecorder
from gym.core import Env, Wrapper

class VideoRecordingWrapper(Wrapper):
    def __init__(self, env: Env, video_recorder: VideoRecorder):
        super().__init__(env)
        self.__video_recorder = video_recorder
        self.__started = False
    
    def reset(self, **kwargs):
        if self.__started:
            self.__video_recorder.stop_recording()
        obs, info = self.env.reset(**kwargs)
        self.__video_recorder.start_recording()
        self.__started = True
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.__video_recorder.stop_recording()
            self.__started = False
        return obs, reward, terminated, truncated, info
    
    def close(self):
        if self.__started:
            self.__video_recorder.stop_recording()
            self.__started = False