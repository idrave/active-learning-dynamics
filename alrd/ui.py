import time
import keyboard

class KeyboardListener:
    """
    Note: keyboard package requires root access on linux
    """
    def __init__(self):
        pass
    
    def which_pressed(self, keys):
        pressed = []
        for k in keys:
            if keyboard.is_pressed(k):
                pressed.append(k)
        return pressed


if __name__ == "__main__":
    listener = KeyboardListener()
    for _ in range(10):
        k = listener.which_pressed(['w+a', 'a', 's', 'd'])  
        print(k)
        time.sleep(0.5)
