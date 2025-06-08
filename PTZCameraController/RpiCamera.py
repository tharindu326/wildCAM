from picamera2 import Picamera2
import cv2
import threading
import time

class FrameReader():
    def __init__(self,size):
        self.size=size
        self.queue = [None for _ in range(self.size)]
        self.offset = 0
    def pushQueue(self,data):
        self.offset = (self.offset + 1) % self.size
        self.queue[self.offset] = data
    def popQueue(self):
        self.offset = self.size -1 if self.offset -1 <0 else self.offset -1
        return self.queue[self.offset]

class Camera():
    is_running = False
    window_name = "Arducam PTZ Camera Controller Preview"
    frame = FrameReader(5)
    
    def start(self,width=640,length=360):
        self.is_running = True
        self.capture_ = threading.Thread(target=self.capture_thread, args=(width,length,))  # Fixed function name
        self.capture_.setDaemon(True)
        self.capture_.start()
        
    def stop(self): 
        self.is_running = False
        self.capture_.join()
        
    def close(self):
        if(hasattr(self,"cam")):
            self.cam.stop()
            self.cam.close()
            
    def capture_thread(self, width, length):
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_still_configuration(main={"size": (width, length),"format": "RGB888"}))
        self.cam.start()
        while self.is_running == True:
            buf = self.cam.capture_array()
            self.frame.pushQueue(buf)
            
    def getFrame(self):
        return self.frame.popQueue()

if __name__ == "__main__":
    tmp = Camera()
    tmp.start()
    tmp.display_preview()  # Add this to actually show the preview
    tmp.stop()
    tmp.close()