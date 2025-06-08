# encoding: UTF-8

import cv2
import os
import sys
import time
from PTZCameraController.RpiCamera import Camera as RpiCamera
from PTZCameraController.Focuser import Focuser
from PTZCameraController.AutoFocus import AutoFocus


class ArduCamPTZ:
    def __init__(self, focuser_channel=1, manual_control=False):
        self.camera = RpiCamera()
        self.focuser = Focuser(focuser_channel)
        self.auto_focus = AutoFocus(self.focuser, self.camera)
        self.image_count = 0
        self.manual_control = manual_control
        
        # Default step sizes
        self.motor_step = 5
        self.focus_step = 100
        self.zoom_step = 100
        
        # Ensure IR cut is disabled at startup
        if self.get_ircut():
            self.toggle_ircut()
            
        if self.manual_control:
            print("Manual Controls Enabled:")
            print("w/s - Move Up/Down")
            print("a/d - Move Left/Right") 
            print("Arrow Keys - Focus (Left/Right) and Zoom (Up/Down)")
            print("Space - Toggle IR Cut")
            print("Enter - Autofocus")
            print("c - Capture Image")
            print("r - Reset Focus and Zoom")
        
    def start_preview(self, width=1280, height=720):
        self.camera.start(width, height)
        
    def stop_preview(self):
        self.camera.stop()
        
    def close(self):
        self.camera.close()
        
    def set_focus(self, value):
        self.focuser.set(Focuser.OPT_FOCUS, value)
        
    def get_focus(self):
        return self.focuser.get(Focuser.OPT_FOCUS)
        
    def focus_in(self, steps=None):
        step_size = steps if steps is not None else self.focus_step
        current_focus = self.get_focus()
        self.set_focus(current_focus - step_size)
        
    def focus_out(self, steps=None):
        step_size = steps if steps is not None else self.focus_step
        current_focus = self.get_focus()
        self.set_focus(current_focus + step_size)
        
    def reset_focus(self):
        self.focuser.reset(Focuser.OPT_FOCUS)
        
    def set_zoom(self, value):
        self.focuser.set(Focuser.OPT_ZOOM, value)
        
    def get_zoom(self):
        return self.focuser.get(Focuser.OPT_ZOOM)
        
    def zoom_in(self, steps=None):
        step_size = steps if steps is not None else self.zoom_step
        current_zoom = self.get_zoom()
        self.set_zoom(current_zoom + step_size)
        
    def zoom_out(self, steps=None):
        step_size = steps if steps is not None else self.zoom_step
        current_zoom = self.get_zoom()
        self.set_zoom(current_zoom - step_size)
        
    def reset_zoom(self):
        self.focuser.reset(Focuser.OPT_ZOOM)
        
    def set_motor_x(self, value):
        self.focuser.set(Focuser.OPT_MOTOR_X, value)
        
    def get_motor_x(self):
        return self.focuser.get(Focuser.OPT_MOTOR_X)
        
    def set_motor_y(self, value):
        self.focuser.set(Focuser.OPT_MOTOR_Y, value)
        
    def get_motor_y(self):
        return self.focuser.get(Focuser.OPT_MOTOR_Y)
        
    def move_left(self, steps=None):
        step_size = steps if steps is not None else self.motor_step
        current_x = self.get_motor_x()
        self.set_motor_x(current_x + step_size)
        
    def move_right(self, steps=None):
        step_size = steps if steps is not None else self.motor_step
        current_x = self.get_motor_x()
        self.set_motor_x(current_x - step_size)
        
    def move_up(self, steps=None):
        step_size = steps if steps is not None else self.motor_step
        current_y = self.get_motor_y()
        self.set_motor_y(current_y - step_size)
        
    def move_down(self, steps=None):
        step_size = steps if steps is not None else self.motor_step
        current_y = self.get_motor_y()
        self.set_motor_y(current_y + step_size)
        
    def set_position(self, x, y):
        self.set_motor_x(x)
        self.set_motor_y(y)
        
    def get_position(self):
        return (self.get_motor_x(), self.get_motor_y())
        
    def set_ircut(self, state):
        current_state = self.get_ircut()
        if state != current_state:
            self.toggle_ircut()
            
    def get_ircut(self):
        return bool(self.focuser.get(Focuser.OPT_IRCUT))
        
    def toggle_ircut(self):
        current_state = self.focuser.get(Focuser.OPT_IRCUT)
        self.focuser.set(Focuser.OPT_IRCUT, current_state ^ 0x0001)
        
    def start_autofocus(self):
        self.auto_focus.startFocus()
            
    def start_autofocus_alternative(self):
        self.auto_focus.startFocus2()
        
    def reset_all(self):
        self.reset_zoom()
        time.sleep(0.5)
        self.reset_focus()
        
    def get_status(self):
        return {
            'focus': self.get_focus(),
            'zoom': self.get_zoom(),
            'motor_x': self.get_motor_x(),
            'motor_y': self.get_motor_y(),
            'position': self.get_position(),
            'ircut': self.get_ircut()
        }
        
    def set_step_sizes(self, motor_step=None, focus_step=None, zoom_step=None):
        if motor_step is not None:
            self.motor_step = motor_step
        if focus_step is not None:
            self.focus_step = focus_step
        if zoom_step is not None:
            self.zoom_step = zoom_step
            
    def capture_image(self, filename=None):
        if filename is None:
            filename = f"image{self.image_count}.jpg"
            self.image_count += 1
            
        frame = self.camera.getFrame()
        cv2.imwrite(filename, frame)
        return filename
        
    def get_frame(self):
        return self.camera.getFrame()
        
    def process_key(self, keyCode):
        if not self.manual_control:
            return False
            
        if keyCode == ord('s'):
            self.move_down()
            return True
        elif keyCode == ord('w'):
            self.move_up()
            return True
        elif keyCode == ord('d'):
            self.move_right()
            return True
        elif keyCode == ord('a'):
            self.move_left()
            return True
        elif keyCode == ord('r'):
            self.reset_all()
            return True
        elif keyCode == 32:  # Space
            self.toggle_ircut()
            return True
        elif keyCode == 13:  # Enter
            self.start_autofocus()
            return True
        elif keyCode == ord('c'):
            filename = self.capture_image()
            return True
        elif keyCode == 81:  # Up arrow
            self.zoom_in()
            return True
        elif keyCode == 83:  # Down arrow
            self.zoom_out()
            return True
        elif keyCode == 82:  # Right arrow
            self.focus_out()
            return True
        elif keyCode == 84:  # Left arrow
            self.focus_in()
            return True
        
        return False


if __name__ == "__main__":
    camera = ArduCamPTZ(manual_control=True)
    
    camera.start_preview(1280, 720)
    time.sleep(2)
    print("Initial status:", camera.get_status())
    
    window_name = "PTZ Camera Preview"
    
    while True:
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow(window_name, frame)
            
        keyCode = cv2.waitKey(1) & 0xFF
        if keyCode == ord('q'):
            break
            
        camera.process_key(keyCode)
    
    cv2.destroyWindow(window_name)
    camera.stop_preview()
    camera.close()