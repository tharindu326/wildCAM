# encoding: UTF-8

import cv2
import os
import sys
import time
from PTZCameraController.RpiCamera import Camera as RpiCamera
from PTZCameraController.Focuser import Focuser
from PTZCameraController.AutoFocus import AutoFocus


class ArduCamPTZ:
    def __init__(self, focuser_channel=1):
        self.camera = RpiCamera()
        self.focuser = Focuser(focuser_channel)
        self.auto_focus = AutoFocus(self.focuser, self.camera)
        self.image_count = 0
        
        # Default step sizes (can be modified as needed)
        self.motor_step = 5
        self.focus_step = 5
        self.zoom_step = 100
        
    def start_preview(self, width=1280, height=720):
        self.camera.start_preview(width, height)
        
    def stop_preview(self):
        self.camera.stop_preview()
        
    def close(self):
        self.camera.close()
        
    def set_focus(self, value):
        """
        Set absolute focus value
        Args:
            value (int): Focus value to set
        """
        self.focuser.set(Focuser.OPT_FOCUS, value)
        
    def get_focus(self):
        """
        Get current focus value
        Returns:
            int: Current focus value
        """
        return self.focuser.get(Focuser.OPT_FOCUS)
        
    def focus_in(self, steps=None):
        """
        Move focus inward (decrease focus value)
        Args:
            steps (int, optional): Number of steps to move. Uses default if None.
        """
        step_size = steps if steps is not None else self.focus_step
        current_focus = self.get_focus()
        self.set_focus(current_focus - step_size)
        
    def focus_out(self, steps=None):
        """
        Move focus outward (increase focus value)
        Args:
            steps (int, optional): Number of steps to move. Uses default if None.
        """
        step_size = steps if steps is not None else self.focus_step
        current_focus = self.get_focus()
        self.set_focus(current_focus + step_size)
        
    def reset_focus(self):
        """Reset focus to initial position"""
        self.focuser.reset(Focuser.OPT_FOCUS)
        
    # Zoom Control Methods
    def set_zoom(self, value):
        """
        Set absolute zoom value
        Args:
            value (int): Zoom value to set
        """
        self.focuser.set(Focuser.OPT_ZOOM, value)
        
    def get_zoom(self):
        """
        Get current zoom value
        Returns:
            int: Current zoom value
        """
        return self.focuser.get(Focuser.OPT_ZOOM)
        
    def zoom_in(self, steps=None):
        """
        Zoom in (increase zoom value)
        Args:
            steps (int, optional): Number of steps to zoom. Uses default if None.
        """
        step_size = steps if steps is not None else self.zoom_step
        current_zoom = self.get_zoom()
        self.set_zoom(current_zoom + step_size)
        
    def zoom_out(self, steps=None):
        """
        Zoom out (decrease zoom value)
        Args:
            steps (int, optional): Number of steps to zoom. Uses default if None.
        """
        step_size = steps if steps is not None else self.zoom_step
        current_zoom = self.get_zoom()
        self.set_zoom(current_zoom - step_size)
        
    def reset_zoom(self):
        """Reset zoom to initial position"""
        self.focuser.reset(Focuser.OPT_ZOOM)
        
    # Motor Control Methods (Pan/Tilt)
    def set_motor_x(self, value):
        """
        Set absolute motor X position (horizontal movement)
        Args:
            value (int): Motor X position value
        """
        self.focuser.set(Focuser.OPT_MOTOR_X, value)
        
    def get_motor_x(self):
        """
        Get current motor X position
        Returns:
            int: Current motor X position
        """
        return self.focuser.get(Focuser.OPT_MOTOR_X)
        
    def set_motor_y(self, value):
        """
        Set absolute motor Y position (vertical movement)
        Args:
            value (int): Motor Y position value
        """
        self.focuser.set(Focuser.OPT_MOTOR_Y, value)
        
    def get_motor_y(self):
        """
        Get current motor Y position
        Returns:
            int: Current motor Y position
        """
        return self.focuser.get(Focuser.OPT_MOTOR_Y)
        
    def move_left(self, steps=None):
        """
        Move camera left (increase motor X)
        Args:
            steps (int, optional): Number of steps to move. Uses default if None.
        """
        step_size = steps if steps is not None else self.motor_step
        current_x = self.get_motor_x()
        self.set_motor_x(current_x + step_size)
        
    def move_right(self, steps=None):
        """
        Move camera right (decrease motor X)
        Args:
            steps (int, optional): Number of steps to move. Uses default if None.
        """
        step_size = steps if steps is not None else self.motor_step
        current_x = self.get_motor_x()
        self.set_motor_x(current_x - step_size)
        
    def move_up(self, steps=None):
        """
        Move camera up (decrease motor Y)
        Args:
            steps (int, optional): Number of steps to move. Uses default if None.
        """
        step_size = steps if steps is not None else self.motor_step
        current_y = self.get_motor_y()
        self.set_motor_y(current_y - step_size)
        
    def move_down(self, steps=None):
        """
        Move camera down (increase motor Y)
        Args:
            steps (int, optional): Number of steps to move. Uses default if None.
        """
        step_size = steps if steps is not None else self.motor_step
        current_y = self.get_motor_y()
        self.set_motor_y(current_y + step_size)
        
    def set_position(self, x, y):
        """
        Set absolute camera position
        Args:
            x (int): Motor X position
            y (int): Motor Y position
        """
        self.set_motor_x(x)
        self.set_motor_y(y)
        
    def get_position(self):
        """
        Get current camera position
        Returns:
            tuple: (motor_x, motor_y) position
        """
        return (self.get_motor_x(), self.get_motor_y())
        
    # IR Cut Filter Control
    def set_ircut(self, state):
        """
        Set IR cut filter state
        Args:
            state (bool): True to enable IR cut filter, False to disable
        """
        current_state = self.get_ircut()
        if state != current_state:
            self.focuser.set(Focuser.OPT_IRCUT, current_state ^ 0x0001)
            
    def get_ircut(self):
        """
        Get IR cut filter state
        Returns:
            bool: True if IR cut filter is enabled, False otherwise
        """
        return bool(self.focuser.get(Focuser.OPT_IRCUT))
        
    def toggle_ircut(self):
        """Toggle IR cut filter state"""
        current_state = self.focuser.get(Focuser.OPT_IRCUT)
        self.focuser.set(Focuser.OPT_IRCUT, current_state ^ 0x0001)
        
    # Mode Control
    def set_mode(self, adjust_mode):
        """
        Set camera mode
        Args:
            adjust_mode (bool): True for Adjust mode, False for Fix mode
        """
        current_mode = self.get_mode()
        if adjust_mode != current_mode:
            self.focuser.set(Focuser.OPT_MODE, current_mode ^ 0x0001)
            self.focuser.waitingForFree()
            
    def get_mode(self):
        """
        Get current camera mode
        Returns:
            bool: True if in Adjust mode, False if in Fix mode
        """
        return bool(self.focuser.get(Focuser.OPT_MODE))
        
    def toggle_mode(self):
        """Toggle between Adjust and Fix modes"""
        current_mode = self.focuser.get(Focuser.OPT_MODE)
        self.focuser.set(Focuser.OPT_MODE, current_mode ^ 0x0001)
        self.focuser.waitingForFree()
        
    def start_autofocus(self):
        """
        Start autofocus operation
        Note: Only works in Adjust mode
        """
        if self.get_mode():
            self.auto_focus.startFocus()
        else:
            print("Warning: Autofocus only works in Adjust mode. Switch to Adjust mode first.")
            
    def start_autofocus_alternative(self):
        """
        Start alternative autofocus operation
        Note: Only works in Adjust mode
        """
        if self.get_mode():
            self.auto_focus.startFocus2()
        else:
            print("Warning: Autofocus only works in Adjust mode. Switch to Adjust mode first.")
            
    def capture_image(self, filename=None):
        if filename is None:
            filename = f"image{self.image_count}.jpg"
            self.image_count += 1
            
        frame = self.camera.getFrame()
        cv2.imwrite(filename, frame)
        return filename
        
    def get_frame(self):
        return self.camera.getFrame()
        
    # Utility Methods
    def reset_all(self):
        """Reset zoom and focus to initial positions"""
        self.reset_zoom()
        time.sleep(0.5)
        self.reset_focus()
        
    def get_status(self):
        """
        Get comprehensive camera status
        Returns:
            dict: Dictionary containing all current camera settings
        """
        return {
            'focus': self.get_focus(),
            'zoom': self.get_zoom(),
            'motor_x': self.get_motor_x(),
            'motor_y': self.get_motor_y(),
            'position': self.get_position(),
            'ircut': self.get_ircut(),
            'mode': 'Adjust' if self.get_mode() else 'Fix',
            'mode_bool': self.get_mode()
        }
        
    def set_step_sizes(self, motor_step=None, focus_step=None, zoom_step=None):
        """
        Set default step sizes for movement operations
        Args:
            motor_step (int, optional): Default motor step size
            focus_step (int, optional): Default focus step size
            zoom_step (int, optional): Default zoom step size
        """
        if motor_step is not None:
            self.motor_step = motor_step
        if focus_step is not None:
            self.focus_step = focus_step
        if zoom_step is not None:
            self.zoom_step = zoom_step


if __name__ == "__main__":
    camera = ArduCamPTZ()
    
    try:
        camera.start_preview(1280, 720)
        print("Current status:", camera.get_status())
        # Set to adjust mode for autofocus
        camera.set_mode(True)
        # Move camera
        camera.move_left(10)
        camera.move_up(5)
        # Adjust zoom and focus
        camera.zoom_in(200)
        camera.focus_out(10)
        # Perform autofocus
        camera.start_autofocus()
        # Capture image
        filename = camera.capture_image()
        print(f"Image saved as: {filename}")
        # Toggle IR cut filter
        camera.toggle_ircut()
        print("Final status:", camera.get_status())
        
    finally:
        # Clean up
        camera.stop_preview()
        camera.close()