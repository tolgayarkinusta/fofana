"""Motor control module for PWM-based dual motor control."""
from typing import Tuple

class MotorController:
    def __init__(self):
        """Initialize motor controller for dual motor setup."""
        self.left_motor_pwm = 0
        self.right_motor_pwm = 0
        
    def set_motor_speeds(self, left_pwm: int, right_pwm: int) -> None:
        """Set PWM values for both motors.
        
        Args:
            left_pwm: PWM value for left motor (-100 to 100)
            right_pwm: PWM value for right motor (-100 to 100)
        """
        self.left_motor_pwm = max(-100, min(100, left_pwm))
        self.right_motor_pwm = max(-100, min(100, right_pwm))
        # TODO: Implement actual PWM control
        
    def get_motor_speeds(self) -> Tuple[int, int]:
        """Get current PWM values for both motors.
        
        Returns:
            Tuple of (left_pwm, right_pwm) values
        """
        return self.left_motor_pwm, self.right_motor_pwm
