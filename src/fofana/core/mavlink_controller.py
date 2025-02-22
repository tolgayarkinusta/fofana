"""
MAVLink protokolü kullanan motor kontrol modülü.

Bu modül, OrangeCube üzerinden motorların PWM kontrolünü sağlar:
- İki motorun ileri/geri hareketi için PWM değerleri (1000-2000 arası)
- Servo pin 5: Sol motor
- Servo pin 6: Sağ motor
- Arm/disarm güvenlik kontrolleri
- MAVLink üzerinden haberleşme
"""
from pymavlink import mavutil
from typing import Optional

class USVController:
    def __init__(self, connection_string: str = "COM10", baud: int = 57600):
        """Initialize USV controller with MAVLink connection.
        
        Args:
            connection_string: Serial port for OrangeCube connection
            baud: Baud rate for serial communication
        """
        print("Trying to connect to OrangeCube...")
        self.master = mavutil.mavlink_connection(connection_string, baud=baud)
        print("Connected to MAVLink")
        print("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        print("Heartbeat found!")
        
    def arm_vehicle(self) -> None:
        """Arm the vehicle's motors."""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0
        )
        print("Waiting for vehicle to arm...")
        self.master.motors_armed_wait()
        print("Armed!")
        
    def disarm_vehicle(self) -> None:
        """Disarm the vehicle's motors."""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, 0, 0, 0, 0, 0, 0
        )
        print("Waiting for vehicle to disarm...")
        self.master.motors_disarmed_wait()
        print("Disarmed!")
        
    def set_motor_speed(self, motor: str, pwm_value: int, debug: bool = False) -> None:
        """Set motor speed using PWM value.
        
        Args:
            motor: 'left' or 'right' motor
            pwm_value: PWM value (typically 1000-2000)
            debug: Whether to print motor outputs
        """
        # Left motor on pin 5, right motor on pin 6
        servo_pin = 5 if motor.lower() == 'left' else 6
        self.set_servo(servo_pin, pwm_value)
        
        if debug:
            self.print_motor_outputs()
            
    def set_servo(self, pin: int, pwm_value: int) -> None:
        """Set servo PWM value.
        
        Args:
            pin: Servo pin number (5-8)
            pwm_value: PWM value (1000-2000)
        """
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            pin,
            pwm_value,
            0, 0, 0, 0, 0
        )
            
    def stop_motors(self) -> None:
        """Stop both motors by setting neutral PWM."""
        self.set_motor_speed('left', 1500)
        self.set_motor_speed('right', 1500)
        
    def print_motor_outputs(self) -> None:
        """Print current motor servo outputs."""
        print(self.master.recv_match(type='SERVO_OUTPUT_RAW', blocking=True))
        
    def set_mode(self, mode_name: str) -> None:
        """Set vehicle control mode.
        
        Args:
            mode_name: Mode name (e.g., 'STABILIZE', 'MANUAL', 'DEPTH_HOLD')
        """
        mode_id = self.master.mode_mapping()[mode_name]
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        print(f"Vehicle mode changed to {mode_name}")
