import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self,
                 vehicle_mass,
                 brake_deadband,
                 decel_limit,
                 accel_limit,
                 wheel_radius,
                 wheel_base,
                 steer_ratio,
                 min_speed,
                 max_lat_accel,
                 max_steer_angle):

        self.yaw_controller = YawController(
            wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        kp = .3
        ki = .3
        kd = .0
        mn = .0
        mx = .2
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = .5 
        ts = .02
        self.lowpass = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, linear_velocity_cmd, angular_velocity_cmd, current_velocity, enabled):
        if not enabled:
            self.throttle_controller.reset()
            return .0, .0, .0
    
        current_velocity = self.lowpass.filt(current_velocity)

        steer = self.yaw_controller.get_steering(
            linear_velocity_cmd, angular_velocity_cmd, current_velocity)

        current_time = rospy.get_time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Velocity error
        vel_error = linear_velocity_cmd - current_velocity
        
        throttle = self.throttle_controller.step(vel_error, dt)
        brake = 0.

        # Brake if we want to hold the car stopped.
        if linear_velocity_cmd == .0 and current_velocity < .1:
            throttle = .0
            brake = 400 # N*m
        # Or if we want to go slower and we can't lower the throttle anymore.
        elif throttle < .1 and vel_error < .0:
            throttle = .0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steer
