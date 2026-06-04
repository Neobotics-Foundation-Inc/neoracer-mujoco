"""
Constants for the sim. Edit these to change the car or the physics.
Everything else reads from here, so there are no magic numbers buried in the code.
"""

# physics
GRAVITY = [0, 0, -9.81]
TIMESTEP = 0.005
GROUND_FRICTION = [1.5, 0.1, 0.1]   # [sliding, torsional, rolling]
WHEEL_FRICTION = [1.5, 0.1, 0.1]

# car body (simple box)
BODY_SIZE = [1.2, 0.5, 0.18]        # half-sizes
BODY_MASS = 20
RIDE_HEIGHT = 0.45                  # spawn height of the chassis

# wheels
WHEEL_RADIUS = 0.22
WHEEL_WIDTH = 0.08                  # cylinder half-length
WHEEL_MASS = 1
WHEEL_X = 0.8                       # front/back offset from center
WHEEL_Y = 0.45                      # left/right offset from center
WHEEL_Z = -0.22                     # how far the axles drop below the chassis

# drive + steering
MOTOR_GEAR = 8                      # ctrl value -> wheel torque
STEER_LIMIT = 0.6                   # max front wheel angle (rad)
STEER_KP = 40                       # steering servo stiffness

# oval track
TRACK_STRAIGHT = 4.0                # half-length of each straight
TRACK_RADIUS = 3.0                  # radius of the curved ends
LANE_HALF = 0.9                     # half lane width
WALL_HEIGHT = 0.2
