"""
The car's digital twin: from what the car is to what the simulators run.

    parts.Car           what the car is: its parts as datasheets and
                        measurements give them (parts.py). neoracer.NEORACER
                        is the NeoRacer, each value with its source.
      derive.model      the MuJoCo model's numbers (ModelParams)
      derive.settings   what the simulators need beside the model: the
                        LiDAR's scan, the encoder's filter, the steer limit
                        (Settings)
    mjcf.car_xml        ModelParams written into the car's XML

Each step is a pure function from one type to the next. A variant of the
car is a variant of its parts: dataclasses.replace(NEORACER, motor=...).
"""
