"""
A car's specification and what the simulators run, derived from it. Pure
Python with numpy, no MuJoCo, so any program can read the car.

    provenance.Sourced  a value with its source, its status (measured,
                        documented, assumed, estimated or chosen) and the
                        values other sources give
    spec.Spec           what the car is, part by part, every value Sourced.
                        neoracer.NEORACER is the NeoRacer.
      derive.derive     the simulators' parameters (DerivedParams): the
                        MuJoCo model's numbers (ModelParams), the LiDAR's
                        scan, the encoder's filter, full lock
    mjcf.car_xml        ModelParams written into the car's XML
    parts               what the cars' models share, such as the solver's
                        options (SOLVER)

Each step is a pure function from one type to the next. A variant of the
car is a variant of its parts: dataclasses.replace(NEORACER, motor=...).
"""
