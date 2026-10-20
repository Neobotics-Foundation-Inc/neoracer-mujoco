# RACECAR Neo V2 chassis model

`emb_sc_chassis.blend` is a Blender model of the LC Racing EMB-SC chassis that the RACECAR Neo V2 is built on, with the LC Racing part numbers (L60xx) in its object and collection names. It was reconstructed from photos and is not manufacturing CAD.

- **Units and frame:** millimetres. The origin is the middle of the wheelbase, the axles are at z = 14, and the floor at the tyre contact is at z = −18.7.
- **Rigging:** each wheel turns about a `WheelSpin_*` pivot under a `SteerPivot_*` pivot. The `POSE_CONTROLS` object has a `steer_deg` property that steers the front wheels.
- **Dimensions:** some differ from the car. The wheelbase is 225 mm and the tyres are 65.4 mm across, where the car measures 230 mm and 72 mm. The car's numbers are in `src/neoracer_spec/racecar_neo_v2.py`, from its measurement worksheet. Use this model for the shapes and the layout of the chassis, not for its dimensions.

The file is a Git LFS object, like the meshes in `assets/meshes/`.
