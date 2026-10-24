# RACECAR Neo V2 models

`emb_sc_chassis.blend` is a Blender model of the LC Racing EMB-SC chassis that the RACECAR Neo V2 is built on, with the LC Racing part numbers (L60xx) in its object and collection names. It was reconstructed from photos and is not manufacturing CAD.

- **Units and frame:** millimeters. The origin is the middle of the wheelbase, the axles are at z = 14, and the floor at the tire contact is at z = −18.7.
- **Rigging:** each wheel turns about a `WheelSpin_*` pivot under a `SteerPivot_*` pivot. The `POSE_CONTROLS` object has a `steer_deg` property that steers the front wheels.
- **Dimensions:** some differ from the car. The wheelbase is 225 mm and the tires are 65.4 mm across, where the car measures 230 mm and 72 mm. The car's numbers are in `src/neoracer_spec/racecar_neo_v2.py`, from its measurement worksheet. Use this model for the shapes and the layout of the chassis, not for its dimensions.

`racecar-neo-v2-body.step` is the CAD model of the payload, everything on the chassis' body posts: the printed body, the electronics in it, the LiDAR, the camera and the WiFi adapter. It was exported from Autodesk software on 2026-10-21. RacecarSim's model of the payload is this one, 6 mm higher.

- **Units and frame:** millimeters. +y is the car's forward, +x its right and +z up. The y is the car's x from its rear axle less 130 mm, and z = 0 is the underside of the `floor` part, which the measured heights of the LiDAR and the camera put 96 mm above the floor.
- **Antennas:** the WiFi adapter's are partial, as their names say. They reach 260 mm above the floor, where the car's reach 275.

Both files are Git LFS objects, like the meshes in `assets/meshes/`. The STEP file is 77 MB and nothing in the repo reads it, so `.lfsconfig` leaves it out of clones. Fetch it with:

```
git -c lfs.fetchexclude= lfs pull --include="cad/**/*.step"
```
