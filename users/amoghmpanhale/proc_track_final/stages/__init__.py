"""The five stages of building a track, one file each, named in the order they
run:

  s1_control_points.py  scatter a ring of random points
  s2_centerline.py      spline through them, resample evenly in meters
  s3_width.py           give the corridor a varying half-width
  s4_validation.py      check the result against four rules
  s5_repair.py          nudge the control points that broke a rule

track.py one directory up is the loop that calls them. Read that first.
"""
