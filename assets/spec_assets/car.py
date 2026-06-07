import os

import mujoco

# Path to the bundled car model so callers don't need to know the layout.
XML_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xml_assets"
)
CAR_XML = os.path.join(XML_ASSETS_DIR, "car.xml")


def load_car_spec(xml_path: str = CAR_XML) -> mujoco.MjSpec:
    """Load a car XML into an (uncompiled) MjSpec."""
    return mujoco.MjSpec.from_file(xml_path)


def build_car(xml_path: str = CAR_XML):
    """Load a car XML, compile it, and return (spec, model, data)."""
    spec = load_car_spec(xml_path)
    model = spec.compile()
    data = mujoco.MjData(model)
    return spec, model, data
