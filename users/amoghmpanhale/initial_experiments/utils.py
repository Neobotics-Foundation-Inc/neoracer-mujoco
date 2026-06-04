import mujoco

DEFAULT_PHYSICS = {
    "gravity": [0, 0, -9.81],
    "ground_friction": [1.5, 0.05, 0.01],
    "wheel_friction": [1.5, 0.05, 0.01],
}


def load_car(xml_path):
    '''
    Loads the car model from the specified XML path.
    Args:
        xml_path (str): The path to the XML file containing the car model.
    Returns:
        model (mujoco.MjModel): The loaded Mujoco model.
        data (mujoco.MjData): The data associated with the loaded model.
    '''
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data