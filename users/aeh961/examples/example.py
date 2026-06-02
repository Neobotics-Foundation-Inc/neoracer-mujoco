from neoracer.car_spec import build_demo_car_spec
from neoracer.env import run_viewer


def main():
    spec = build_demo_car_spec()
    run_viewer(spec)


if __name__ == "__main__":
    main()