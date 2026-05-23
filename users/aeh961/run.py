import os

print("\nMuJoCo Experiments")
print("1 - Falling cube")
print("2 - Moving car")
print("3 - Car with wheels")
print("4 - Steering")
print("5 - Checkpoints")

choice = input("\nSelect experiment: ")

files = {
    "1":"01_test.py",
    "2":"02_car.py",
    "3":"03_simple_car_with_wheels.py",
    "4":"04_steering.py",
    "5":"05_checkpoint_system.py"
}

if choice in files:
    os.system(f"mjpython {files[choice]}")
else:
    print("Invalid choice")