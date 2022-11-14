import os.path

from setuptools import setup

# Store simulators path before installing
with open(os.path.join(os.path.dirname(__file__), "uitb/utils/__simulatorsdir__.py"), "w") as f:
    f.write("SIMULATORS_DIR = " + repr(os.path.normpath(os.path.abspath('simulators'))))

setup(
   name='uitb',
   version='0.2.0',
   author='Aleksi Ikkala',
   author_email='aleksi.ikkala@gmail.com',
   packages=['uitb'],
   package_data={'': ['**']},
   url='https://github.com/aikkala/user-in-the-box',
   license='LICENSE',
   description='Modeling and simulating HCI interaction tasks in MuJoCo',
   long_description=open('README.md').read(),
   python_requires='>=3.8',
   install_requires=[
       "gym>=0.26.0",
       "pygame",
       "mujoco>=2.2.0",
       #"stable_baselines3>=1.4.0", 
       "stable_baselines3 @ git+https://github.com/carlosluis/stable-baselines3.git@fix_tests#egg=stable_baselines3-2.0.0a0",
       "torch",
       "wandb", "tensorboard",
       "numpy", "matplotlib", "scipy",
       "opencv-python",
       "ruamel.yaml",
   ],
)