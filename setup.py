from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'superquadric_grasp_system'

# Get launch files if they exist
launch_files = glob('launch/*.py')
config_files = glob('config/*.yaml')

# Build data_files list conditionally
data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

# Only add launch files if they exist
if launch_files:
    data_files.append((os.path.join('share', package_name, 'launch'), launch_files))

# Only add config files if they exist
if config_files:
    data_files.append((os.path.join('share', package_name, 'config'), config_files))

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gejan',
    maintainer_email='gejan@student.ethz.ch',
    description='Superquadric-based grasp planning system with modular pose estimation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = superquadric_grasp_system.perception_node:main',
        ],
    },
)