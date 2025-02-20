# This script allows my code to be installed using `pip3 install -e .`

from setuptools import find_packages, setup
from itertools import chain

setup(
    name="metroid_ii_rl",
    description="Metroid II Pufferlib Gymnasium env. for RL",
    long_description_content_type="text/markdown",
    # version=open('Metroid-II-RL/version.py').read().split()[-1].strip("'"),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # TODO make pyboy a dependency eventually (when my changes and the
        # metroid wrapper get pulled in. manual install from my fork needed
        # 'pyboy<2.0.0',
        'gymnasium>=0.29.1',
        'numpy',
    ],

    # entry_points = {
    #     'console_scripts': [
    #         'pokegym.play = pokegym.environment:play'
    #     ]
    # },
    python_requires=">=3.10",
    license="MIT",
    # @pdubs: Put your info here
    author="Ben LaGreca",
    author_email="benlagreca02@gmail.com",
    # url="https://github.com/PufferAI/pokegym",
    keywords=["Metroid II", "AI", "RL"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
