from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="HW1",
    author="Sergey Nefedov",
    install_requires=[
        "click==7.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.1",
        "dataclasses==0.8",
        "pyyaml==3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.1.5",
        "pytest==6.0.1"
    ],
    license="MIT",
)
