from setuptools import setup

REQUIRED_PACKAGES = [
    "netCDF4==1.4.0",
    "tensorflow==1.12.0",
    "numpy==1.14.2"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="deep_climate_emulator",
    version="1.0.0",
    license="Apache 2.0",
    author="Ted Weber",
    author_email="weber.ted2@gmail.com",
    description="Residual Network to approximate precipitation fields in "
                "Earth System Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hutchresearch/deep_climate_emulator",
    packages=["emulator"],
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    zip_safe=True,
)
