import setuptools


REQUIRED_PACKAGES = ["tensorflow==1.12.0", "numpy==1.15.4", "netCDF4==1.4.2"]

setuptools.setup(
    name="deep_climate_emulator",
    version="1.0.0",
    license="Apache License 2.0",
    author="Ted Weber",
    author_email="weber.ted2@gmail.com",
    description="Residual Network to approximate precipitation fields in "
                "Earth System Models",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hutchresearch/deep_climate_emulator",
    packages=['emulator'],
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    zip_safe=True,
	classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
)
