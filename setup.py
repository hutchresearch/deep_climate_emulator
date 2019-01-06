import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep_climate_emulator",
    version="1.0.0",
    license="MIT",
    author="Ted Weber",
    author_email="weber.ted2@gmail.com",
    description="Residual Network to approximate precipitation fields in "
                "Earth System Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hutchresearch/deep_climate_emulator",
    packages=['emulator'],
    install_requires=[
        'netCDF4',
        'numpy',
        'tensorflow'
    ],
    include_package_data=True,
    zip_safe=True,
	classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
)
