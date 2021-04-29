from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Sta663-DCMMs",
    version="2.2.8",
    description="Dynamic Count Mixture Models",
    author="Daniel Deng",
    author_email="currurant@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url="https://github.com/Currurant/DCMMs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ], install_requires=['pandas', 'numpy', 'matplotlib', 'statsmodels', 'scipy', 'pybats'],
    include_package_data=True,
    python_requires='>=3.6'#'Examples_data/*.pickle', 'Examples_data/*.csv',
)
