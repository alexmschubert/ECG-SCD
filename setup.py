import setuptools

# with open("README.md", "r", encoding="utf-8") as f:
#     long_description = f.read()

setuptools.setup(
    name="ekg-scd",
    version="0.0.0a0",
    author="John Luby, Alexander Schubert, Luke Frymire",
    author_email="alexander_schubert@berkeley.edu",
    description="Internal package for predicting Sudden Cardiac Death from EKGs.",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    classifiers=[
        "License :: Other/Proprietary License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6"
)
