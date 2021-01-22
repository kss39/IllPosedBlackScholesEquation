import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ipbse",
    version="0.1.5",
    author="Tianyang Wang",
    author_email="wangty.kss@gmail.com",
    description="IPBSE solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kss39/IllPosedBlackScholesEquation/tree/prod",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'scipy', 'pandas', 'tqdm'
    ],
    python_requires='>=3.8',
)
