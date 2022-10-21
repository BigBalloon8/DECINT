import setuptools
import node
setuptools.setup(
    name="DECINT-run",
    version=f"{node.__version__}",
    author="Chris Rae",
    author_email="raecd123@gmail.com",
    packages=setuptools.find_packages(),
    install_requires=["ecdsa", "click", "requests"],
    entry_points={
        "console_scripts":[
            "DECINT-run = ./src/DECINT/DECINT:run"
        ]}
)