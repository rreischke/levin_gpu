from pathlib import Path

from setuptools import setup

__version__ = "0.1.0.dev0"

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="levin-gpu",
    version=__version__,
    author="Robert Reischke",
    author_email="reischke@posteo.net",
    url="https://github.com/rreischke/levin_bessel",
    description="GPU-accelerated Levin integration of oscillatory integrals over Bessel functions, written in JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license_files=("LICENSE.txt",),
    packages=["levin"],
    python_requires=">=3.10",
    install_requires=[
        "jax>=0.4.30",
        "numpy>=2",
    ],
    extras_require={
        "cuda": ["jax[cuda12]>=0.4.30"],
        "test": ["pytest>=7", "scipy>=1.10", "mpmath>=1.3"],
        "notebook": ["jupyter", "matplotlib", "scipy>=1.10"],
        # The original C++ implementation, used as a reference in the tests.
        "reference": ["pylevin"],
    },
    zip_safe=False,
)
