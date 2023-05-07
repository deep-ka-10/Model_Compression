from setuptools import setup, find_packages

setup(
    name='modelcompression',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'tensorflow-model-optimization',
        'numpy'
    ]
)
