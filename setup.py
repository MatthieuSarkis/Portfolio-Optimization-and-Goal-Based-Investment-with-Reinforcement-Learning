from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='Soft Actor Critic Trading Bot',
   version='1.0',
   description='Soft Actor Critic Trading Bot',
   license="Apache License 2.0",
   long_description=long_description,
   author='Matthieu Sarkis',
   author_email='matthieu.sarkis@gmail.com',
   #url="",
   packages=['src'],
   install_requires=['numpy',
                     'pandas',
                     'jupyter',
                     'sklearn',
                     'matplotlib',
                     'seaborn',
                     'torch',
                     'yfinance',
                     'gym',],
)