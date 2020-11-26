#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages

setup(name='wavelets_selection',
      version='0.1',
      description='Wavelets selector for time series',
      url='https://github.com/anabatistaaraujo/wavalets_selection',
      author='Ana Araujo e Talison Melo',
      author_email='anacsbatista87@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'pywt',
          'scipy',
          'pandas',
          'matplotlib',
          'numpy',
          'sklearn',
          'warnings'
      ],
      zip_safe=False)

