from setuptools import setup, find_packages

setup(name='gridds',
      description='Grid Data Science Toolkit',
      version='0.0.1',
      author='Alexander Ladd',
      author_email='ladd12@llnl.gov',
      url='',
      license='NOT_FOR_EXTERNAL_USE',
      packages=find_packages(),
      install_requires=[
        'pandas',
        'numpy',
      ],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        ]
      )
