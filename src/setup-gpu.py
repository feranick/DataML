from setuptools import setup, find_packages

setup(
    name='DataML',
    packages=find_packages(),
    install_requires=['numpy', 'keras', 'h5py', 'tensorflow-gpu'],
    entry_points={'console_scripts' : ['DataML=DataML:DataML','ConvertLabel=ConvertLabel:ConvertLabel']},
    py_modules=['DataML','ConvertLabel','libDataML'],
    version='20181106b',
    description='Multilabel machine learning for combined experimental data',
    long_description= """ Multilabel machine learning for combined experimental data """,
    author='Nicola Ferralis',
    author_email='ferralis@mit.edu',
    url='https://github.com/feranick/DataML',
    download_url='https://github.com/feranick/DataML',
    keywords=['Machine learning', 'physics'],
    license='GPLv2',
    platforms='any',
    classifiers=[
     'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
     'Development Status :: 5 - Production/Stable',
     'Programming Language :: Python',
     'Programming Language :: Python :: 3',
     'Programming Language :: Python :: 3.5',
     'Programming Language :: Python :: 3.6',
     'Intended Audience :: Science/Research',
     'Topic :: Scientific/Engineering :: Chemistry',
     'Topic :: Scientific/Engineering :: Physics',
     ],
)
