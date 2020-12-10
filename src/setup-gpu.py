from setuptools import setup, find_packages

setup(
    name='DataML',
    packages=find_packages(),
    install_requires=['numpy', 'h5py', 'tensorflow-gpu','pydot'],
    entry_points={'console_scripts' : ['DataML=DataML:DataML','ConvertLabel=ConvertLabel:ConvertLabel','GetClasses=GetClasses:GetClasses']},
    py_modules=['DataML','ConvertLabel','GetClasses','libDataML'],
    version='20201012a',
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
     'Programming Language :: Python :: 3.6',
     'Programming Language :: Python :: 3.7',
     'Programming Language :: Python :: 3.7',
     'Intended Audience :: Science/Research',
     'Topic :: Scientific/Engineering :: Chemistry',
     'Topic :: Scientific/Engineering :: Physics',
     ],
)
