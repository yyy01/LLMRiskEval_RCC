import setuptools

with open('README.md') as f:
    _LONG_DESCRIPTION = f.read()

setuptools.setup(
    name='rcctool',
    version='0.0.1',
    description='RCCtool',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='ZJUM3',
    url='https://github.com/ZJUM3/LLMEval_RCC',
    packages=setuptools.find_packages(),
    install_requires=[ ],
    extras_require={
        'test': ['pytest']
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='LLM evaluation',
)