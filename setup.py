import setuptools

with open('README.md') as f:
    _LONG_DESCRIPTION = f.read()

setuptools.setup(
    name='rcctool',
    version='1.0.0',
    description='RCCtool',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='yyy01',
    url='https://github.com/yyy01/LLMRiskEval_RCC',
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