from setuptools import setup, find_packages

setup(
    name='spelt',  # Required
    version='0.1.0',  # Required, follows semantic versioning
    author='Jake Swann',  # Optional
    author_email='jake.swann.22@ucl.ac.uk',  # Optional
    description='SpElT - a package for analysis of spatial electrophysiology data',  # Optional
    long_description=open('README.md').read(),  # Optional, long description read from the README file
    long_description_content_type='text/markdown',  # Optional, to specify the content type of the long description
    url='https://github.com/jakeswann1/SpElT',  # Optional, URL for the package's homepage or source repository
    packages=find_packages(),  # Required, automatically find packages in the directory
    classifiers=[  # Optional, additional metadata about your package
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Optional, specify the Python versions supported
    install_requires=[  # Optional, list of dependencies
        'numpy',
        'scipy',
        'spikeinterface',
        'pandas',
        'matplotlib',
        'seaborn',
        'pynapple'
        # Add other dependencies as required
    ],
    entry_points={},  # Optional, if you have console scripts to expose
    include_package_data=True,  # Optional, include additional files specified in MANIFEST.in
    package_data={  # Optional, include additional data files
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
        # And include any *.msg files found in the 'hello' package, too:
        'hello': ['*.msg'],
    },
    project_urls={  # Optional, additional links for the project
        'Bug Reports': 'https://github.com/jakeswann1/SpElT/issues',
        'Source': 'https://github.com/jakeswann1/SpElT',
    },
)
