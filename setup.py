from setuptools import setup, find_packages


with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='pypnf',
    version='0.0.3',
    description='A Package for Point and Figure Charting',
    keywords=['Point and Figure', 'PnF', 'Sentiment Indicator'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/swaschke/pypnf',
    author='Stefan Waschke',
    author_email='swaschke.pypnf@gmail.com',
    license='GPL2',
    packages=['pypnf'],
    # packages=find_packages(),
    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
                 'Operating System :: OS Independent',
                 'Topic :: Office/Business :: Financial'
                 ],
    install_requires=['numpy>=1.20', 'tabulate>=0.8.9'],
    python_requires='>=3.6',
)