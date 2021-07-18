from setuptools import setup, find_packages


with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='pypnf',
    version='0.0.1',
    description='A Package for Point and Figure Charting',
    keywords=['Point and Figure','PnF', 'Sentiment Indicator'],
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/swaschke/pypnf',
    author='Stefan Waschke',
    author_email='swaschke.pypnf@gmail.com',
    packages=find_packages(),
    classifiers=['Development Status :: 1 - Planning'],
    install_requires=[],
    python_requires='>=3.6',
)