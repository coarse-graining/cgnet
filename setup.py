from setuptools import setup, find_packages

NAME = 'cgnet'
VERSION = '0.0'


def read(filename):
    import os
    BASE_DIR = os.path.dirname(__file__)
    filename = os.path.join(BASE_DIR, filename)
    with open(filename, 'r') as fi:
        return fi.read()

def readlist(filename):
    rows = read(filename).split("\n")
    rows = [x.strip() for x in rows if x.strip()]
    return list(rows)

setup(
    name=NAME,
    version=VERSION,
    author="Nick Charron, Brooke Husic, Dominik Lemm, Jiang Wang",
    author_email="husic@zedat.fu-berlin.de",
    url='https://github.com/coarse-graining/cgnet',
    #download_url='https://github.com/coarse-graining/cgnet/tarball/master',
    #long_description=read('README.md'),
    license='BSD-3-Clause',
    packages=find_packages(),
    zip_safe=True,
    entry_points={
        'console_scripts': [
            '%s = %s.cli.main:main' % (NAME, NAME),
        ],
    },
    )
