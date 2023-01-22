import setuptools
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSIONFILE="topictuner/__init__.py"
getversion = re.search( r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setuptools.setup(
     python_requires='>=3',                                   # Minimum Python version
     name='topicmodeltuner',                                  # Package name
     version=new_version,                                     # Version
     author="Dan Robinson",                                   # Author name
     author_email="drob707@gmail.com",                        # Author mail
     description="HDBSCAN Tuning for BERTopic Models",        # Short package description
     long_description=long_description,                       # Long package description
     long_description_content_type="text/markdown",
     url="https://github.com/drob-xx/TopicTuner",             # Url to Git Repo
     download_url = 'https://github.com/drob-xx/TopicTuner/archive/refs/tags/'+new_version+'.tar.gz',
     packages=setuptools.find_packages(),                     # Searches throughout all dirs for files to include
     include_package_data=True,                               # Must be true to include files depicted in MANIFEST.in
     license_files=["LICENSE"],                               # License file
     install_requires=["bertopic", "loguru"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: OS Independent",
     ],
 )