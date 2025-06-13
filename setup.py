from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e.'

def get_requirements(file_path:str)->List[str]:
    requirements_list=[]
    with open(file_path) as file_obj:
        requirements_list=file_obj.readlines()
        requirements_list=[req.replace('\n',"")for req in requirements_list]

        if HYPEN_E_DOT in requirements_list:
            requirements_list.remove(HYPEN_E_DOT)        

    return requirements_list

setup(
    name='Personality-Detection',
    version='0.1',
    author='Ratnesh',
    author_email='off.rkv@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)