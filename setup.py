from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
    returns the list of requirements from the requirements.txt file
    """
    HYPHEN_E_DOT = '-e .'
    requirements =[]

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name = 'Diamond_Price_Prediction',
    version = '0.0.1',
    author = 'Randhir Singh',
    author_email = 'randhirsingh7777777@gmail.com',
    description = 'ML Regression problem for predicting diamond price',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')

)