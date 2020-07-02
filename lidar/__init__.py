"""
A package to work with automotive lidar data.

# Reading this documentation.
The main features of this package are the "Dataset" class consisting of many
"Frame" objects.

# Installation

``` bash
 pip install git+https://gitlab.v2c2.at/sensor-fdir/lidar
```

# Getting started

* install VS Code
* install the remote development extension in VS code (ms-vscode-remote.vscode-remote-extensionpack)
* install and run docker desktop
* clone the repository (you can use VC code for that as well)
* open the folder in VS code
* VS code asks to open the remote development environment - say yes
* The docker image is download on the first start, so this may take a while

## Gitlab access to the docker registry

On your very first time accessing our gitlab docker registry you need to do the following steps:

1.) generate an access token: 
* in gitlab => User settings => Access token
* generate a token with "read_registy" rights
* store the token savely (KeePass)

2.) In the terminal login to our registry:

``` bash
docker login registry-gitlab.v2c2.at -u <vornamenachname> -p <token>
```


# Contribution

* add Issues in gitlab
* generate a branch and implement new features
* write tests for the new features
* make a merge request in gitlab


## Development

* use docstrings everywhere. The documentation in "doc" is generated with pdoc.
* Have a look at the Makefile and the available make commands
* use typehints when declaring a function, class or method.
* VS code settings in the dev container take care of linting and code formatting.
* write tests for every method/function wich manipulates data
* for example useage you can have a look at the tests
* every 0.x release needs to have 100% code coverage with tests


"""

from .dataset import Dataset
from .frame import Frame
