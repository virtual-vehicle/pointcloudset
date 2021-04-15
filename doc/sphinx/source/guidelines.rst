Guidelines
================================

* use docstrings everywhere. The documentation in &#34;doc&#34; is generated with pdoc with $make doc
* Write tests for every method/function which manipulates data.
* Have a look at the Makefile and the available make commands.
* Use typehints when declaring a function, class or method.
* VS code settings in the dev container take care of linting with mypy and flake8 and code formatting with black.
* every 0.x release needs to have 100% code coverage with tests