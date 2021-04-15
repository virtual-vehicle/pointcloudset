.PHONY: doc doccoverage
doc:
	sphinx-apidoc -f -e -o ./doc/sphinx/source/python-api ./lidar --module-first && cd doc/sphinx/ && make html

doccoverage:
	docstr-coverage lidar --skipmagic

test:
	coverage run -m pytest
	pytest --nbval notebooks/usage.ipynb
	python -m coverage report -i
	python -m coverage html -i

sort-imports:
	isort .

clean:
	py3clean .
	cd doc/sphinx/ && make $@

black:
	black . --exclude=notebooks


build:
	python3 setup.py sdist bdist_wheel