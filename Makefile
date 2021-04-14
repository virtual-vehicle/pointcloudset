.PHONY: doc docserver doccoverage
doc:
	pdoc --force --html . --output-dir doc

doccoverage:
	docstr-coverage lidar --skipmagic

docserver:
	pdoc --http : .

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

html:
	cd doc/sphinx/ && make $@