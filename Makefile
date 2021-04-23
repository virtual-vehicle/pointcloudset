.PHONY: doc doccoverage
doc:
	sphinx-apidoc --no-toc -f -e -o ./doc/sphinx/source/python-api ./lidar --module-first && cd doc/sphinx/ && make html

doccoverage:
	docstr-coverage lidar --skipmagic

test:
	pytest --cov=lidar tests
	pytest --current-env --nbval-lax doc/sphinx/source/tutorial_notebooks/usage.ipynb
	pytest --current-env --nbval-lax doc/sphinx/source/tutorial_notebooks/reading_las.ipynb
	pytest --current-env --nbval-lax tests/plot/plot2d_test.ipynb
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