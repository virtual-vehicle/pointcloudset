.PHONY: doc doccoverage
doc:
	sphinx-apidoc --no-toc --module-first -f -e -o ./doc/sphinx/source/python-api ./pointcloudset pointcloudset/config.py pointcloudset/io/dataset/commandline.py pointcloudset/io/dataset/convert_bag2dataset.py && cd doc/sphinx/ && make html

doccoverage:
	docstr-coverage pointcloudset --skipmagic

test:
	pytest --cov=pointcloudset tests
	pytest --current-env --nbval-lax doc/sphinx/source/tutorial_notebooks/usage.ipynb
	pytest --current-env --nbval-lax doc/sphinx/source/tutorial_notebooks/reading_las_pcd.ipynb
	pytest --current-env --nbval-lax tests/notebooks/test_plot_plane.ipynb
	pytest --current-env --nbval-lax tests/notebooks/test_readme.ipynb
	python -m coverage report -i
	python -m coverage html -i

mypy:
	mypy -p pointcloudset --ignore-missing-imports


sort-imports:
	isort .

clean:
	py3clean .
	cd doc/sphinx/ && make $@
	rm -r doc/sphinx/source/python-api

black:
	black . --exclude=notebooks


build:
	python3 setup.py sdist bdist_wheel