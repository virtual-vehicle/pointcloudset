.PHONY: doc doccoverage
doc:
	sphinx-apidoc --no-toc --module-first -f -e -o ./doc/sphinx/source/python-api ./src/pointcloudset pointcloudset/config.py src/pointcloudset/io/dataset/commandline.py src/pointcloudset/io/dataset/convert_rosbagconvert.py && cd doc/sphinx/ && make html

doccoverage:
	docstr-coverage pointcloudset --skipmagic

test:
	pytest --cov=pointcloudset tests
	pytest --cov-append --current-env --nbval-lax doc/sphinx/source/tutorial_notebooks/usage.ipynb
	pytest --cov-append --current-env --nbval-lax doc/sphinx/source/tutorial_notebooks/reading_las_pcd.ipynb
	pytest --cov-append --current-env --nbval-lax tests/notebooks/test_plot_plane.ipynb
	pytest --cov-append --current-env --nbval-lax tests/notebooks/test_readme.ipynb
	pytest --cov-append --current-env --nbval-lax tests/notebooks/test_animate.ipynb
	python -m coverage report -i
	python -m coverage html -i

ruff:
	ruff check pointcloudset

ruff-fix:
	ruff check pointcloudset --fix

mypy:
	mypy -p pointcloudset --ignore-missing-imports


sort-imports:
	ruff check --select I --fix .
	ruff format .

clean:
	py3clean .
	cd doc/sphinx/ && make $@
	rm -r doc/sphinx/source/python-api
