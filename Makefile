.PHONY: doc docserver doccoverage
doc:
	pdoc --force --html . --output-dir doc

doccoverage:
	docstr-coverage lidar --skipmagic

docserver:
	pdoc --http : .	

test:
	coverage run -m pytest
	/opt/conda/bin/python -m coverage report -i 
	/opt/conda/bin/python -m coverage html -i

sort-imports:
	isort -rc .	

clean:
	rm --force --recursive build \
	rm --force --recursive dist \
	rm --force --recursive *.egg-info \
	rm --force --recursive .pytest_cache
