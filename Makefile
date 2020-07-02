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
	py3clean .
	
