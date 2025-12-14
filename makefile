.PHONY: install
install:
	poetry lock
	poetry synk

.PHONY: run
run:
	python run.py