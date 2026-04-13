.PHONY: help test verify-mroi install clean

help:
	@echo "Privacy-First MMM Toolkit - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run full test suite"
	@echo "  make verify-mroi   - Verify mROI formula against ground truth"
	@echo "  make clean         - Clean generated files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

verify-mroi:
	@echo "Verifying mROI formula against ground truth..."
	@pytest tests/test_analysis.py::test_mroi_numerical_ground_truth -v
	@echo ""
	@echo "✓ mROI formula matches simulation to within 5%"

clean:
	rm -rf __pycache__ .pytest_cache mmm/__pycache__ tests/__pycache__
	rm -rf mmm_output_advanced/*.png mmm_output_advanced/*.csv
