# GammaFlow Test Suite

Comprehensive test suite for GammaFlow using pytest.

## Test Organization

### `conftest.py`
- Pytest configuration and fixtures
- Reusable test data (spectra, time series, etc.)
- Parametrized fixtures for testing multiple scenarios

### `test_calibration.py`
Tests for `EnergyCalibration` class:
- Creation and validation
- Properties and ref counting
- Copy and detach operations
- Factory methods (from_coefficients)
- String representation

**95 tests** covering all calibration functionality.

### `test_spectrum.py`
Tests for `Spectrum` class:
- Creation and validation
- All properties (counts, edges, uncertainty, etc.)
- Arithmetic operations (+, -, *, /)
- Uncertainty propagation
- Calibration methods
- Energy operations (slicing, rebinning, integration)
- Analysis methods (normalize, statistics)
- Copy/detach operations
- Numpy interface (__array__, __len__, __getitem__)
- String representation

**120+ tests** covering all spectrum functionality.

### `test_time_series.py`
Tests for `SpectralTimeSeries` class:
- Creation (empty, from spectra, shared/independent modes)
- Properties (counts, spectra, timestamps, etc.)
- Shared memory behavior
- Calibration operations
- Vectorized operations (background subtraction, normalization)
- Per-spectrum operations (apply_to_each, filter)
- Time operations (slice, rebin, integrate)
- Analysis methods (mean, sum)
- Numpy protocol (__array__, __len__, __getitem__, __iter__)
- Copy-on-write behavior
- Edge cases and error handling

**150+ tests** covering all time series functionality.

### `test_integration.py`
Integration tests for complete workflows:
- Complete analysis workflows
- Shared calibration in realistic scenarios
- Real-world scenarios (peak search, dead time correction, batch processing)
- Error handling in integrated scenarios
- Performance characteristics
- Mixed object/array operations

**30+ tests** covering end-to-end workflows.

### `test_basic.py`
Quick smoke tests for basic functionality (original simple tests).

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_spectrum.py
pytest tests/test_calibration.py
pytest tests/test_time_series.py
pytest tests/test_integration.py
```

### Run Specific Test Class
```bash
pytest tests/test_spectrum.py::TestSpectrumArithmetic
```

### Run Specific Test
```bash
pytest tests/test_spectrum.py::TestSpectrumArithmetic::test_add_spectra
```

### Run with Verbose Output
```bash
pytest -v
```

### Run with Output Capture Disabled
```bash
pytest -s
```

### Run and Stop at First Failure
```bash
pytest -x
```

### Run Last Failed Tests
```bash
pytest --lf
```

### Run Tests Matching Pattern
```bash
pytest -k "calibration"  # Runs tests with "calibration" in name
pytest -k "not slow"     # Skip slow tests
```

## Coverage

### Run with Coverage
```bash
pytest --cov=gammaflow
```

### Generate HTML Coverage Report
```bash
pytest --cov=gammaflow --cov-report=html
open htmlcov/index.html  # View report
```

### Generate Coverage Report with Missing Lines
```bash
pytest --cov=gammaflow --cov-report=term-missing
```

## Test Statistics

| Test File | Number of Tests | Coverage |
|-----------|----------------|----------|
| test_calibration.py | ~95 | EnergyCalibration class |
| test_spectrum.py | ~120 | Spectrum class |
| test_time_series.py | ~150 | SpectralTimeSeries class |
| test_integration.py | ~30 | End-to-end workflows |
| **Total** | **~395** | **All functionality** |

## Test Categories

### Unit Tests
Test individual methods and functions in isolation:
- `test_calibration.py` - all tests
- `test_spectrum.py` - most tests
- `test_time_series.py` - most tests

### Integration Tests
Test interactions between components:
- `test_integration.py` - all tests
- Some tests in `test_time_series.py`

### Edge Cases
Tests for boundary conditions and error handling:
- Empty inputs
- Single-element inputs
- Very large inputs
- Invalid inputs
- Incompatible operations

## Fixtures

Defined in `conftest.py`:

### Energy Calibration Fixtures
- `uncalibrated_edges` - None edges
- `simple_edges` - Simple 5-element array
- `calibration_linear` - Linear calibration
- `calibration_quadratic` - Quadratic calibration

### Spectrum Fixtures
- `simple_counts` - 4-element array
- `random_counts` - Random Poisson data
- `spectrum_uncalibrated` - Basic uncalibrated spectrum
- `spectrum_calibrated` - Calibrated spectrum
- `spectrum_with_uncertainty` - Explicit uncertainties
- `spectrum_large` - Large 1024-bin spectrum

### Time Series Fixtures
- `small_spectra_list` - 10 spectra, 64 bins
- `time_series_small` - Small time series
- `calibrated_spectra_list` - 20 calibrated spectra
- `time_series_calibrated` - Calibrated time series
- `mixed_calibration_spectra` - Different calibrations

### Parametrized Fixtures
- `calibration_model` - Test both 'polynomial' and 'linear'
- `normalization_method` - Test 'area', 'peak', 'live_time'
- `shared_calibration_mode` - Test both True and False

## Writing New Tests

### Template for New Test Class
```python
class TestNewFeature:
    """Test new feature."""
    
    def test_basic_functionality(self, fixture_name):
        """Test basic use case."""
        # Arrange
        obj = create_test_object()
        
        # Act
        result = obj.method()
        
        # Assert
        assert result == expected
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ExpectedError):
            # code that should raise error
    
    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_multiple_inputs(self, input, expected):
        """Test multiple input values."""
        assert function(input) == expected
```

### Best Practices
1. Use descriptive test names
2. One assertion per test (when possible)
3. Use fixtures for common setups
4. Test both success and failure cases
5. Test edge cases and boundary conditions
6. Use parametrize for multiple similar tests
7. Add docstrings explaining what's tested

## Continuous Integration

To run tests in CI/CD:

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=gammaflow --cov-report=xml --cov-report=term

# Coverage should be > 90%
```

## Troubleshooting

### Import Errors
If you get import errors, make sure GammaFlow is installed:
```bash
pip install -e .
```

### Fixture Not Found
Make sure you're running from the project root and `conftest.py` is in the tests directory.

### Slow Tests
Run without slow tests:
```bash
pytest -m "not slow"
```

### Random Failures
Some tests use random data with fixed seeds. If a test fails randomly, check if the seed needs updating.

## Future Test Additions

Potential areas for additional tests:
- [ ] Performance benchmarks
- [ ] Memory usage tests
- [ ] Visualization tests (when plotting is added)
- [ ] File I/O tests (when I/O is added)
- [ ] Listmode data tests (when added)
- [ ] Stress tests with very large datasets
- [ ] Multithreading safety tests
- [ ] Property-based tests using hypothesis

## Summary

The GammaFlow test suite provides comprehensive coverage of all functionality:
- **~395 tests** covering every class and method
- **Unit tests** for individual components
- **Integration tests** for complete workflows
- **Edge case tests** for robustness
- **Fixtures** for reusable test data
- **Parametrization** for efficiency

Run `pytest` to verify everything works!

