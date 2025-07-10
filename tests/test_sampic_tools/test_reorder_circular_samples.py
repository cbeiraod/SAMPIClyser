import numpy as np
import pytest

from sampiclyser.sampic_tools import reorder_circular_samples_with_trigger


# Sample data for testing
@pytest.fixture
def sample_linear():
    trig = np.array([0, 0, 1, 1, 0, 0])
    samp = np.arange(6)
    return trig, samp


@pytest.fixture
def sample_wrap():
    trig = np.array([1, 0, 0, 0, 1])
    samp = np.arange(5)
    return trig, samp


@pytest.fixture
def sample_non_contiguous():
    trig = np.array([1, 0, 1, 0, 0])
    samp = np.arange(5)
    return trig, samp


@pytest.fixture
def sample_no_ones():
    trig = np.array([0, 0, 0])
    samp = np.arange(3)
    return trig, samp


@pytest.fixture
def sample_mismatch():
    trig = np.array([1, 0, 0])
    samp = np.arange(4)
    return trig, samp


@pytest.fixture
def bad_shape():
    trig = np.array([[1, 3], [0, 3], [0, 3]])
    samp = np.arange(4)
    return trig, samp


def test_linear_run_reorder(sample_linear):
    trig, samp = sample_linear
    trig_out, samp_out, start_mask = reorder_circular_samples_with_trigger(trig, samp, reorder_samples=True)
    # The block of ones [1,1] moves to end
    assert np.array_equal(trig_out, [0, 0, 0, 0, 1, 1])
    # Samples shifted identically
    assert np.array_equal(samp_out, np.roll(samp, 2))
    # Start Mask correctly computed
    assert np.array_equal(start_mask, [0, 0, 1, 0, 0, 0])


def test_linear_run_no_reorder_samples(sample_linear):
    trig, samp = sample_linear
    trig_out, samp_out, start_mask = reorder_circular_samples_with_trigger(trig, samp, reorder_samples=False)
    # Trigger reordered but samples unchanged
    assert np.array_equal(trig_out, [0, 0, 0, 0, 1, 1])
    assert np.array_equal(samp_out, samp)
    # Start Mask correctly computed
    assert np.array_equal(start_mask, [0, 0, 1, 0, 0, 0])


def test_wrap_run_reorder(sample_wrap):
    trig, samp = sample_wrap
    trig_out, samp_out, start_mask = reorder_circular_samples_with_trigger(trig, samp, reorder_samples=True)
    # Wrap block of ones placed at end
    assert np.array_equal(trig_out, [0, 0, 0, 1, 1])
    assert np.array_equal(samp_out, np.roll(samp, -1))
    # Start Mask correctly computed
    assert np.array_equal(start_mask, [0, 0, 0, 0, 1])


def test_no_ones_raises(sample_no_ones):
    trig, samp = sample_no_ones
    with pytest.raises(ValueError):
        reorder_circular_samples_with_trigger(trig, samp, reorder_samples=True)


def test_non_contiguous_raises(sample_non_contiguous):
    trig, samp = sample_non_contiguous
    with pytest.raises(ValueError):
        reorder_circular_samples_with_trigger(trig, samp, reorder_samples=True)


def test_shape_mismatch_raises(sample_mismatch):
    trig, samp = sample_mismatch
    with pytest.raises(ValueError):
        reorder_circular_samples_with_trigger(trig, samp, reorder_samples=True)


def test_bad_shape_raises(bad_shape):
    trig, samp = bad_shape
    with pytest.raises(ValueError):
        reorder_circular_samples_with_trigger(trig, samp, reorder_samples=True)
