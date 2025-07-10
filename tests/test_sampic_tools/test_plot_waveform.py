import matplotlib.pyplot as plt
import numpy as np
import pytest

from sampiclyser.sampic_tools import plot_waveform


# A no-op interpolation stub for speed; bypass actual heavy interpolation
def dummy_interp(x_orig, y_orig, **kwargs):
    return x_orig, y_orig


@pytest.fixture(autouse=True)
def patch_apply_interpolation(monkeypatch):
    """Monkey-patch apply_interpolation_method to a no-op for testing."""
    import sampiclyser.sampic_tools as sampic_tools

    monkeypatch.setattr(sampic_tools, "apply_interpolation_method", dummy_interp)
    yield


@pytest.fixture
def simple_waveform():
    # 8-sample buffer with a 2-wide trigger block in the middle
    samp = np.arange(8, dtype=float)
    trig = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=int)
    return samp, trig


def test_basic_scatter_only(simple_waveform):
    samp, trig = simple_waveform
    fig, ax = plt.subplots()
    plot_waveform(
        ax=ax,
        hid=0,
        channel=5,
        baseline=0.0,
        samp_arr=samp,
        trig_arr=trig,
        period=1.0,
        color="C0",
        interp_kwargs={},
        label_channel=False,
        label_hit=False,
        reorder_circular_buffer=False,
        reorder_samp_arr=False,
        plot_sample_types=False,
        plot_buffer_start=False,
        explicit_labels=False,
        time_scale=1.0,
    )
    # No lines, only one scatter for all samples
    assert len(ax.lines) == 0
    # All 8 points plotted as dots
    assert len(ax.collections) == 1


def test_interpolation_and_combined_labels(simple_waveform):
    samp, trig = simple_waveform
    fig, ax = plt.subplots()
    plot_waveform(
        ax=ax,
        hid=3,
        channel=2,
        baseline=10.0,
        samp_arr=samp,
        trig_arr=trig,
        period=0.5,
        color="r",
        interp_kwargs={"interpolation_method": "sinc", "interpolation_factor": 2, "interpolation_parameter": 4},
        label_channel=True,
        label_hit=True,
        reorder_circular_buffer=False,
        reorder_samp_arr=False,
        plot_sample_types=False,
        plot_buffer_start=False,
        explicit_labels=False,
        time_scale=2.0,
    )
    # One interpolated line and one scatter
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1
    # Check line label
    line = ax.lines[0]
    assert line.get_label() == "Hit 3 - Channel 2"


def test_interpolation_and_hit_labels(simple_waveform):
    samp, trig = simple_waveform
    fig, ax = plt.subplots()
    plot_waveform(
        ax=ax,
        hid=3,
        channel=2,
        baseline=10.0,
        samp_arr=samp,
        trig_arr=trig,
        period=0.5,
        color="r",
        interp_kwargs={"interpolation_method": "sinc", "interpolation_factor": 2, "interpolation_parameter": 4},
        label_channel=False,
        label_hit=True,
        reorder_circular_buffer=False,
        reorder_samp_arr=False,
        plot_sample_types=False,
        plot_buffer_start=False,
        explicit_labels=False,
        time_scale=2.0,
    )
    # One interpolated line and one scatter
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1
    # Check line label
    line = ax.lines[0]
    assert line.get_label() == "Hit 3"


def test_interpolation_and_channel_labels(simple_waveform):
    samp, trig = simple_waveform
    fig, ax = plt.subplots()
    plot_waveform(
        ax=ax,
        hid=3,
        channel=2,
        baseline=10.0,
        samp_arr=samp,
        trig_arr=trig,
        period=0.5,
        color="r",
        interp_kwargs={"interpolation_method": "sinc", "interpolation_factor": 2, "interpolation_parameter": 4},
        label_channel=True,
        label_hit=False,
        reorder_circular_buffer=False,
        reorder_samp_arr=False,
        plot_sample_types=False,
        plot_buffer_start=False,
        explicit_labels=False,
        time_scale=2.0,
    )
    # One interpolated line and one scatter
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1
    # Check line label
    line = ax.lines[0]
    assert line.get_label() == "Channel 2"


def test_interpolation_and_none_labels(simple_waveform):
    samp, trig = simple_waveform
    fig, ax = plt.subplots()
    plot_waveform(
        ax=ax,
        hid=3,
        channel=2,
        baseline=10.0,
        samp_arr=samp,
        trig_arr=trig,
        period=0.5,
        color="r",
        interp_kwargs={"interpolation_method": "sinc", "interpolation_factor": 2, "interpolation_parameter": 4},
        label_channel=False,
        label_hit=False,
        reorder_circular_buffer=False,
        reorder_samp_arr=False,
        plot_sample_types=False,
        plot_buffer_start=False,
        explicit_labels=False,
        time_scale=2.0,
    )
    # One interpolated line and one scatter
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1
    # Check line label
    line = ax.lines[0]
    assert line.get_label() == "_child0"


def test_sample_type_markers(simple_waveform):
    samp, trig = simple_waveform
    fig, ax = plt.subplots()
    plot_waveform(
        ax=ax,
        hid=1,
        channel=7,
        baseline=0.0,
        samp_arr=samp,
        trig_arr=trig,
        period=1.0,
        color="g",
        interp_kwargs={},
        label_channel=False,
        label_hit=False,
        reorder_circular_buffer=False,
        reorder_samp_arr=False,
        plot_sample_types=True,
        plot_buffer_start=False,
        explicit_labels=True,
        time_scale=1.0,
    )
    # No line, but two scatters: one for non-triggers, one for triggers
    assert len(ax.lines) == 0
    assert len(ax.collections) == 2


def test_buffer_start_marker_and_reorder(simple_waveform):
    samp, trig = simple_waveform
    fig, ax = plt.subplots()
    plot_waveform(
        ax=ax,
        hid=2,
        channel=4,
        baseline=0.0,
        samp_arr=samp,
        trig_arr=trig,
        period=1.0,
        color="b",
        interp_kwargs={},
        label_channel=False,
        label_hit=False,
        reorder_circular_buffer=True,
        reorder_samp_arr=True,
        plot_sample_types=True,
        plot_buffer_start=True,
        explicit_labels=True,
        time_scale=1.0,
    )
    # Should produce three collections: buffer start, non-trigger, trigger
    assert len(ax.collections) == 3
    # The first scatter must contain exactly one marker '>' at the buffer start
    start_coll = ax.collections[0]
    assert start_coll.get_offsets().shape[0] == 1
    # Build legend
    ax.legend()
    # Check that buffer start label is present in legend
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Buffer start" in labels


def test_invalid_inputs():
    # Mismatched lengths
    samp = np.arange(5)
    trig = np.arange(6) % 2
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        plot_waveform(
            ax=ax,
            hid=0,
            channel=0,
            baseline=0.0,
            samp_arr=samp,
            trig_arr=trig,
            period=1.0,
            color="C0",
            interp_kwargs={},
            label_channel=False,
            label_hit=False,
            reorder_circular_buffer=False,
            reorder_samp_arr=False,
            plot_sample_types=False,
            plot_buffer_start=False,
            explicit_labels=False,
            time_scale=1.0,
        )

    # No ones in trig_arr
    trig2 = np.zeros(4, int)
    samp2 = np.arange(4)
    with pytest.raises(ValueError):
        plot_waveform(
            ax=ax,
            hid=0,
            channel=0,
            baseline=0.0,
            samp_arr=samp2,
            trig_arr=trig2,
            period=1.0,
            color="C0",
            interp_kwargs={},
            label_channel=False,
            label_hit=False,
            reorder_circular_buffer=True,
            reorder_samp_arr=True,
            plot_sample_types=False,
            plot_buffer_start=False,
            explicit_labels=False,
            time_scale=1.0,
        )
