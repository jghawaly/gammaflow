"""
RADAI synthetic gamma-ray detection dataset loader.

The RADAI dataset consists of simulated list-mode gamma-ray measurements
in HDF5 format from a mobile 2"x4"x16" NaI(Tl) detector traversing
synthetic urban environments. Each file contains hundreds of runs, each
providing per-photon event data with energy (keV), inter-event time
(microseconds), and optional per-photon source attribution.

Three splits are available:

- ``training`` — 300 runs, ~3600 s each, full per-photon ground truth
- ``testing``  — 300 runs, ~3600 s each, no ground truth
- ``developer`` — 1950 runs, ~200 s each, full ground truth, high SNR

Requires ``h5py`` (not a core gammaflow dependency).  Install with::

    pip install h5py

For more information see:
    Ghawaly Jr, J. M., Archer, D. E., Nicholson, A. D., Peplow, D. E.,
    Prins, N. J., Joshi, T. H. Y., ... & Nachtsheim, A. C. (2025).
    RADAI: A Large-Scale Realistic Dataset for Radiation Detection
    Algorithm Development. IEEE Transactions on Nuclear Science.

Typical usage::

    from gammaflow.datasets import RADAIDataset

    ds = RADAIDataset("/path/to/radai")

    # Simple: load a single run
    listmode, meta = ds.load_run(0, split="training")

    # Filter by photon source ID
    lm_bg, _ = ds.load_run(0, source_ids=[0])        # background only
    lm_cs, _ = ds.load_run(0, source_ids=[7])         # Cs-137 only

    # Filter by background component
    lm_norm, _ = ds.load_run(0, background_ids=[1, 2, 3])  # K-40 + U + Th

    # SNR-based source windows
    windows = ds.compute_source_windows(0, snr_threshold=0.5)
    lm_win, win = ds.load_source_window(0, window_index=0, snr_threshold=0.5)

    # Background segments (no detectable source)
    lm_clean, meta = ds.load_background_segments(0, snr_threshold=0.5)

    # Convenience: directly as SpectralTimeSeries
    ts, meta = ds.load_run_as_time_series(0, integration_time=1.0)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Generator

import numpy as np

from gammaflow.core.listmode import ListMode
from gammaflow.core.time_series import SpectralTimeSeries

try:
    import h5py

    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


SPLITS = ("training", "testing", "developer")

_DEFAULT_VERSION = "v4.3"
_DEFAULT_SNR_THRESHOLD = 0.5
_DEFAULT_TIME_BIN_S = 1.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SourceEncounter:
    """
    Pre-computed metadata for a single source encounter within a run.

    Read directly from the HDF5 ``sources/`` group — no computation
    required.

    Attributes
    ----------
    source_id : int
        Source ID from the dataset's ``source_ids`` mapping.
    source_name : str
        Human-readable source name.
    time_ms : float
        Time of closest approach in milliseconds from run start.
    distance_m : float
        Distance at closest approach in meters.
    shielding_id : int
        Shielding configuration ID.
    shielding_name : str
        Human-readable shielding name.
    activity : float
        Source activity (Bq).
    standoff : float
        Standoff distance in meters.
    location_id : int
        Source location identifier within the block.
    snr_peak : float
        Pre-computed peak signal-to-noise ratio.
    snr_integral : float
        Pre-computed integral signal-to-noise ratio.
    """

    source_id: int
    source_name: str
    time_ms: float
    distance_m: float
    shielding_id: int
    shielding_name: str
    activity: float
    standoff: float
    location_id: int
    snr_peak: float
    snr_integral: float

    @property
    def time_s(self) -> float:
        """Time of closest approach in seconds."""
        return self.time_ms / 1000.0


@dataclass
class SourceWindow:
    """
    A contiguous time region where a source has SNR above threshold.

    Computed by :meth:`RADAIDataset.compute_source_windows` from the
    per-photon source IDs and a user-specified SNR threshold.

    Attributes
    ----------
    source_id : int
        Source ID whose photons define this window.
    source_name : str
        Human-readable source name.
    start_s : float
        Window start time in seconds from run start.
    end_s : float
        Window end time in seconds from run start.
    peak_snr : float
        Maximum per-bin SNR within this window.
    encounter : SourceEncounter or None
        The nearest pre-computed source encounter, if matched.
    """

    source_id: int
    source_name: str
    start_s: float
    end_s: float
    peak_snr: float
    encounter: Optional[SourceEncounter]

    @property
    def duration_s(self) -> float:
        """Window duration in seconds."""
        return self.end_s - self.start_s

    @property
    def center_s(self) -> float:
        """Window center time in seconds."""
        return (self.start_s + self.end_s) / 2.0


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


class RADAIDataset:
    """
    Loader for the RADAI synthetic gamma-ray detection dataset.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the RADAI HDF5 files.
    version : str
        Version suffix for filenames (default ``'v4.3'``).
        Files are expected as ``{split}_{version}.h5``.

    Attributes
    ----------
    data_dir : Path
        Resolved dataset directory.
    source_map : dict
        Mapping from source ID to source name.
    background_map : dict
        Mapping from background ID to background component name.
    shielding_map : dict
        Mapping from shielding ID to shielding name.

    Examples
    --------
    >>> ds = RADAIDataset("/path/to/radai")
    >>> listmode, meta = ds.load_run(0, split="training")
    >>> print(meta["n_events"], listmode.total_time)
    """

    def __init__(self, data_dir: Union[str, Path], version: str = _DEFAULT_VERSION):
        if not _H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required for RADAIDataset. Install with: pip install h5py"
            )

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        self.version = version

        # Discover which split files are present
        self._split_paths: Dict[str, Path] = {}
        for split in SPLITS:
            path = self._resolve_split_path(split)
            if path is not None:
                self._split_paths[split] = path

        if not self._split_paths:
            raise FileNotFoundError(
                f"No RADAI HDF5 files found in {self.data_dir}. "
                f"Expected files like training_{version}.h5"
            )

        # Load ID → name mappings from the first available file
        self.source_map: Dict[int, str] = {}
        self.background_map: Dict[int, str] = {}
        self.shielding_map: Dict[int, str] = {}
        self._load_mappings()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_split_path(self, split: str) -> Optional[Path]:
        """Return the HDF5 path for *split*, or ``None`` if not found."""
        expected = self.data_dir / f"{split}_{self.version}.h5"
        if expected.exists():
            return expected
        # Fallback: glob for any version of this split
        candidates = sorted(self.data_dir.glob(f"{split}_*.h5"))
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _get_split_path(self, split: str) -> Path:
        """Return the HDF5 path for *split*, raising if not found."""
        if split not in SPLITS:
            raise ValueError(
                f"Unknown split '{split}'. Valid splits: {SPLITS}"
            )
        if split not in self._split_paths:
            raise FileNotFoundError(
                f"No HDF5 file found for split '{split}' in {self.data_dir}"
            )
        return self._split_paths[split]

    def _has_ground_truth(self, split: str) -> bool:
        """Whether *split* has per-photon IDs and source metadata."""
        return split != "testing"

    def _require_ground_truth(self, split: str, operation: str) -> None:
        """Raise ``ValueError`` if *split* lacks ground truth."""
        if not self._has_ground_truth(split):
            raise ValueError(
                f"{operation} requires per-photon ground truth, "
                f"which is not available in the '{split}' split."
            )

    def _load_mappings(self) -> None:
        """Read ID → name mappings from root HDF5 attributes."""
        path = next(iter(self._split_paths.values()))
        with h5py.File(path, "r") as f:
            source_ids = f.attrs["source_ids"]
            source_names = f.attrs["source_names"]
            self.source_map = {
                int(sid): str(sn) for sid, sn in zip(source_ids, source_names)
            }

            if "background_ids" in f.attrs:
                bg_ids = f.attrs["background_ids"]
                bg_names = f.attrs["background_names"]
                self.background_map = {
                    int(bid): str(bn) for bid, bn in zip(bg_ids, bg_names)
                }

            if "source_shielding_ids" in f.attrs:
                sh_ids = f.attrs["source_shielding_ids"]
                sh_names = f.attrs["source_shielding_names"]
                self.shielding_map = {
                    int(sid): str(sn) for sid, sn in zip(sh_ids, sh_names)
                }

    def _read_source_encounters(self, run_group) -> List[SourceEncounter]:
        """Build ``SourceEncounter`` list from an HDF5 run group."""
        if "sources" not in run_group:
            return []

        src = run_group["sources"]
        ids = src["id"][:]
        times = src["time"][:]
        distances = src["distance"][:]

        # Optional fields — read if present, else default
        shieldings = src["shielding"][:] if "shielding" in src else np.zeros_like(ids)
        activities = src["activity"][:] if "activity" in src else np.zeros(len(ids), dtype=np.float32)
        standoffs = src["standoff"][:] if "standoff" in src else np.zeros(len(ids), dtype=np.float32)
        location_ids = src["location_id"][:] if "location_id" in src else np.zeros_like(ids)

        snr_peak = np.zeros(len(ids), dtype=np.float32)
        snr_integral = np.zeros(len(ids), dtype=np.float32)
        if "snr" in src:
            if "peak" in src["snr"]:
                snr_peak = src["snr"]["peak"][:]
            if "integral" in src["snr"]:
                snr_integral = src["snr"]["integral"][:]

        encounters = []
        for i in range(len(ids)):
            sid = int(ids[i])
            encounters.append(
                SourceEncounter(
                    source_id=sid,
                    source_name=self.source_map.get(sid, f"Unknown({sid})"),
                    time_ms=float(times[i]),
                    distance_m=float(distances[i]),
                    shielding_id=int(shieldings[i]),
                    shielding_name=self.shielding_map.get(int(shieldings[i]), "Unknown"),
                    activity=float(activities[i]),
                    standoff=float(standoffs[i]),
                    location_id=int(location_ids[i]),
                    snr_peak=float(snr_peak[i]),
                    snr_integral=float(snr_integral[i]),
                )
            )
        return encounters

    @staticmethod
    def _find_contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous True regions in a boolean array.

        Returns a list of ``(start_idx, end_idx)`` pairs (end exclusive).
        """
        if not np.any(mask):
            return []
        diff = np.diff(mask.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        # Handle edge: mask starts True
        if mask[0]:
            starts = np.concatenate([[0], starts])
        # Handle edge: mask ends True
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])
        return list(zip(starts.tolist(), ends.tolist()))

    # ------------------------------------------------------------------
    # Run listing and metadata
    # ------------------------------------------------------------------

    def list_runs(self, split: str = "training") -> np.ndarray:
        """
        List available run indices for a split.

        Parameters
        ----------
        split : str
            One of ``'training'``, ``'testing'``, ``'developer'``.

        Returns
        -------
        np.ndarray
            Sorted array of integer run indices (0-based).
        """
        path = self._get_split_path(split)
        with h5py.File(path, "r") as f:
            run_keys = list(f["runs"].keys())
        indices = np.array(
            sorted(int(k.lstrip("run")) for k in run_keys), dtype=np.int64
        )
        return indices

    def get_run_metadata(
        self, run_id: int, split: str = "training"
    ) -> Dict[str, Any]:
        """
        Get metadata for a specific run.

        Parameters
        ----------
        run_id : int
            Run index.
        split : str
            Dataset split.

        Returns
        -------
        dict
            Keys include ``run_id``, ``split``, ``start_timestamp``,
            ``end_timestamp``, ``seed``, ``n_events``, and
            ``source_encounters`` (list of :class:`SourceEncounter`,
            if ground truth is available).
        """
        path = self._get_split_path(split)
        with h5py.File(path, "r") as f:
            run_key = f"run{run_id}"
            if run_key not in f["runs"]:
                raise KeyError(f"Run {run_id} not found in '{split}' split")
            run = f["runs"][run_key]

            meta = {
                "run_id": run_id,
                "split": split,
                "start_timestamp": float(run.attrs.get("start_timestamp", 0.0)),
                "end_timestamp": float(run.attrs.get("end_timestamp", 0.0)),
                "seed": int(run.attrs["seed"]) if "seed" in run.attrs else None,
                "n_events": run["listmode"]["dt"].shape[0],
            }

            if self._has_ground_truth(split):
                meta["source_encounters"] = self._read_source_encounters(run)
            else:
                meta["source_encounters"] = []

        return meta

    def get_source_encounters(
        self, run_id: int, split: str = "training"
    ) -> List[SourceEncounter]:
        """
        Get source encounter metadata for a run.

        Parameters
        ----------
        run_id : int
            Run index.
        split : str
            Dataset split (must have ground truth).

        Returns
        -------
        list of SourceEncounter
            One entry per source encounter, sorted by time of closest
            approach.

        Raises
        ------
        ValueError
            If the split has no ground truth (testing).
        """
        self._require_ground_truth(split, "get_source_encounters")
        meta = self.get_run_metadata(run_id, split)
        encounters = meta["source_encounters"]
        encounters.sort(key=lambda e: e.time_ms)
        return encounters

    # ------------------------------------------------------------------
    # Core data loading
    # ------------------------------------------------------------------

    def load_run(
        self,
        run_id: int,
        split: str = "training",
        source_ids: Optional[List[int]] = None,
        background_ids: Optional[List[int]] = None,
    ) -> Tuple[ListMode, Dict[str, Any]]:
        """
        Load a single run as a :class:`~gammaflow.ListMode` object.

        Parameters
        ----------
        run_id : int
            Run index.
        split : str
            Dataset split.
        source_ids : list of int, optional
            Keep only photons whose ``id`` is in this list.  Requires
            ground truth (not available for testing split).
            Common values: ``[0]`` for background-only, ``[7]`` for
            Cs-137 (bare).
        background_ids : list of int, optional
            Keep only photons whose ``background_id`` is in this list.
            Requires ground truth.  Can be combined with *source_ids*
            (logical AND).

        Returns
        -------
        listmode : ListMode
            List-mode event data (time_deltas in seconds, energies
            in keV).
        metadata : dict
            Run metadata including source encounters if available.

        Raises
        ------
        ValueError
            If filtering requested on testing split.
        KeyError
            If *run_id* not found.
        """
        filtering = source_ids is not None or background_ids is not None
        if filtering:
            self._require_ground_truth(split, "Photon filtering")

        path = self._get_split_path(split)
        with h5py.File(path, "r") as f:
            run_key = f"run{run_id}"
            if run_key not in f["runs"]:
                raise KeyError(f"Run {run_id} not found in '{split}' split")
            run = f["runs"][run_key]

            lm = run["listmode"]
            dt_us = lm["dt"][:]  # uint16, microseconds
            energy = lm["energy"][:]  # float32, keV

            # Compute absolute times FIRST (before filtering), since
            # dt[i] is the time since event i-1.  Filtering removes
            # events, which makes the raw dt values meaningless for
            # the remaining events.
            abs_times_us = np.cumsum(dt_us.astype(np.uint32))

            # Build filter mask
            if filtering:
                mask = np.ones(len(dt_us), dtype=bool)
                if source_ids is not None:
                    ids = lm["id"][:]
                    mask &= np.isin(ids, source_ids)
                if background_ids is not None:
                    bg_ids = lm["background_id"][:]
                    mask &= np.isin(bg_ids, background_ids)
                abs_times_us = abs_times_us[mask]
                energy = energy[mask]

            # Build metadata
            meta = {
                "run_id": run_id,
                "split": split,
                "start_timestamp": float(run.attrs.get("start_timestamp", 0.0)),
                "end_timestamp": float(run.attrs.get("end_timestamp", 0.0)),
                "seed": int(run.attrs["seed"]) if "seed" in run.attrs else None,
                "n_events": len(energy),
            }
            if self._has_ground_truth(split):
                meta["source_encounters"] = self._read_source_encounters(run)
            else:
                meta["source_encounters"] = []

        # Convert absolute times to time deltas in seconds
        abs_times_s = abs_times_us.astype(np.float64) * 1e-6
        time_deltas_s = np.empty_like(abs_times_s)
        if len(abs_times_s) > 0:
            time_deltas_s[0] = abs_times_s[0]
            time_deltas_s[1:] = np.diff(abs_times_s)

        listmode = ListMode(time_deltas_s, energy.astype(np.float64))
        return listmode, meta

    def load_runs(
        self,
        run_ids: Optional[List[int]] = None,
        split: str = "training",
        source_ids: Optional[List[int]] = None,
        background_ids: Optional[List[int]] = None,
    ) -> Generator[Tuple[ListMode, Dict[str, Any]], None, None]:
        """
        Generator yielding ``(listmode, metadata)`` for multiple runs.

        Parameters
        ----------
        run_ids : list of int, optional
            Run indices to load.  If ``None``, loads all runs in the
            split.
        split : str
            Dataset split.
        source_ids : list of int, optional
            Photon source ID filter (see :meth:`load_run`).
        background_ids : list of int, optional
            Photon background ID filter (see :meth:`load_run`).

        Yields
        ------
        listmode : ListMode
        metadata : dict
        """
        if run_ids is None:
            run_ids = self.list_runs(split).tolist()

        for rid in run_ids:
            yield self.load_run(
                rid, split=split, source_ids=source_ids, background_ids=background_ids
            )

    # ------------------------------------------------------------------
    # SNR-based source windows
    # ------------------------------------------------------------------

    def compute_source_windows(
        self,
        run_id: int,
        split: str = "training",
        time_bin_s: float = _DEFAULT_TIME_BIN_S,
        snr_threshold: float = _DEFAULT_SNR_THRESHOLD,
    ) -> List[SourceWindow]:
        """
        Compute SNR-defined source windows for a run.

        Bins the list-mode data into time steps, computes per-bin
        SNR = S / sqrt(S + B) for each source, and identifies
        contiguous regions where SNR exceeds *snr_threshold*.

        Parameters
        ----------
        run_id : int
            Run index.
        split : str
            Dataset split (must have ground truth).
        time_bin_s : float
            Width of time bins in seconds (default 1.0).
        snr_threshold : float
            Minimum SNR to consider a source "present" (default 0.5).

        Returns
        -------
        list of SourceWindow
            Windows sorted by start time.

        Raises
        ------
        ValueError
            If the split has no ground truth.
        """
        self._require_ground_truth(split, "compute_source_windows")

        path = self._get_split_path(split)
        with h5py.File(path, "r") as f:
            run_key = f"run{run_id}"
            if run_key not in f["runs"]:
                raise KeyError(f"Run {run_id} not found in '{split}' split")
            run = f["runs"][run_key]

            lm = run["listmode"]
            dt_us = lm["dt"][:]
            photon_ids = lm["id"][:]

            # Get source encounters for matching
            encounters = self._read_source_encounters(run)

        # Compute absolute times in seconds
        abs_times_s = np.cumsum(dt_us.astype(np.float64)) * 1e-6
        end_time = abs_times_s[-1] if len(abs_times_s) > 0 else 0.0

        # Create time bin edges
        time_edges = np.arange(0.0, end_time + time_bin_s, time_bin_s)

        # Bin background counts (id == 0)
        bg_mask = photon_ids == 0
        bg_counts, _ = np.histogram(abs_times_s[bg_mask], bins=time_edges)

        # Find unique non-zero source IDs present in this run
        unique_src_ids = np.unique(photon_ids[photon_ids > 0])

        windows = []
        for sid in unique_src_ids:
            sid = int(sid)
            src_mask = photon_ids == sid
            src_counts, _ = np.histogram(abs_times_s[src_mask], bins=time_edges)

            # SNR = S / sqrt(S + B), guarding against division by zero
            total = src_counts + bg_counts
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.where(total > 0, src_counts / np.sqrt(total), 0.0)

            # Find contiguous regions above threshold
            above = snr >= snr_threshold
            regions = self._find_contiguous_regions(above)

            source_name = self.source_map.get(sid, f"Unknown({sid})")

            for start_idx, end_idx in regions:
                start_s = float(time_edges[start_idx])
                end_s = float(time_edges[min(end_idx, len(time_edges) - 1)])
                peak = float(np.max(snr[start_idx:end_idx]))

                # Match to nearest encounter
                matched_enc = None
                if encounters:
                    window_center = (start_s + end_s) / 2.0
                    best_dist = float("inf")
                    for enc in encounters:
                        if enc.source_id == sid:
                            dist = abs(enc.time_s - window_center)
                            if dist < best_dist:
                                best_dist = dist
                                matched_enc = enc

                windows.append(
                    SourceWindow(
                        source_id=sid,
                        source_name=source_name,
                        start_s=start_s,
                        end_s=end_s,
                        peak_snr=peak,
                        encounter=matched_enc,
                    )
                )

        windows.sort(key=lambda w: w.start_s)
        return windows

    def load_source_window(
        self,
        run_id: int,
        window_index: int = 0,
        split: str = "training",
        time_bin_s: float = _DEFAULT_TIME_BIN_S,
        snr_threshold: float = _DEFAULT_SNR_THRESHOLD,
        source_ids: Optional[List[int]] = None,
        background_ids: Optional[List[int]] = None,
    ) -> Tuple[ListMode, SourceWindow]:
        """
        Load list-mode data for an SNR-defined source window.

        Parameters
        ----------
        run_id : int
            Run index.
        window_index : int
            Index into the list returned by
            :meth:`compute_source_windows` (default 0 = first window).
        split : str
            Dataset split (must have ground truth).
        time_bin_s : float
            Time bin width for SNR computation (seconds).
        snr_threshold : float
            SNR threshold for window definition.
        source_ids : list of int, optional
            Additional photon filtering on the extracted window.
        background_ids : list of int, optional
            Additional photon filtering on the extracted window.

        Returns
        -------
        listmode : ListMode
            Time-sliced list-mode data within the window.
        window : SourceWindow
            The source window metadata.
        """
        windows = self.compute_source_windows(
            run_id, split=split, time_bin_s=time_bin_s, snr_threshold=snr_threshold
        )
        if not windows:
            raise ValueError(
                f"No source windows found in run {run_id} at "
                f"SNR threshold {snr_threshold}"
            )
        if window_index >= len(windows):
            raise IndexError(
                f"window_index {window_index} out of range "
                f"(run {run_id} has {len(windows)} windows)"
            )

        window = windows[window_index]

        lm, _ = self.load_run(
            run_id, split=split, source_ids=source_ids, background_ids=background_ids
        )
        sliced = lm.slice_time(t_min=window.start_s, t_max=window.end_s)
        return sliced, window

    def load_background(
        self,
        run_id: int,
        split: str = "training",
    ) -> Tuple[ListMode, Dict[str, Any]]:
        """
        Load only background photons from a run.

        Uses the per-photon ``id`` column: background photons have
        ``id == 0``.  This is equivalent to
        ``load_run(run_id, source_ids=[0])``.

        Parameters
        ----------
        run_id : int
            Run index.
        split : str
            Dataset split (must have ground truth).

        Returns
        -------
        listmode : ListMode
            Background-only list-mode data.
        metadata : dict
            Run metadata.
        """
        return self.load_run(run_id, split=split, source_ids=[0])

    # ------------------------------------------------------------------
    # Convenience: SpectralTimeSeries
    # ------------------------------------------------------------------

    def load_run_as_time_series(
        self,
        run_id: int,
        split: str = "training",
        integration_time: float = 1.0,
        stride_time: Optional[float] = None,
        energy_bins: Optional[Union[int, np.ndarray]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        source_ids: Optional[List[int]] = None,
        background_ids: Optional[List[int]] = None,
    ) -> Tuple[SpectralTimeSeries, Dict[str, Any]]:
        """
        Load a run and bin it into a :class:`~gammaflow.SpectralTimeSeries`.

        Convenience wrapper around :meth:`load_run` followed by
        :meth:`SpectralTimeSeries.from_list_mode`.

        Parameters
        ----------
        run_id : int
            Run index.
        split : str
            Dataset split.
        integration_time : float
            Spectrum integration time in seconds (default 1.0).
        stride_time : float, optional
            Time between spectrum starts.  Defaults to
            *integration_time* (non-overlapping).
        energy_bins : int or array-like, optional
            Number of energy bins or explicit bin edges.
        energy_range : tuple of float, optional
            ``(min_keV, max_keV)`` for energy binning.
        source_ids : list of int, optional
            Photon source ID filter.
        background_ids : list of int, optional
            Photon background ID filter.

        Returns
        -------
        time_series : SpectralTimeSeries
        metadata : dict
        """
        lm, meta = self.load_run(
            run_id, split=split, source_ids=source_ids, background_ids=background_ids
        )

        ts = SpectralTimeSeries.from_list_mode(
            lm,
            integration_time=integration_time,
            stride_time=stride_time,
            energy_bins=energy_bins,
            energy_range=energy_range,
        )
        return ts, meta

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def load_diagnostics(
        self,
        run_id: int,
        split: str = "training",
    ) -> Dict[str, np.ndarray]:
        """
        Load per-second diagnostic metrics for a run.

        These metrics quantify how much the background varies over time,
        useful for identifying dynamic segments (rain, block transitions).

        Parameters
        ----------
        run_id : int
            Run index.
        split : str
            Dataset split.

        Returns
        -------
        dict
            Keys:

            - ``'timestamps_ms'`` — 1 Hz timestamps in milliseconds
            - ``'gross_count_rate_variation'`` — absolute gross count
              rate variation per second
            - ``'spectral_count_rate_variation'`` — normalized spectral
              shape variation per second
        """
        path = self._get_split_path(split)
        with h5py.File(path, "r") as f:
            run_key = f"run{run_id}"
            if run_key not in f["runs"]:
                raise KeyError(f"Run {run_id} not found in '{split}' split")
            diag = f["runs"][run_key]["diagnostics"]

            result = {}
            if "variation_timestamps" in diag:
                result["timestamps_ms"] = diag["variation_timestamps"][:].astype(np.float64)
            if "gross_count_rate_variation" in diag:
                result["gross_count_rate_variation"] = diag["gross_count_rate_variation"][:]
            if "spectral_count_rate_variation" in diag:
                result["spectral_count_rate_variation"] = diag["spectral_count_rate_variation"][:]

        return result

    # ------------------------------------------------------------------
    # Detector kinematics
    # ------------------------------------------------------------------

    def load_detector_kinematics(
        self,
        run_id: int,
        split: str = "training",
    ) -> Dict[str, np.ndarray]:
        """
        Load detector position and kinematics data for a run.

        Parameters
        ----------
        run_id : int
            Run index.
        split : str
            Dataset split.

        Returns
        -------
        dict
            Keys: ``'time_ms'``, ``'distance_m'``, ``'velocity_mps'``,
            ``'acceleration_mps2'`` (all numpy arrays at ~10 Hz).
        """
        path = self._get_split_path(split)
        with h5py.File(path, "r") as f:
            run_key = f"run{run_id}"
            if run_key not in f["runs"]:
                raise KeyError(f"Run {run_id} not found in '{split}' split")
            pos = f["runs"][run_key]["detector"]["position"]

            result = {"time_ms": pos["time"][:].astype(np.float64)}
            if "distance" in pos:
                result["distance_m"] = pos["distance"][:].astype(np.float64)
            if "velocity" in pos:
                result["velocity_mps"] = pos["velocity"][:].astype(np.float64)
            if "acceleration" in pos:
                result["acceleration_mps2"] = pos["acceleration"][:].astype(np.float64)

        return result

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"RADAIDataset(data_dir='{self.data_dir}'"]
        for split in SPLITS:
            if split in self._split_paths:
                parts.append(f"{split}='{self._split_paths[split].name}'")
        return ", ".join(parts) + ")"
