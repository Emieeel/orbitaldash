"""Trajectory input models and normalization helpers.

This module defines lightweight containers for trajectory-aware visualization,
where each frame can provide geometry, orbital coefficients, and optional
metadata such as energies along a reaction coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .geometry import standardize_geometry


@dataclass
class TrajectoryFrame:
    """Single trajectory frame.

    Attributes:
        geometry: Molecular geometry (any input accepted by ``standardize_geometry``).
        mo_coeff: MO coefficient matrix of shape (n_ao, n_mo).
        basis: Basis specification or PySCF mol-like object.
        mo_energies: Optional per-orbital energies (n_mo,).
        mo_occ: Optional per-orbital occupations (n_mo,).
        metadata: Optional per-frame metadata. Common keys: ``energy``,
            ``reaction_coordinate``, ``label``.
        custom_eval: Optional callable ``custom_eval(grid) -> (P, n_mo)``.
    """

    geometry: Any
    mo_coeff: np.ndarray
    basis: Optional[Any] = None
    mo_energies: Optional[np.ndarray] = None
    mo_occ: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    custom_eval: Optional[Any] = None


@dataclass
class TrajectoryData:
    """Normalized trajectory container."""

    frames: list[TrajectoryFrame]
    metadata: dict[str, Any] = field(default_factory=dict)
    global_basis: Optional[Any] = None


def _as_optional_array(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    return np.asarray(values)


def _normalize_frame(raw: Any, default_basis: Optional[Any]) -> TrajectoryFrame:
    if isinstance(raw, TrajectoryFrame):
        frame = raw
    elif isinstance(raw, Mapping):
        frame = TrajectoryFrame(
            geometry=raw.get("geometry"),
            mo_coeff=np.asarray(raw.get("mo_coeff")),
            basis=raw.get("basis"),
            mo_energies=_as_optional_array(raw.get("mo_energies")),
            mo_occ=_as_optional_array(raw.get("mo_occ")),
            metadata=dict(raw.get("metadata") or {}),
            custom_eval=raw.get("custom_eval"),
        )
    else:
        raise ValueError("Each trajectory frame must be a TrajectoryFrame or mapping")

    if frame.geometry is None:
        raise ValueError("Each trajectory frame must include 'geometry'")
    if frame.mo_coeff is None:
        raise ValueError("Each trajectory frame must include 'mo_coeff'")

    frame.mo_coeff = np.asarray(frame.mo_coeff)
    frame.basis = frame.basis if frame.basis is not None else default_basis
    frame.geometry = standardize_geometry(frame.geometry)

    if frame.mo_energies is not None:
        frame.mo_energies = np.asarray(frame.mo_energies).reshape(-1)
    if frame.mo_occ is not None:
        frame.mo_occ = np.asarray(frame.mo_occ).reshape(-1)

    return frame


def standardize_trajectory(
    trajectory: Any,
    default_basis: Optional[Any] = None,
) -> TrajectoryData:
    """Normalize trajectory-like input into ``TrajectoryData``.

    Accepted inputs:
        - ``TrajectoryData``
        - sequence of ``TrajectoryFrame``
        - sequence of dict-like frames
        - dict with keys ``frames`` and optional ``metadata``, ``global_basis``
    """
    if isinstance(trajectory, TrajectoryData):
        data = trajectory
        data.frames = [_normalize_frame(f, data.global_basis or default_basis) for f in data.frames]
        return data

    if isinstance(trajectory, Mapping):
        raw_frames = trajectory.get("frames")
        if raw_frames is None:
            raise ValueError("trajectory mapping must include 'frames'")
        metadata = dict(trajectory.get("metadata") or {})
        global_basis = trajectory.get("global_basis", default_basis)
        frames = [_normalize_frame(f, global_basis) for f in raw_frames]
        if not frames:
            raise ValueError("trajectory must contain at least one frame")
        return TrajectoryData(frames=frames, metadata=metadata, global_basis=global_basis)

    if isinstance(trajectory, Sequence):
        frames = [_normalize_frame(f, default_basis) for f in trajectory]
        if not frames:
            raise ValueError("trajectory must contain at least one frame")
        return TrajectoryData(frames=frames)

    raise ValueError("Unsupported trajectory input type")


def frame_energies(trajectory: TrajectoryData) -> np.ndarray:
    """Return per-frame energies if present, else NaN array."""
    energies = []
    for f in trajectory.frames:
        e = f.metadata.get("energy")
        energies.append(float(e) if e is not None else np.nan)
    return np.asarray(energies, dtype=float)


def frame_reaction_coordinates(trajectory: TrajectoryData) -> np.ndarray:
    """Return reaction coordinates if present, else frame indices."""
    coords = []
    has_any = False
    for i, f in enumerate(trajectory.frames):
        rc = f.metadata.get("reaction_coordinate")
        if rc is None:
            coords.append(float(i))
        else:
            has_any = True
            coords.append(float(rc))
    if has_any:
        return np.asarray(coords, dtype=float)
    return np.arange(len(trajectory.frames), dtype=float)
