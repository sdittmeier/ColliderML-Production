#!/usr/bin/env python3
"""
Run all EDM4HEP to HDF5 conversions in sequence, driven by a YAML config.
"""

import argparse
import gc
import time
from pathlib import Path
import yaml
import logging
import logging.config
from tqdm import tqdm
import pandas as pd
from pyedm4hep import EDM4hepEventBatch
import awkward as ak
import psutil
import os

def log_memory(label: str):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()
    logger.info(f"MEMORY [{label}]: RSS={mem_info.rss / 1024 / 1024:.2f} MB, VMS={mem_info.vms / 1024 / 1024:.2f} MB, Global Used={vm.used / 1024 / 1024:.2f} MB, Global Free={vm.free / 1024 / 1024:.2f} MB, Global Percent={vm.percent}%")

from convert_particles import convert_particles, build_particles_df_with_parents_and_vertex, write_particles_with_selection
from convert_calorimeter import process_calohits_batch, write_calohits_with_selection
# Reuse per-event tracks processing from dedicated module
from convert_tracks import process_event_for_tracks
from convert_digihits import convert_digihits, process_event_for_digihits, write_digihits_with_selection

from utils.path_utils import make_dir, get_run_paths
from utils.driver import iterate_and_process_chunks, local_events_for_run
from utils.track_utils import (
    load_root_file,
    build_hdf5_tracks,
    write_tracks_with_selection,
    normalize_tracksummary_df,
)

logger = logging.getLogger(__name__)


def _get_objects(config: dict) -> list[str]:
    objs = config.get("objects", ["tracker_hits", "tracks", "particles", "calo_hits"])  # default set
    return [obj.lower() for obj in objs]


def _compute_paths(config: dict) -> tuple[Path, Path, str, str]:
    campaign = config["campaign"]
    dataset = config["dataset"]
    version = config["version"]
    common_cfg = config.get("common", {})

    # Path layout differs between two callers:
    # - Production NERSC pipeline writes each campaign's runs under
    #   ``<output_base_dir>/<campaign>/<dataset>/<version>/runs/<N>/``,
    #   so convert_all needs the campaign-keyed prefix.
    # - The colliderml.simulate driver writes flat to
    #   ``<output_base_dir>/runs/<N>/`` (no campaign prefix), which is
    #   what get_run_paths() expects out of the box.
    # The YAML can declare ``input_base_dir`` explicitly to override the
    # campaign-keyed default. The docker_test configs set it to the same
    # value as ``common.output_base_dir`` so the simulate driver works.
    if "input_base_dir" in config:
        input_base_dir = Path(config["input_base_dir"])
    else:
        input_base_dir = Path(common_cfg["output_base_dir"]) / campaign / dataset / version
    output_base_dir = Path(config.get("h5_output_dir", common_cfg["output_base_dir"]))
    dataset_base = f"{campaign}/{dataset}/{version}"
    dataset_name_dot = dataset_base.replace("/", ".")
    return input_base_dir, output_base_dir, dataset_base, dataset_name_dot


def _prepare_output_dirs(output_base_dir: Path, dataset_base: str, output_format: str = 'hdf5') -> tuple[Path, Path, Path, Path]:
    """
    Prepare output directories for conversion results.
    
    Args:
        output_base_dir: Base output directory
        dataset_base: Dataset path (campaign/dataset/version)
        output_format: Output format - 'hdf5' (default) or 'parquet'
    
    Returns:
        Tuple of (particles_out_dir, trkhits_out_dir, tracks_out_dir, calo_out_dir)
    """
    format_subdir = output_format if output_format in ['hdf5', 'parquet'] else 'hdf5'
    particles_out_dir = make_dir(output_base_dir, f"{dataset_base}/{format_subdir}/truth/particles")
    trkhits_out_dir = make_dir(output_base_dir, f"{dataset_base}/{format_subdir}/reco/tracker_hits")
    tracks_out_dir = make_dir(output_base_dir, f"{dataset_base}/{format_subdir}/reco/tracks")
    calo_out_dir = make_dir(output_base_dir, f"{dataset_base}/{format_subdir}/reco/calo_hits")
    return particles_out_dir, trkhits_out_dir, tracks_out_dir, calo_out_dir


def _process_chunk_for_all(
    run_dirs: list[Path],
    start_event: int,
    end_event: int,
    start_run: int,
    start_local: int,
    end_run: int,
    end_local: int,
    *,
    run_size: int,
    objects: list[str],
    dataset_name_dot: str,
    particles_out_dir: Path,
    trkhits_out_dir: Path,
    tracks_out_dir: Path,
    calo_out_dir: Path,
    particles_columns_keep: list[str] | None,
    digihits_columns_keep: list[str] | None,
    min_particle_energy: float | None,
    min_tracker_hits: int | None,
    min_calo_hits: int | None,
    digihits_measurements_columns: list[str] | None,
    tracksummary_file: str,
    # new optional selection for tracks output
    tracks_columns_keep: list[str] | None = None,
    calo_columns_keep: list[str] | None = None,
    # calorimeter thresholds
    ecal_energy_threshold: float = 5.0e-5,
    hcal_energy_threshold: float = 2.5e-4,
    time_min: float = -1.0,
    time_max: float = 10.0,
    output_format: str = 'hdf5',
    row_group_size: int | None = None,
    preserve_unmatched_particle_id: bool = False,
) -> None:
    chunk_start_time = time.time()
    logger.info(f"Starting chunk processing for events {start_event}-{end_event}")
    log_memory("Start Chunk")

    particles_frames: list[pd.DataFrame] = []
    digihits_frames: list[pd.DataFrame] = []
    tracks_frames: list[pd.DataFrame] = []
    calo_frames: list[pd.DataFrame] = []
    seen_pairs_tracks: set[tuple[int, int]] = set()
    seen_pairs_particles: set[tuple[int, int]] = set()
    seen_pairs_hits: set[tuple[int, int]] = set()
    seen_pairs_calo: set[tuple[int, int]] = set()

    run_processing_time = 0.0
    
    for abs_run in tqdm(range(start_run, end_run + 1), leave=False):
        run_start_time = time.time()
        run_dir = run_dirs[abs_run]
        edm4hep_path = Path(run_dir) / "edm4hep.root"
        if not edm4hep_path.exists():
            logger.warning(f"Missing EDM4hep file: {edm4hep_path}")
            continue

        local_start, local_stop = local_events_for_run(
            start_run=start_run,
            start_local=start_local,
            end_run=end_run,
            end_local=end_local,
            abs_run=abs_run,
            run_size=run_size,
        )
        local_count = max(0, local_stop - local_start)
        local_events = (local_start, local_stop)

        # Info log: per-run context
        if local_count > 0:
            local_events_str = f"{local_start}-{local_stop-1} (n={local_count})"
        else:
            local_events_str = "<empty>"
        logger.info(
            f"Run {abs_run}: dir={run_dir} edm4hep={edm4hep_path} local_events={local_events_str}"
        )

        # Load only local events for this run to reduce I/O
        batch_load_start = time.time()
        # Prefer passing a range to activate entry_start/stop in the loader
        batch = EDM4hepEventBatch(str(edm4hep_path), events=local_events, condense_calo=False)
        logger.debug(
            f"EDM4hep batch load for run {abs_run} (events={local_count}): {time.time() - batch_load_start:.3f}s"
        )

        tracksummary_arrays = None
        track_fitting_df_run = None
        digihits_run_df = None

        # Process calorimeter FIRST so we have accurate calo hit counts for particle filtering
        if "calo_hits" in objects:
            calo_start_time = time.time()
            # Load calorimeter data once per run
            calo_fetch_start = time.time()
            batch._ensure_loaded("calo_hits")
            batch._ensure_loaded("calo_contributions")
            calo_hits_all = batch.get_calo_hits_df()
            calo_contributions_all = batch.get_calo_contributions_df()
            logger.debug(f"Loaded calorimeter DataFrames for run {abs_run} in {time.time() - calo_fetch_start:.3f}s")
            
            if calo_hits_all is not None and not calo_hits_all.empty and \
               calo_contributions_all is not None and not calo_contributions_all.empty:
                # Process entire run at once (vectorized, no event loop!)
                run_calo_df = process_calohits_batch(
                    calo_hits_all,
                    calo_contributions_all,
                    ecal_energy_threshold=ecal_energy_threshold,
                    hcal_energy_threshold=hcal_energy_threshold,
                    time_min=time_min,
                    time_max=time_max,
                )
                
                if not run_calo_df.empty:
                    # Update event_ids to global numbering
                    run_calo_df['event_id'] = run_calo_df['event_id'] + abs_run * run_size
                    
                    # Track which events we processed
                    for local_event_num in run_calo_df['event_id'].unique():
                        local_event = local_event_num - abs_run * run_size
                        pair = (abs_run, local_event)
                        if pair in seen_pairs_calo:
                            logger.error(
                                f"Overlap detected for calo_hits on (run,local_event)=({abs_run},{local_event})"
                            )
                        seen_pairs_calo.add(pair)
                    
                    calo_frames.append(run_calo_df)
                    logger.info(
                        f"Run {abs_run}: calo_hits rows={len(run_calo_df)} events={run_calo_df['event_id'].nunique()}"
                    )
            else:
                logger.warning(f"Missing or empty calorimeter data for run {abs_run}")
            
            calo_time = time.time() - calo_start_time
            logger.debug(f"Calorimeter processing for run {abs_run}: {calo_time:.3f}s")

        if "particles" in objects:
            particles_start_time = time.time()
            particles_root_path = Path(run_dir) / "particles.root"
            digi_particles_df_run = None
            if particles_root_path.exists():
                try:
                    included_columns = [
                        "event_id",
                        "vx",
                        "vy",
                        "vz",
                        "px",
                        "py",
                        "pz",
                        "vertex_primary",
                        "perigee_d0",
                        "perigee_z0",
                    ]
                    digi_particles_df_run = load_root_file(str(particles_root_path), included_columns=included_columns, events=local_events)
                except Exception as e:
                    logger.warning(f"Failed to load particles.root at {particles_root_path}: {e}")

            if local_count > 0:
                df_run = build_particles_df_with_parents_and_vertex(
                    batch,
                    str(edm4hep_path),
                    digi_particles_df_run,
                    local_events=local_events,
                    min_particle_energy=min_particle_energy,
                    min_tracker_hits=min_tracker_hits,
                    min_calo_hits=min_calo_hits,
                )
                if not df_run.empty and "event_id" in df_run.columns:
                    # Update event_id in-place (no copy needed)
                    global_event_nums = df_run["event_id"] + abs_run * run_size
                    df_run["event_id"] = global_event_nums
                if not df_run.empty:
                    for le in range(local_events[0], local_events[1]):
                        pair = (abs_run, le)
                        if pair in seen_pairs_particles:
                            logger.error(f"Overlap detected for particles on (run,local_event)=({abs_run},{le})")
                        seen_pairs_particles.add(pair)
                    particles_frames.append(df_run)
                    logger.info(
                        f"Run {abs_run}: particles rows={len(df_run)} events={df_run.event_id.nunique() if 'event_id' in df_run.columns else 'n/a'}"
                    )
            particles_time = time.time() - particles_start_time
            logger.debug(f"Particles processing for run {abs_run}: {particles_time:.3f}s")
        
        if "tracker_hits" in objects:
            digihits_start_time = time.time()
            measurements_path = Path(run_dir) / "measurements.root"
            if measurements_path.exists():
                try:
                    included_columns = (
                        digihits_measurements_columns
                        if digihits_measurements_columns is not None
                        else [
                        "event_nr",
                            "volume_id",
                            "layer_id",
                            "surface_id",
                            "rec_gx",
                            "rec_gy",
                            "rec_gz",
                            "true_x",
                            "true_y",
                            "true_z",
                        ]
                    )
                    digi_measurements_df_all = load_root_file(str(measurements_path), included_columns=included_columns)
                except Exception as e:
                    logger.error(f"Failed to load measurements for run {abs_run}: {e}")
                    digi_measurements_df_all = pd.DataFrame()
                hits_fetch_start = time.time()
                hits_all = batch.get_tracker_hits_df()
                logger.debug(f"Loaded tracker hits DataFrame for run {abs_run} in {time.time() - hits_fetch_start:.3f}s")
                evs_for_run = []
                for local_event_num in range(local_events[0], local_events[1]):
                    ev_hits = hits_all[hits_all.event_id == local_event_num] if not hits_all.empty else None
                    # Boolean indexing creates a view; process_event_for_digihits will copy if needed
                    ev_meas = digi_measurements_df_all[digi_measurements_df_all.event_id == local_event_num]
                    ev_df = process_event_for_digihits(abs_run * run_size + local_event_num, local_event_num, ev_meas, ev_hits)
                    if not ev_df.empty:
                        pair = (abs_run, local_event_num)
                        if pair in seen_pairs_hits:
                            logger.error(
                                f"Overlap detected for tracker_hits on (run,local_event)=({abs_run},{local_event_num})"
                            )
                        seen_pairs_hits.add(pair)
                        digihits_frames.append(ev_df)
                        evs_for_run.append(ev_df)
                if evs_for_run:
                    digihits_run_df = pd.concat(evs_for_run, ignore_index=True)
                    logger.info(
                        f"Run {abs_run}: tracker_hits rows={len(digihits_run_df)} events={len(evs_for_run)}"
                    )
            else:
                logger.warning(f"Missing measurements file: {measurements_path}")
            digihits_time = time.time() - digihits_start_time
            logger.debug(f"Digihits processing for run {abs_run}: {digihits_time:.3f}s")

        if "tracks" in objects and digihits_run_df is not None:
            tracks_load_start = time.time()
            ts_path = Path(run_dir) / tracksummary_file
            track_fitting_df_run = None
            if ts_path.exists():
                try:
                    included_tracksummary_columns = [
                        "event_id",
                        "event_nr",
                        "track_id",
                        "track_nr",
                        "measurementIDs",
                        "nMeasurements",
                        "majorityParticleId_particle",
                        "eLOC0_fit",
                        "eLOC1_fit",
                        "ePHI_fit",
                        "eTHETA_fit",
                        "eQOP_fit",
                        "eT_fit",
                        "t_d0",
                        "t_z0",
                        "t_phi",
                        "t_theta",
                        "t_charge",
                        "t_p",
                        "t_pT",
                        "t_time",
                    ]
                    df_raw = load_root_file(
                        str(ts_path),
                        included_columns=included_tracksummary_columns,
                        events=local_events,
                    )
                    track_fitting_df_run = normalize_tracksummary_df(df_raw)
                except Exception as e:
                    logger.warning(f"Failed to load tracksummary at {ts_path}: {e}")
            tracks_load_time = time.time() - tracks_load_start
            logger.debug(f"Track summary loading for run {abs_run}: {tracks_load_time:.3f}s")

            tracks_proc_start_time = time.time()
            run_tracks_rows = 0
            for local_event_num in range(local_events[0], local_events[1]):
                global_event_num = abs_run * run_size + local_event_num
                try:
                    per_ev_start = time.time()
                    # Boolean indexing creates a view; function will copy if needed
                    event_df = process_event_for_tracks(
                        run_dir=Path(run_dir),
                        local_event_num=local_event_num,
                        global_event_num=global_event_num,
                        track_fitting_df_event=(
                            track_fitting_df_run[track_fitting_df_run["event_id"] == local_event_num].copy()
                            if track_fitting_df_run is not None and "event_id" in track_fitting_df_run.columns
                            else pd.DataFrame()
                        ),
                        digihits_run_df=digihits_run_df,
                    )
                    logger.debug(
                        f"Tracks event processed run={abs_run} local={local_event_num} rows={0 if event_df is None else len(event_df)} in {time.time() - per_ev_start:.3f}s"
                    )
                except Exception as e:
                    logger.warning(
                        f"Tracks processing failed for (run,local)=({abs_run},{local_event_num}): {e}"
                    )
                    continue
                if event_df is None or event_df.empty:
                    continue
                pair = (abs_run, local_event_num)
                if pair in seen_pairs_tracks:
                    logger.error(
                        f"Overlap detected for tracks on (run,local_event)=({abs_run},{local_event_num})"
                    )
                seen_pairs_tracks.add(pair)
                tracks_frames.append(event_df)
                run_tracks_rows += len(event_df)
            tracks_proc_time = time.time() - tracks_proc_start_time
            logger.debug(f"Tracks processing for run {abs_run}: {tracks_proc_time:.3f}s")
            logger.info(
                f"Run {abs_run}: tracks rows={run_tracks_rows} events={local_count}"
            )
        
        run_time = time.time() - run_start_time
        run_processing_time += run_time
        logger.debug(f"Run {abs_run} total processing time: {run_time:.3f}s")
        
        # Delete batch object and force garbage collection to free memory
        del batch
        gc.collect()
        log_memory(f"End Run {abs_run}")

    # File writing phase
    writing_start_time = time.time()
    expected_events = end_event - start_event + 1
    
    # Determine file extension based on output format
    file_ext = '.parquet' if output_format == 'parquet' else '.h5'
    
    if "particles" in objects and particles_frames:
        particles_write_start = time.time()
        log_memory("Before Particles Concat")
        particles_all = pd.concat(particles_frames, ignore_index=True)
        log_memory("After Particles Concat")
        particles_out = Path(particles_out_dir) / (
            f"{dataset_name_dot}.truth.particles.events{start_event}-{end_event}{file_ext}"
        )
        processed_events_particles = len(seen_pairs_particles)
        if processed_events_particles != expected_events:
            logger.warning(
                f"Particles chunk events expected={expected_events}, processed={processed_events_particles}"
            )
        logger.info(f"Writing particles to: {particles_out} (rows={len(particles_all)})")
        write_particles_with_selection(particles_all, str(particles_out), columns_keep=particles_columns_keep, output_format=output_format, row_group_size=row_group_size)
        if particles_out.exists():
            logger.info(f"Wrote particles file: {particles_out}")
        else:
            logger.warning(f"Particles file not created (possibly filtered to empty): {particles_out}")
        particles_write_time = time.time() - particles_write_start
        logger.debug(f"Particles file writing time: {particles_write_time:.3f}s")
        
    if "tracker_hits" in objects and digihits_frames:
        digihits_write_start = time.time()
        log_memory("Before TrackerHits Concat")
        digihits_all = pd.concat(digihits_frames, ignore_index=True)
        log_memory("After TrackerHits Concat")
        digihits_all = pd.concat(digihits_frames, ignore_index=True)
        trkhits_out = Path(trkhits_out_dir) / (
            f"{dataset_name_dot}.reco.tracker_hits.events{start_event}-{end_event}{file_ext}"
        )
        processed_events_hits = len(seen_pairs_hits)
        if processed_events_hits != expected_events:
            logger.warning(
                f"Tracker hits chunk events expected={expected_events}, processed={processed_events_hits}"
            )
        logger.info(f"Writing tracker hits to: {trkhits_out} (rows={len(digihits_all)})")
        write_digihits_with_selection(digihits_all, str(trkhits_out),
                                     columns_keep=digihits_columns_keep,
                                     output_format=output_format,
                                     row_group_size=row_group_size,
                                     preserve_unmatched_particle_id=preserve_unmatched_particle_id)
        if trkhits_out.exists():
            logger.info(f"Wrote tracker hits file: {trkhits_out}")
        else:
            logger.warning(
                f"Tracker hits file not created (possibly filtered to empty): {trkhits_out}"
            )
        digihits_write_time = time.time() - digihits_write_start
        logger.debug(f"Tracker hits file writing time: {digihits_write_time:.3f}s")
        
    if "tracks" in objects and tracks_frames:
        tracks_write_start = time.time()
        log_memory("Before Tracks Concat")
        tracks_all = pd.concat(tracks_frames, ignore_index=True)
        log_memory("After Tracks Concat")
        tracks_out = Path(tracks_out_dir) / (
            f"{dataset_name_dot}.reco.tracks.events{start_event}-{end_event}{file_ext}"
        )
        processed_events_tracks = len(seen_pairs_tracks)
        if processed_events_tracks != expected_events:
            logger.warning(
                f"Tracks chunk events expected={expected_events}, processed={processed_events_tracks}"
            )
        logger.info(f"Writing tracks to: {tracks_out} (rows={len(tracks_all)})")
        write_tracks_with_selection(tracks_all, str(tracks_out), columns_keep=tracks_columns_keep, output_format=output_format, row_group_size=row_group_size)
        if tracks_out.exists():
            logger.info(f"Wrote tracks file: {tracks_out}")
        else:
            logger.warning(f"Tracks file not created (possibly filtered to empty): {tracks_out}")
        tracks_write_time = time.time() - tracks_write_start
        logger.debug(f"Tracks file writing time: {tracks_write_time:.3f}s")

    if "calo_hits" in objects and calo_frames:
        calo_write_start = time.time()
        log_memory("Before CaloHits Concat")
        calo_all = pd.concat(calo_frames, ignore_index=True)
        log_memory("After CaloHits Concat")
        calo_out = Path(calo_out_dir) / (
            f"{dataset_name_dot}.reco.calo_hits.events{start_event}-{end_event}{file_ext}"
        )
        processed_events_calo = len(seen_pairs_calo)
        if processed_events_calo != expected_events:
            logger.warning(
                f"Calo hits chunk events expected={expected_events}, processed={processed_events_calo}"
            )
        logger.info(f"Writing calo hits to: {calo_out} (rows={len(calo_all)})")
        write_calohits_with_selection(calo_all, str(calo_out), columns_keep=calo_columns_keep, output_format=output_format, row_group_size=row_group_size)
        if calo_out.exists():
            logger.info(f"Wrote calo hits file: {calo_out}")
        else:
            logger.warning(f"Calo hits file not created (possibly filtered to empty): {calo_out}")
        calo_write_time = time.time() - calo_write_start
        logger.debug(f"Calo hits file writing time: {calo_write_time:.3f}s")

    writing_time = time.time() - writing_start_time
    chunk_total_time = time.time() - chunk_start_time
    
    logger.info(f"Chunk {start_event}-{end_event} timing summary:")
    logger.info(f"  Run processing: {run_processing_time:.3f}s")
    logger.info(f"  File writing: {writing_time:.3f}s")
    logger.info(f"  Total chunk time: {chunk_total_time:.3f}s")


 


def convert_all(config: dict, chunk_index: int | None = None) -> None:
    logger.debug(
        f"Starting conversion with config: campaign={config.get('campaign')}, dataset={config.get('dataset')}, version={config.get('version')}"
    )

    input_base_dir, output_base_dir, dataset_base, dataset_name_dot = _compute_paths(config)
    chunk_size = int(config.get("chunk_size", 1000))
    run_size = int(config.get("run_size", 10))
    objects = _get_objects(config)
    
    # Extract output format from config (default to hdf5 for backward compatibility)
    output_format = config.get("output_format", "hdf5")

    logger.debug(f"Input base directory: {input_base_dir}")
    logger.debug(f"Output base directory: {output_base_dir}")
    logger.debug(f"Output format: {output_format}")
    logger.debug(f"Objects to convert: {objects}")

    start_time = time.time()

    run_dirs = get_run_paths(input_base_dir)
    logger.info(f"Found {len(run_dirs)} runs. chunk_size={chunk_size}, run_size={run_size}, chunk_index={chunk_index}, output_format={output_format}")

    particles_out_dir, trkhits_out_dir, tracks_out_dir, calo_out_dir = _prepare_output_dirs(output_base_dir, dataset_base, output_format)

    particles_columns_keep = config.get("particles_columns_keep")
    digihits_columns_keep = config.get("digihits_columns_keep")
    tracksummary_file = config.get("tracksummary_file", "tracksummary_ambi.root")
    tracks_columns_keep = config.get("tracks_columns_keep")
    calo_columns_keep = config.get("calo_columns_keep")
    min_particle_energy = config.get("min_particle_energy")
    min_tracker_hits = config.get("min_tracker_hits")
    min_calo_hits = config.get("min_calo_hits")
    digihits_measurements_columns = config.get("digihits_measurements_columns")
    
    # Extract calorimeter thresholds from config
    calo_config = config.get("calorimeter", {})
    ecal_energy_threshold = calo_config.get("ecal_energy_threshold", 5.0e-5)
    hcal_energy_threshold = calo_config.get("hcal_energy_threshold", 2.5e-4)
    time_min = calo_config.get("time_min", -1.0)
    time_max = calo_config.get("time_max", 10.0)

    row_group_size = config.get("row_group_size")  # None means PyArrow default (single row group)
    preserve_unmatched_particle_id = bool(config.get("preserve_unmatched_particle_id", False))

    processing_start_time = time.time()
    
    iterate_and_process_chunks(
        run_dirs=run_dirs,
        run_size=run_size,
        chunk_size=chunk_size,
        config=config,
        chunk_index=chunk_index,
        process_chunk_fn=lambda start_event, end_event, start_run, start_local, end_run, end_local: _process_chunk_for_all(
            run_dirs,
            start_event,
            end_event,
            start_run,
            start_local,
            end_run,
            end_local,
            run_size=run_size,
            objects=objects,
            dataset_name_dot=dataset_name_dot,
            particles_out_dir=particles_out_dir,
            trkhits_out_dir=trkhits_out_dir,
            tracks_out_dir=tracks_out_dir,
            calo_out_dir=calo_out_dir,
            particles_columns_keep=particles_columns_keep,
            digihits_columns_keep=digihits_columns_keep,
            min_particle_energy=min_particle_energy,
            min_tracker_hits=min_tracker_hits,
            min_calo_hits=min_calo_hits,
            digihits_measurements_columns=digihits_measurements_columns,
            tracksummary_file=tracksummary_file,
            tracks_columns_keep=tracks_columns_keep,
            calo_columns_keep=calo_columns_keep,
            ecal_energy_threshold=ecal_energy_threshold,
            hcal_energy_threshold=hcal_energy_threshold,
            time_min=time_min,
            time_max=time_max,
            output_format=output_format,
            row_group_size=row_group_size,
            preserve_unmatched_particle_id=preserve_unmatched_particle_id,
        ),
    )

    processing_time = time.time() - processing_start_time
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"\nConversion timing summary:")
    logger.info(f"  Setup time: {processing_start_time - start_time:.2f}s")
    logger.info(f"  Processing time: {processing_time:.2f}s")
    logger.info(f"  Total conversion time: {total_time:.2f}s")
    logger.debug("Conversion process completed successfully")


def main():
    main_start_time = time.time()

    # Accept the same baseline args every other pipeline stage takes
    # (--output, --events, --seed, --output-subdir) even though we only
    # need --config + --chunk-index ourselves. The colliderml.simulate
    # driver passes --output/--output-subdir/--seed to every stage in
    # the manifest; rejecting them here breaks the pipeline.
    parser = argparse.ArgumentParser(description="Convert all EDM4HEP data to HDF5 (config-driven)")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--chunk-index", type=int, default=None, help="Optional chunk index to process (for distributed runs)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (ignored — the YAML config controls output paths)")
    parser.add_argument("--events", "-n", type=int, default=None,
                        help="Number of events (ignored — converts whatever the prior stages produced)")
    parser.add_argument("--seed", type=str, default=None,
                        help="Random seed (ignored — conversion is deterministic)")
    parser.add_argument("--output-subdir", type=str, default=None,
                        help="Output subdirectory (ignored — the YAML config controls output paths)")
    args = parser.parse_args()
    
    config_load_start = time.time()
    logger.debug(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger.debug("Config loaded successfully")
    config_load_time = time.time() - config_load_start
    logger.debug(f"Config loading time: {config_load_time:.3f}s")
    
    # One-liner logging control: honor simple config key "log_level" (default INFO)
    level_name = str(config.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        force=True,
    )
    
    logger.debug("Starting convert_all function")
    convert_all(config, chunk_index=args.chunk_index)
    logger.debug("convert_all function completed")
    
    main_total_time = time.time() - main_start_time
    logger.info(f"Total script execution time: {main_total_time:.2f}s")

if __name__ == "__main__":
    main()
