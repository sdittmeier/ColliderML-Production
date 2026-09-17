#!/usr/bin/env python3
"""
Parquet writing utilities for EDM4HEP to Parquet conversion.

This module provides reusable functions for writing particle physics data
to Parquet format with proper handling of nested/ragged arrays.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def optimize_dtypes_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes for Parquet storage.
    
    Converts to smaller types where possible to reduce file size:
    - float64 -> float32 (where appropriate)
    - int64 -> int32 (where values fit)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    df = df.copy()
    
    for col in df.columns:
        if col == "event_id":
            # Keep event_id as int64 for safety
            continue

        dtype = df[col].dtype

        # Preserve unsigned integer types (e.g. detector enums, geometry IDs, particle IDs)
        if pd.api.types.is_unsigned_integer_dtype(dtype):
            continue

        # Downcast signed integers (except cell_id which needs int64)
        if pd.api.types.is_integer_dtype(dtype) and col != "cell_id":
            try:
                # Check if all values fit in int32
                col_min = df[col].min() if len(df) > 0 else 0
                col_max = df[col].max() if len(df) > 0 else 0
                if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype("int32")
            except Exception:
                pass

        # Downcast floats
        elif pd.api.types.is_float_dtype(dtype):
            try:
                df[col] = df[col].astype("float32")
            except Exception:
                pass
                
    return df


def group_by_event_to_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group DataFrame by event_id and aggregate all other columns into lists.
    Uses pandas groupby for simplicity and correct handling of nested lists.
    
    For columns that already contain lists (like hit_ids), this creates nested lists:
    - hit_ids: list[int] per row → list[list[int]] per event
    
    Args:
        df: Input DataFrame with event_id column and per-particle/hit data
        
    Returns:
        Grouped DataFrame with one row per event and list columns
    """
    if df.empty:
        logger.warning("group_by_event_to_lists received empty DataFrame")
        return df
    
    if 'event_id' not in df.columns:
        raise ValueError("DataFrame must have 'event_id' column for grouping")
    
    # Get the original column names (excluding event_id)
    original_columns = [col for col in df.columns if col != 'event_id']
    
    # Use pandas groupby with list aggregation
    # This correctly handles both scalar columns and columns that already contain lists
    grouped = df.groupby('event_id', as_index=False).agg(list)
    
    # Rename columns to remove any suffixes that pandas might add
    # Build rename dict: map any modified column names back to originals
    rename_dict = {}
    for orig_col in original_columns:
        # Find the column in grouped that corresponds to this original column
        # It might be 'col' or 'col_list' or similar
        for grouped_col in grouped.columns:
            if grouped_col != 'event_id' and (grouped_col == orig_col or grouped_col.startswith(orig_col)):
                if grouped_col != orig_col:
                    rename_dict[grouped_col] = orig_col
                break
    
    if rename_dict:
        grouped = grouped.rename(columns=rename_dict)
    
    logger.debug(f"Grouped {len(df)} rows into {len(grouped)} events")
    
    return grouped


def write_parquet_table(
    df: pd.DataFrame,
    output_file: str,
    compression: str = 'snappy',
    optimize_dtypes: bool = True,
    schema_overrides: dict[str, pa.DataType] | None = None,
    row_group_size: int | None = None,
) -> None:
    """
    Write a DataFrame to Parquet format.
    
    Args:
        df: DataFrame to write (should already be grouped by event if needed)
        output_file: Path to output Parquet file
        compression: Compression codec ('snappy', 'gzip', 'zstd', 'none')
        optimize_dtypes: Whether to optimize dtypes before writing
    """
    if df.empty:
        logger.warning(f"Skipping write of empty DataFrame to {output_file}")
        return

    try:
        if schema_overrides:
            # Build Arrow arrays column-by-column, applying overrides where defined.
            arrays: list[pa.Array] = []
            names: list[str] = []
            for col in df.columns:
                col_data = df[col].to_list()
                override_type = schema_overrides.get(col)
                if override_type is not None:
                    arr = pa.array(col_data, type=override_type)
                else:
                    arr = pa.array(col_data)
                arrays.append(arr)
                names.append(col)
            table = pa.Table.from_arrays(arrays, names=names)
        else:
            # Optimize dtypes in pandas and let Arrow infer types.
            if optimize_dtypes:
                df = optimize_dtypes_for_parquet(df)
            table = pa.Table.from_pandas(df)

        pq.write_table(
            table,
            output_file,
            compression=compression,
            use_dictionary=True,
            row_group_size=row_group_size,
        )

        logger.debug(f"Wrote Parquet file: {output_file} ({len(df)} rows)")

    except Exception as e:
        logger.error(f"Failed to write Parquet file {output_file}: {e}")
        raise


def build_parquet_from_flat_df(
    df: pd.DataFrame,
    output_file: str,
    compression: str = 'snappy',
    schema_overrides: dict[str, pa.DataType] | None = None,
    row_group_size: int | None = None,
    nullable_integer_columns: set[str] | None = None,
) -> None:
    """
    Build Parquet file from a flat DataFrame (with per-particle/hit rows).
    
    This is the main entry point for converting flat data to Parquet format:
    1. Groups by event_id into lists
    2. Optimizes dtypes
    3. Writes to Parquet
    
    Args:
        df: Flat DataFrame with event_id and per-particle/hit columns
        output_file: Path to output Parquet file
        compression: Compression codec to use
    """
    if df.empty:
        logger.warning(f"Skipping empty DataFrame for {output_file}")
        return

    # Keep the legacy zero sentinel unless a caller explicitly requests nulls.
    if schema_overrides:
        df = df.copy()
        for col, dtype in schema_overrides.items():
            if col not in df.columns:
                continue
            value_type = getattr(dtype, "value_type", None)
            if value_type is not None and pa.types.is_integer(value_type):
                if nullable_integer_columns and col in nullable_integer_columns:
                    df[col] = df[col].astype(object).where(df[col].notna(), None)
                else:
                    df[col] = df[col].fillna(0)

    # Group by event
    grouped = group_by_event_to_lists(df)

    # Write to Parquet with optional schema overrides
    write_parquet_table(
        grouped,
        output_file,
        compression=compression,
        optimize_dtypes=not bool(schema_overrides),
        schema_overrides=schema_overrides,
        row_group_size=row_group_size,
    )


