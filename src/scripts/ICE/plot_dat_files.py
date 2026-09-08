#!/usr/bin/env python3
#______________________________________________________________________
#
#   Plots the total mass, momentum, kinetic energy and internal energy
#   from the dat files written by sus.  You need to add
#
#      <save label="TotalMass"/>
#      <save label="KineticEnergy"/>
#      <save label="TotalIntEng"/>
#      <save label="TotalMomentum"/>
#
#   to the ups file.
#
#   If the simulation has more than one material, the DataArchiver also
#   writes a per-material dat file for each of these labels
#   (<Label>_<matlIndex>.dat, e.g. TotalMass_0.dat, TotalMass_1.dat, ...)
#   alongside the combined <Label>.dat.  This script auto-detects however
#   many of those per-material files exist and overlays them, alongside
#   the combined total, on each plot.
#______________________________________________________________________


import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

SCALAR_LABELS = ["TotalMass", "TotalIntEng", "KineticEnergy"]

#______________________________________________________________________
#   User-editable plot settings -- change these to adjust scales and
#   formatting without touching the plotting logic below.

FIGURE_SIZE      = (11, 8)   # inches
LINE_WIDTH       = 1.5
REL_LINESTYLE    = "--"      # style for the relative-difference traces
GRID             = True
LEGEND_FONTSIZE  = "small"

# Cycle these colors through the total + per-material traces on every
# panel.  Set to None to fall back to matplotlib's default color cycle.
COLOR_CYCLE = None
# COLOR_CYCLE = ["black", "tab:red", "tab:blue", "tab:green"]

# Per-panel y-axis scale ("linear" or "log") and axis limits.  Leave a
# limit as None to autoscale that panel.
PANEL_SETTINGS = {
    "TotalMass":     { "yscale": "linear", "xlim": None, "ylim": None },
    "TotalIntEng":   { "yscale": "linear", "xlim": None, "ylim": None },
    "KineticEnergy": { "yscale": "linear", "xlim": None, "ylim": None },
    "TotalMomentum": { "yscale": "linear", "xlim": None, "ylim": None },
}
#______________________________________________________________________

def set_color_cycle( axes_list ):
    """Apply COLOR_CYCLE to every axis in <axes_list>, if one is set --
    call this before plotting.  A twin axis (ax.twinx()) starts its own
    color cycle, so pass it too if you want its traces to match."""

    if COLOR_CYCLE is None:
        return

    for ax in axes_list:
        ax.set_prop_cycle( color=COLOR_CYCLE )

#______________________________________________________________________

def apply_panel_settings( ax,
                          base ):
    """Apply this quantity's y-scale/limits from PANEL_SETTINGS -- call
    this after plotting.  Anything left as None autoscales."""

    settings = PANEL_SETTINGS[base]

    ax.set_yscale( settings["yscale"] )

    if settings["xlim"] is not None:
        ax.set_xlim( settings["xlim"] )

    if settings["ylim"] is not None:
        ax.set_ylim( settings["ylim"] )

#______________________________________________________________________

def read_rows( path ):
    """Read a whitespace-separated dat file, returning a list of tuples
    of floats -- one tuple per non-blank, non-comment line."""

    rows = []

    with open( path ) as f:
        for line in f:
            line = line.strip()

            if line == "" or line.startswith( "#" ):
                continue

            parts  = line.split()
            values = []
            for part in parts:
                values.append( float( part ) )
            rows.append( tuple( values ) )

    return rows

#______________________________________________________________________

def column( rows,
           index ):
    """Extract column <index> from a list of row-tuples, as a list of
    floats."""

    values = []
    for row in rows:
        values.append( row[index] )
    return values

#______________________________________________________________________

def relative_difference( values ):
    """1.0 - value/initial_value for every entry -- the diagnostic used
    to spot conservation-law drift over a run."""

    init = values[0]
    rel  = []
    for v in values:
        rel.append( 1.0 - v/init )
    return rel

#______________________________________________________________________

def find_material_files( directory,
                         base ):
    """Return the (matlIndex, path) pairs for <base>_<N>.dat files in
    <directory>, sorted numerically by N.  Empty if the simulation has
    only one material."""

    pattern = re.compile( "^%s_([0-9]+)\\.dat$" % re.escape( base ) )
    found   = []

    for entry in directory.iterdir():
        m = pattern.match( entry.name )
        if m:
            found.append( (int( m.group( 1 ) ), entry) )

    found.sort( key=lambda pair: pair[0] )
    return found

#______________________________________________________________________

def plot_scalar_label( ax,
                       directory,
                       base,
                       ylabel ):
    """Plot the combined total and every per-material dat file for one
    scalar <save> label (TotalMass, TotalIntEng, KineticEnergy) on <ax>,
    with a relative-difference trace on a twin y-axis."""

    total_path = directory / ("%s.dat" % base)
    if not total_path.is_file():
        print( "%s not found -- skipping" % total_path )
        return

    ax2 = ax.twinx()
    set_color_cycle( [ax, ax2] )

    total_rows = read_rows( total_path )
    time       = column( total_rows, 0 )
    value      = column( total_rows, 1 )
    rel        = relative_difference( value )

    ax.plot( time, value, label="total", linewidth=LINE_WIDTH )
    ax2.plot( time, rel, REL_LINESTYLE, label="total rel", linewidth=LINE_WIDTH )

    for matl_index, path in find_material_files( directory, base ):
        rows  = read_rows( path )
        mtime = column( rows, 0 )
        mvalue = column( rows, 1 )
        mrel  = relative_difference( mvalue )

        label = "mat %d" % matl_index
        ax.plot( mtime, mvalue, label=label, linewidth=LINE_WIDTH )
        ax2.plot( mtime, mrel, REL_LINESTYLE, label=("%s rel" % label), linewidth=LINE_WIDTH )

    ax.set_ylabel( ylabel )
    ax2.set_ylabel( "Relative Difference" )
    ax.grid( GRID )
    apply_panel_settings( ax, base )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend( lines1 + lines2, labels1 + labels2, fontsize=LEGEND_FONTSIZE )

#______________________________________________________________________

def plot_momentum( ax,
                   directory ):
    """Plot the combined total and every per-material TotalMomentum dat
    file's x/y/z components on <ax>.  No relative-difference trace --
    momentum components cross zero, so a relative difference isn't
    meaningful."""

    base = "TotalMomentum"
    axes_columns = [(1, "x"), (2, "y"), (3, "z")]

    total_path = directory / ("%s.dat" % base)
    if not total_path.is_file():
        print( "%s not found -- skipping" % total_path )
        return

    set_color_cycle( [ax] )

    total_rows = read_rows( total_path )
    time = column( total_rows, 0 )

    for col_index, axis_name in axes_columns:
        label = "total %s" % axis_name
        ax.plot( time, column( total_rows, col_index ), label=label, linewidth=LINE_WIDTH )

    for matl_index, path in find_material_files( directory, base ):
        rows  = read_rows( path )
        mtime = column( rows, 0 )

        for col_index, axis_name in axes_columns:
            label = "mat %d %s" % (matl_index, axis_name)
            ax.plot( mtime, column( rows, col_index ), label=label, linewidth=LINE_WIDTH )

    ax.set_ylabel( "total Momentum" )
    ax.grid( GRID )
    apply_panel_settings( ax, base )
    ax.legend( fontsize=LEGEND_FONTSIZE )

#______________________________________________________________________

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TotalMass/TotalIntEng/KineticEnergy/TotalMomentum dat files, "
                    "overlaying any per-material dat files found alongside the combined totals."
    )
    parser.add_argument( "directory", type=Path, nargs="?", default=Path( "." ),
                         help="directory containing the dat files (default: current directory)" )
    parser.add_argument( "-o", "--output", type=Path, default=None,
                         help="save the figure to this file instead of opening an interactive window" )

    args = parser.parse_args()

    #     bulletproofing
    if not args.directory.is_dir():
        parser.error( "%s is not a directory" % args.directory )

    return args

#______________________________________________________________________

def main():
    args = parse_args()

    fig, axes = plt.subplots( 2, 2, figsize=FIGURE_SIZE )

    plot_scalar_label( axes[0][0], args.directory, "TotalMass", "total mass" )
    plot_scalar_label( axes[0][1], args.directory, "TotalIntEng", "Total Internal Energy" )
    plot_scalar_label( axes[1][0], args.directory, "KineticEnergy", "Kinetic Energy" )
    plot_momentum( axes[1][1], args.directory )

    for row in axes:
        for ax in row:
            ax.set_xlabel( "time" )

    fig.tight_layout()

    if args.output is None:
        plt.show()
    else:
        fig.savefig( args.output )
        print( "Wrote %s" % args.output )

    return 0


if __name__ == "__main__":
    raise SystemExit( main() )
