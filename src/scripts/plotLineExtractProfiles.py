#!/usr/bin/env python3
#______________________________________________________________________
#  plotLineExtractProfiles.py
#
#  Python port of plotLineExtractProfiles.gp, for comparison.
#
#  RUN: src/scripts/combineLineExtractData.sh prior to running this script
#
#  Reads one file per timestep from <uda>/<lineName>/<level>/timesteps/
#  ( e.g. "t00062" ) and writes one PNG per PAGE, per timestep, where a
#  page is a 2x2 window of up to 4 panels.  <level> defaults to "L-0".
#
#  X_line/                    << lineName
#  `-- L-0                    << level
#      `-- timesteps
#          |-- t00001
#          |-- t00049
#          `-- t00097
#
#  Where X_line/L-0/timesteps/t00097 contains a "#"-commented header
#  line naming each column, followed by one data row per cell, e.g.:
#      # X_CC  Y_CC  Z_CC  Timestep  Time [s]  press_CC_0  delP_MassX_0 ...
#      2.5e-04 0.0   0.0   62        1.0e-05    1.013e+05   0.0 ...
#
#  Usage:
#      ./plotLineExtractProfiles.py
#      ./plotLineExtractProfiles.py --uda StickMPMICE.uda.000 --line-name domain
#      ./plotLineExtractProfiles.py --uda StickMPMICE.uda.000 --line-name domain --level L-1
#______________________________________________________________________

import argparse
import functools
import multiprocessing
import os
import sys

import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt

#______________________________________________________________________
#  PLOT OPTIONS
#______________________________________________________________________
FIGSIZE         = ( 12, 9 )       # inches
LINE_MARKER     = "."
LINE_MARKERSIZE = 3
LINE_WIDTH      = 0.5
GRID_LINESTYLE  = ":"
GRID_LINEWIDTH  = 0.5

#______________________________________________________________________
#  PAGE DEFINITIONS
#
#  COL_X -- the data column ( 1-based, matching the file's own column
#           numbering ) used as the x axis of every panel
#  PAGES -- one entry per PNG to make per timestep.  "suffix" names the
#           output file; "cols" lists 1-4 data columns, one per panel --
#           a page with fewer than 4 columns just leaves the remaining
#           2x2 grid cell(s) blank.
#
#  Panel titles are NOT typed here -- they are pulled automatically from
#  the column descriptor in the data file's header line (see below).
#______________________________________________________________________
COL_X = 1

PAGES = [
    { "suffix": "A", "cols": [6, 8, 9, 10] },              #  <<< Change these column numbers
    #{ "suffix": "B", "cols": [10, 13, 16] },              #  <<< Change these column numbers
]

#______________________________________________________________________
#  Y_RANGES -- optional fixed y-axis limits, keyed by data column number.
#  A column with no entry here autoscales as usual.  E.g.:
#      Y_RANGES = { 6: ( 100000, 102000 ) }
#  fixes column 6's y axis to [100000, 102000] on every panel that plots it.
#______________________________________________________________________
Y_RANGES = {
    # col, yMin, yMax
    6: ( 100000, 102000 ),                              #  <<< Change these to fix the y_range
}

#______________________________________________________________________

def require_dir( path,
                 description ):

    """Exit with a clear, specific error message if path is not an
    existing directory, instead of letting a later os.listdir()/open()
    fail with an opaque traceback."""

    if not os.path.isdir( path ):
        sys.exit( "Error: %s not found: %s" % ( description, path ) )

#______________________________________________________________________

def column_descriptors( header_line ):
    """Split a timestep file's "#"-commented header line into one
    descriptor per data column ( 1-based -- descriptors[0] is column 1 ).

    "Time [s]" is collapsed to "Time_s" first so it counts as the single
    column it actually is, rather than splitting into two words.
    """
    normalized = header_line.lstrip( "#" ).strip()
    normalized = normalized.replace( "Time [s]", "Time_s" )

    return normalized.split()

#______________________________________________________________________

def read_timestep_file( filepath ):

    """Read a timestep file exactly once, returning ( rows, time_val ).

    rows is a list of data rows, each row a list of floats ( 1-based
    column N is row[N - 1] ), skipping "#"-commented lines.  time_val is
    column 5 ( "Time [s]" ) from the first data row, kept as the
    original string ( not round-tripped through float() ).
    """

    rows = []
    time_val = None

    f = open( filepath, "r" )
    for line in f:
        if line.startswith( "#" ):
            continue

        fields = line.split()

        if time_val is None:
            time_val = fields[4]

        row = []
        for field in fields:
            row.append( float( field ) )
        rows.append( row )
    f.close()

    return rows, time_val

#______________________________________________________________________

def extract_xy( rows,
                col_x,
                col_y ):

    """Pull two already-parsed 1-based columns out of rows ( see
    read_timestep_file() )."""

    x_values = []
    y_values = []

    for row in rows:
        x_values.append( row[col_x - 1] )
        y_values.append( row[col_y - 1] )

    return x_values, y_values

#______________________________________________________________________

def plot_page( rows,
               outfile,
               col_x,
               cols,
               titles,
               xlabel,
               timestep_id,
               time_val ):

    """Write one 2x2-grid PNG for a single page: one panel per entry in
    cols ( up to 4 ), any remaining grid cell left blank."""

    fig, axes = plt.subplots( 2, 2, figsize=FIGSIZE )
    fig.suptitle( "timestep: %s      time = %s [s]" % ( timestep_id, time_val ) )

    flat_axes = []
    for row in range( 2 ):
        for col in range( 2 ):
            flat_axes.append( axes[row][col] )

    for panel in range( len( flat_axes ) ):
        ax = flat_axes[panel]

        if panel < len( cols ):
            col = cols[panel]
            x_values, y_values = extract_xy( rows, col_x, col )

            ax.plot( x_values, y_values, marker=LINE_MARKER, markersize=LINE_MARKERSIZE, linewidth=LINE_WIDTH )

            ax.set_title( titles[panel] )
            ax.set_xlabel( xlabel )
            ax.grid( True, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH )

            if col in Y_RANGES:
                ax.set_ylim( Y_RANGES[col] )
        else:
            ax.axis( "off" )

    fig.tight_layout()
    fig.savefig( outfile )
    plt.close( fig )

#______________________________________________________________________

def parse_args():
    parser = argparse.ArgumentParser( description="Plot Uintah lineExtract timestep profiles." )
    parser.add_argument( "--uda",       default=".",
                          help="path to the uda directory" )

    parser.add_argument( "--line-name", default=".", dest="line_name",
                          help="lineExtract line name (subdirectory of the uda)" )

    parser.add_argument( "--level",     default="L-0",
                          help="AMR level subdirectory to read (default: L-0)" )

    parser.add_argument( "--jobs", "-n", type=int, default=4,
                          help="worker processes to use (default: all CPUs -- %(4)s here)" )
    return parser.parse_args()

#______________________________________________________________________

def process_timestep( file_name,
                      data_dir,
                      out_dir,
                      descriptors,
                      xlabel ):

    """Read one timestep file and write all of its PAGES PNGs.  Runs in
    a worker process when called via multiprocessing.Pool.map()."""

    infile = os.path.join( data_dir, file_name )
    print( "    Working on %s" % infile )

    rows, time_val = read_timestep_file( infile )

    for page in PAGES:
        outfile = os.path.join( out_dir, "%s_%s.png" % ( page["suffix"], file_name ) )

        titles = []
        for col in page["cols"]:
            titles.append( descriptors[col - 1] )

        plot_page( rows,
                   outfile,
                   COL_X,
                   page["cols"],
                   titles,
                   xlabel,
                   file_name,
                   time_val )

#______________________________________________________________________

def describe_columns( descriptors ):

    """Return a "N: name" listing of every column, one per line, for use
    in error messages."""

    lines = []
    for i in range( len( descriptors ) ):
        lines.append( "  %d: %s" % ( i + 1, descriptors[i] ) )

    return "\n".join( lines )

#______________________________________________________________________

def validate_columns( descriptors ):

    """Exit with a clear error message if COL_X or any PAGES column
    number falls outside the columns this dataset actually has, instead
    of failing deep inside a worker process with an opaque
    multiprocessing traceback."""

    n_cols = len( descriptors )

    if not ( 1 <= COL_X <= n_cols ):
        sys.exit( "Error: COL_X=%d is out of range -- this dataset has %d columns:\n%s" %
                  ( COL_X, n_cols, describe_columns( descriptors ) ) )

    for page in PAGES:
        for col in page["cols"]:
            if not ( 1 <= col <= n_cols ):
                sys.exit( "Error: PAGES page %r references column %d, but this dataset has only %d columns:\n%s" %
                          ( page["suffix"], col, n_cols, describe_columns( descriptors ) ) )

#______________________________________________________________________

def main():
    args = parse_args()

    require_dir( args.uda, "uda directory" )

    line_dir = os.path.join( args.uda, args.line_name )
    require_dir( line_dir, "line-name directory" )

    level_dir = os.path.join( line_dir, args.level )
    require_dir( level_dir, "level directory" )

    data_dir = os.path.join( level_dir, "timesteps" )
    require_dir( data_dir, "timesteps directory" )

    out_dir = os.path.join( level_dir, "plots" )

    if not os.path.isdir( out_dir ):
        os.makedirs( out_dir )

    file_names = sorted( os.listdir( data_dir ) )

    if len( file_names ) == 0:
        sys.exit( "Error: no timestep files found in %s" % data_dir )

    sample_file = open( os.path.join( data_dir, file_names[0] ), "r" )
    header_line = sample_file.readline()
    sample_file.close()

    descriptors = column_descriptors( header_line )
    validate_columns( descriptors )
    xlabel = descriptors[COL_X - 1]

    worker = functools.partial( process_timestep,
                                data_dir    =data_dir,
                                out_dir     =out_dir,
                                descriptors =descriptors,
                                xlabel      =xlabel )

    with multiprocessing.Pool( args.jobs ) as pool:
        pool.map( worker, file_names )

    print( "Done.  PNGs written to %s/" % out_dir )

#______________________________________________________________________

if __name__ == "__main__":
    main()
