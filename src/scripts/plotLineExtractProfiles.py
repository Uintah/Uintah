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
import os

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
    { "suffix": "A", "cols": [6, 7, 8, 9] },              #  <<< Change these column numbers
    { "suffix": "B", "cols": [10, 13, 16] },              #  <<< Change these column numbers
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

def read_time_value( filepath ):

    """Return column 5 ( "Time [s]" ), as a string, from the first data
    row of a timestep file."""

    f = open( filepath, "r" )

    time_val = None
    for line in f:
        if not line.startswith( "#" ):
            fields = line.split()
            time_val = fields[4]
            break

    f.close()
    return time_val

#______________________________________________________________________

def read_xy( filepath,
            col_x,
            col_y ):

    """Return ( x_values, y_values ) for the given 1-based data columns,
    skipping "#"-commented lines."""

    x_values = []
    y_values = []

    f = open( filepath, "r" )
    for line in f:
        if line.startswith( "#" ):
            continue
        fields = line.split()
        x_values.append( float( fields[col_x - 1] ) )
        y_values.append( float( fields[col_y - 1] ) )
    f.close()

    return x_values, y_values

#______________________________________________________________________

def plot_page( infile,
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
            x_values, y_values = read_xy( infile, col_x, col )

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
    return parser.parse_args()

#______________________________________________________________________

def main():
    args = parse_args()

    data_dir = os.path.join( args.uda, args.line_name, args.level, "timesteps" )
    out_dir  = os.path.join( args.uda, args.line_name, args.level, "plots" )

    if not os.path.isdir( out_dir ):
        os.makedirs( out_dir )

    file_names = sorted( os.listdir( data_dir ) )

    sample_file = open( os.path.join( data_dir, file_names[0] ), "r" )
    header_line = sample_file.readline()
    sample_file.close()

    descriptors = column_descriptors( header_line )
    xlabel = descriptors[COL_X - 1]

    for file_name in file_names:
        infile = os.path.join( data_dir, file_name )
        print( "    Working on %s" % infile )

        time_val = read_time_value( infile )

        for page in PAGES:
            outfile = os.path.join( out_dir, "%s_%s.png" % ( page["suffix"], file_name ) )

            titles = []
            for col in page["cols"]:
                titles.append( descriptors[col - 1] )

            plot_page( infile,
                       outfile,
                       COL_X,
                       page["cols"],
                       titles,
                       xlabel,
                       file_name,
                       time_val )

    print( "Done.  PNGs written to %s/" % out_dir )

#______________________________________________________________________

if __name__ == "__main__":
    main()
