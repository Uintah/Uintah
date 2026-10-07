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
#  Which columns get plotted, the page layout, and the y-axis ranges are
#  read from an XML config file ( plotLineExtractProfiles.xml, next to
#  this script, by default ) rather than edited here -- see --config
#  below and the comments in that file.
#
#  Usage:
#      ./plotLineExtractProfiles.py
#      ./plotLineExtractProfiles.py --uda StickMPMICE.uda.000 --line-name domain
#      ./plotLineExtractProfiles.py --uda StickMPMICE.uda.000 --line-name domain --level L-1
#      ./plotLineExtractProfiles.py --config myOther.xml
#______________________________________________________________________

import argparse
import functools
import multiprocessing
import os
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt

#______________________________________________________________________

def require_dir( path,
                 description ):

    """Exit with a clear, specific error message if path is not an
    existing directory, instead of letting a later os.listdir()/open()
    fail with an opaque traceback."""

    if not os.path.isdir( path ):
        sys.exit( "Error: %s not found: %s" % ( description, path ) )

#______________________________________________________________________

def split_list( text,
               description ):

    """Split a comma-separated string ( e.g. "press_CC_0, 8, delP_MassX_0" )
    into a list of stripped, non-empty tokens.  Each token is resolved
    against the data file's own column descriptors later, by
    resolve_column() -- not here, since descriptors aren't known yet
    when the config file is first read."""

    values = []
    for piece in text.split( "," ):
        piece = piece.strip()
        if piece != "":
            values.append( piece )

    if len( values ) == 0:
        sys.exit( "Error: %s: no values found" % description )

    return values

#______________________________________________________________________

def parse_float( text,
                 description ):

    """Parse text as a float, exiting with a clear error message if it
    isn't one."""

    try:
        return float( text )
    except ValueError:
        sys.exit( "Error: %s: %r is not a number" % ( description, text ) )

#______________________________________________________________________

def is_blank( text ):

    """True if text is None, or is all whitespace."""

    if text is None:
        return True
    return text.strip() == ""

#______________________________________________________________________

def apply_attr( elem,
                attr,
                style,
                key,
                as_float=False,
                description=None ):

    """If elem has attr set, copy its value into style[key] -- parsed as
    a float first when as_float is true.  Leaves style untouched if
    elem doesn't have attr."""

    value = elem.get( attr )
    if value is None:
        return

    if as_float:
        style[key] = parse_float( value, description )
    else:
        style[key] = value

#______________________________________________________________________

def load_plot_options( root ):

    """Read the <plotOptions> block into a style dict with the keys
    plot_page() expects.  Falls back to this script's built-in defaults
    for any option, or the whole block, that's missing from the XML."""

    style = {
        "figsize"         : ( 12, 9 ),
        "line_marker"     : ".",
        "line_markersize" : 3,
        "line_width"      : 0.5,
        "grid_linestyle"  : ":",
        "grid_linewidth"  : 0.5,
    }

    options = root.find( "plotOptions" )
    if options is None:
        return style

    figsize = options.find( "figsize" )
    if figsize is not None:
        width  = parse_float( figsize.get( "width",  "12" ), "plotOptions/figsize@width" )
        height = parse_float( figsize.get( "height", "9"  ), "plotOptions/figsize@height" )
        style["figsize"] = ( width, height )

    line = options.find( "line" )
    if line is not None:
        apply_attr( line, "marker",     style, "line_marker" )
        apply_attr( line, "markersize", style, "line_markersize", as_float=True,
                   description="plotOptions/line@markersize" )
        apply_attr( line, "width",      style, "line_width", as_float=True,
                   description="plotOptions/line@width" )

    grid = options.find( "grid" )
    if grid is not None:
        apply_attr( grid, "linestyle", style, "grid_linestyle" )
        apply_attr( grid, "linewidth", style, "grid_linewidth", as_float=True,
                   description="plotOptions/grid@linewidth" )

    return style

#______________________________________________________________________

def load_pages( root ):

    """Read every <page> under <pages> into a list of
    { "suffix": str, "cols": [ref, ...] } dicts, in document order.  Each
    ref is still the raw text from the config file at this point -- a
    column descriptor name ( e.g. "press_CC_0" ) or a 1-based column
    number -- resolve_column() turns it into a column index later."""

    pages_elem = root.find( "pages" )
    if pages_elem is None:
        sys.exit( "Error: config file is missing a <pages> block" )

    pages = []
    for page in pages_elem.findall( "page" ):
        suffix = page.get( "suffix" )
        if is_blank( suffix ):
            sys.exit( "Error: a <page> is missing its required suffix attribute" )

        cols_text = page.get( "cols" )
        if cols_text is None:
            sys.exit( "Error: <page suffix=%r> is missing its required cols attribute" % suffix )

        cols = split_list( cols_text, "<page suffix=%r> cols" % suffix )
        if len( cols ) > 4:
            sys.exit( "Error: <page suffix=%r> lists %d columns, but a page holds at most 4" %
                      ( suffix, len( cols ) ) )

        pages.append( { "suffix": suffix, "cols": cols } )

    if len( pages ) == 0:
        sys.exit( "Error: config file's <pages> block has no <page> entries" )

    return pages

#______________________________________________________________________

def load_y_ranges( root ):

    """Read every <range> under <yRanges> into a { ref: ( min, max ) }
    dict, keyed by the raw column reference text ( a descriptor name or
    a 1-based column number ) -- resolve_column() turns each key into a
    column index later, once descriptors are known."""

    y_ranges = {}

    ranges_elem = root.find( "yRanges" )
    if ranges_elem is None:
        return y_ranges

    for range_elem in ranges_elem.findall( "range" ):
        col_ref = range_elem.get( "col" )
        if is_blank( col_ref ):
            sys.exit( "Error: a <range> is missing a valid col attribute" )
        col_ref = col_ref.strip()

        y_min = parse_float( range_elem.get( "min", "" ), "<range col=%r> min" % col_ref )
        y_max = parse_float( range_elem.get( "max", "" ), "<range col=%r> max" % col_ref )

        if y_min >= y_max:
            sys.exit( "Error: <range col=%r> has min >= max ( %s, %s )" % ( col_ref, y_min, y_max ) )

        y_ranges[col_ref] = ( y_min, y_max )

    return y_ranges

#______________________________________________________________________

def load_config( path ):

    """Read the plot-configuration XML file at path, returning
    ( style, col_x_ref, pages, y_ranges ) -- see load_plot_options(),
    load_pages(), and load_y_ranges() for the shape of each.  col_x_ref,
    every page's cols, and every y_ranges key are still raw text at this
    point ( a column descriptor name or a 1-based column number ) --
    resolve_config() turns them into column indices once the data
    file's own descriptors are known."""

    if not os.path.isfile( path ):
        sys.exit( "Error: config file not found: %s" % path )

    try:
        tree = ET.parse( path )
    except ET.ParseError as error:
        sys.exit( "Error: could not parse config file %s: %s" % ( path, error ) )

    root = tree.getroot()

    col_x_elem = root.find( "colX" )
    if col_x_elem is None or is_blank( col_x_elem.text ):
        sys.exit( "Error: config file is missing a <colX> value" )
    col_x_ref = col_x_elem.text.strip()

    style    = load_plot_options( root )
    pages    = load_pages( root )
    y_ranges = load_y_ranges( root )

    return style, col_x_ref, pages, y_ranges

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
               time_val,
               style,
               y_ranges ):

    """Write one 2x2-grid PNG for a single page: one panel per entry in
    cols ( up to 4 ), any remaining grid cell left blank."""

    fig, axes = plt.subplots( 2, 2, figsize=style["figsize"] )
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

            ax.plot( x_values, y_values, marker=style["line_marker"], markersize=style["line_markersize"], linewidth=style["line_width"] )

            ax.set_title( titles[panel] )
            ax.set_xlabel( xlabel )
            ax.grid( True, linestyle=style["grid_linestyle"], linewidth=style["grid_linewidth"] )

            if col in y_ranges:
                ax.set_ylim( y_ranges[col] )
        else:
            ax.axis( "off" )

    fig.tight_layout()
    fig.savefig( outfile )
    plt.close( fig )

#______________________________________________________________________

def parse_args():
    default_config = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ),
                                   "plotLineExtractProfiles.xml" )

    parser = argparse.ArgumentParser( description="Plot Uintah lineExtract timestep profiles." )
    parser.add_argument( "--uda",       default=".",
                          help="path to the uda directory" )

    parser.add_argument( "--line-name", default=".", dest="line_name",
                          help="lineExtract line name (subdirectory of the uda)" )

    parser.add_argument( "--level",     default="L-0",
                          help="AMR level subdirectory to read (default: L-0)" )

    parser.add_argument( "--jobs", "-n", type=int, default=4,
                          help="worker processes to use (default: all CPUs -- %(4)s here)" )

    parser.add_argument( "--config",    default=default_config,
                          help="path to the plot-configuration XML file (default: next to this script)" )
    return parser.parse_args()

#______________________________________________________________________

def process_timestep( file_name,
                      data_dir,
                      out_dir,
                      descriptors,
                      xlabel,
                      col_x,
                      pages,
                      style,
                      y_ranges ):

    """Read one timestep file and write all of its page PNGs.  Runs in
    a worker process when called via multiprocessing.Pool.map()."""

    infile = os.path.join( data_dir, file_name )
    print( "    Working on %s" % infile )

    rows, time_val = read_timestep_file( infile )

    for page in pages:
        outfile = os.path.join( out_dir, "%s_%s.png" % ( page["suffix"], file_name ) )

        titles = []
        for col in page["cols"]:
            titles.append( descriptors[col - 1] )

        plot_page( rows,
                   outfile,
                   col_x,
                   page["cols"],
                   titles,
                   xlabel,
                   file_name,
                   time_val,
                   style,
                   y_ranges )

#______________________________________________________________________

def describe_columns( descriptors ):

    """Return a "N: name" listing of every column, one per line, for use
    in error messages."""

    lines = []
    for i in range( len( descriptors ) ):
        lines.append( "  %d: %s" % ( i + 1, descriptors[i] ) )

    return "\n".join( lines )

#______________________________________________________________________

def resolve_column( ref,
                    descriptors,
                    description ):

    """Resolve one column reference -- a descriptor name, matched
    exactly against descriptors, or a 1-based column number -- to its
    1-based column index.  Exits with a clear error, listing every
    column this dataset actually has, if ref matches neither.  Runs
    late ( after the data file's header has been read ), instead of
    failing deep inside a worker process with an opaque multiprocessing
    traceback."""

    for i in range( len( descriptors ) ):
        if descriptors[i] == ref:
            return i + 1

    if ref.lstrip( "-" ).isdigit():
        col = int( ref )
        if 1 <= col <= len( descriptors ):
            return col
        sys.exit( "Error: %s: column number %d is out of range -- this dataset has %d columns:\n%s" %
                  ( description, col, len( descriptors ), describe_columns( descriptors ) ) )

    sys.exit( "Error: %s: %r is not a column name or number in this dataset. Available columns:\n%s" %
              ( description, ref, describe_columns( descriptors ) ) )

#______________________________________________________________________

def resolve_config( col_x_ref,
                    pages_raw,
                    y_ranges_raw,
                    descriptors ):

    """Turn the raw ( name-or-number ) column references loaded from the
    config file into the 1-based column indices the rest of the script
    works with, now that descriptors ( from the data file's own header
    line ) are available to match names against.  Returns
    ( col_x, pages, y_ranges ) in the same shape main() used to get
    straight out of load_config()."""

    col_x = resolve_column( col_x_ref, descriptors, "colX" )

    pages = []
    for page in pages_raw:
        cols = []
        for ref in page["cols"]:
            cols.append( resolve_column( ref, descriptors, "page %r" % page["suffix"] ) )
        pages.append( { "suffix": page["suffix"], "cols": cols } )

    y_ranges = {}
    for ref in y_ranges_raw:
        col = resolve_column( ref, descriptors, "yRanges range %r" % ref )
        y_ranges[col] = y_ranges_raw[ref]

    return col_x, pages, y_ranges

#______________________________________________________________________

def main():
    args = parse_args()

    style, col_x_ref, pages_raw, y_ranges_raw = load_config( args.config )

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
    col_x, pages, y_ranges = resolve_config( col_x_ref, pages_raw, y_ranges_raw, descriptors )
    xlabel = descriptors[col_x - 1]

    worker = functools.partial( process_timestep,
                                data_dir    =data_dir,
                                out_dir     =out_dir,
                                descriptors =descriptors,
                                xlabel      =xlabel,
                                col_x       =col_x,
                                pages       =pages,
                                style       =style,
                                y_ranges    =y_ranges )

    with multiprocessing.Pool( args.jobs ) as pool:
        pool.map( worker, file_names )

    print( "Done.  PNGs written to %s/" % out_dir )

#______________________________________________________________________

if __name__ == "__main__":
    main()
