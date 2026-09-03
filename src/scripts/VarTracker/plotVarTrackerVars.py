#!/usr/bin/env python3
#______________________________________________________________________
#
# Reads the parsedVarTracker/<timestep>/<task>/<var>_<matl> output produced by parseVarTrackerVars.py
# and writes one X-Y plot per (timestep, variable)..
#
# Each variable gets its own <var>/template.gp plus its own <var>/regenerate_plots.sh, driven by
# gnuplot variables (outfile, plottitle, xlab, ylab, plotcmd).
#
# Hand-tune a variable's template.gp and re-run that variable's own regenerate_plots.sh
#______________________________________________________________________


import argparse
import shlex
import subprocess
from pathlib import Path

# gnuplot terminal for each supported output format. Both are cairo-based
# terminals sharing the same font/color syntax, but pngcairo's `size` is in
# pixels while pdfcairo's is in inches (its default unit) -- so the two
# need different `size` values, not just a different terminal name.
PNG_PDF_TERMINAL = {
    "png": "pngcairo noenhanced color font \"Times-Roman,8\" size 800,600",
    "pdf": "pdfcairo noenhanced color font \"Times-Roman,8\" size 8,6",
}

#______________________________________________________________________

def read_block( path ):
    """Read one parsedVarTracker/<ts>/<task>/<var>_<matl> file, returning
    its a list of (x, y, z, value) floats. Comment lines
    are ignored
    """
    points = []

    with open( path ) as f:
        for line in f:
            line = line.strip()
            
            if line == "" or line.startswith( "#" ):
                continue

            parts = line.split( "," )
            x = float( parts[0] )
            y = float( parts[1] )
            z = float( parts[2] )
            value = float( parts[3] )
            points.append( (x, y, z, value) )

    return points

#______________________________________________________________________

def read_time( path ):
    """Read the "# Time: <value>" comment line returning the physical time as a string, or
    None if the file has no such line """
    
    prefix = "# Time:"

    with open( path ) as f:
        for line in f:
            line = line.strip()
            if line.startswith( prefix ):
                return line[len( prefix ):].strip()

    return None

#______________________________________________________________________

def varying_axis( points ):
    """Return which of "x", "y", "z" varies.-- VarTracker
    samples along a single spatial direction.
    Falls back to "index" (point order) if none of them do.
    """
    
    if len( points ) < 2:
        return "index"

    xs = set()
    ys = set()
    zs = set()
    for x, y, z, value in points:
        xs.add( x )
        ys.add( y )
        zs.add( z )

    if len( xs ) > 1:
        return "x"
    elif len( ys ) > 1:
        return "y"
    elif len( zs ) > 1:
        return "z"
    else:
        return "index"

#______________________________________________________________________

def axis_column( axis ):
    """Map an axis name to the gnuplot column to use in a `using` clause:
    1-indexed for x/y/z, or gnuplot's pseudocolumn 0 (point index) for the
    "index" fallback."""
    
    columns = { "x": "1", "y": "2", "z": "3", "index": "0" }
    return columns[axis]

#______________________________________________________________________

def gnuplot_quote( s ):
    """Escape a string for embedding in a single-quoted gnuplot string
    literal (a literal single quote is written doubled: '')."""
    
    return s.replace( "'", "''" )

#______________________________________________________________________

def bash_dquote_escape( s ):
    """Escape a string for embedding in a double-quoted bash string.
    Double quotes are still needed (rather than single quotes)"""
    
    s = s.replace( "\\", "\\\\" )
    s = s.replace( "\"", "\\\"" )
    return s

#______________________________________________________________________

def timestep_sort_key( name ):
    """Sort timestep directory names numerically when possible, so "10"
    sorts after "2" instead of before it."""
    
    if name.isdigit():
        return (0, int( name ))
    else:
        return (1, name)

#______________________________________________________________________

def discover_blocks( parsed_dir,
                     out_dir ):
    """Walk parsedVarTracker/<ts>/<task>/<var>_<matl> and group data by
    (timestep, variable) -> list of (task, matl, block_file, points), in
    timestep then task order. """
    groups = {}

    timestep_dirs = []
    for entry in parsed_dir.iterdir():
        if not entry.is_dir():
            continue

        if entry.resolve() == out_dir.resolve():
            continue
        timestep_dirs.append( entry )
        
    timestep_dirs.sort( key=lambda d: timestep_sort_key( d.name ) )

    for ts_dir in timestep_dirs:
        timestep = ts_dir.name

        for task_dir in sorted( ts_dir.iterdir() ):
            if not task_dir.is_dir():
                continue

            task = task_dir.name

            for block_file in sorted( task_dir.iterdir() ):
                if not block_file.is_file():
                    continue

                var, matl = block_file.name.rsplit( "_", 1 )
                points    = read_block( block_file )
                
                if not points:
                    continue

                key = (timestep, var)
                if key not in groups:
                    groups[key] = []
                groups[key].append( (task, matl, block_file, points) )

    return groups

#______________________________________________________________________

def build_labels( series ):
    """One legend label per (task, matl, block_file, points) entry in
    series -- just the task name, unless that task appears more than once
    (multiple materials found for it), in which case " (matl X)"
    disambiguates."""

    task_counts = {}
    for task, matl, block_file, points in series:
        if task in task_counts:
            task_counts[task] += 1
        else:
            task_counts[task] = 1

    labels = []
    for task, matl, block_file, points in series:
        if task_counts[task] > 1:
            labels.append( "%s (matl %s)" % (task, matl) )
        else:
            labels.append( task )

    return labels

#______________________________________________________________________

def write_gnuplotTemplate( out_dir,
                    var,
                    fmt ):
    """Write <var>/template.gp -- the template every plot of this one
    variable is rendered through. Hand-tune the styling here -- it applies
    to every timestep's plot of this variable"""

    lines = []
    lines.append( "# Template for %s, from plotVarTrackerVarsGnuplot.py -- hand-tune the" % var )
    lines.append( "# styling below, then re-run regenerate_plots.sh to apply it to every" )
    lines.append( "# timestep's plot of this variable." )
    lines.append( "" )
    lines.append( "set macros" )
    lines.append( "" )

    # noenhanced: task/variable names contain "_" and "::", which gnuplot's
    # default "enhanced text" mode would otherwise render as subscripts.

    lines.append( "set terminal %s" % PNG_PDF_TERMINAL[fmt] )
    lines.append( "set output outfile" )
    lines.append( "set title plottitle" )
    lines.append( "set xlabel xlab" )
    lines.append( "set ylabel ylab" )
  #  lines.append( "set key outside" )
    lines.append( "set grid" )
    lines.append( "" )
    lines.append( "plot @plotcmd" )

    template_path = out_dir / var / "template.gp"
    template_path.parent.mkdir( parents=True, exist_ok=True )
    
    with open( template_path, "w" ) as f:
        for line in lines:
            f.write( line + "\n" )

    return template_path

#______________________________________________________________________

def build_plot_commandList( out_dir,
                            timestep,
                            var,
                            series,
                            fmt ):
    """Work out this (timestep, variable)'s output path and the gnuplot
    variable assignments (as one `-e`-ready string) that drive template.gp

    for it: outfile, plottitle, xlab, ylab, and plotcmd -- one line per
    (task, matl, block_file, points) entry in series, legend = task names.

    Paths are written in terms of the $parsed_dir/$out_dir bash variables.
    """
    first_task, first_matl, first_file, first_points = series[0]

    axis   = varying_axis( first_points )
    column = axis_column( axis )
    labels = build_labels( series )

    time = read_time( first_file )

    if time is None:
        plottitle = "%s -- Timestep %s" % (var, timestep)
    else:
        plottitle = "%s -- Timestep %s (Time=%s)" % (var, timestep, time)

    var_dir = out_dir / var
    plot_path = var_dir / ( "%s.%s" % (timestep, fmt) )

    plot_terms = []
    i = 0
    for task, matl, block_file, points in series:
    
        data_ref = "$parsed_dir/%s/%s/%s_%s" % (timestep, task, var, matl)
        term = "'%s' using %s:4 with linespoints title '%s'" % (
            gnuplot_quote( data_ref ),
            column,
            gnuplot_quote( labels[i] ),
        )
        plot_terms.append( term )
        i += 1

    plotcmd = ", ".join( plot_terms )

    outfile_ref = "$out_dir/%s/%s.%s" % (var, timestep, fmt)

    commandList = (
        "outfile='%s'; plottitle='%s'; xlab='%s'; ylab='%s'; plotcmd=\"%s\""
        % (
            gnuplot_quote( outfile_ref ),
            gnuplot_quote( plottitle ),
            gnuplot_quote( axis ),
            gnuplot_quote( var ),
            plotcmd,
        )
    )

    return plot_path, commandList

#______________________________________________________________________

def write_regenerate_script( parsed_dir,
                             out_dir,
                             var,
                             commandLists ):
    """Write <out_dir>/<var>/regenerate_plots.sh. This script will regenerate
    the plots, using the template.gp gnuplot script"""

    script_path = out_dir / var / "regenerate_plots.sh"

    lines = []
    lines.append( "#!/bin/bash" )
    lines.append( "# Regenerates every %s plot from template.gp -- hand-tune" % var )
    lines.append( "# template.gp (styling applies to all of this variable's plots), then" )
    lines.append( "# re-run this script." )
    lines.append( "" )
    
    lines.append( "parsed_dir=%s" % shlex.quote( str( parsed_dir.resolve() ) ) )
    lines.append( "out_dir=%s" % shlex.quote( str( out_dir.resolve() ) ) )
    lines.append( "template=\"$out_dir/%s/template.gp\"" % var )
    lines.append( "" )
    
    lines.append( "commandLists=(" )
    for commandList in commandLists:
        lines.append( "  \"%s\"" % bash_dquote_escape( commandList ) )
    lines.append( ")" )
    
    lines.append( "" )
    lines.append( "for commandList in \"${commandLists[@]}\"; do" )
    lines.append( "  gnuplot -e \"$commandList\" \"$template\"" )
    lines.append( "done" )

    #__________________________________
    #   Write the lines above into a script

    with open( script_path, "w" ) as f:
        for line in lines:
            f.write( line + "\n" )

    script_path.chmod( 0o755 )

    return script_path

#______________________________________________________________________

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot parseVarTrackerVars.py output with gnuplot: one X-Y plot per (timestep, variable), one line per task."
    )
    parser.add_argument( "parsed_dir",    type=Path, nargs="?", default=Path( "parsedVarTracker" ),
                          help="directory produced by parseVarTrackerVars.py (default: parsedVarTracker)" )

    parser.add_argument( "-o", "--output", type=Path,           default=None,
                          help="directory to write plots into "
                               "(default: <parsed_dir>/plots)" )

    parser.add_argument( "-f", "--format", choices=["png", "pdf"], default="png",
                          help="output image format (default: png)" )

    args = parser.parse_args()

    #     bulletproofing
    if not args.parsed_dir.is_dir():
        parser.error( "%s is not a directory" % args.parsed_dir )

    if args.output is None:
        args.output = args.parsed_dir / "plots"

    return args

#______________________________________________________________________

def main():
    args = parse_args()

    groups = discover_blocks( args.parsed_dir, args.output )

    if not groups:
        print( "No data found under %s" % args.parsed_dir )
        return 1

    keys = list( groups.keys() )
    keys.sort( key=lambda key: (timestep_sort_key( key[0] ), key[1]) )

    template_vars = []
    for timestep, var in keys:
        if var not in template_vars:
            template_vars.append( var )

    commandLists_by_var = {}
    for var in template_vars:
        write_gnuplotTemplate( args.output, var, args.format )
        commandLists_by_var[var] = []

    #__________________________________
    #   gnuplot cmds
    plot_count = 0
    for key in keys:
        timestep, var = key
        series = groups[key]

        plot_path, commandList = build_plot_commandList( args.output,
                                                        timestep,
                                                        var,
                                                        series,
                                                        args.format )

        plot_path.parent.mkdir( parents=True, exist_ok=True )

        commandLists_by_var[var].append( commandList )
        plot_count += 1
    #__________________________________
    #   regenerate script
    script_paths = []
    for var in template_vars:
    
        script_path = write_regenerate_script( args.parsed_dir, 
                                               args.output, 
                                               var, 
                                               commandLists_by_var[var] )
        script_paths.append( script_path )
        subprocess.run( [str( script_path )], check=True )

    print( "Wrote %d plot(s) to %s" % (plot_count, args.output) )
    print( "Hand-tune a variable's template.gp and re-run its own regenerate_plots.sh (e.g. %s) to regenerate just its plots." % script_paths[0] )

    return 0


if __name__ == "__main__":
    raise SystemExit( main() )
