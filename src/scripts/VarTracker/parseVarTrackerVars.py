#!/usr/bin/env python3
#______________________________________________________________________

# Parses sus output from a run with the VarTracker option enabled, extracting
# the text for specific (task, variable, material) combinations per timestep.
#
# Algorithm:
#   scan every line of the log once, recording (line_number, ...) for each
#   Timestep / "execution of" task / "Variable:" entry
#
#   for each recorded timestep, in order:
#     bounds = [this timestep's line, the next timestep's line - 1]
#              (or end of file, for the last timestep)
#
#     tasks_here = tasks whose line falls within bounds
#
#     for each task name we're looking for:
#       if exactly one of tasks_here matches that name:
#         task_bounds = [that task's line, the next task's line - 1]
#                       (or the timestep's own upper bound, for the last task)
#
#         for each (variable, material) we're looking for:
#           find the first recorded variable matching name + material whose
#           line falls within task_bounds
#           if found:
#             write out the lines from that variable's line through the
#             first blank line after it (or the task's upper bound, if no
#             blank line appears first) to parsedVarTracker/<ts>/<task>/<var>_<matl>
#______________________________________________________________________

import argparse
from pathlib import Path
from typing import NamedTuple

#__________________________________

class Timestep(NamedTuple):
    line: int
    id: str
    time: str

#__________________________________

class Task(NamedTuple):
    line: int
    name: str

#__________________________________

class Variable(NamedTuple):
    line: int
    name: str
    description: str
    matl: str

#______________________________________________________________________

def awk_field(  fields,
                idx ):
    """awk-style field access: out-of-range yields "" instead of raising."""
    if idx < len( fields ):
        return fields[idx]
    else:
        return ""

#______________________________________________________________________

def parse_log(lines):
    """Extract the timesteps, tasks, and variables referenced in the sus output.

    Each line is prefixed with its line number before splitting on
    whitespace, so field indices line up  ($1 line number, $3 timestep id, $6 task name, $4/$9/$10 for
    variable name/description/material) regardless of the out exact
    column layout.
    """
    timesteps = []
    tasks = []
    variables = []

    for line_num, raw in enumerate( lines, start=1 ):
        fields = ( "%d %s" % (line_num, raw) ).split()

        # An exact field match, not a substring search -- task names like
        # ICE::actuallyComputeStableTimestep also contain "Timestep" and
        # would otherwise be misdetected as timestep boundaries.
        
        if awk_field( fields, 1 ) == "Timestep":
            # The header also carries the physical time as one "Time=<value>"
            # token (e.g. "Timestep 0  Time=0  Next delT=1e-08  ..."); strip
            # the "Time=" prefix to get just the value.
            
            time_field = awk_field( fields, 3 )
            
            if time_field.startswith( "Time=" ):
                time = time_field[len( "Time=" ):]
            else:
                time = time_field

            timesteps.append( Timestep( line_num,
                                        awk_field( fields, 2 ),
                                        time ) )

        if "execution of" in raw:
            name = awk_field( fields, 5 ).replace( ",", "", 1 )
            tasks.append( Task( line_num, name ) )

        if "Variable:" in raw:
            name = awk_field( fields, 3 ).replace( ",", "", 1 )
            variables.append( Variable( line_num,
                                        name,
                                        awk_field( fields, 8 ),
                                        awk_field( fields, 9 ) ) )

    return timesteps, tasks, variables

#______________________________________________________________________

def distinct_sorted( values ):
    """Return the sorted distinct values from values, via an explicit loop
    rather than a set/comprehension (keeps this file's style consistent)."""
    seen = []
    for value in values:
        if value not in seen:
            seen.append( value )

    seen.sort()
    return seen

#______________________________________________________________________

def is_wanted_timestep( timestep,
                        ts_low,
                        ts_high ):
    """True if this Timestep's id falls within [ts_low, ts_high]
    (inclusive), compared numerically -- not as strings, since "10" < "9"
    as a string comparison."""
    return ts_low <= int( timestep.id ) <= ts_high

#______________________________________________________________________

def span( entries,
          index,
          end_of_range ):
    """Line-number range (lo, hi) covered by entries[index], up to the next entry."""
    lo = entries[index].line

    is_last_entry = index + 1 >= len( entries )
    if is_last_entry:
        # No next entry to bound this one, so the caller tells us where the
        # enclosing range (the timestep, or the whole file) ends.
        hi = end_of_range
    else:
        # This entry runs up to just before the next one starts.
        hi = entries[index + 1].line - 1

    return lo, hi

#______________________________________________________________________

def write_numbered_lines( path,
                          lines,
                          lo,
                          hi ):
    """Write lines[lo..hi] (1-based, inclusive), each prefixed with its line number."""
    with open( path, "w" ) as f:
        for line_num in range( lo, hi + 1 ):
            f.write( "%d %s\n" % (line_num, lines[line_num - 1]) )

#______________________________________________________________________

def strip_rank_prefix( raw ):
    """Drop the leading per-rank integer, e.g. "0  Variable: ..." ->
    "Variable: ...". 
    """
    parts = raw.split( maxsplit=1 )
    if len( parts ) == 2 and parts[0].isdigit():
        return parts[1]
    return raw

#______________________________________________________________________

def remove_IntVector_format( content ):
    """Turn a data line's "[int 16, 10, 0]: 8.51e-01" into " 16, 10, 0, 8.51e-01".
    Left untouched if it doesn't look like an index data line (e.g. the
    "Variable: ..." header line keeps its own colon).
    """
    if "[int" in content:
        content = content.replace( "[int", "" ).replace( "]", "" )
        content = content.replace( ":", ",", 1 )
    return content

#______________________________________________________________________

def write_variable_block( path,
                          lines,
                          lo,
                          hi,
                          time ):
    """Write lines[lo..hi] (1-based, inclusive) as plain content.
    "# Variable: ..." header line.
    "# " and followed by a "# Time: <time>" comment and a
    "# x,  y,  z,  value" column-header comment.
    """
    with open( path, "w" ) as f:
        for line_num in range( lo, hi + 1 ):
            content = strip_rank_prefix( lines[line_num - 1] )
            content = remove_IntVector_format( content )

            if line_num == lo:
                f.write( "# " + content + "\n" )
                f.write( "# Time: %s\n" % time )
                f.write( "# x,  y,  z,  value\n" )
            else:
                f.write( content + "\n" )

#______________________________________________________________________

def write_csv( path,
               rows ):
    """Write one comma-joined row per line; each row is a Timestep/Task/Variable."""
    with open( path, "w" ) as f:
        for row in rows:
            fields = []
            for value in row:
                fields.append( str( value ) )

            line = ",".join( fields )
            f.write( line + "\n" )

#______________________________________________________________________

def write_index_files( out_dir,
                       lines,
                       timesteps,
                       tasks,
                       variables ):
    """Write the same intermediate index files the original bash script produced."""
    write_numbered_lines( out_dir / "out_nl",
                          lines,
                          1,
                          len( lines ) )
    write_csv( out_dir / "timesteps",   timesteps )
    write_csv( out_dir / "allTasks",    tasks )
    write_csv( out_dir / "allVarsFile", variables )


#______________________________________________________________________

def remove_intermediate_files( out_dir ):
    """Delete the temp files"""
    
    for name in ( "out_nl", "timesteps", "allTasks", "allVarsFile" ):
        (out_dir / name).unlink( missing_ok=True )


#______________________________________________________________________
def find_unique_task_index( candidates,
                            name ):
    """Return the index of the sole candidate whose name matches, else None."""
    matching_indices = []
    for i, task in enumerate( candidates ):
        if name in task.name:
            matching_indices.append( i )

    if len( matching_indices ) == 1:
        return matching_indices[0]
    else:
        return None
#______________________________________________________________________

def find_variable( variables,
                   name,
                   matl,
                   lo,
                   hi ):
    """Return the first Variable matching name/matl within [lo, hi], or None."""
    for v in variables:
        name_matches = name in v.name
        matl_matches = matl in v.matl
        in_range = lo <= v.line <= hi

        if name_matches and matl_matches and in_range:
            return v

    return None

#______________________________________________________________________

def extract_variable( task_dir,
                      lines,
                      v,
                      var,
                      matl,
                      task_hi,
                      time ):
    """Write the numbered lines of a matched variable's block to task_dir.

    A variable's block always ends
    at the first blank line after it. The next recorded Variable entry is
    not a reliable boundary -- a requested-but-missing variable in between
    leaves no entry of its own, just a WARNING line, so two recorded
    entries can be several lines apart in the raw log. The scan is capped
    at the enclosing task's own extent as a safety bound.
    """
    lo = v.line
    hi = lo
    while hi < task_hi and lines[hi].strip() != "":
        hi += 1

    fname = task_dir / ( "%s_%s" % (var, matl) )
    write_variable_block( fname, lines, lo, hi, time )


#______________________________________________________________________

def parse_args():
    """Parse the output file argument plus the task/variable/material/timestep
    flags. tasks/variables/materials each default to None here -- when the
    user doesn't override one, main() fills it in with every distinct value
    actually found in the log, rather than a fixed list.

    timesteplow/timestephigh/timestep also default to None here; main()
    resolves them into a [ts_low, ts_high] range once it knows the log's
    actual timestep ids (needed to resolve "last timestep"). --timestep is
    a shorthand for "just this one timestep" and cannot be combined with
    --timesteplow/--timestephigh.
    """

    parser = argparse.ArgumentParser(
        description="Parse sus VarTracker output into per-timestep/task/variable extracts."
    )

    parser.add_argument( "sus_output", type=Path, help="sus output file to parse" )

    parser.add_argument( "-t", "--task",      action="append", dest="tasks",
                          help="task name to search for (repeatable; default: every task found in the log)" )

    parser.add_argument( "-v", "--variable",  action="append", dest="variables",
                          help="variable name to search for (repeatable; default: every variable found in the log)" )

    parser.add_argument( "-m", "--material",  action="append", dest="materials",
                          help="material index to search for (repeatable; default: every material found in the log)" )

    parser.add_argument( "-tlow", "--timesteplow",   type=int, default=None,
                          help="sets start output timestep to int (default: 0)" )

    parser.add_argument( "-thigh", "--timestephigh", type=int, default=None,
                          help="sets end output timestep to int (default: last timestep found in the log)" )

    parser.add_argument( "-timestep", "--timestep",  type=int, default=None,
                          help="only output this one timestep; cannot be combined with "
                               "--timesteplow/--timestephigh (default: unset, i.e. all timesteps)" )

    args = parser.parse_args()

    if not args.sus_output.is_file():
        parser.error( "%s is not a file" % args.sus_output )

    timestep_range_given = args.timesteplow is not None or args.timestephigh is not None
    if args.timestep is not None and timestep_range_given:
        parser.error( "--timestep cannot be combined with --timesteplow/--timestephigh" )

    return args

#______________________________________________________________________

def main():
    args = parse_args()

    lines = args.sus_output.read_text().splitlines()
    total_lines = len( lines )

    out_dir = Path( "parsedVarTracker" )
    out_dir.mkdir( exist_ok=True )

    #__________________________________
    #     One pass over the output creates the three indices everything below searches.
    timesteps, tasks, variables = parse_log( lines )

    #__________________________________
    #     Default to every task/variable/material actually found in the log
    #     for any flag the user didn't specify -- -t/-v/-m still narrow the
    #     search when given, same as before.
    if args.tasks is None:
        task_names = []
        for t in tasks:
            task_names.append( t.name )
        args.tasks = distinct_sorted( task_names )

    if args.variables is None:
        variable_names = []
        for v in variables:
            variable_names.append( v.name )
        args.variables = distinct_sorted( variable_names )

    if args.materials is None:
        material_values = []
        for v in variables:
            material_values.append( v.matl )
        args.materials = distinct_sorted( material_values )

    #__________________________________
    #     Resolve the [ts_low, ts_high] timestep range: --timestep (if
    #     given) selects just that one timestep; otherwise --timesteplow/
    #     --timestephigh narrow the range, each falling back to the log's
    #     actual first/last timestep when not given -- which is exactly
    #     "all timesteps" when neither flag was given at all.
    if args.timestep is not None:
        ts_low  = args.timestep
        ts_high = args.timestep
    else:
        if args.timesteplow is not None:
            ts_low = args.timesteplow
        else:
            ts_low = 0

        if args.timestephigh is not None:
            ts_high = args.timestephigh
        else:
            ts_high = 0
            for t in timesteps:
                ts_int = int( t.id )
                if ts_int > ts_high:
                    ts_high = ts_int

    #__________________________________
    #     Create intermediate files (out_nl, timesteps, allTasks, allVarsFile)
    write_index_files( out_dir, 
                       lines, 
                       timesteps,
                       tasks, 
                       variables )

    #__________________________________
    #     Echo every timestep that will be processed, mirroring the
    #     original script's summary line.
    timestep_summaries = []

    for t in timesteps:
        if is_wanted_timestep( t, ts_low, ts_high ):
            timestep_summaries.append( "%s,%s" % (t.line, t.id) )

    print( " ".join( timestep_summaries ) )

    #__________________________________
    #     Walk each timestep, then each task of interest within it, then each
    #     (variable, material) combination of interest within that task.
    
    for t_idx, timestep in enumerate( timesteps ):

        if not is_wanted_timestep( timestep, ts_low, ts_high ):
            continue

        # Line range this timestep's output occupies in the log. Computed
        # against the full, unfiltered timesteps list (via t_idx from the
        # enumerate above) so a non-contiguous -T selection still gets the
        # correct boundaries -- not the boundaries of whatever timestep
        # happens to be next among only the ones that were selected.
        timestep_lo, timestep_hi = span( timesteps, t_idx, total_lines )

        print( "__________________________________%s" % timestep.id )
        print( "Timestep line number %s & %s " % (timestep_lo, timestep_hi) )

        # Only tasks that occurred within this timestep are candidates below.
        ts_tasks = []
        for t in tasks:
            if timestep_lo <= t.line <= timestep_hi:
                ts_tasks.append( t )
        #__________________________________
        
        for find_task in args.tasks:
            print( "  %s" % find_task )

            task_dir = out_dir / timestep.id / find_task
            task_dir.mkdir( parents=True, exist_ok=True )

            # Bail on this task if it's missing, or ran more than once, in
            # this timestep -- either way there's no single unambiguous span.
            task_idx = find_unique_task_index( ts_tasks, 
                                               find_task )
            if task_idx is None:
                print( "    Warning:  Did not find %s in this timestep %s" % (find_task, timestep.id) )
                continue
            #__________________________________
            # Line range this task's own output occupies, within the timestep.
            task_lo, task_hi = span( ts_tasks, 
                                     task_idx, 
                                     timestep_hi )
                                     
            print( "  Task extents %s & %s " % (task_lo, task_hi) )

            #__________________________________
            # For every (variable, material) combination we care about, find
            # its entry within this task's span and dump that variable's block.
            for var in args.variables:
                for matl in args.materials:
                    print( "      working on variable (%s) matl (%s)" % (var, matl), end="" )

                    v = find_variable( variables,
                                       var,
                                       matl,
                                       task_lo,
                                       task_hi )
                    if v is None:
                        print( " ... not found" )
                        continue

                    extract_variable( task_dir,
                                      lines,
                                      v,
                                      var,
                                      matl,
                                      task_hi,
                                      timestep.time )
                    print( " ... found" )

    remove_intermediate_files( out_dir )

    return 0


if __name__ == "__main__":
    raise SystemExit( main() )
