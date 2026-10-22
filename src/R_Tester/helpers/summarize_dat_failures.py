#!/usr/bin/env python3
#______________________________________________________________________
#
#   summarize_dat_failures.py
#
#   Walks a Uintah regression-test results tree, finds every
#   "compare_sus_runs.log.txt" file, and for each test whose dat-file
#   comparison FAILED, reports the worst (max) absolute and relative
#   differences seen -- per offending .dat file, and rolled up per test.
#
#   Usage:
#       summarize_dat_failures.py [root_dir] [options]
#______________________________________________________________________


import argparse
import os
import re
import sys

LOGFILE_NAME = "compare_sus_runs.log.txt"

COMPARING_RE = re.compile( "^Comparing (\\S+\\.dat)\\.\\.\\.(PASSED|FAILED)\\s*$", re.MULTILINE )

# Each pattern captures both the error magnitude and the UDA1 time at
# which it occurred, so the matching line can be found in the dat file.
REL_RE    = re.compile( "greatest relative error \\(%([^)]+)\\) at:\\s*\\n"
                         "\\s*time\\s+value:\\s*\\n"
                         "\\s*UDA1:\\s*(\\S+)\\s*/" )
ABS_RE    = re.compile( "greatest absolute error:\\s*(\\S+)\\s*\\n"
                         "\\s*time\\s+value:\\s*\\n"
                         "\\s*UDA1:\\s*(\\S+)\\s*/" )
XXDIFF_RE = re.compile( "xxdiff\\s+(\\S+)" )

# compare_dats bails out before comparing anything if a dat file present in
# the new run is absent from the gold standard uda -- the names are listed
# space/newline-separated inside the parens.
MISSING_RE = re.compile( "ERROR: Missing dat files \\(([^)]*)\\) in the gold standard uda" )

#______________________________________________________________________

class DatFailure:
    """One FAILED dat-file comparison, with its worst absolute and
    relative error and the timestep (dat-file line number) each
    occurred at."""

    __slots__ = ("component", "test", "datfile", "max_abs", "timestep_abs", "max_rel", "timestep_rel")

    def __init__( self,
                  component,
                  test,
                  datfile,
                  max_abs,
                  timestep_abs,
                  max_rel,
                  timestep_rel ):
        self.component    = component
        self.test         = test
        self.datfile      = datfile
        self.max_abs      = max_abs
        self.timestep_abs = timestep_abs
        self.max_rel      = max_rel
        self.timestep_rel = timestep_rel

#______________________________________________________________________

class MissingDat:
    """One dat file that compare_dats reported as absent from the gold
    standard uda, so no comparison could even be attempted."""

    __slots__ = ("component", "test", "datfile")

    def __init__( self,
                  component,
                  test,
                  datfile ):
        self.component = component
        self.test      = test
        self.datfile   = datfile

#______________________________________________________________________

def find_logs( root ):
    """Return the path of every compare_sus_runs.log.txt under <root>,
    skipping restart subdirectories -- a restart's dat-file comparison
    is just the parent test's comparison run twice and would only
    double-count the same failures."""

    logs = []
    for dirpath, dirnames, filenames in os.walk( root ):
        if "restart" in dirnames:
            dirnames.remove( "restart" )

        if LOGFILE_NAME in filenames:
            logs.append( os.path.join( dirpath, LOGFILE_NAME ) )

    return logs

#______________________________________________________________________

def identify_test( logpath,
                    root ):
    """Derive (component, test name) from the log's path.

    Expected shape relative to root:
        <Component>-results/<test>/compare_sus_runs.log.txt
    Falls back gracefully if the tree doesn't match that shape."""

    rel   = os.path.relpath( logpath, root )
    parts = rel.split( os.sep )[:-1]        # drop the filename itself

    if len( parts ) >= 2:
        component = parts[0]
        test      = parts[1]
    elif len( parts ) == 1:
        component = ""
        test      = parts[0]
    else:
        component = ""
        test      = os.path.basename( root )

    return component, test

#______________________________________________________________________

def extract_dat_section( text ):
    """Return the slice of <text> covering the dat-file comparison
    block, or None if it has no such section."""

    start_m = re.search( "^Comparing dat files\\s*$", text, re.MULTILINE )
    if not start_m:
        return None

    start = start_m.end()
    end_m = re.search( "^(Comparing udas|Summary:)\\s*$", text[start:], re.MULTILINE )

    if end_m:
        end = start + end_m.start()
    else:
        end = len( text )

    return text[start:end]

#______________________________________________________________________

def find_timestep( datfile,
                    time_str ):
    """Return the 1-indexed line number (timestep) in <datfile> whose
    first column equals <time_str> exactly, or None if the file is
    missing or no line matches."""

    if time_str is None:
        return None

    if not os.path.isfile( datfile ):
        return None

    with open( datfile, "r", errors="replace" ) as f:
        lineno = 0
        for line in f:
            lineno += 1
            fields = line.split()
            if not fields:
                continue
            if fields[0] == time_str:
                return lineno

    return None

#______________________________________________________________________

def parse_log( logpath,
                root ):
    """Return (failures, missing) -- the DatFailure and MissingDat
    objects found in one compare_sus_runs.log.txt."""

    with open( logpath, "r", errors="replace" ) as f:
        text = f.read()

    section = extract_dat_section( text )
    if not section:
        return [], []

    component, test = identify_test( logpath, root )
    
    #__________________________________
    #  Are there any missing dat files?
    missing   = []
    missing_m = MISSING_RE.search( section )
    if missing_m:
        for datfile in missing_m.group( 1 ).split():
            missing.append( MissingDat( component, test, datfile ) )

    #__________________________________
    #   Find comparison failures

    markers = list( COMPARING_RE.finditer( section ) )
    results = []

    for i in range( len( markers ) ):
        m       = markers[i]
        datfile = m.group( 1 )
        status  = m.group( 2 )

        if status != "FAILED":
            continue

        if i + 1 < len( markers ):
            block_end = markers[i + 1].start()
        else:
            block_end = len( section )
        block = section[m.end():block_end]

        rel_m = REL_RE.search( block )
        abs_m = ABS_RE.search( block )
        xx_m  = XXDIFF_RE.search( block )

        if xx_m:
            uda1_path = xx_m.group( 1 )
        else:
            uda1_path = None

        if rel_m:
            max_rel      = float( rel_m.group( 1 ) )
            rel_time_str = rel_m.group( 2 )
        else:
            max_rel      = float( "nan" )
            rel_time_str = None

        if abs_m:
            max_abs      = float( abs_m.group( 1 ) )
            abs_time_str = abs_m.group( 2 )
        else:
            max_abs      = float( "nan" )
            abs_time_str = None

        if uda1_path:
            timestep_abs = find_timestep( uda1_path, abs_time_str )
            timestep_rel = find_timestep( uda1_path, rel_time_str )
        else:
            timestep_abs = None
            timestep_rel = None

        results.append( DatFailure( component, test, datfile, max_abs, timestep_abs, max_rel, timestep_rel ) )

    return results, missing

#______________________________________________________________________

def collect( root ):
    """Parse every compare_sus_runs.log.txt under <root> and return
    (failures, missing) -- the combined lists of DatFailure and
    MissingDat objects."""

    failures = []
    missing  = []
    for logpath in sorted( find_logs( root ) ):
        log_failures, log_missing = parse_log( logpath, root )
        failures.extend( log_failures )
        missing.extend( log_missing )

    return failures, missing

#______________________________________________________________________

def per_test_summary( failures ):
    """Roll up <failures> per (component, test): the count of failing
    dat files and the worst absolute/relative error seen."""

    summary = {}

    for fa in failures:
        key = (fa.component, fa.test)

        if key not in summary:
            summary[key] = {"count": 0, "max_abs": 0.0, "timestep_abs": None, "max_rel": 0.0, "timestep_rel": None}
        s = summary[key]

        s["count"] += 1

        if fa.max_abs == fa.max_abs:            # NaN-safe
            if fa.max_abs > s["max_abs"]:
                s["max_abs"]      = fa.max_abs
                s["timestep_abs"] = fa.timestep_abs

        if fa.max_rel == fa.max_rel:
            if fa.max_rel > s["max_rel"]:
                s["max_rel"]      = fa.max_rel
                s["timestep_rel"] = fa.timestep_rel

    return summary

#______________________________________________________________________

def fmt_num( x ):
    """Format a float with 6 significant digits."""

    return "%.6g" % x

#______________________________________________________________________

def fmt_timestep( timestep ):
    """Format a timestep index, or "-" if it's unknown."""

    if timestep is None:
        return "-"
    return str( timestep )

#______________________________________________________________________

def sort_key( sort_mode ):
    """Return a key function for sorting a list of DatFailure by
    <sort_mode>: "abs", "rel", or the default "test" ordering."""

    def key_abs( fa ):
        if fa.max_abs == fa.max_abs:            # NaN-safe
            return -fa.max_abs
        return float( "inf" )

    def key_rel( fa ):
        if fa.max_rel == fa.max_rel:
            return -fa.max_rel
        return float( "inf" )

    def key_test( fa ):
        return (fa.component, fa.test, fa.datfile)

    if sort_mode == "abs":
        return key_abs
    elif sort_mode == "rel":
        return key_rel
    else:
        return key_test

#______________________________________________________________________

def print_text_table( rows,
                       headers ):
    """Print <rows> (and <headers>) as a left-justified, space-padded
    plain-text table."""

    widths = []
    for h in headers:
        widths.append( len( h ) )

    for row in rows:
        for i in range( len( row ) ):
            cell_len = len( str( row[i] ) )
            if cell_len > widths[i]:
                widths[i] = cell_len

    header_cells = []
    for i in range( len( headers ) ):
        header_cells.append( headers[i].ljust( widths[i] ) )
    print( "  ".join( header_cells ) )

    rule_cells = []
    for w in widths:
        rule_cells.append( "-" * w )
    print( "  ".join( rule_cells ) )

    for row in rows:
        row_cells = []
        for i in range( len( row ) ):
            row_cells.append( str( row[i] ).ljust( widths[i] ) )
        print( "  ".join( row_cells ) )

#______________________________________________________________________

def print_markdown_table( rows,
                           headers ):
    """Print <rows> (and <headers>) as a GitHub-flavored Markdown
    table."""

    print( "| " + " | ".join( headers ) + " |" )

    rule_cells = []
    for h in headers:
        rule_cells.append( "---" )
    print( "|" + "|".join( rule_cells ) + "|" )

    for row in rows:
        row_cells = []
        for cell in row:
            row_cells.append( str( cell ) )
        print( "| " + " | ".join( row_cells ) + " |" )

#______________________________________________________________________

def print_summary_table( failures,
                          printer ):
    """Print the per-test rollup: failing-dat-file count and worst
    absolute/relative error, sorted worst-relative-error first."""

    summary  = per_test_summary( failures )
    sum_rows = []

    for key in sorted( summary.keys() ):
        component, test = key
        s = summary[key]

        sum_rows.append( [component, test, s["count"],
                          fmt_num( s["max_abs"] ), fmt_timestep( s["timestep_abs"] ),
                          fmt_num( s["max_rel"] ), fmt_timestep( s["timestep_rel"] )] )

    sum_rows.sort( key=lambda r: -float( r[5] ) )

    print( "Per-test summary (worst relative error first)" )
    print( "=" * 47 )
    printer( sum_rows, ["Component", "Test", "#FailedDats",
                        "MaxAbsErr", "TimestepAbs", "MaxRelErr(%)", "TimestepRel"] )
    print()

#______________________________________________________________________

def print_detail_table( failures,
                         printer,
                         sort_mode ):
    """Print one row per failing dat file, sorted by <sort_mode>."""

    failures = sorted( failures, key=sort_key( sort_mode ) )

    detail_rows = []
    for fa in failures:
        detail_rows.append( [fa.component, fa.test, fa.datfile,
                             fmt_num( fa.max_abs ), fmt_timestep( fa.timestep_abs ),
                             fmt_num( fa.max_rel ), fmt_timestep( fa.timestep_rel )] )

    print( "Detailed dat-file failures (sorted by %s)" % sort_mode )
    print( "=" * 47 )
    printer( detail_rows, ["Component", "Test", "DatFile",
                           "MaxAbsErr", "TimestepAbs", "MaxRelErr(%)", "TimestepRel"] )

#______________________________________________________________________

def print_missing_table( missing,
                          printer ):
    """Print one row per dat file reported missing from the gold
    standard uda."""

    missing_rows = []
    for md in sorted( missing, key=lambda m: (m.component, m.test, m.datfile) ):
        missing_rows.append( [md.component, md.test, md.datfile] )

    print( "Dat files missing from the gold standard" )
    print( "=" * 47 )
    printer( missing_rows, ["Component", "Test", "DatFile"] )
    print()

#______________________________________________________________________

def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize dat-file comparison failures from Uintah regression-test logs."
    )
    parser.add_argument( "root", nargs="?", default=".",
                         help="directory to search recursively for %s (default: cwd)" % LOGFILE_NAME )
    parser.add_argument( "--sort", choices=["test", "abs", "rel"], default="test",
                         help="sort order for the detail table (default: test)" )
    parser.add_argument( "--format", choices=["text", "markdown"], default="text",
                         help="output table style (default: text)" )
    parser.add_argument( "--summary-only", action="store_true",
                         help="only print the per-test rollup table, not every dat file" )
    parser.add_argument( "--component", default=None,
                         help="only include tests under <component>-results (e.g. MPM)" )

    args = parser.parse_args()
    return args

#______________________________________________________________________

def main():
    args = parse_args()

    root = os.path.abspath( args.root )
    if not os.path.isdir( root ):
        sys.exit( "error: not a directory: %s" % root )

    failures, missing = collect( root )

    if args.component:
        wanted = args.component.rstrip( "-results" ) + "-results"

        kept_failures = []
        for fa in failures:
            if fa.component == wanted:
                kept_failures.append( fa )
        failures = kept_failures

        kept_missing = []
        for md in missing:
            if md.component == wanted:
                kept_missing.append( md )
        missing = kept_missing

    if not failures and not missing:
        return 0

    if args.format == "markdown":
        printer = print_markdown_table
    else:
        printer = print_text_table

    if missing:
        print_missing_table( missing, printer )

    if not failures:
        return 0

    print_summary_table( failures, printer )

    if args.summary_only:
        return 0

    print_detail_table( failures, printer, args.sort )

    return 0


if __name__ == "__main__":
    raise SystemExit( main() )
