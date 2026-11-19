#!/usr/bin/env python3

#______________________________________________________________________
# summarize_test_times.py
#
# Reports how long each regression test took, grouped by component, by
# reading the test_times.csv files that runSusTests.py writes (one per
# component, in <component>-results/test_times.csv).
#
# Usage:  summarize_test_times.py [-d dir] [-s] [csv_file ...]
#   -d dir     build directory (opt or dbg) that holds the
#              <component>-results directories (default: current dir)
#   -s         sort the tests in each component, slowest first
#   csv_file   read these files instead of searching dir
#
#   e.g. from the opt or dbg directory:
#     TestScripts/helpers/summarize_test_times.py
#
# Output, one block per component:
#   Performing <component>-<build> tests.
#   ====================================
#   <n>  <testName>            Timer: hh:mm:ss
#   Tests Total: hh:mm:ss  (<count> tests)
# followed by a total over all components.
#
# Tests restarted from a checkpoint are reported as <testName>(restart),
# postProcessUda tests as <testName>(postProcess).
#
# Note: "Tests Total" is the sum of the test times.  It is a little less
# than the "Component Timer" in summary_log, which also counts set-up
# work done between tests.
#______________________________________________________________________

import argparse
import csv
import glob
import os
import sys
from os import path

# start_from value  ->  tag appended to the test name
TAGS = { "checkpoint"     : "(restart)",
         "postProcessUda" : "(postProcess)" }

# columns this script needs from test_times.csv
REQUIRED_COLUMNS = [ "component", "build", "test", "start_from", "seconds" ]

#______________________________________________________________________
# Format a number of seconds as hh:mm:ss.  Fractions of a second are
# dropped, matching the "Test Timer:" lines.  Hours do not wrap at 24.
def hms( seconds ):

  total   = int( seconds )
  hours   = total // 3600
  minutes = ( total % 3600 ) // 60
  secs    = total % 60

  return "%02d:%02d:%02d" % (hours, minutes, secs)

#______________________________________________________________________
# Return e.g. "1 test" or "5 tests"
def plural( count,
            word ):

  if count == 1:
    return "1 %s" % word
  else:
    return "%d %ss" % (count, word)

#______________________________________________________________________
# Sort key: the time a test took
def get_seconds( row ):
  return row["seconds"]

#______________________________________________________________________
# Return the test_times.csv files under build_dir, in component order
def find_csv_files( build_dir ):

  pattern = path.join( build_dir,
                       "*-results",
                       "test_times.csv" )
  return sorted( glob.glob( pattern ) )

#______________________________________________________________________
# Read one test_times.csv file and return its rows as a list of dicts.
# "seconds" is converted to a float.  Rows that cannot be used are
# reported and skipped.  Returns None if the file itself is unusable.
def read_csv_file( csv_file ):

  rows = []

  try:
    f = open( csv_file,
              newline = "" )
  except OSError as e:
    print( "  ERROR: cannot read %s (%s)" % (csv_file, e),
           file = sys.stderr )
    return None

  with f:
    reader = csv.DictReader( f )

    #__________________________________
    # Bulletproofing: are the needed columns present?
    if reader.fieldnames is None:
      print( "  WARNING: %s is empty, skipping it" % csv_file,
             file = sys.stderr )
      return rows

    for column in REQUIRED_COLUMNS:
      if column not in reader.fieldnames:
        print( "  ERROR: %s has no '%s' column" % (csv_file, column),
               file = sys.stderr )
        return None

    #__________________________________
    # Read the rows; line 1 is the header so data starts on line 2
    line_num = 1
    for row in reader:
      line_num += 1

      try:
        row["seconds"] = float( row["seconds"] )
      except ( TypeError, ValueError ):
        print( "  WARNING: %s line %d: bad seconds value (%s), skipping it" % (csv_file, line_num, row["seconds"]),
               file = sys.stderr )
        continue

      rows.append( row )

  return rows

#______________________________________________________________________
# Sort key for a group of tests: when its first test finished.  The
# "finished" column is "YYYY-mm-dd HH:MM:SS", which sorts as text.
# Files without that column give "" and keep the order they were read in.
def get_first_finished( group ):

  tests = group[1]
  if "finished" in tests[0] and tests[0]["finished"] is not None:
    return tests[0]["finished"]
  return ""

#______________________________________________________________________
# Group rows by (component, build).  Tests keep the order they were read
# in; components are put in the order they ran, using the "finished"
# timestamps.  (Sorting by file name would not match the run order, e.g.
# MPM-results sorts before MPMICE-results but MPMICE runs first.)
def group_by_component( rows ):

  groups = {}
  for row in rows:
    key = ( row["component"], row["build"] )

    if key not in groups:
      groups[key] = []
    groups[key].append( row )

  ordered = sorted( groups.items(),
                    key = get_first_finished )
  return ordered

#______________________________________________________________________
# Print one component block and return the sum of its test times
def print_component( key,
                     tests,
                     sort_by_time ):

  component = key[0]
  build     = key[1]

  if sort_by_time:
    tests = sorted( tests,
                    key = get_seconds,
                    reverse = True )

  print( "Performing %s-%s tests." % (component, build) )
  print( "====================================" )

  total = 0.0
  n     = 0
  for row in tests:
    n     += 1
    total += row["seconds"]

    name = row["test"]
    if row["start_from"] in TAGS:
      name = name + TAGS[ row["start_from"] ]

    print( "%-4d %-52s Timer: %s" % (n, name, hms( row["seconds"] )) )

  print( "Tests Total: %s  (%s)" % (hms( total ), plural( n, "test" )) )
  return total

#______________________________________________________________________

def main():

  parser = argparse.ArgumentParser( description = "Summarize regression test times from the test_times.csv files written by runSusTests.py" )

  parser.add_argument( "-d",
                       dest    = "build_dir",
                       default = ".",
                       help    = "build directory (opt or dbg) holding the <component>-results directories (default: current directory)" )

  parser.add_argument( "-s",
                       dest   = "sort_by_time",
                       action = "store_true",
                       help   = "sort the tests in each component, slowest first" )

  parser.add_argument( "csv_files",
                       nargs = "*",
                       help  = "test_times.csv files to read instead of searching the build directory" )

  args = parser.parse_args()

  #__________________________________
  # Bulletproofing: which files to read?
  csv_files = args.csv_files

  if len( csv_files ) == 0:
    if not path.isdir( args.build_dir ):
      print( "  ERROR: build directory (%s) does not exist" % args.build_dir,
             file = sys.stderr )
      return 1

    csv_files = find_csv_files( args.build_dir )

    if len( csv_files ) == 0:
      print( "  ERROR: no */test_times.csv files found under %s" % path.abspath( args.build_dir ),
             file = sys.stderr )
      return 1

  #__________________________________
  # Read every file
  rows = []
  for csv_file in csv_files:
    file_rows = read_csv_file( csv_file )

    if file_rows is None:
      return 1
    rows.extend( file_rows )

  if len( rows ) == 0:
    print( "  ERROR: no test times found in %s" % " ".join( csv_files ),
           file = sys.stderr )
    return 1

  #__________________________________
  # Report each component, then the total over all of them
  groups = group_by_component( rows )

  grand_total = 0.0
  for group in groups:
    grand_total += print_component( group[0],
                                    group[1],
                                    args.sort_by_time )

  print( "" )
  print( "All Tests Total: %s  (%s, %s)" % (hms( grand_total ), plural( len( rows ), "test" ), plural( len( groups ), "component" )) )
  return 0

#______________________________________________________________________
#______________________________________________________________________

if __name__ == "__main__":
  try:
    rc = main()
    sys.stdout.flush()
  except BrokenPipeError:
    # output was piped to something like "head" that stopped reading;
    # point stdout at /dev/null so Python does not complain on exit
    devnull = open( "/dev/null",
                    "w" )
    os.dup2( devnull.fileno(),
             sys.stdout.fileno() )
    rc = 0
  sys.exit( rc )
