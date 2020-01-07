#! /usr/bin/perl
#______________________________________________________________________
#   script that performs the <uda>/*.dat 
#   exit return values:
#      0:   The dat files are withn allowable errors
#      1:   The dat files exceed tolerances
#      2:   Error


if (@ARGV <= 4) {
  print "Usage: compare_dat_files {abs error allowed} {rel error allowed} {uda directory 1} {uda directory 2} {dat file names}\n";
  print "  Now exiting...\n";
  exit(2);
}

$allowable_abs_error = $ARGV[0];
$allowable_rel_error = $ARGV[1];

$uda_dir1 = $ARGV[2];
$uda_dir2 = $ARGV[3];

# Strip trailing / (if any) from dir name:

$uda_dir1 = $1 if( $uda_dir1 =~ /(.*)\/$/ );
$uda_dir2 = $1 if( $uda_dir2 =~ /(.*)\/$/ );

print "Using absolute tolerance: " . $allowable_abs_error . "\n";
print "Using relative tolerance: " . $allowable_rel_error . "\n \n";

$max_rel_error = 0;
$max_abs_error = 0;
$first_rel_error = 0;
$first_abs_error = 0;

$max_rel_error_time1 = 0;
$max_rel_error_time2 = 0;
$max_abs_error_time1 = 0;
$max_abs_error_time2 = 0;

$first_rel_error_time1 = 0;
$first_rel_error_time2 = 0;
$first_abs_error_time1 = 0;
$first_abs_error_time2 = 0;

$max_rel_error_value1 = 0;
$max_rel_error_value2 = 0;
$max_abs_error_value1 = 0;
$max_abs_error_value2 = 0;

$first_rel_error_value1 = 0;
$first_rel_error_value2 = 0;
$first_abs_error_value1 = 0;
$first_abs_error_value2 = 0;

$failed = 0;
$lineno = 0;

#__________________________________

foreach $datfile (@ARGV[4 .. @ARGV-1]) {

  $datfilename1 = $uda_dir1 . "/" . $datfile;
  $datfilename2 = $uda_dir2 . "/" . $datfile;

  if (!open(IN1, $datfilename1)) {
    print "Could not open " . $datfilename1 . "\n";
    $failed = 1;
  }
  if (!open(IN2, $datfilename2)) {
    print "Could not open " . $datfilename2 . "\n";
    $failed = 1;
  }

  print "Comparing " . $datfile . "...";
  $detected_rel_error = 0;
  $detected_abs_error = 0;
  $max_rel_error = 0;
  $max_abs_error = 0;

  #__________________________________

  while (($line1 = <IN1>) && ($line2 = <IN2>)) {
    $lineno = $lineno + 1;
    
    ($time1, @values1) = getTimeAndValue($line1);
    ($time2, @values2) = getTimeAndValue($line2);

    if (@values1 != @values2) {
      print "Values1 " . @values1 . " Values2 " . @values2;
      print " on line number " . $lineno . "\n";

      print "Error: the dat files do not have the same number of values per line\n";
      exit(2);
    }
    
    # prepend the time onto the value list to be compared as values
    unshift(@values1, $time1);
    unshift(@values2, $time2);

    #__________________________________
    # loop over each value and check them
    
    foreach $value1 (@values1) {
      $value2  = shift(@values2);
      $max_abs = abs($value1);

      if (abs($value2) > $max_abs) {
        $max_abs = abs($value2);
      }

      if ($max_abs != 0) {
        #__________________________________
        # do absolute comparisons
        $abs_error = abs($value1 - $value2);
        if ($abs_error > $allowable_abs_error) {

          if ($detected_abs_error == 0) {
            $first_abs_error = $abs_error;
            $first_abs_error_time1 = $time1;
            $first_abs_error_time2 = $time2;
            $first_abs_error_value1 = $value1;
            $first_abs_error_value2 = $value2;
            $detected_abs_error = 1;
          }
          if ($abs_error > $max_abs_error) {
            $max_abs_error = $abs_error;
            $max_abs_error_time1 = $time1;
            $max_abs_error_time2 = $time2;
            $max_abs_error_value1 = $value1;
            $max_abs_error_value2 = $value2;
          }
        }

        #__________________________________
        # do relative comparisons
        $rel_error = abs($value1 - $value2) / $max_abs;
        if ($rel_error > $allowable_rel_error) {
        
          if ($detected_rel_error == 0) {
            $first_rel_error = $rel_error;
            $first_rel_error_time1 = $time1;
            $first_rel_error_time2 = $time2;
            $first_rel_error_value1 = $value1;
            $first_rel_error_value2 = $value2;
            $detected_rel_error = 1;
          }
          if ($rel_error > $max_rel_error) {
            $max_rel_error = $rel_error;
            $max_rel_error_time1 = $time1;
            $max_rel_error_time2 = $time2;
            $max_rel_error_value1 = $value1;
            $max_rel_error_value2 = $value2;
          }
        }
      }
    }
  }  # while loop

  #__________________________________
  if ($detected_rel_error != 0 || $detected_abs_error != 0) {
    print "FAILED\n";

    if ($detected_rel_error != 0) {
      print "    greatest relative error (%" . $max_rel_error * 100 . ") at:\n";
      print "                    time                   value:\n";
      print "             UDA1: " . $max_rel_error_time1 . " / " . $max_rel_error_value1 . "\n";
      print "             UDA2: " . $max_rel_error_time2 . " / " . $max_rel_error_value2 . "\n";

      if ($max_rel_error != $first_rel_error) {
        print "    and first significant relative error (%" . $first_rel_error * 100 . ") at:\n";
        print "                    time                   value:\n";
        print "             UDA1: " . $first_rel_error_time1 . " / " . $first_rel_error_value1 . "\n";
        print "             UDA2: " . $first_rel_error_time2 . " / " . $first_rel_error_value2 . "\n";
      }
    }

    if ($detected_abs_error != 0) {
      print "\n";
      print "    greatest absolute error: " . $max_abs_error . "\n";
      print "                    time                   value:\n";
      print "             UDA1: " . $max_abs_error_time1 . " / " . $max_abs_error_value1 . "\n";
      print "             UDA2: " . $max_abs_error_time2 . " / " . $max_abs_error_value2 . "\n";

      if ($max_abs_error != $first_abs_error) {
        print "    and first significant absolute error: %" . $first_abs_error * 100 . "\n";
        print "                    time                   value:\n";
        print "             UDA1: " . $first_abs_error_time1 . " / " . $first_abs_error_value1 . "\n";
        print "             UDA2: " . $first_abs_error_time2 . " / " . $first_abs_error_value2 . "\n";
      }
    }
    print "\nSuggested command to compare these files:\n\n";
    print "   xxdiff " . $datfilename1 . " \\\n          " . $datfilename2 . "\n\n";
    $failed = 1;
  }
  else {
    print "PASSED\n";
  }

  close(IN1);
  close(IN2);
}

if ($failed != 0) {
  print "\nOne or more dat files are not within allowable error.\n";
  exit(1);
}
else {
  print "\nDat files are all within allowable error.\n";
  exit(0);
}

#______________________________________________________________________

sub getTimeAndValue {
  my($line) = @_;
  my(@vals) = ();
  
  #  This is fragile and may not work for all possible number formats   -Todd
  $number_reg_exp = "(-?\\d+(?:\\.\\d+)?(?:[Ee][+-]?\\d+(?:\\.\\d+)?)?)";
  
  #__________________________________
  # ignore header lines. return 0 0
  if ($line =~'[:alpha:]') {
    $time = 0;
    push(@vals,0);
    return ($time, @vals);
  }

  if ( $line =~ "$number_reg_exp" ) {
    $time = $1;
    $line = $';
    
    while ( $line =~ "$number_reg_exp" ) {
      push(@vals, $1);
      $line = $';
    }

    if (@vals >= 1) {
      return ($time, @vals);
    }
  }
  
  print "Error parsing line:\n" . @_[0] . "\n";
  exit(2);
}





