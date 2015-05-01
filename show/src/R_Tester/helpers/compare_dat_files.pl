#!/usr/bin/perl

if (@ARGV <= 4) {
    print "usage: compare_dat_files {abs error allowed} {rel error allowed} {uda directory 1} {uda directory 2} {dat file names}\n";
}



$allowable_abs_error = $ARGV[0];
$allowable_rel_error = $ARGV[1];
$uda_dir1 = $ARGV[2];
$uda_dir2 = $ARGV[3];

print "Using absolute tolerance:" . $allowable_abs_error . "\n";
print "Using relative tolerance:" . $allowable_rel_error . "\n";


$greatest_rel_error = 0;
$greatest_abs_error = 0;
$first_significant_rel_error = 0;
$first_significant_abs_error = 0;
$greatest_rel_error_time1 = 0;
$greatest_rel_error_time2 = 0;
$greatest_abs_error_time1 = 0;
$greatest_abs_error_time2 = 0;
$first_significant_rel_error_time1 = 0;
$first_significant_rel_error_time2 = 0;
$first_significant_abs_error_time1 = 0;
$first_significant_abs_error_time2 = 0;

$greatest_rel_error_value1 = 0;
$greatest_rel_error_value2 = 0;
$greatest_abs_error_value1 = 0;
$greatest_abs_error_value2 = 0;
$first_significant_rel_error_value1 = 0;
$first_significant_rel_error_value2 = 0;
$first_significant_abs_error_value1 = 0;
$first_significant_abs_error_value2 = 0;
$failed = 0;
$lineno = 0;
foreach $datfile (@ARGV[4 .. @ARGV-1]) {
    $datfilename1 = $uda_dir1 . "/" . $datfile;
    $datfilename2 = $uda_dir2 . "/" . $datfile;
    if (!open(IN1, $datfilename1)) {
	print "Could not open " . $datfilename1 . "\n";
    }
    if (!open(IN2, $datfilename2)) {
	print "Could not open " . $datfilename2 . "\n";
    }
    print "Comparing " . $datfile . "... ";
    $has_significant_rel_error = 0;
    $has_significant_abs_error = 0;
    $greatest_rel_error = 0;
    $greatest_abs_error = 0;
    
    while (($line1 = <IN1>) && ($line2 = <IN2>)) {
        $lineno = $lineno + 1;
	($time1, @values1) = getTimeAndValue($line1);
	($time2, @values2) = getTimeAndValue($line2);
	if (@values1 != @values2) {

            print "Values1 " . @values1 . " Values2 " . @values2;
            print " on line number " . $lineno . "\n";
            
	    print "Error: the dat files do not have the same number of values per line\n";
	    exit(1);
	}
	# prepend the time onto the value list to be compared as values
	unshift(@values1, $time1);
	unshift(@values2, $time2);
	foreach $value1 (@values1) {
	    $value2 = shift(@values2);
	    $max_abs = abs($value1);
	    if (abs($value2) > $max_abs) {
		$max_abs = abs($value2);
	    }
	    if ($max_abs != 0) {
                # do absolute comparisons
                $abs_error = abs($value1 - $value2);
		if ($abs_error > $allowable_abs_error) {
		    if ($has_significant_abs_error == 0) { 
                        $first_significant_abs_error = $abs_error;
			$first_significant_abs_error_time1 = $time1;
			$first_significant_abs_error_time2 = $time2;
			$first_significant_abs_error_value1 = $value1;
			$first_significant_abs_error_value2 = $value2;
			$has_significant_abs_error = 1;
		    }
                    if ($abs_error > $greatest_abs_error) {
                        $greatest_abs_error = $abs_error;
                        $greatest_abs_error_time1 = $time1;
                        $greatest_abs_error_time2 = $time2;
                        $greatest_abs_error_value1 = $value1;
                        $greatest_abs_error_value2 = $value2;
                    }
		}

                # do relative comparisons
                $rel_error = abs($value1 - $value2) / $max_abs;
                if ($rel_error > $allowable_rel_error) {
		    if ($has_significant_rel_error == 0) { 
                        $first_significant_rel_error = $rel_error;
			$first_significant_rel_error_time1 = $time1;
			$first_significant_rel_error_time2 = $time2;
			$first_significant_rel_error_value1 = $value1;
			$first_significant_rel_error_value2 = $value2;
			$has_significant_rel_error = 1;
		    }
                    if ($rel_error > $greatest_rel_error) {
                        $greatest_rel_error = $rel_error;
                        $greatest_rel_error_time1 = $time1;
                        $greatest_rel_error_time2 = $time2;
                        $greatest_rel_error_value1 = $value1;
                        $greatest_rel_error_value2 = $value2;
                    }
		}
	    }
	}
    }

    if ($has_significant_rel_error != 0 || $has_significant_abs_error != 0) {
	print "*** failed\n";
        if ($has_significant_rel_error != 0) {
            print "\tgreatest relative error: %" . $greatest_rel_error * 100;
            print "\n\tat times: " . $greatest_rel_error_time1 . " / ";
            print $greatest_rel_error_time2 . "\n";
            print "\tvalues: " . $greatest_rel_error_value1 . " / ";
            print $greatest_rel_error_value2 . "\n";

            if ($greatest_rel_error != $first_significant_rel_error) {
                print "\tand first signifant relative error: %" . $first_significant_rel_error * 100 . "\n";
                print "\tat times: " . $first_significant_rel_error_time1 . " / ";
                print $first_significant_rel_error_time2 . "\n";
                
                print "\tvalues: " . $first_significant_rel_error_value1 . " / ";
                print $first_significant_rel_error_value2 . "\n";
            }
        }
        if ($has_significant_abs_error != 0) {
            print "\n\tgreatest absolute error: " . $greatest_abs_error;
            print "\n\tat times: " . $greatest_abs_error_time1 . " / ";
            print $greatest_abs_error_time2 . "\n";
            print "\tvalues: " . $greatest_abs_error_value1 . " / ";
            print $greatest_abs_error_value2 . "\n";

            if ($greatest_abs_error != $first_significant_abs_error) {
                print "\tand first signifant absolute error: %" . $first_significant_abs_error * 100 . "\n";
                print "\tat times: " . $first_significant_abs_error_time1 . " / ";
                print $first_significant_abs_error_time2 . "\n";
                
                print "\tvalues: " . $first_significant_abs_error_value1 . " / ";
                print $first_significant_abs_error_value2 . "\n";
            }
        }
        print "\nThe following is the suggested command to compare these files:\n";
	print "xxdiff\\\n" . $datfilename1 . "\\\n" . $datfilename2 . "\n\n";
	$failed = 1;
    }
    else {
	print "good\n";
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

sub getTimeAndValue {
    my($line) = @_;
    my(@vals) = ();
    $number_reg_exp = "(-?\\d+(?:\\.\\d+)?(?:e-?\\d+(?:\\.\\d+)?)?)";
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
    exit(1);
}





