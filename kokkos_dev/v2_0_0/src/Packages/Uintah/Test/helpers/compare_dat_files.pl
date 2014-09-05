#!/usr/bin/perl

if (@ARGV <= 4) {
    print "usage: compare_dat_files {abs error allowed} {rel error allowed} {uda directory 1} {uda directory 2} {dat file names}\n";
}

$allowable_abs_error = $ARGV[0];
$allowable_rel_error = $ARGV[1];
$uda_dir1 = $ARGV[2];
$uda_dir2 = $ARGV[3];
$greatest_rel_error = 0;
$greatest_error_time1 = 0;
$greatest_error_time2 = 0;
$first_significant_error_time1 = 0;
$first_significant_error_time2 = 0;
$failed = 0;

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
    $has_significant_error = 0;
    
    while (($line1 = <IN1>) && ($line2 = <IN2>)) {
	($time1, @values1) = getTimeAndValue($line1);
	($time2, @values2) = getTimeAndValue($line2);
	if (@values1 != @values2) {
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
		if (abs($value1 - $value2) > $allowable_abs_error) {
		    $rel_error = abs($value1 - $value2) / $max_abs;
		}
		if ($rel_error > $greatest_rel_error) {
		    $greatest_rel_error = $rel_error;
		    $greatest_error_time1 = $time1;
		    $greatest_error_time2 = $time2;
		    if ($has_significant_error == 0 && \ 
			$rel_error > $allowable_rel_error) {
			$first_significant_error_time1 = $time1;
			$first_significant_error_time2 = $time2;
			$has_significant_error = 1;
		    }
		}
	    }
	}
    }

    if ($has_significant_error != 0) {
	print "*** failed\n";
	print "\tgreatest relative error: %" . $greatest_rel_error * 100;
	print "\n\tat times: " . $greatest_error_time1 . " / ";
	print $greatest_error_time2 . "\n\tand first signifant error at: ";
	print $first_significant_error_time1 . " / ";
	print $first_significant_error_time2 . "\n";
	print "\nThe following is the suggested command to compare these files:\n";

	print "xdiff\\\n" . $datfilename1 . "\\\n" . $datfilename2 . "\n\n";
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
    $number_reg_exp = "(-?\\d+(?:.\\d+)?(?:e-?\\d+(?:.\\d+)?)?)";
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





