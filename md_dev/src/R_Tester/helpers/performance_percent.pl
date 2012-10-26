#!/usr/bin/perl

$testfilename = $ARGV[0];
$comparefilename = $ARGV[1];

$test_time = 0;
$compare_time = 0;

open(TESTFILE, $testfilename);
open(COMPAREFILE, $comparefilename);

@test_text = <TESTFILE>;
$test_text = join("", @test_text);
if ($test_text =~ /real\s*(\w+\.\w+)/) {
    $test_time = $1;
}

@compare_text = <COMPAREFILE>;
$compare_text = join("", @compare_text);
if ($compare_text =~ /real\s*(\w+\.\w+)/) {
    $compare_time = $1;
}

print STDERR "New test time " . $test_time . "\n";
print STDERR "Old test time " . $compare_time . "\n";

if ($compare_time == 0 && $test_time > 0) {
    $percent = 99999;
}
elsif ($test_time == 0 && $compare_time > 0) {
    $percent = -99999;
}
elsif ($test_time > 0 && $compare_time > 0) {
    $percent = int(($test_time - $compare_time) / 
		   $compare_time * 100 + 0.5);
}
else {
    $percent = 0; # both are zero
}
 
if ($percent > 0) {
    print STDERR "Total time increased (worsened) by %" . $percent . "\n";
}
elsif ($percent < 0) {
    print STDERR "Total time decreased (improved) by %" . -$percent . "\n";
}
else {
    print STDERR "Total time stayed the same.\n"
}
print "$percent $test_time";
exit 0

