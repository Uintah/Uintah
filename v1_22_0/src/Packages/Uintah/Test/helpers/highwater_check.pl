#!/usr/bin/perl

$testfilename = $ARGV[0];
$comparefilename = $ARGV[1];

$test_highwater = 0;
$compare_highwater = 0;

open(TESTFILE, $testfilename);
open(COMPAREFILE, $comparefilename);

@test_text = <TESTFILE>;
$test_text = join("", @test_text);
while ($test_text =~ /highwater alloc:\s*(\w+)\s/) {
    $test_highwater += $1;
    $test_text = $';
}

@compare_text = <COMPAREFILE>;
$compare_text = join("", @compare_text);
while ($compare_text =~ /highwater alloc:\s*(\w+)\s/) {
    $compare_highwater += $1;
    $compare_text = $';
}

print "New total memory highwater " . $test_highwater . "\n";
print "Old total memory highwater " . $compare_highwater . "\n";

if ($compare_highwater == 0 && $test_highwater > 0) {
    $percent = 99999;
}
else if ($test_highwater == 0 && $compare_highwater > 0) {
    $percent = -99999;
}
if ($test_highwater > 0 && $compare_highwater > 0) {
    $percent = floor(($test_highwater - $compare_highwater) / 
		     $compare_highwate * 100 + 0.5);
}
 
if ($percent > 0) {
    print "Memory usage increased (worsened) by %" . $percent;
}
else ($percent < 0) {
    print "Memory usage decreased (improved) by %" . -$percent;
}
else {
    print "Memory usage stayed the same."
}
exit $percent;

