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

print STDERR "New total memory highwater " . $test_highwater . "\n";
print STDERR "Old total memory highwater " . $compare_highwater . "\n";

if ($compare_highwater == 0 && $test_highwater > 0) {
    $percent = 99999;
    $abs = 0;
}
elsif ($test_highwater == 0 && $compare_highwater > 0) {
    $percent = -99999;
    $abs = 0;
}
elsif ($test_highwater > 0 && $compare_highwater > 0) {
    $percent = int(($test_highwater - $compare_highwater) / 
		   $compare_highwater * 100 + 0.5);
    $abs = $test_highwater - $compare_highwater;
}
else {
    $percent = 0; # both are zero
    $abs = 0;
}
 
if ($percent > 0) {
    print STDERR "Memory usage increased (worsened) by %" . $percent . " " . $abs . " bytes\n";
}
elsif ($percent < 0) {
    print STDERR "Memory usage decreased (improved) by %" . -$percent . " " . $abs . " bytes\n";
}
else {
    print STDERR "Memory usage stayed the same.\n"
}
print "$percent $abs $test_highwater";
exit 0

