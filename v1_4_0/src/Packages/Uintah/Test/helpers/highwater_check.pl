#!/bin/perl

$testfilename = $ARGV[0];
$comparefilename = $ARGV[1];

open(TESTFILE, $testfilename);
open(COMPAREFILE, $comparefilename);

@test_text = <TESTFILE>;
$test_text = join("", @test_text);
$test_text =~ /highwater alloc:\s*(\w+)\s/;
$test_highwater = $1;

@compare_text = <COMPAREFILE>;
$compare_text = join("", @compare_text);
$compare_text =~ /highwater alloc:\s*(\w+)\s/;
$compare_highwater = $1;

print "New memory highwater " . $test_highwater . "\n";
print "Old memory highwater " . $compare_highwater . "\n";
if ($test_highwater > 1.5 * $compare_highwater) {
    print "New memory exceeds old memory in highwater usage by more than 50%" . "\n";
    exit 1;
}
exit 0;
