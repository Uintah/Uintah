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
if ($test_highwater > 1.5 * $compare_highwater) {
    print "New memory exceeds old memory in highwater usage by more than 50%" . "\n";
    exit 1;
}
exit 0;
