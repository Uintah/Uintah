#!/usr/bin/perl

# standalone converter to create a NRRD header for a vff file.
# The vff file format has a header, follwed by a line with a formfeed,
# followed by the actual raw data.

$inf=$ARGV[0];
$out=$ARGV[1];

open(INF,"<$inf") || die "\n Can't open $inf !\n Wrong filename base?\n ...";

my $y, $y, $z, $num;

while (<INF>)
{
  $num++;
  $line = $_;
  if ($line=~ /^size=/)
  {
    $size_line = $line;
    chomp $size_line;
    $size_line=~ s/size=//;
    $size_line=~ s/\;//;
    ($x, $y, $z) = split(' ',$size_line);
  }
  if ($line=~ /\f/){
    $line_skip = $num;
    last;
  }
  $prev_line = $line;
}
close(INF);

open(OUT,">$out") || die "\n Can't open $out !\n Wrong filename base?\n ...";
print OUT "NRRD0001\n";
print OUT "type: short\n";
print OUT "dimension: 3\n";
print OUT "sizes: $x $y $z\n";
print OUT "spacings: 1 1 1\n";
print OUT "data file: $inf\n";
print OUT "endian: big\n";
print OUT "encoding: raw\n";
print OUT "line skip: $line_skip\n";
close(OUT);
