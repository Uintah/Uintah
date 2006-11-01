#!/usr/bin/perl

# use module
use XML::Simple;
use Data::Dumper;
$xml = new XML::Simple(forcearray => 1);


# read XML file
$data = $xml->XMLin("$ARGV[0]");

# Reading the meta data and finding out the number of tests

my $i = 0;
my $j = 0;

foreach $e (@{$data->{Test}})
{
    $test_title = $e->{Meta}->[0]->{Title}->[0];
    $int_command = $e->{Meta}->[0]->{interactive}->[0];
    $output_file = $e->{Meta}->[0]->{outFile}->[0];
    
    @uda_files = @{$e->{udaFile}};

    if ($output_file eq "")
    {
	$output_file = "tmp_out";
    }

    $test_title =~ tr/" "/"_"/;  # This replaces the spaces with undescore in the test title 

    `rm -f $test_title.dat`; # test_title is used as the file name, this will delete any pre-existing files with the same name 

    $final_command = "$int_command -o $output_file -uda ";

    for ($k = 0 ; $k<=$#uda_files; $k++)
    {
	print "$final_command $uda_files[$k]\n";
	`$final_command $uda_files[$k]`;
	print "more $output_file >> $test_title.dat\n";
	`more $output_file >> $test_title.dat`;  # This will append the new global error to the test_title file
    }
    
    # The gnuplot script creation
    open(gpFile, ">$test_title.gp");
    print gpFile "set term postscript\n";
    print gpFile "set output \"$test_title.ps\"\n";
    print gpFile "plot \'$test_title.dat\' with line\n";

    close(gpFile);
    `gnuplot $test_title.gp`;
}

