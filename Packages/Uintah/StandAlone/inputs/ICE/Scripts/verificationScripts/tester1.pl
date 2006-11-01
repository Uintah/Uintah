#!/usr/bin/perl

# use module
use XML::Simple;
use Data::Dumper;
# create object
$xml = new XML::Simple(forcearray => 1);
#$xml = new XML::Simple;

#print $ARGV[0];


# read XML file
$data = $xml->XMLin("$ARGV[0]");


# Reading the meta data and finding out the number of tests

my $i=0;

foreach $e (@{$data->{Test}})
{
     $test_title[$i]=$e->{Meta}->[0]->{Title}->[0];
     $test_upsF[$i]=$e->{Meta}->[0]->{upsFile}->[0];
     @tmp = split(/\//,$test_upsF[$i]);   # This is to get the file name only without the full path preceding it
     $base_filename[$i] = $tmp[$#tmp];       # This will be used in constructing the test_file_name
     $base_filename[$i]=~ s/.ups//;
     $test_pbsF[$i]=$e->{Meta}->[0]->{pbsFile}->[0];
     $int_command[$i]=$e->{Meta}->[0]->{interactive}->[0];
     $i++;
}

$num_of_tests=$i;


#print $test_pbsF[0];
# Extracting info out of the config file

my $i=0;
my $line;
my $tmpline;
#my @reqired_lines;

open(MYDATA, "$ARGV[0]") or die("$ARGV[0], File not found");

while ($line=<MYDATA>)
{
    if ($line=~ /\<content\>/)
    {
	$j=0;
	while (($line=<MYDATA>) !~ /\<\/content\>/)
	{
	    $req_lines[$i][$j]=$line;  
	    $j++;
	}
	$i++;
    }
}

close(MYDATA);

#print $required_lines[0][0];
#print $required_lines[1][0];
#print $num_of_tests;
#print Dumper(@required_lines);


open(statsFile,">$ARGV[0]".".stat");

# Creating new ups files for each test

for ($i=0;$i<$num_of_tests;$i++)
{
    open(inpFile, $test_upsF[$i]) or die("$test_upsF[$i], File Not Found");

    my $test_ups;
    my $testI;

    @required_lines = (@{$req_lines[$i]});  # This is just assigning the first set of req_lines into the required_lines

#    print $required_lines[0];
    $test_ups = $base_filename[$i]."_$test_title[$i]".".ups";
    $udaFilename = $base_filename[$i]."_$test_title[$i]".".uda";
    $test_pbs = $base_filename[$i]."_$test_title[$i]".".pbs";  
    

    $int = $int_command[$i];

    open(outFile, ">$test_ups") or die("$test_ups, File cannot be created\n");

    while($line=<inpFile>)
    {
	if ($line =~ /\<filebase\>/)
	{
	   $line= "<filebase>$udaFilename</filebase>\n";
	}
	
	if ($line =~ /($required_lines[0])/)
	{
	    print outFile @required_lines;
	    while(($line =<inpFile>) !~ /($required_lines[$#required_lines])/)
	    {
		# Do nothing; we are basically skipping those lines (not including them in the out file)
		# There has to be a better way, but this will work for now.
	    }
	    $line=<inpFile>;
	}
	print outFile $line;
    }
    
    close(inpFile);
    close(outFile);

    if ( $int eq "")
    {
	open(inpFile, $test_pbsF[$i]) or die("$test_pbsF[$i], File Not Found\n");

	open (outFile, ">$test_pbs") or die("$test_pbs, File cannot be created\n");
	
	while($line=<inpFile>)
	{

	    if ($line =~ m/set\s*LAMJOB/)
	    {
		@tmp_arr = split(" ",$line);
		foreach $tmp_var (@tmp_arr)
		{
		    if ($tmp_var =~ /\.ups/)
		    {
			$tmp_var =~ s/\S*/$test_ups/;  # This replaces the ups file argument  in the pbs file
		    }
		    
		}
		$line = join(' ',@tmp_arr)."'"."\n";
	    }
	    
	    if($line =~ m/set\s*OUT/)
	    {
		$line = "set OUT = \"out.$base_filename[$i]"."_$test_title[$i]".".000\"\n";
	    }

	    print outFile $line;
	    
	}
    }
    close(outFile);

    print statsFile "Test Name : "."$test_title[$i]"."\n";
    print statsFile "Input file(ups) : "."$test_ups"."\n";
    print statsFile "Ouput file(uda) : "."$udaFilename"."\n";
    
    if ($int eq "")
    {
	print statsFile "Queue file (pbs) : "."test_pbs"."\n";
	$tmp=`qsub $test_pbs`;
    }
    else 
    {
	print statsFile "Command Used (interactive) : "."$int $test_ups"."\n";
	$now = time();
	$tmp=`$int $test_ups`;
	$fin = time()-$now;
	print  statsFile "Running Time : ".$fin."\n";
    }

    print statsFile "---------------------------------------------\n";
    
}


