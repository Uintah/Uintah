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
     @tmp = split(/\//,$test_upsF[$i]);   # This is to split the file name using / as delimiters to get rid of the full path preceding the file name
     $base_filename[$i] = $tmp[$#tmp];    # The array built from previous line using the split command has the file name as its last entry (so we are grabbing that)
     $base_filename[$i]=~ s/.ups//;       # Removing the extension .ups so that we can use this to build our uda file names
     $test_pbsF[$i]=$e->{Meta}->[0]->{pbsFile}->[0];        # Reading the pbs file name for the queue
     $int_command[$i]=$e->{Meta}->[0]->{interactive}->[0];  # Reading the interactive command


     $study[$i]=$e->{Meta}->[0]->{Study}->[0];
     $errFile[$i]=$study[$i];
     $errFile[$i] =~ tr/" "/"_"/;   # Replacing spaces with underscores. This obviates the necessity to specify errFile tag which was originally planned.

#$e->{Meta}->[0]->{errFile}->[0];

     $x[$i]=$e->{Meta}->[0]->{x}->[0];
     $compCommand[$i]=$e->{Meta}->[0]->{compCommand}->[0];
     if($compCommand[$i])
     {
	 `echo 0 > .$errFile[$i].tmp`;    # This is to create a tmp file that has the number of tests under the current genre
     }
     $i++;     
}

$num_of_tests=$i;


#print @compare_conf;
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
    
    if ($compCommand[$i])
    {
	$tmp_err = `cat .$errFile[$i].tmp`;
	chomp($tmp_err);
	
	$tmp_err++;
	`echo $tmp_err > .$errFile[$i].tmp`;
    }

    open(inpFile, $test_upsF[$i]) or die("$test_upsF[$i], File Not Found");

    my $test_ups;
    my $testI;

    @required_lines = (@{$req_lines[$i]});  # This is just assigning the first set of req_lines into the required_lines

#    print $required_lines[0];
    $test_ups = $base_filename[$i]."_$test_title[$i]".".ups";
    $udaFilename = $base_filename[$i]."_$test_title[$i]".".uda";
    $test_pbs = $base_filename[$i]."_$test_title[$i]".".pbs";
    $compFilename = $base_filename[$i]."_$test_title[$i]"."_comp".".xml";
    

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


# Modifying the pbs file 

# *******Note***********
# If the <interactive> tag and the <pbsFile> are both given then <interactive> will get preference
# **********************


    if ( $int eq "")
    {
	open(inpFile, $test_pbsF[$i]) or die("$test_pbsF[$i], File Not Found\n");

	open (outFile, ">$test_pbs") or die("$test_pbs, File cannot be created\n");
	
	while($line=<inpFile>)
	{

	    # Set LAMJOB to point to the new ups file

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
	    
	    # Set OUT file to correspond to the new ups name

	    if($line =~ m/set\s*OUT/)
	    {
		$line = "set OUT = \"out.$base_filename[$i]"."_$test_title[$i]".".000\"\n";
	    }

	    # Right before we exit we need to schedule the batch_compare script
#	    $string =~ s/^\s+//;
#	    $string =~ s/\s+$//;
	    if ($compCommand[$i])
	    {
		if($line =~ m/^\s*exit/)
		{
		    $line = "analyze_results.pl $compFilename \nexit \n";
		}
	    }
	    print outFile $line;
	    
	}
    }
    close(outFile);

############
# Creation of the compare config file
############

    if($compCommand[$i])
    {
	`rm -fr $compFilename`;
	
	`echo \\<start\\> >> $compFilename`;
	`echo \\<Test\\>  >> $compFilename`;
	`echo \\<Meta\\>  >> $compFilename`;
	`echo \\<Title\\>$study[$i]\\</Title\\>  >> $compFilename`;
	`echo \\<Interactive\\>$compCommand[$i]\\</Interactive\\>  >> $compFilename`;
	`echo \\<Launcher\\>$ARGV[0]\\</Launcher\\> >> $compFilename`;
	`echo \\</Meta\\>  >> $compFilename`;
	`echo \\<x\\>$x[$i]\\</x\\>  >> $compFilename`;
	`echo \\<udaFile\\>$udaFilename\\</udaFile\\>  >> $compFilename`;
	`echo \\</Test\\>  >> $compFilename`;
	`echo \\</start\\> >> $compFilename`;
    }

    print statsFile "Test Name : "."$test_title[$i]"."\n";
    print statsFile "Input file(ups) : "."$test_ups"."\n";
    print statsFile "Ouput file(uda) : "."$udaFilename"."\n";
    
    if ($int eq "")
    {
	print statsFile "Queue file (pbs) : "."$test_pbs"."\n";
	$tmp=`qsub $test_pbs`;
    }
    else 
    {
	print statsFile "Command Used (interactive) : "."$int $test_ups"."\n";
	$now = time();
	$tmp=`$int $test_ups`;
	if($compCommand[$i])
	{
	    $tmp=`batch_compare $compFilename`;
	}
	$fin = time()-$now;
	print  statsFile "Running Time : ".$fin."\n";
    }

    print statsFile "---------------------------------------------\n";
    
}


close(statsFile);




# Extract only unique compare conf file names 

#my %hash = map { $_ => 1 } @compare_conf;
#my @array2 = sort keys %hash;

# Loop through the array of unique compare conf files and add the footer to them

#foreach $compFile (@array2)
#{
#    `echo \\</Test\\>  >> $compFile`;
#    `echo \\</start\\> >> $compFile`;
#}

#print Dumper(keys %hash);

