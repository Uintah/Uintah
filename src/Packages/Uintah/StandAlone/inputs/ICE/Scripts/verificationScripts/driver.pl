#!/usr/bin/perl

# use module
use XML::Simple;
use Data::Dumper;
use Cwd;
# create object
$xml = new XML::Simple(forcearray => 1);
#$xml = new XML::Simple;

#print $ARGV[0];


# read XML file
$data = $xml->XMLin("$ARGV[0]");


# Reading the meta data and finding out the number of tests

my $i=0;

$test_title = $data->{Meta}->[0]->{Title}->[0];
@emails = @{$data->{Meta}->[0]->{Email}};
@testFiles = @{$data->{testFile}};


$curr_dir = cwd;

####################
# Loop through each test
####################

for($i = 0; $i<=$#testFiles; $i++)
{
    # Split the testFile tag value by "/" to seperate the path and the testFileName
    @tmp1 = split(/\//, $testFiles[$i]);

    # The last element in the array tmp1 will be the testFileName
    $testFileName = $tmp1[$#tmp1];
    
    # Check if extension is xml or tst for the testFileName
    @tmp = split(/\./,$testFileName);

    # Removing the last element in the tmp1 array
    # Last element is the file name, we are going to construct the path
    pop(@tmp1);

    # Now we join the array using the "/" to construct the path 
    # The path is used to change into that dir and launch the appropriate script

    $path_var[$i] = join('/',@tmp1);

#    print "The path variable is:".$path_var[$i]."\n";
#    print "File name :".$testFileName."\n";
    
    # Check if the path_var is not empty and then change into it
    if ($path_var[$i])
    {
	chdir($path_var[$i]);
    }

    if ( ($tmp[$#tmp] eq "xml") || ($tmp[$#tmp] eq "XML") )
    {
	print "Launching driver.pl $testFileName & \n";
	system("driver.pl $testFileName &");
    }
    elsif(($tmp[$#tmp] eq "tst") || ($tmp[$#tmp] eq "TST"))
    {
	print "Launching run_tests.pl $testFileName\n";
	`run_tests.pl $testFileName`;
    }
    
    # Change back into the current working dir to proceed with     
    # the other tests
    chdir($curr_dir);

}

while(1)
{

    # Number of tests scheduled by this script
    $failed_cnt = $#testFiles+1;

    print "Number of tests:".$failed_cnt."\n";

    ## For each directory (each test) we have to check if the DONE file is ready
    # The DONE file signifies that the tests in that tree have completed
    # When all the tests scheduled by this script is DONE we will exit this infinite while loop


    foreach $fl_name (@testFiles)
    {
	if (-e "$fl_name.DONE")
	{
	    $failed_cnt--;
	    print "Updated unfinished count:".$failed_cnt."\n";
	}
    }
 
    

    if(0 == $failed_cnt  )
    {
	
#	`echo DONE> $ARGV[0].DONE`;    # Creating the DONE file for the parent script to read 

	##########
	### Cleaning the DONE files in the child directories (to prepare for the next regression tester)
	##########
	my $i = 0;
	foreach $fl_name (@testFiles)
	{
	    if (-e "$fl_name.DONE")
	    {
		open(doneFile, "$fl_name.DONE");
		while($line=<doneFile>)
		{
#		    print $line;
		    if ($line !~ m/Title/)  # If the line is not a Title then we add the path in front of it
		    {
			chomp($line);
			if ($path_var[$i])
			{
			    $write_line = "$path_var[$i]"."\/"."$line";
			}
			else
			{
			    $write_line = $line;
			}
		    }
		    else
		    {
			chomp($line);
			$write_line = $line;
		    }
		    `echo \"$write_line\" >> $ARGV[0].TMP`;
		}
		close(doneFile);
		print "Cleaning the"." $fl_name.DONE"."\n";
		`rm -f $fl_name.DONE`;
	    }
	    $i++;
	}
	`mv $ARGV[0].TMP $ARGV[0].DONE`;
	foreach $email (@emails) # This sends email to all recipients when the tests are completed
	{
	    print "A email shall be sent to $email\n";
	    # Send an email ;
	    `cat $ARGV[0].DONE | mail -s Test $email`;
	}
	
	last;   # Break out of the loop all tests have completed successfully	
    }

    print "Gonna Sleep for 1 min\nzzzzzzzzzzzzzzzzzzzz.......\n";
    sleep(60);  # This puts the script in sleep for x mins

}
