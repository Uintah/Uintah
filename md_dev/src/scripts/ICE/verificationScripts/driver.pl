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

my $clean = 0;

chomp($ARGV[1]);

if ($ARGV[1] eq "clean") 
{
  $clean = 1;
}

# Reading the meta data and finding out the number of tests

my $i=0;

$test_title = $data->{Meta}->[0]->{Title}->[0];
@emails     = @{$data->{Meta}->[0]->{Email}};
@testFiles  = @{$data->{testFile}};
@newPath    = @{$data->{path}};

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

#  print "The path variable is:".$path_var[$i]."\n";
#  print "File name :".$testFileName."\n";
  
  # Check if the path_var is not empty and then change into it
  if ($path_var[$i])
  {
    chdir($path_var[$i]);
  }

  if ( ($tmp[$#tmp] eq "xml") || ($tmp[$#tmp] eq "XML") )
  {
    if ($clean == 1)
    {
      print "cleaning the temp files \n";
      system("rm -f *.tmp *.stat *.results *.DONE .*.tmp *.dat");
      
      print "Launching driver.pl $testFileName  clean & \n";
      @args = ("driver.pl","$testFileName clean &");
      system("@args")==0 or die("ERROR(driver.pl): @args failed");      
    }
    else
    {
      print "Launching driver.pl $testFileName & \n";
      #__________________________________
      # update the path
      $orgPath = $ENV{"PATH"};
      $ENV{"PATH"} = "@newPath:$curr_dir/bin:$orgPath";
      $Path = $ENV{"PATH"};
      print "path $Path\n";
      
      @args = ("driver.pl","$testFileName &");
      system("@args")==0 or die("ERROR(driver.pl): @args failed");
    }
  }
  elsif(($tmp[$#tmp] eq "tst") || ($tmp[$#tmp] eq "TST"))
  {
    if ($clean == 1)
    {
      print "cleaning the temporary files \n";
      system("/bin/rm -f *.tmp *.stat *.results *.DONE .*.tmp *.dat");
      print "Press Return to exit \n";
    }
    else 
    {
      # clean out any section of the configuration file that is commented out
      #WARNING this doesn't work for single lines that are commented out
      system("/bin/rm -f $testFileName.clean");
      $cmd = "sed  /'<!--'/,/'-->'/d < $testFileName > $testFileName.clean \n";
      system("$cmd");
      
      print "\n\n Launching run_tests.pl $testFileName.clean\n\n";
      @args = ("run_tests.pl","$testFileName.clean");
      system("@args")==0  or die("ERROR(driver.pl): @args failed");
      
      system("/bin/rm -f $testFileName.clean");
    }
  }
  
  # Change back into the current working dir to proceed with     
  # the other tests
  chdir($curr_dir);
}

if ($clean == 1)
{
  exit(0);   #   No need to wait for jobs to complete, its a clean job. So exit the script
}

#__________________________________
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
    open(finalDone, ">$ARGV[0].TMP");
    
    open(htmlFile, ">$ARGV[0].htm ");
    print htmlFile "<html>\n";

    open(fileList, ">$ARGV[0].results");
    print fileList "$ARGV[0].htm\n";

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
            $html_write = "<img src=\"$write_line\" alt=\"$line\"><BR>";
            print fileList "$write_line\n";
          }
          else
          {
            chomp($line);
            $write_line = $line;
            $html_write = "<B>$line</B><BR>";
          }
          print finalDone "$write_line\n";
          print htmlFile "$html_write \n";
        }
        close(doneFile);
        print "Cleaning up the"." $fl_name.DONE"."\n";
        `rm -f $fl_name.DONE`;
        `rm -f $fl_name.results`;
#       `rm -f $fl_name.DONE`;
      }
      $i++;
    }
    close(htmlFile);
    close(fileList);

    close (finalDone);
    `mv $ARGV[0].TMP $ARGV[0].DONE`;
    print htmlFile "</html>\n";
    $i = 1;
    foreach $email (@emails) # This sends email to all recipients when the tests are completed
    {
      print "A email shall be sent to $email\n";
      if (1==$i)
      {
        print "Adding the line the TMP file $i\n";
        `echo \"The testing framework tree is at $curr_dir\">>$ARGV[0].TMP`;
        $i = 0;
      }
      `tar  zcvf results.tar.gz  -T $ARGV[0].results`;
      # Send an email ;
      `cat $ARGV[0].TMP | mail -s Test -a results.tar.gz  $email`;
    }
    
    last;   # Break out of the loop all tests have completed successfully   
  }
  
  $secs = 120;

  print "Gonna Sleep for $secs seconds\nzzzzzzzzzzzzzzzzzzzz.......\n";
  sleep($secs);  # This puts the script in sleep for x mins

}
