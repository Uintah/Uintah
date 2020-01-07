#!/usr/bin/env perl


#______________________________________________________________________
#  run_tests.pl
#  Perl script used to modify an input file and run the tests listed in
#  the tst file.
#
# Algorithm
#   - read in each test in the tst file
#   - make a symbolic link to the comparison utility used to compute the L2nomr
#   - read in the replacement patterns for each test and all tests
#   - perform global replacements on the ups file
#
#   Loop over tests
#     - create a new ups file
#     - change the output uda name
#     - replace lines in ups file
#     - run the test
#
#     if(post Process cmd )
#       -run analyze_results.pl <tst file> < test number>
#     endif
#   end Loop
#
#______________________________________________________________________
use strict;
use warnings;
use XML::LibXML;
use Data::Dumper;
use Time::HiRes qw/time/;
use File::Basename;
use Cwd;
use lib dirname (__FILE__);  # needed to find local Utilities.pm
use Utilities qw( cleanStr modify_batchScript read_file write_file );

# removes white spaces from variable
sub  trim { my $s = shift; $s =~ s/^\s+|\s+$//g; return $s };

my $tstFile           = $ARGV[0];
my $config_files_path = $ARGV[1];

# read XML file into a dom tree
my $doc = XML::LibXML->load_xml(location => $tstFile);

#__________________________________
# copy gnuplot script   OPTIONAL
my $gpFile = cleanStr( $doc->findvalue( '/start/gnuplot/script' ) );

if( length $gpFile > 0 ){
  $gpFile    = $config_files_path."/".$gpFile;
  system("cp -f $gpFile . > /dev/null 2>&1");
  print "  gnuplot script used in postProcess ($gpFile)\n";
}

#__________________________________
# copy batch script and modify the template  OPTIONAL

my $batchCmd    = cleanStr( $doc->findvalue( '/start/batchScheduler/submissionCmd' ) );
my $batchScript = cleanStr( $doc->findvalue( '/start/batchScheduler/template' ) );

if( length $batchScript > 0 ){
  my $cmd = "cp -f $config_files_path"."/"."$batchScript" . " . > /dev/null 2>&1";
  system( $cmd );
  print "  Batch script template used to submit jobs ($batchScript)\n";

  my @nodes = $doc->findnodes('/start/batchScheduler/batchReplace');
  modify_batchScript( $batchScript, @nodes );
  print "\n";
}


#__________________________________
# set exitOnCrash flag    OPTIONAL
my $exitOnCrash = "true";
if( $doc->exists( '/start/exitOnCrash' ) ){
  $exitOnCrash = cleanStr( $doc->findvalue( '/start/exitOnCrash' ) );
}
$exitOnCrash = trim(uc($exitOnCrash));
print "  Exit on crash or timeout ($exitOnCrash)\n";


#__________________________________
# set sus timeout value    OPTIONAL
my $timeout = 24*60*60;
if( $doc->exists( '/start/susTimeout_minutes' ) ){
  $timeout = cleanStr( $doc->findvalue( '/start/susTimeout_minutes' ) );
}
print "  Simulation timeout: $timeout minutes\n";


#__________________________________
# determine the ups basename
my $upsFile      = cleanStr( $doc->findvalue( '/start/upsFile' ) );
$upsFile         = basename( $upsFile );
my $ups_basename = basename( $upsFile, ".ups" );        # Removing the extension .ups so that we can use this to build our uda file names

if ( ! -e $upsFile ){
  print "\n\nERROR(run_tests.pl): $upsFile, File Not Found";
  print " Now exiting\n";
  exit
}

#__________________________________
# Globally, replace lines & values in the main ups file before loop over tests.
print "  \nReplacing lines and values in base ups file\n";

foreach my $rp ( $doc->findnodes('/start/AllTests/replace_lines/*') ){
  system("replace_XML_line", "$rp", "$upsFile") ==0 ||  die("Error replacing_XML_line $rp in file $upsFile \n $@");
  print "\treplace_XML_line $rp\n";
}

foreach my $X ($doc->findnodes('/start/AllTests/replace_values/entry')) {
  my $xmlPath = $X->{path};
  my $value   = $X->{value};
  print "\treplace_XML_value $xmlPath $value\n";
  system("replace_XML_value", "$xmlPath", "$value", "$upsFile")==0 ||  die("Error: replace_XML_value $xmlPath $value $upsFile \n $@");
}


#__________________________________
#     loop over tests
my $statsFile;
open( $statsFile,">out.stat");

my $nTest = 0;
foreach my $test_dom ($doc->findnodes('/start/Test')) {

  my $test_title  = cleanStr( $test_dom->findvalue('Title') );
  my $test_ups    = $ups_basename."_$test_title".".ups";
  my $test_output = "out.".$test_title;
  my $uda         = $ups_basename."_$test_title".".uda";

  #__________________________________
  # change the uda filename in each ups file
  print "\n-------------------------------------------------- $test_title\n";
  print "Now modifying $test_ups\n";

  system(" cp $upsFile $test_ups");
  my $fn = "<filebase>".$uda."</filebase>";
  system("replace_XML_line", "$fn", "$test_ups")==0 ||  die("Error replace_XML_line $fn in file $test_ups \n $@");
  print "\treplace_XML_line $fn\n";

  #__________________________________
  # replace lines in test_ups
  foreach my $rpl ( $test_dom->findnodes('replace_lines/*') ) {
    print "\treplace_XML_line $rpl\n";
    system("replace_XML_line", "$rpl", "$test_ups")==0 ||  die("Error replacing_XML_line $rpl in file $test_ups \n $@");
  }

  #__________________________________
  # replace any values in the ups files
  foreach my $X ( $test_dom->findnodes('replace_values/entry') ) {
    my $xmlPath = $X->{path};
    my $value   = $X->{value};

    print "\treplace_XML_value $xmlPath $value\n";
    system("replace_XML_value", "$xmlPath", "$value", "$test_ups")==0 ||  die("Error: replace_XML_value $xmlPath $value $test_ups \n $@");
  }

  #__________________________________
  #  replace any batch script values per test
  my $test_batch = undef;

  if( length $batchScript > 0 ){
    my ($basename, $parentdir, $ext) = fileparse($batchScript, qr/\.[^.]*$/);
    $test_batch = "batch_$test_title$ext";
    system(" cp $batchScript $test_batch" );

    modify_batchScript( $test_batch, $test_dom->findnodes('batchReplace') );
  }

  #bulletproofing
  system("xmlstarlet val --err $test_ups") == 0 ||  die("\nERROR: $upsFile, contains errors.\n");

  #__________________________________
  # print meta data and run sus command
  my $sus_cmd_0 = $test_dom->findnodes('sus_cmd');

  print $statsFile "Test Name :     "."$test_title"."\n";
  print $statsFile "(ups) :         "."$test_ups"."\n";
  print $statsFile "(uda) :         "."$uda"."\n";
  print $statsFile "output:         "."$test_output"."\n";
  print $statsFile "Command Used :  "."$sus_cmd_0 $test_ups"."\n";

  my $now = time();
  my @sus_cmd = ("$sus_cmd_0 ","$test_ups ","> $test_output 2>&1");

  my $rc = 0;
  if( length $batchScript > 0 ){
    submitBatchScript( $test_title, $batchCmd, $test_batch, $statsFile, @sus_cmd );
  }else{
    $rc = runSusCmd( $timeout, $exitOnCrash, $statsFile, @sus_cmd );
  }

  my $fin = time()-$now;
  printf $statsFile ("Running Time :  %.3f [secs]\n", $fin);

  #__________________________________
  #  execute post process command
  my $postProc_cmd = undef;
  $postProc_cmd = $test_dom->findvalue('postProcess_cmd');

  if( $rc == 0 && length $postProc_cmd != 0){

    my @cmd = ("analyze_results.pl","$tstFile", "$nTest", "$uda");
    print $statsFile "postProcessCmd:  "."$postProc_cmd"." -uda ".$uda."\n";

    if ( $exitOnCrash eq "TRUE" ) {
      system("@cmd")==0 or die("ERROR(run_tests.pl): \t\tFailed running: (@cmd)\n");
    }else{
      system("@cmd");
    }
  }
  print $statsFile "---------------------------------------------\n";
  $nTest++;
}  # all tests loop

close($statsFile);


#______________________________________________________________________
#   subroutines
#______________________________________________________________________
  #
sub runSusCmd {
  my( $timeout, $exitOnCrash, $statsFile, @sus_cmd ) = @_;

  print "\tLaunching: (@sus_cmd)\n";
  my @cmd = (" timeout --preserve-status $timeout @sus_cmd ");

  my $rc = -9;
  if ( $exitOnCrash eq "TRUE" ) {

    $rc = system("@cmd");

    if ( $rc != 0 && $rc != 36608 ){
      die("ERROR(run_tests.pl): \t\tFailed running: (@sus_cmd)\n");
      return 1;
    }

  }else{
    $rc = system("@cmd");
  }

  #__________________________________
  #  Warn user if sus didn't run successfully
  if( $rc == 36608 ) {
    print "\t\tERROR the simulation has timed out.\n";
    print $statsFile "\t\tERROR the simulation has timed out.\n";
  }
  elsif ($rc != 0 ){
    print "\t\tERROR the simulation crashed. (rc = $rc)\n";
    print $statsFile "\t\tERROR the simulation crashed. (rc = $rc)\n";
  }

  return $rc;

};


#______________________________________________________________________

sub submitBatchScript{
  my( $test_title, $batchCmd, $test_batch, $statsFile, @sus_cmd ) = @_;

  #__________________________________
  # concatenate sus cmd to batch script
  open(my $fh, '>>', $test_batch) or die "Could not open file '$test_batch' $!";
  print $fh "\n ", @sus_cmd, "\n";
  close $fh;

  #__________________________________
  # edit batch script
  my $data  = read_file($test_batch);

  #  change job name
  my $tag   = "\\[jobName\\]";
  my $value = $test_title;
  $data     =~ s/$tag/"$value"/g;

  # change the job output name
  $tag      = "\\[output\\]";
  $value    = "job-". $test_title . ".out";
  $data     =~ s/$tag/"$value"/g;

  # remove white spaces before and after "="
  # Slurm doesn't like white spaces
  $data     =~ s{\s+=}{=}g;
  $data     =~ s{=\s+}{=}g;

  #print "$data";
  write_file($test_batch, $data);

  #__________________________________
  # concatenate postProcess cmd to batch script  TODO

  #__________________________________
  # submit batch script

  print "\t Submitting batch script: ", $batchCmd, " " , $test_batch, "\n";
  my @cmd = ( "$batchCmd", "$test_batch" );
  system("@cmd")==0 or die("ERROR(run_tests.pl): \t\tFailed running: (@cmd)\n");
};
