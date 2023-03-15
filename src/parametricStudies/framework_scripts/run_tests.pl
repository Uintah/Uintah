#!/usr/bin/env perl
#
# The MIT License
#
# Copyright (c) 1997-2023 The University of Utah
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#

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
use diagnostics;
use XML::LibXML;
use Data::Dumper;
use Time::HiRes qw/time/;
use File::Basename;
use Cwd;
use lib dirname (__FILE__);  # needed to find local Utilities.pm
use Utilities qw( cleanStr setPath modify_xml_file modify_batchScript read_file write_file runPreProcessCmd runSusCmd submitBatchScript );

# removes white spaces from variable
sub  trim { my $s = shift; $s =~ s/^\s+|\s+$//g; return $s };

my $tstFile           = $ARGV[0];
my $config_files_path = $ARGV[1];

# read XML file into a dom tree
my $tst_dom = XML::LibXML->load_xml(location => $tstFile);

#__________________________________
# copy gnuplot script   OPTIONAL
my $gpFile = cleanStr( $tst_dom->findvalue( '/start/gnuplot/script' ) );

if( length $gpFile > 0 ){
  $gpFile    = $config_files_path."/".$gpFile;
  system("cp -f $gpFile . > /dev/null 2>&1");
  print "  gnuplot script used in postProcess ($gpFile)\n";
}

#__________________________________
# copy batch script and modify the template  OPTIONAL

my $batchCmd    = cleanStr( $tst_dom->findvalue( '/start/batchScheduler/submissionCmd' ) );
my $batchScript = cleanStr( $tst_dom->findvalue( '/start/batchScheduler/template' ) );

if( length $batchScript > 0 ){

  $batchScript = setPath( $batchScript, $config_files_path ) ;

  my $cmd = "cp -f $batchScript" . " . > /dev/null 2>&1";
  system( $cmd );
  
  print "  Batch script template used to submit jobs ($batchScript)\n";
}


#__________________________________
# set exitOnCrash flag    OPTIONAL
my $exitOnCrash = "true";
if( $tst_dom->exists( '/start/exitOnCrash' ) ){
  $exitOnCrash = cleanStr( $tst_dom->findvalue( '/start/exitOnCrash' ) );
}
$exitOnCrash = trim(uc($exitOnCrash));
print "\tExit on crash or timeout ($exitOnCrash)\n";


#__________________________________
# set sus timeout value    OPTIONAL
my $timeout = 24*60*60;
if( $tst_dom->exists( '/start/susTimeout_minutes' ) ){
  $timeout = cleanStr( $tst_dom->findvalue( '/start/susTimeout_minutes' ) );
}
print "\tSimulation timeout: $timeout minutes\n";


#__________________________________
# determine the ups basename
my $upsFile      = cleanStr( $tst_dom->findvalue( '/start/upsFile' ) );
$upsFile         = basename( $upsFile );
my $ups_basename = basename( $upsFile, ".ups" );        # Removing the extension .ups so that we can use this to build our uda file names

if ( ! -e $upsFile ){
  print "\n\nERROR(run_tests.pl): $upsFile, File Not Found";
  print " Now exiting\n";
  exit
}

#______________________________________________________________________
# Globally, replace lines & values in the main ups file before loop over tests.
print "\n\tReplacing lines and values in base ups file\n";

my @allTests_nodes = $tst_dom->findnodes('/start/AllTests');

modify_xml_file( $upsFile, @allTests_nodes );

runPreProcessCmd( $upsFile, "null", @allTests_nodes); 



#______________________________________________________________________
#     loop over tests

my $statsFile;
open( $statsFile,">out.stat");

my $nTest = 0;
foreach my $test_node ($tst_dom->findnodes('/start/Test')) {

  my $test_title  = cleanStr( $test_node->findvalue('Title') );
  my $test_ups    = $test_title.".ups";
  my $test_output = "out.".$test_title;
  my $uda         = $test_title.".uda";

  #__________________________________
  # change the uda filename in each ups file
  print "\n-------------------------------------------------- TEST: $test_title\n";
  print "Now modifying $test_ups\n";

  system(" cp $upsFile $test_ups");
  my $fn = "<filebase>".$uda."</filebase>";
  system("replace_XML_line", "$fn", "$test_ups")==0 ||  die("Error replace_XML_line $fn in file $test_ups \n $@");
  print "\treplace_XML_line $fn\n";

  modify_xml_file( $test_ups, $test_node );
  
  runPreProcessCmd(  "null", "$test_ups", $test_node);
  
  #__________________________________
  #  replace any batch script values per test
  my $test_batch = undef;

  if( length $batchScript > 0 ){
  
    my ($basename, $parentdir, $ext) = fileparse($batchScript, qr/\.[^.]*$/);
    $test_batch = "batch_$test_title$ext";
    system(" cp $batchScript $test_batch" );

    my @nodes = $tst_dom->findnodes('/start/batchScheduler/batchReplace');
    modify_batchScript( $test_batch, @nodes );
    modify_batchScript( $test_batch, $test_node->findnodes('batchReplace') );
  }

  #__________________________________
  # print meta data and run sus command
  my $sus_cmd_0 = $test_node->findnodes('sus_cmd');

  print $statsFile "Test Name :     "."$test_title"."\n";
  print $statsFile "(ups) :         "."$test_ups"."\n";
  print $statsFile "(uda) :         "."$uda"."\n";
  print $statsFile "output:         "."$test_output"."\n";
  print $statsFile "Command Used :  "."$sus_cmd_0 $test_ups"."\n";

  my $now = time();
  my @sus_cmd = ("$sus_cmd_0 ","$test_ups ","> $test_output 2>&1");

  my $rc = 0;
  if( length $batchScript > 0 ){
    submitBatchScript( 0, $test_title, $batchCmd, $test_batch, $statsFile, @sus_cmd );
  }else{
    $rc = runSusCmd( $timeout, $exitOnCrash, $statsFile, @sus_cmd );
  }

  my $fin = time()-$now;
  printf $statsFile ("Running Time :  %.3f [secs]\n", $fin);

  #__________________________________
  #  execute post process command
  my $postProc_cmd = undef;
  $postProc_cmd = $test_node->findvalue('postProcess_cmd');

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


