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
#  run_tests_restart.pl
#  Perl script used to modify an input file and run the tests listed in
#  the tst file.
#
# Algorithm
#   - define the input.xml and timestep.xml files
#   - define the checkpoint index based on <checkpoint_timestep>
#   - read in the index.xml and timestep.xml files into Libxml DOM document array
#   - read in the replacement patterns for each test and all tests
#   - perform global replacements on the input.xml and timestep.xml file
#
#   Loop over tests
#     change the output uda name
#     replace lines in ups file
#     run the test
#     if( batchScript)
#       - modify the batchscript file
#       - submit batch script file
#
#     if(post Process cmd )
#       -run analyze_results.pl <tst file> < test number>
#     endif
#   end Loop
#
#  Useful LibXML websites:
#  https://metacpan.org/pod/XML::LibXML::NodeList
#  https://metacpan.org/pod/XML::LibXML
#  https://metacpan.org/dist/XML-LibXML/view/lib/XML/LibXML/Node.pod
#  https://metacpan.org/pod/XML::LibXML::Document
#  xpath sandbox
#  https://grantm.github.io/perl-libxml-by-example/_static/xpath-sandbox/xpath-sandbox.html
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
use Utilities qw( cleanStr setPath modify_xml_file modify_xml_files modify_batchScript read_file write_file runSusCmd submitBatchScript print_XML_ElementTree );

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
print "  Exit on crash or timeout ($exitOnCrash)\n";


#__________________________________
# set sus timeout value    OPTIONAL
my $timeout = 24*60*60;
if( $tst_dom->exists( '/start/susTimeout_minutes' ) ){
  $timeout = cleanStr( $tst_dom->findvalue( '/start/susTimeout_minutes' ) );
}
print "  Simulation timeout: $timeout minutes\n";


#__________________________________
# determine the restart.uda/input.xml basename
my $restartUda      = cleanStr( $tst_dom->findvalue('/start/restart_uda/uda') );
$restartUda         = basename( $restartUda );
my $input_xml       = $restartUda."/input.xml";
my $input_xml_mod0  = $restartUda."/input.xml.mod-0";

#__________________________________
#  timestep.xml
my $timestep          = cleanStr( $tst_dom->findvalue('/start/restart_uda/checkpoint_timestep') );
my $timestep_xml      = $restartUda."/checkpoints/".$timestep."/timestep.xml";
my $timestep_xml_mod0 = $restartUda."/checkpoints/".$timestep."/timestep.xml.mod-0";

#  bulletproofing
if ( ! -e $input_xml || ! -e $timestep_xml){
  print "\n\nERROR(run_tests_restart.pl): $input_xml, or $timestep_xml files Not Found";
  print " Now exiting\n";
  exit
}


#__________________________________
# determing the checkpoint index
my $ckPt_index = 0;
my $ckPt_index_xml     = $restartUda."/checkpoints/index.xml";
my $ckPt_index_xml_dom = XML::LibXML->load_xml(location => $ckPt_index_xml, , no_blanks => 1);

foreach my $timestep_node ($ckPt_index_xml_dom->findnodes('/Uintah_DataArchive/timesteps/timestep')) {
  my $cmp = $timestep."/timestep.xml";
  if(  $timestep_node->getAttribute('href') eq $cmp ){
    last;          # stop looking and exit the foreach loop
  }
  $ckPt_index += 1;
}


#__________________________________
# read XML files into a dom trees and put them into an array

my $input_xml_dom    = XML::LibXML->load_xml(location => $input_xml,    , no_blanks => 1);
my $timestep_xml_dom = XML::LibXML->load_xml(location => $timestep_xml, , no_blanks => 1);

my @editFiles     = ($input_xml, $timestep_xml);
my @editFiles_dom = ($input_xml_dom, $timestep_xml_dom );

#__________________________________
# Globally, replace lines & values in the input.xml & timestep_xml files before looping over tests.
print "--------------------------------------------------";
print "  Replacing xml lines and values in input.xml and timestep.xml files\n";

my @allTests_node = $tst_dom->findnodes('/start/AllTests');

modify_xml_files( \@editFiles, \@editFiles_dom, @allTests_node );  # passing two references

system(" cp $input_xml    $input_xml_mod0" );
system(" cp $timestep_xml $timestep_xml_mod0" );

#______________________________________________________________________
#     loop over tests
#__________________________________

my $statsFile;
open( $statsFile,">out.stat");

my $nTest = 0;
foreach my $test_node ($tst_dom->findnodes('/start/Test')) {

  my $test_title        = cleanStr( $test_node->findvalue('Title') );
  my $test_input_xml    = $restartUda."/".$test_title."_input.xml";
  my $test_timestep_xml = $restartUda."/checkpoints/".$timestep."/".$test_title."timestep.xml";
  my $test_output       = "out.".$test_title;
  my $test_diff         = "diff.".$test_title;
  my $uda               = $test_title.".uda";

  #__________________________________
  # change the uda filename in each ups file
  print "\n-------------------------------------------------- TEST: $test_title\n";
  print "   Now modifying $input_xml and $timestep_xml\n";

  my $fn = "<filebase>".$uda."</filebase>";
  system("replace_XML_line", "$fn", "$input_xml")==0 ||  die("Error replace_XML_line $fn in file $input_xml \n $@");
  print "\treplace_XML_line $fn $input_xml\n";

  modify_xml_files( \@editFiles, \@editFiles_dom, $test_node );  # passing two references


  #__________________________________
  #  create a diff file per test.
  #  This diff will be applied in the batch script
  #  Jobs can start out of order.
  system(" diff -u     $input_xml_mod0   $input_xml    >> $test_diff");
  system(" diff -u  $timestep_xml_mod0   $timestep_xml >> $test_diff");

  #__________________________________
  #  replace any batch script values per test
  my $test_batch = undef;

  if( length $batchScript > 0 ){

    my ($basename, $parentdir, $ext) = fileparse($batchScript, qr/\.[^.]*$/);
    $test_batch = "batch_$test_title$ext";
    system(" cp $batchScript $test_batch" );

    my @batch_nodes = $tst_dom->findnodes('/start/batchScheduler/batchReplace');
    modify_batchScript( $test_batch, @batch_nodes );
    modify_batchScript( $test_batch, $test_node->findnodes('batchReplace') );
  }

  #__________________________________
  # print meta data and run sus command
  my $sus_cmd_0 = $test_node->findnodes('sus_cmd');

  $sus_cmd_0 =~ s/-restart//;             # remove any restart spec
  $sus_cmd_0 =~ s/-t [0-9] //;

  my @sus_cmd = ("$sus_cmd_0 ","-restart -t", $ckPt_index," ",$restartUda,"> $test_output 2>&1");

  print $statsFile "Test Name :       "."$test_title"."\n";
  print $statsFile "(input.xml) :     "."$input_xml"."\n";
  print $statsFile "(timestep.xml) :  "."$timestep_xml"."\n";
  print $statsFile "(timestep) :      "."$timestep"."\n";
  print $statsFile "(uda) :           "."$uda"."\n";
  print $statsFile "(diff file)       "."$test_diff"."\n";
  print $statsFile "checkpoint index  "."$ckPt_index"."\n";
  print $statsFile "output:           "."$test_output"."\n";
  print $statsFile "Command Used :    "."@sus_cmd"."\n";

  my $now = time();

  my $rc = 0;
  if( length $batchScript > 0 ){
    submitBatchScript( 1, $test_title, $batchCmd, $test_batch, $statsFile, @sus_cmd );
  }else{
    $rc = runSusCmd( $timeout, $exitOnCrash, $statsFile, @sus_cmd );
  }

  # reverse the changes to input.xml and timestep.xml
  system(" cp $input_xml_mod0    $input_xml");
  system(" cp $timestep_xml_mod0 $timestep_xml");

  my $fin = time()-$now;
  printf $statsFile ("Running Time :  %.3f [secs]\n", $fin);

  #__________________________________
  #  execute post process command   OPTIONAL
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

