#!/usr/bin/env perl
#______________________________________________________________________
#  analyze_results.pl
#  Perl script used to call the comparison utility for each test and plot the results
#
# Algorithm
#   - find the basename of the ups file-used to determine the uda name
#   - read in the test information into an array
#   - multiple tests to analyze or one
#
#   Loop over tests
#     - run the postProcess cmd on the uda
#     - concatenate the results to a master L2norm.dat
#   end Loop
#
#   - If there is a gnuplot section is present then execute it
#
#______________________________________________________________________
use strict;
use warnings;
use XML::LibXML;
use Data::Dumper;
use File::Which;
use File::Basename;
use lib dirname (__FILE__);  # needed to find local Utilities.pm
use Utilities;

my $tstFile = $ARGV[0];
my $testNum = $ARGV[1];

# read XML file into a dom tree
my $doc = XML::LibXML->load_xml(location => $tstFile);

#__________________________________
# Find the basename of the ups file
my $upsFile      = Utilities::cleanStr( $doc->findvalue( '/start/upsFile' ) );
my $ups_basename = $upsFile;
$ups_basename    =~ s/.ups//;                               # Removing the extension .ups so that we can use this to build our uda file names

#__________________________________
# Find the test and post processing command to execute
my $nTest        = -1;
my $test_dom     = undef;
my $X            = "";
my $testTitle    = "";
my $postProc_cmd = "";

# loop over all tests xml nodes in tst file
foreach $test_dom ($doc->findnodes('/start/Test')) {
  $nTest +=1;
  if( $nTest != $testNum ){  
    next;
  }
  
  # Pull out the entries from the test dom tree
  $X            = $test_dom->findvalue('x');
  $testTitle    = $test_dom->findvalue('Title');
  $postProc_cmd = $test_dom->findvalue('postProcess_cmd');
}

# remove leading spaces from command
$postProc_cmd=~ s/^\s+//;                     

# change command from a scalar to array for easier parsing
my @cmd_A = ( split(/ /, $postProc_cmd) );

# prune out the command options
my $cmd_trim = $cmd_A[0];

my $cmd = which($cmd_trim);

# command basename
my $cmd_basename = basename( $postProc_cmd );

#__________________________________
# bulletproofing
if ( ! defined $cmd ){
  my $mypath = $ENV{"PATH"};
  print "\n\n__________________________________\n";
  print "ERROR:analyze_results:\n";
  print "The comparison utility: ($cmd_basename)";
  print " doesn't exist in path \n $mypath \n Now exiting\n\n\n";
  die
}

system("ln -fs $cmd > /dev/null 2>&1");

#__________________________________
# Run the comparison tests

my $comp_output = "out.$X.cmp";
my $uda         = $ups_basename."_$testTitle".".uda";

print "\n\tLaunching: ($cmd_basename -o $comp_output -uda $uda)\n\n";
`rm -f $comp_output`;

my @args = ("$cmd_basename", "-o", "$comp_output", "-uda", "$uda");
system("@args")==0 or die("ERROR(analyze_Analysis.pl):\tFailed running: (@args)) failed: $@");

# concatenate results
my $L2norm = `cat $comp_output`;
chomp($L2norm);
`echo $X $L2norm >> L2norm.dat`;
#system("rm $comp_output");
  

#______________________________________________________________________
# Plot results gnuplot script         OPTIONAL
my $gpData = $doc->find( '/start/gnuplot' );

if ( length $gpData ) {
  my $gpFile = $doc->findvalue( '/start/gnuplot/script' );        # if a user provides a gnuplot file

  # modify the plot script
  my $title = $doc->findvalue( '/start/gnuplot/title' );
  my $xlabel= $doc->findvalue( '/start/gnuplot/xlabel' );
  my $ylabel= $doc->findvalue( '/start/gnuplot/ylabel' );

  system("sed", "-i", "s/#title/set title \"$title\"/g", "$gpFile");
  system("sed", "-i", "s/#xlabel/set xlabel \"$xlabel\"/g", "$gpFile");
  system("sed", "-i", "s/#ylabel/set ylabel \"$ylabel\"/g", "$gpFile");

  print "Now plotting using the modified gnuplot script ($gpFile) \n";
  system("gnuplot $gpFile > gp.out 2>&1");
}

