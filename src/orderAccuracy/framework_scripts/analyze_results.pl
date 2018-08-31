#!/usr/bin/perl
#______________________________________________________________________
#  analyze_results.pl
#  Perl script used to call the comparison utility to comput the L2norm
#  for each test and plot the results
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
#   - If there is a gnuplot script then plot the L2norm.dat
#
#  Perl Dependencies:  
#    libxml-simple-perl
#    libxml-dumper-perl
#______________________________________________________________________
use strict; 
use warnings;
use XML::Simple;
use Data::Dumper;
use File::Which;
use File::Basename;


my $xml = new XML::Simple(forcearray => 1, suppressempty => "");
my $tstFile = $ARGV[0];
my $testNum = $ARGV[1];

# read tst file
my $data = $xml->XMLin("$tstFile");

#__________________________________
# find the basename of the ups file
my $upsFile      =$data->{upsFile}->[0];
my $ups_basename = $upsFile;
$ups_basename    =~ s/.ups//;                               # Removing the extension .ups so that we can use this to build our uda file names

#__________________________________
# load details of each test into arrays
my @test_title   = ();
my @sus_cmd      = ();
my @x            = ();
my @postProc_cmd = ();

my $i = 0;
foreach my $e (@{$data->{Test}}){
  $test_title[$i]     =$e->{Title}->[0];                    # test title
  $sus_cmd[$i]        =$e->{sus_cmd}->[0];                  # sus command
  $x[$i]              =$e->{x}->[0];
  $postProc_cmd[$i]   =$e->{postProcess_cmd}->[0];          #post processing command
  $i++;     
}
my $num_of_tests=$i -1;


#__________________________________
# define the looping limits
my $startLoop = $testNum;
my $endLoop   = $testNum;
if ( $testNum eq "all"){
  $startLoop = 0;
  $endLoop   = $num_of_tests;
}

#__________________________________
# main loop
my $k = 0;
for ($k = $startLoop; $k<=$endLoop; $k++){
  
  $postProc_cmd[$k]=~ s/^\s+//;                     # remove leading spaces from command
  
  # change command from a scalar to array for easier parsing
  my @cmd_A = ( split(/ /, $postProc_cmd[$k]) );

  # prune out the command options
  my $cmd_trim = @cmd_A[0];
  
  my $cmd = which($cmd_trim);
  
  # command basename
  my $cmd_basename = basename( $postProc_cmd[$k] );
  
  
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
  my $comp_output = "out.$x[$k].cmp";
  my $uda  = $ups_basename."_$test_title[$k]".".uda";
  
  print "\nLaunching:  $cmd_basename -o $comp_output -uda $uda\n\n";
  `rm -f $comp_output`;  
  
  my @args = ("$cmd_basename","-o","$comp_output","-uda","$uda");
  system("@args")==0 or die("ERROR(analyze_Analysis.pl): @args failed: $?");
  
  # concatenate results
  my $L2norm = `cat $comp_output`;
  chomp($L2norm);
  `echo $x[$k] $L2norm >> L2norm.dat`;
  #system("rm $comp_output");
}

#______________________________________________________________________
# Plot results gnuplot script
my $gpData = $data->{gnuplot}[0];
my $gpFile = $gpData->{script}[0];        # if a user provides a gnuplot file


if ( length $gpFile ) {

  # modify the plot script
  my $title = $gpData->{title}[0];
  my $xlabel= $gpData->{xlabel}[0];
  my $ylabel= $gpData->{ylabel}[0];
  
  system("sed", "-i", "s/#title/set title \"$title\"/g", "$gpFile");
  system("sed", "-i", "s/#xlabel/set xlabel \"$xlabel\"/g", "$gpFile");
  system("sed", "-i", "s/#ylabel/set ylabel \"$ylabel\"/g", "$gpFile");
 
  print "Now plotting Analysis with $gpFile \n";
  system("gnuplot $gpFile > gp.out 2>&1");
}

