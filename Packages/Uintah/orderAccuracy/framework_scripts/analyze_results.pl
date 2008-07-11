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
#     - run the comparisonUtility on the uda
#     - concatenate the results to a master L2norm.dat
#   end Loop
#
#   - If there is a gnuplot script then plot the L2norm.dat
#
#  Perl Dependencies:  
#    libxml-simple-perl
#    libxml-dumper-perl
#______________________________________________________________________

use XML::Simple;
use Data::Dumper;
$xml = new XML::Simple(forcearray => 1);
$tstFile = $ARGV[0];
$testNum = $ARGV[1];

# read tst file
$data = $xml->XMLin("$tstFile");

#__________________________________
# find the basename of the ups file
$upsFile         =$data->{upsFile}->[0];
my $ups_basename = $upsFile;
$ups_basename    =~ s/.ups//;                               # Removing the extension .ups so that we can use this to build our uda file names

#__________________________________
# load details of each test into arrays
foreach $e (@{$data->{Test}}){
  $test_title[$i]     =$e->{Title}->[0];                    # test title
  $sus_cmd[$i]        =$e->{sus_cmd}->[0];                  # sus command
  $study[$i]          =$e->{Study}->[0];                    #Study Name
  $x[$i]              =$e->{x}->[0];
  $compUtil_cmd[$i]   =$e->{compare_cmd}->[0];              #comparison utility command
  $i++;     
}
$num_of_tests=$i -1;


#__________________________________
# define the looping limits
if ( $testNum eq "all"){
  $startLoop = 0;
  $endLoop   = $num_of_tests;
}else{
  $startLoop = $testNum;
  $endLoop   = $testNum;
}

#__________________________________
# main loop
$k = 0;
for ($k = $startLoop; $k<=$endLoop; $k++){
  
  #__________________________________
  # bulletproofing
  my @stripped_cmd = split(/ /,$compUtil_cmd[$k]);  # remove command options
  system("which $stripped_cmd[0] >& /dev/null")==0 or die("ERROR(analyze_Analysis.pl): \tThe comparison script/command ($compUtil_cmd) could not be found\n");

  #__________________________________
  # Run the comparison tests
  my $comp_output = "out.$x.cmp";
  $uda  = $ups_basename."_$test_title[$k]".".uda";
  
  print "\nLaunching:  $compUtil_cmd[$k] -o $comp_output -uda $uda\n\n";
  `rm -f $comp_output`;  
  
  @args = ("$compUtil_cmd[$k]","-o","$comp_output","-uda","$uda");
  system("@args")==0 or die("ERROR(analyze_Analysis.pl): @args failed: $?");
  
  # concatenate results
  $L2norm = `cat $comp_output`;
  chomp($L2norm);
  `echo $x[$k] $L2norm >> L2norm.dat`;
  system("rm $comp_output");
}

#______________________________________________________________________
# Plot results gnuplot script
$gnuplotFile = $data->{gnuplotFile}[0];        # if a user provides a gnuplot file


if ( $gnuplotFile ne "") {
  print "Now plotting Analysis with $gnuplotFile \n";
  `gnuplot $gnuplotFile >&gp.out`;
}

