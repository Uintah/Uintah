#!/usr/bin/perl -w


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
#  Perl Dependencies:
#    libxml-simple-perl
#    libxml-dumper-perl
#______________________________________________________________________
use strict; 
use warnings;
use XML::LibXML;
use Data::Dumper;
use Time::HiRes qw/time/;
use File::Basename;
use File::Which;
use Cwd;

# removes white spaces from variable
sub  trim { my $s = shift; $s =~ s/^\s+|\s+$//g; return $s };

my $tstFile           = $ARGV[0];
my $config_files_path = $ARGV[1];

# read XML file into a dom tree
my $doc = XML::LibXML->load_xml(location => $tstFile);

#__________________________________
# copy gnuplot script   OPTIONAL
my $gpFile = cleanStr( $doc->findvalue( '/start/gnuplot/script' ) );
if( defined $gpFile ){
  $gpFile    = $config_files_path."/".$gpFile;
  system("cp -f $gpFile .");
  print "  gnuplot script used in postProcess ($gpFile)\n";
}

#__________________________________
# set exitOnCrash flag    OPTIONAL
my $exitOnCrash = "true";
$exitOnCrash = cleanStr( $doc->findvalue( '/start/exitOnCrash' ) );
$exitOnCrash = trim(uc($exitOnCrash));
print "  Exit order of accuracy scripts on crash or timeout ($exitOnCrash)\n";


#__________________________________
# set sus timeout value    OPTIONAL
my $timeout = 24*60*60;
$timeout = cleanStr( $doc->findvalue( '/start/susTimeout_minutes' ) );
print "  Simulation timeout: $timeout seconds\n";


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
  my $udaFilename = $ups_basename."_$test_title".".uda";

  #__________________________________
  # change the uda filename in each ups file
  print "\n--------------------------------------------------\n";
  print "Now modifying $test_ups\n";

  system(" cp $upsFile $test_ups");
  my $fn = "<filebase>".$udaFilename."</filebase>";
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

  #bulletproofing
  system("xmlstarlet val --err $test_ups") == 0 ||  die("\nERROR: $upsFile, contains errors.\n");

  #__________________________________
  # print meta data and run sus command
  my $sus_cmd = $test_dom->findnodes('sus_cmd');

  print $statsFile "Test Name :     "."$test_title"."\n";
  print $statsFile "(ups) :         "."$test_ups"."\n";
  print $statsFile "(uda) :         "."$udaFilename"."\n";
  print $statsFile "output:         "."$test_output"."\n";
  print $statsFile "Command Used :  "."$sus_cmd $test_ups"."\n";

  my $now = time();
  my @args = ("$sus_cmd","$test_ups","> $test_output 2>&1");

  my $rc = runSusCmd( $timeout, $exitOnCrash, $statsFile, @args );

  my $fin = time()-$now;
  printf $statsFile ("Running Time :  %.3f [secs]\n", $fin);

  #__________________________________
  #  execute post process command
  my $postProc_cmd = undef;
  $postProc_cmd = $test_dom->findvalue('postProcess_cmd');

  if( $rc == 0 && length $postProc_cmd != 0){

    print "\nLaunching: analyze_results.pl $tstFile test ($test_title)\n";
    my @cmd = ("analyze_results.pl","$tstFile", "$nTest");
    print $statsFile "postProcessCmd:  "."$postProc_cmd"."\n";

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
  my( $timeout, $exitOnCrash, $statsFile, @args ) = @_;

  print "\tLaunching: (@args)\n";
  my @cmd = (" timeout --preserve-status $timeout @args ");

  my $rc = -9;
  if ( $exitOnCrash eq "TRUE" ) {

    $rc = system("@cmd");

    if ( $rc != 0 && $rc != 36608 ){
      die("ERROR(run_tests.pl): \t\tFailed running: (@args)\n");
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
#
#  Remove any white space or newlines in array elements or scalars
#  (This belongs in a separate common module to avoid duplication -Todd)
sub cleanStr {

  my @inputs = @_;

  my $n   = scalar @inputs;           # number of array elements
  my $len = length $inputs[0];        # number of characters in first element

  # if the first element is empty return ""
  if( $len == 0 ){
    return "";
  }

  #__________________________________
  # if there is one array element return a scalar
  if( $n == 1 ){
    $inputs[0] =~ s/\n//g;        # remove newlines
    $inputs[0] =~ s/ //g;         # remove white spaces
    return $inputs[0];
  }

  #__________________________________
  #  Arrays
  my @result = ();
  my $i = 0;

  foreach $i (@inputs){
    $i =~ s/\n//g;        # remove newlines
    $i =~ s/ //g;         # remove white spaces
    my $l = length $i;

    if ($l > 0){
      push( @result, $i );
    }
  }
  return @result;
};
