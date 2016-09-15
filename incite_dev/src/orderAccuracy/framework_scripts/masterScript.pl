#!/usr/bin/perl -w

#______________________________________________________________________
#  MasterScript.pl:
#
#  input:  path to orderAccuracy directory
#
#  Perl script that controls the order of analysis framework scripts
#  This script reads a master xml configuration file <components.xml> and 
#  for each uintah component listed and runs the order of accuracy tests
#  listed in test_config_files/component/whatToRun.xml file.
#
#  Each OA test should have a corresponding comparison program that returns the L2nom
#
#
#  Algorithm:
#  - create the output directory
#  - read in the configuration file components.xml (contains a list of components to test)
#  - set the path so subsequent command calls work.
#
#  Loop over each Uintah component
#    - create a results directory for that component
#    - read in "whatToRun.xml" (list of tests to run)
#    - add post processing utilities path to PATH
#
#    Loop over each Uintah component test
#      - create a results directory
#      - copy config files and sus to that directory
#      - run the test
#    end loop
#  end loop
#
#  Perl Dependencies:  
#    libxml-simple-perl
#    libxml-dumper-perl
#    xmlstarlet
#
#______________________________________________________________________

use strict;
use XML::Simple;
use Data::Dumper;
use File::Path;
use Cwd;

#__________________________________
# bulletproofing
my @modules = qw(XML::Simple Data::Dumper File::Path );

for(@modules) {
  eval "use $_";
  if ($@) {
    print "\n\nError: Could not find the perl module ($_)\n";
    print " Now exiting\n";
    exit
  }
}
#______________________________________________________________________


my $simple = XML::Simple->new(ForceArray=>1, suppressempty => "");


# Find the path to orderAccuracy scripts and prune /framework_scripts off 
# the end if snecessary
my $me = `dirname $0 |xargs readlink -f --no-newline`;
$me =~ s/\/framework_scripts//;

# Define the paths
my $base_path             = $me;    # path to orderAccuracy scripts
my $config_files_path     = $base_path . "/test_config_files";  # configurations files
my $scripts_path          = $base_path . "/framework_scripts";  # framework scripts
my $postProcessCmd_path   = $base_path . "/postProcessTools";   # postProcessing 
my $here_path             = cwd;

if (! -e $base_path."/framework_scripts" ){
  print "\n\nError: You must specify the path to the orderAccuracy directory ($base_path)\n";
  print " Now exiting\n";
  exit
}

#__________________________________
# create the base testing directory
if (! -e "order_of_accuracy" ){
  system("/bin/rm -rf order_of_accuracy");
  mkdir("order_of_accuracy") || die "cannot mkdir(order_of_accuracy) $!";
}
chdir("order_of_accuracy");
my $curr_path = cwd;

#__________________________________
# read in components.xml
if (! -e $config_files_path . "/components.xml" ){
  print "\n\nError: Could not find $config_files_path/components.xml\n";
  print " Now exiting\n";
  exit
}

my $xml = $simple->XMLin($config_files_path . "/components.xml");
my @components   = @{$xml->{component}};
my $sus_path     = $xml->{sus_path}[0];
my $extraScripts_path = $xml->{scripts_path}[0];


#__________________________________
# add compare_path:sus_path and framework_scripts to the path
my $orgPath = $ENV{"PATH"};
#my $syspath ="/usr/bin/:/usr/sbin:/bin";

$ENV{"PATH"} = "$orgPath:$postProcessCmd_path:$sus_path:$scripts_path:$extraScripts_path:$here_path:.";

# bulletproofing
print "----------------------   \n";
print "Using the following commands:\n";
system("which sus") == 0               || die("\nCannot find the command sus $@");
#system("which octave")  == 0           || die("\nCannot find the command octave.  You may want to comment this out if you're not using octave $@");
#system("which gnuplot") == 0           || die("\nCannot find the command gnuplot.  You may want to comment this out if you're not using octave  $@");
system("which mpirun")  == 0           || die("\nCannot find the command mpirun $@");
system("which xmlstarlet")  == 0       || die("\nCannot find the command xmlstarlet $@");
system("which replace_XML_line")  == 0 || die("\nCannot find the command replace_XML_line $@");
system("which replace_XML_value") == 0 || die("\nCannot find the command replace_XML_value $@");
system("which findReplace")       == 0 || die("\nCannot find the command findReplace $@");

#__________________________________
# loop over each component 
 my $c=0;
 for($c = 0; $c<=$#components; $c++){
   chdir($curr_path);
   
   my $component = $components[$c];
   
   if ( ! -e $component) {
    mkpath($component) || die "cannot mkpath($component) $!";
   }
   chdir($component);
   print "----------------------------------------------------------------  $component \n";
         
   my $fw_path = $config_files_path."/".$component;  # path to component config files
  
   # read whatToRun.xml file into data array
   my $whatToRun = $simple->XMLin($fw_path."/whatToRun.xml");
   
   # add the comparison utilities path to PATH
   my $p   = $whatToRun->{postProcessCmd_path}[0];
   my $orgPath = $ENV{"PATH"};
   $ENV{"PATH"} = "$p:$orgPath";
   
   # additional symbolic links to make
   my $input = $whatToRun->{symbolicLinks}[0];
   my @symLinks = split(/ /,$input);
  
   
   #__________________________________
   # loop over all tests
   #   - make test directories
   #   - copy config_files_path_pathig & input files
   my @tests = @{$whatToRun->{test}};
   my $i=0;
   my $otherFiles = "";
   
   for($i = 0; $i<=$#tests; $i++){
     my $test     = $tests[$i];
     my $testName = $test->{name}[0];
     my $upsFile  = $test->{ups}[0];
     my $tstFile  = $test->{tst}[0];
     
     #remove newline from variable if they exist
     chomp($upsFile);
     chomp($tstFile);
     
     
     if($test->{otherFilesToCopy}[0] ){ 
        $otherFiles= $test->{otherFilesToCopy}[0];
        chomp($otherFiles);
     }
    
     #find a unique testname
     my $count = 0;
     my $testNameOld = $testName;
     $testName = "$testName.$count";
     
     while( -e $testName){
       $testName = "$testNameOld.$count";
       $count +=1;
     }
     
     mkpath($testName) || die "ERROR:masterScript.pl:cannot mkpath($testName) $!";
     
     
     chdir($testName);
     
     print "\n\n=======================================================================================\n";
     print "Test Name: $testName, ups File : $upsFile, tst File: $tstFile other Files: $otherFiles\n";
     print "=======================================================================================\n";
     # bulletproofing
     # do these files exist
     if (! -e $fw_path."/".$upsFile || 
         ! -e $fw_path."/".$tstFile ||
         ! -e $fw_path."/".$otherFiles ){
       print "\n \nERROR:setupFrameWork:\n";
       print "The ups file: \n \t $fw_path/$upsFile \n"; 
       print "or the tst file: \n \t $fw_path/$tstFile \n";
       print "or the other file(s) \n \t $fw_path/$otherFiles \n";
       print "do not exist.  Now exiting\n";
       exit
     }
     
     # copy the config files to the testing directory
     my $testing_path = $curr_path."/".$component."/".$testName;
     chdir($fw_path);
     system("cp -f $upsFile $tstFile $otherFiles $testing_path");
     
     system("echo $here_path $postProcessCmd_path> $testing_path/scriptPath 2>&1");
          
     chdir($testing_path);

     # make a symbolic link to sus
     my $sus = `which sus`;
     system("ln -s $sus > /dev/null 2>&1");

     # make any symbolic Links needed by that component
     my $j = 0;
     foreach $j (@symLinks) {
       if( $j gt "" && $j ne "."){
         system("ln -s $j > /dev/null 2>&1");
       }
     }
     
     # Bulletproofing
     print "\t Checking that the tst file is a properly formatted xml file  \n";
     system("xmlstarlet val --err $tstFile") == 0 ||  die("\nERROR: $tstFile, contains errors.\n");
     
     # clean out any comment in the TST file
     system("xmlstarlet c14n --without-comments $tstFile > $tstFile.clean 2>&1");
     $tstFile = "$tstFile.clean";
     
     
     #__________________________________
     # run the tests
     print "\n\nLaunching: run_tests.pl $testing_path/$tstFile\n\n";
     my @args = (" $scripts_path/run_tests.pl","$testing_path/$tstFile", "$fw_path");
     system("@args")==0  or die("ERROR(masterScript.pl): \tFailed running: (@args) \n\n");

     chdir("..");
   }
   chdir("..");
 }
  # END
