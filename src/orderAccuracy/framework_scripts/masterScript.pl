#!/usr/bin/env perl

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
#    libxml-dumper-perl
#    xmlstarlet
#
#______________________________________________________________________
use strict;
use warnings;
use XML::LibXML;
use Data::Dumper;
use File::Path;
use File::Basename;

use Cwd;
use lib dirname (__FILE__) ."/framework_scripts";    # needed to find local Utilities.pm
use Utilities qw( cleanStr print_XML_ElementTree get_XML_value);

#__________________________________
# bulletproofing
my @modules = qw( Data::Dumper File::Path File::Basename );

for(@modules) {
  eval "use $_";
  if ($@) {
    print "\n\nError: Could not find the perl module ($_)\n";
    print " Now exiting\n";
    exit
  }
}
#______________________________________________________________________


# Find the path to orderAccuracy scripts and prune /framework_scripts off
# the end if snecessary

my $OA_path = `dirname $0 |xargs readlink -f --no-newline`;

# Define the paths
my $src_path              = dirname($OA_path);                # top level src path
my $config_files_path     = $OA_path . "/test_config_files";  # configurations files
my $scripts_path          = $OA_path . "/framework_scripts";  # framework scripts
my $postProcessCmd_path   = $OA_path . "/postProcessTools";   # postProcessing
my $here_path             = cwd;

if (! -e $OA_path."/framework_scripts" ){
  print "\n\nError: You must specify the path to the orderAccuracy directory ($OA_path)\n";
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

# read XML file into a dom tree and parse
my $filename    = $config_files_path . "/components.xml";
my $dom         = XML::LibXML->load_xml(location => $filename , no_blanks => 1);
my $xmlElements = $dom->documentElement;

my @components        = get_XML_value( $xmlElements, 'component' );
my $sus_path          = get_XML_value( $xmlElements, 'sus_path' );
my $extraScripts_path = get_XML_value( $xmlElements, 'scripts_path' );

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

#__________________________________
# loop over each component

 foreach my $compNode ( $xmlElements->findnodes('component') ) {
   chdir($curr_path);

   my $component = cleanStr( $compNode->textContent() );

   if ( ! -e $component) {
    mkpath($component) || die "cannot mkpath($component) $!";
   }
   chdir($component);
   print "----------------------------------------------------------------  $component \n";

   my $fw_path = $config_files_path."/".$component;  # path to component config files

   # read whatToRun.xml file into xml tree and parse
   my $dom         = XML::LibXML->load_xml(location => $fw_path."/whatToRun.xml" , no_blanks => 1);
   my $whatToRun   = $dom->documentElement;

   # add the comparison utilities path to PATH
   my $p        = cleanStr( $whatToRun->findvalue('postProcessCmd_path') );
   my $orgPath  = $ENV{"PATH"};
   $ENV{"PATH"} = "$p:$orgPath";

   # additional symbolic links to make OPTIONAL
   my @symLinks = ();

   if( $whatToRun->exists( 'symbolicLinks' ) ){
     my $sl     = $whatToRun->findvalue('symbolicLinks');
     @symLinks  = split(/ /,$sl);
     @symLinks  = cleanStr(@symLinks);
   }

   #__________________________________
   # loop over all tests
   #   - make test directories
   #   - copy config_files_path_pathig & input files
   my $otherFiles = "";

   foreach my $test ( $whatToRun->findnodes('test') ) {

     my $testName = cleanStr( $test->findvalue('name') );

     my $tstFile  = cleanStr( $test->findvalue('tst') );
     $tstFile     = $fw_path."/".$tstFile;
     my $tst_basename = basename( $tstFile );

     my $dom      = XML::LibXML->load_xml(location => "$tstFile" , no_blanks => 1);
     my $tstData  = $dom->documentElement;
 
                    # Inputs directory default path (src/StandAlone/inputs)
     my $default_path = $src_path . "/StandAlone/inputs/";
     my $inputs_path = get_XML_value( $tstData, 'inputs_path', $default_path );
    
                   # UPS file
     my $ups_tmp  = cleanStr( $tstData->findvalue('upsFile') );
     my $upsFile  = $inputs_path.$component."/".$ups_tmp;

                   # if it's not in the std location look for it
     if ( ! -e $ups_tmp ){
       $upsFile = `find $inputs_path  |grep --word-regexp $ups_tmp`;
       chomp( $upsFile );
     }

                   # Other files needed
     if($test->exists('otherFilesToCopy') ){
       $otherFiles= cleanStr( $test->findvalue('otherFilesToCopy') );
      }

                    # find a unique testname
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
     if (! -e $upsFile ||
         ! -e $tstFile ||
         ! -e $fw_path."/".$otherFiles ){
       print "\n \nERROR:setupFrameWork:\n";
       print "The ups file: \n       \t ($upsFile) \n";
       print "or the tst file: \n     \t ($tstFile)\n";
       print "or the other file(s) \n \t ($fw_path/$otherFiles) \n";
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

     # create any symbolic links requested by that component
     my $j = 0;
     foreach $j (@symLinks) {
       if( -e $j ){
         system("ln -s $j > /dev/null 2>&1");
       }
     }

     # Bulletproofing
     print "\t Checking that the tst file is a properly formatted xml file  \n";
     system("xmlstarlet val --err $tst_basename") == 0 ||  die("\nERROR: $tst_basename, contains errors.\n");


     # clean out any comment in the TST file
     system("xmlstarlet c14n --without-comments $tst_basename > $tst_basename.clean 2>&1");
     $tst_basename = "$tst_basename.clean";


     #__________________________________
     # run the tests
     print "\n\nLaunching: run_tests.pl $testing_path/$tst_basename\n\n";
     my @args = (" $scripts_path/run_tests.pl","$testing_path/$tst_basename", "$fw_path");
     system("@args")==0  or die("ERROR(masterScript.pl): \tFailed running: (@args) \n\n");

     chdir("..");
   }  # loop over tests
   
   chdir("..");
 }

  # END
