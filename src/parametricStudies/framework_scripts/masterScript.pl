#!/usr/bin/env perl
#
# The MIT License
#
# Copyright (c) 1997-2025 The University of Utah
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
#  MasterScript.pl:
##
#  Perl script that controls the parametric study framework scripts
#  This script reads a master xml configuration file <components.xml> and
#  for each uintah component listed and runs the studies
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
#use diagnostics;
use XML::LibXML;
use Data::Dumper;
use File::Path;
use File::Basename;
use File::Spec;

use Cwd;
use lib dirname (__FILE__) ."/framework_scripts";    # needed to find local Utilities.pm
use Utilities qw( cleanStr setPath print_XML_ElementTree get_XML_value);

#__________________________________
# bulletproofing
my @modules = qw( Data::Dumper File::Path File::Basename File::Spec);

for(@modules) {
  eval "use $_";
  if ($@) {
    print "\n\nError: Could not find the perl module ($_)\n";
    print " Now exiting\n";
    exit
  }
}
#______________________________________________________________________


# Find the parametricStudies path and prune /framework_scripts off
# the end if snecessary

my $PS_path = `dirname $0 |xargs readlink -f --no-newline`;

# Define the paths
my $src_path              = dirname($PS_path);                # top level src path
my $config_files_path     = $PS_path . "/test_config_files";  # configurations files
my $scripts_path          = $PS_path . "/framework_scripts";  # framework scripts
my $postProcessCmd_path   = $PS_path . "/postProcessTools";   # postProcessing
my $here_path             = cwd;

if (! -e $PS_path."/framework_scripts" ){
  print "\n\nError: You must specify the path to the parametricStudies directory ($PS_path)\n";
  print " Now exiting\n";
  exit
}

#__________________________________
# create the base testing directory
if (! -e "ps_results" ){
  system("/bin/rm -rf ps_results");
  mkdir("ps_results") || die "cannot mkdir(ps_results) $!";
}
chdir("ps_results");
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
system("which sus") == 0               || die("\nERROR: Cannot find the command sus.  You may want to set <sus_path> in components.xml, or run in Uintah:StandAlone dir $@");
#system("which octave")  == 0           || die("\nCannot find the command octave.  You may want to comment this out if you're not using octave $@");
#system("which gnuplot") == 0           || die("\nCannot find the command gnuplot.  You may want to comment this out if you're not using octave  $@");
system("which mpirun")  == 0           || die("\nERROR: Cannot find the command mpirun $@");
system("which xmlstarlet")  == 0       || die("\nERROR: Cannot find the command xmlstarlet $@");
system("which replace_XML_line")  == 0 || die("\nERROR: Cannot find the command replace_XML_line $@");
system("which replace_XML_value") == 0 || die("\nERROR: Cannot find the command replace_XML_value $@");

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
   my @symLinks;

   if( $whatToRun->exists( 'symbolicLinks' ) ){
     my $sl     = $whatToRun->findvalue('symbolicLinks');

     @symLinks  = split(/ /,$sl);
     @symLinks  = cleanStr(@symLinks);
   }

   #__________________________________
   # loop over all tests
   #   - make test directories
   #   - copy tst, scripts, other files & input files
   my $otherFiles = "";

   
   foreach my $test ( $whatToRun->findnodes('test') ) {

     my $testName = cleanStr( $test->findvalue('name') );

     # tst file can live outside of uintah src tree
     # The postProcessing cmd may be in the same dir as the tst
     # so add that path to PATH
     my $tstFile  = cleanStr( $test->findvalue('tst') );
     $tstFile     = setPath( $tstFile, $fw_path );

     my($vol,$tstPath,$file) = File::Spec->splitpath($tstFile);
     my $orgPath  = $ENV{"PATH"};
     $ENV{"PATH"} = "$orgPath:$tstPath";
     
     my $tst_basename = basename( $tstFile );

     my $dom      = XML::LibXML->load_xml(location => "$tstFile" , no_blanks => 1);
     my $tstData  = $dom->documentElement;

                    # Inputs directory default path (src/StandAlone/inputs)
     my $default_path = $src_path . "/StandAlone/inputs/";
     my $inputs_path = get_XML_value( $tstData, 'inputs_path', $default_path );


                   # UPS file
     my $ups_tmp  = cleanStr( $tstData->findvalue('upsFile') );
     my $upsFile  = setPath( $ups_tmp, $tstPath, $fw_path, $inputs_path.$component );


                   # gnuplot file
     my $gp_tmp  = cleanStr( $tstData->findvalue('/start/gnuplot/script') );
     my $gpFile  = setPath( $gp_tmp, $tstPath, $fw_path, $inputs_path.$component );
     
     
                   # restarts
     my $doRestart  = cleanStr( $tstData->exists('/start/restart_uda') );
     my $restartUda = cleanStr( $tstData->findvalue('/start/restart_uda/uda') );
     
     if( $doRestart ){
       $doRestart = 1;
       $upsFile   = '';
     }
                   # Other files needed.  This could contain wildcards
     if($test->exists('otherFilesToCopy') ){
       $otherFiles = cleanStr( $test->findvalue('otherFilesToCopy') );
       $otherFiles = setPath( $otherFiles, $fw_path, $inputs_path.$component ) ;
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
     unlink( $testNameOld );
     symlink( $testName, $testNameOld  ) || die "ERROR:masterScript.pl:cannot create symlink $!";

     chdir($testName);

     #__________________________________
     # bulletproofing
     # do these files exist
     if ( $doRestart == 0 && (! -e $upsFile || ! -e $tstFile ) ){
       print "\n \nERROR:setupFrameWork:\n";
       print "The ups file: \n        ($upsFile) \n";
       print "or the tst file: \n     ($tstFile)\n";
       print "do not exist.  Now exiting\n";
       exit
     }     
     
     if ( $doRestart == 1 && (! -e $restartUda || ! -e $tstFile ) ){
       print "\n \nERROR:setupFrameWork:\n";
       print "The restart uda: \n     ($restartUda) \n";
       print "or the tst file: \n     ($tstFile)\n";
       print "do not exist.  Now exiting\n";
       exit
     }

     #__________________________________
     # copy the config files to the testing directory
     my $testing_path = $curr_path."/".$component."/".$testName;
     chdir($fw_path);
     
     if( $doRestart ){
       system("rsync -ap --include=checkpoints/** --exclude='t[0-9]*'  $restartUda $testing_path");
     } 
     else {
       system("cp -f $upsFile $testing_path");
     }
     
     system("cp -f  $tstFile $testing_path");
     
     system("cp -f  $gpFile $testing_path");
     
     system("cp -rf $otherFiles $testing_path > /dev/null 2>&1");
     
     system("echo '$here_path:$postProcessCmd_path'> $testing_path/scriptPath 2>&1");

     chdir($testing_path);

     #__________________________________
     # make a symbolic link to sus
     my $sus = `which sus`;
     system("ln -s $sus > /dev/null 2>&1");

     # make a symbolic link to inputs
     system("ln -s $inputs_path > /dev/null 2>&1");

     # create any symbolic links requested by that component
     foreach my $s (@symLinks) {
       if( defined $s ){
         print " creating symbolic link: $s \n";
         system("ln -s $s> /dev/null 2>&1");
       }
     }


     print "\n\n===================================================================================\n";
     print "Test Name      : $testName \n";
     print "ups File       : $upsFile \n";
     print "restartUda     : $restartUda \n";
     print "tst File       : $tstFile \n";
     print "inputs dir     : $inputs_path\n";
     print "sus            : $sus";
     print "other Files    : $otherFiles\n";
     print "gnuplot File   : $gpFile\n";
     print "results path   : $testing_path\n";
     print "=======================================================================================\n";

     # Bulletproofing
     print "Checking that the tst file is a properly formatted xml file  \n";
     system("xmlstarlet val --err $tst_basename") == 0 ||  die("\nERROR: $tst_basename, contains errors.\n");


     # clean out any comment in the TST file
     system("xmlstarlet c14n --without-comments $tst_basename > $tst_basename.clean 2>&1");
     $tst_basename = "$tst_basename.clean";


     #__________________________________
     # run the tests
     if( $doRestart) {                 # restarting
       print "\n\nLaunching: run_tests_restart.pl $tst_basename\n\n";
       my @args = (" $scripts_path/run_tests_restart.pl","$testing_path/$tst_basename", "$fw_path");
       system("@args")==0  or die("ERROR(masterScript.pl): \tFailed running: (@args) \n\n");
     }
     else{
     
       print "\n\nLaunching: run_tests.pl $tst_basename\n\n";
       my @args = (" $scripts_path/run_tests.pl","$testing_path/$tst_basename", "$fw_path");
       system("@args")==0  or die("ERROR(masterScript.pl): \tFailed running: (@args) \n\n");
     }
     
     chdir("..");
   }  # loop over tests

   chdir("..");
 }

  # END
