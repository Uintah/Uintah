#!/usr/bin/perl -w
use strict;
use XML::Simple;
use Data::Dumper;
use Cwd;

my $simple = XML::Simple->new(ForceArray=>1);

# Define the paths
my $base_path          = $ARGV[0];
my $config_files_path  = $base_path . "/test_config_files";  # configurations files
my $scripts_path       = $base_path . "/framework_scripts";  # framework scripts 

print " $base_path \n $config_files_path \n $scripts_path \n";


# create base directory
system("/bin/rm -rf order_of_accuracy");
mkdir("order_of_accuracy") || die "cannot mkdir(order_of_accuracy) $!";
chdir("order_of_accuracy");
my $curr_path = cwd;


# read in components.xml
my $xml = $simple->XMLin($config_files_path . "/components.xml");
my @components = @{$xml->{component}};
my $sus_path   = $xml->{sus_path}[0];

# add sus_path and framework_scripts to the path
my $orgPath = $ENV{"PATH"};
$ENV{"PATH"} = "$sus_path:$scripts_path:$orgPath";

system("which sus") == 0 ||  die("Cannot find the command sus $@");

#__________________________________
# loop over each component 
 my $c=0;
 for($c = 0; $c<=$#components; $c++){

   my $component = $components[$c];
   mkdir($component) || die "cannot mkdir($component) $!";
   chdir($component);
   print "----------------------------------------------  $component \n";
         
   my $fw_path = $config_files_path."/".$component;  # path to framework config files
  
   # read whatToRun.xml file into data array
   my $whatToRun = $simple->XMLin($fw_path."/whatToRun.xml");
   
   # add the comparison utilities path to PATH
   my $p   = $whatToRun->{compareUtil_path}[0];
   my $orgPath = $ENV{"PATH"};
   $ENV{"PATH"} = "$p:$orgPath";
 
   #__________________________________
   # loop over all tests
   #   - make test directories
   #   - copy config_files_path_pathig & input files
   my @tests = @{$whatToRun->{test}};
   my $i=0;
   for($i = 0; $i<=$#tests; $i++){
     my $test     = $tests[$i];
     my $testName = $test->{name}[0];
     my $upsFile  = $test->{ups}[0];
     my $tstFile  = $test->{tst}[0];
    
     mkdir($testName) || die "cannot mkdir($testName) $!";
     chdir($testName);
     
     print "Test Name: $testName, ups File : $upsFile, tst File: $tstFile\n";
     
     # bulletproofing
     # do these files exist
     if (! -e $fw_path."/".$upsFile || 
         ! -e $fw_path."/".$tstFile ){
       print "\n \nERROR:setupFrameWork:\n";
       print "The ups file: \n \t $fw_path."/".upsFile \n"; 
       print "or the tst file: \n \t $fw_path."/".$tstFile \n";
       print "doesn't exist.  Now exiting\n";
       exit
     }
     
     # copy the config files to the testing directory
     my $testing_path = $curr_path."/".$component."/".$testName;
     chdir($fw_path);
     system("cp -f $upsFile $tstFile $testing_path");
      
     my $tst = $simple->XMLin($fw_path."/dx.tst");
     my $gnuplotScript = $tst->{gnuplotFile}->[0];
     system("cp -f $gnuplotScript $testing_path");
      
     print "$testing_path \n";
     
     chdir($testing_path);
     
     # make a symbolic link to sus
     my $sus = `which sus`;
     system("ln -s $sus");
     
     #__________________________________
     # run the tests
     print "\n\n Launching run_tests.pl $testing_path/$tstFile\n\n";
     my @args = (" $scripts_path/run_tests.pl","$testing_path/$tstFile");
     system("@args")==0  or die("ERROR(masterScript.pl): \tFailed running: (@args) \n");
     
     chdir("..");
   }
   chdir("..");
 }
  # END
