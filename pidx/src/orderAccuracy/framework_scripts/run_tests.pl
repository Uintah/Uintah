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
#   - perform global findReplace on the ups file
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

use XML::Simple;
use Data::Dumper;
use Cwd;
# create object
$simple = new XML::Simple(forcearray => 1, suppressempty => "");
$tstFile           = $ARGV[0];
$config_files_path = $ARGV[1];

# read XML file
my $data = $simple->XMLin("$tstFile");

#__________________________________
# copy gnuplot script
my $gpFile = $data->{gnuplot}[0]->{script}[0];
chomp($gpFile);

if($gpFile ne ""){                
  $gpFile    = $config_files_path."/".$gpFile;
  system("cp -f $gpFile .");
}

#__________________________________
# determing the ups basename
$upsFile         =$data->{upsFile}->[0];
chomp($upsFile);
my $ups_basename = $upsFile;
$ups_basename    =~ s/.ups//;                     # Removing the extension .ups so that we can use this to build our uda file names

#__________________________________
# Read in the test data from xml file
my $i = 0;
my @tests = @{$data->{Test}};

#print Dumper(@tests);         #debugging
       
for($i = 0; $i<=$#tests; $i++){
  my $test            =$tests[$i];
  $test_title[$i]     =$test->{Title}[0];          # test title
  $sus_cmd[$i]        =$test->{sus_cmd}[0];        # sus command
  $postProc_cmd[$i]   =$test->{postProcess_cmd}[0];    # comparison utility command
  
  #print Dumper($test);         #debugging
}
$num_of_tests=$#tests;

#__________________________________
# make a symbolic link to the post processing command
# Note Debian doesn't have the --skip-dot option
for ($i=0;$i<=$num_of_tests;$i++){
   if( $postProc_cmd[$i] ne ''){
    my @stripped_cmd = split(/ /,$postProc_cmd[$i]);  # remove command options
    my $cmd = `which --skip-dot $stripped_cmd[0] >&/dev/null`;
    system("ln -fs $cmd >&/dev/null");
  }
}

#__________________________________
# Read in all of the replacement patterns 
# and store them in arrays.
#   There can be global replacement lines and individual test replacement lines
my $nTest=-1;
my $line;
my $insideTest=0;
my $insideAllTest=0;
my $insideComment=0;

open(tstFile, "$ARGV[0]") or die("ERROR(run_tests.pl): $ARGV[0], File not found");

while ($line=<tstFile>){
  $blankLine=0;
  
  if($line=~ /\<!--/){
    $insideComment=1;
  }
  if($line=~ /--\>/){
    $insideComment=0;
  }
  if($line=~ /\<AllTests\>/){
    $insideAllTest=1;
  }
  if($line=~ /\<\/AllTests\>/){
    $insideAllTest=0;
  }
  if($line=~ /\<Test\>/){
    $insideTest=1;
    $nTest ++;
  }
  if($line=~ /\<\/Test\>/){
    $insideTest=0;
  }
  if ($line=~ /^\s*$/ ) {
    $blankLine=1;
  }
  
  
  # inside of <AllTests>
  if($insideAllTest && !$insideComment && !$blankLine){
    if ($line=~ /\<replace_lines\>/){       # find <replace_lines>
      $nLine=0;

      while (($line=<tstFile>) !~ /\<\/replace_lines\>/){
        chomp($line);
        
        if ($line !~ /^\s*$/ ) {      # ignore blank lines
          $global_replaceLines[$nLine]=$line;
          $nLine++;
        }
      }
    }
    
    if ($line=~ /\<replace_values\>/){       # find <replace_values>
      $nLine=0;
      while (($line=<tstFile>) !~ /\<\/replace_values\>/){
        chomp($line);
        
        if ($line !~ /^\s*$/ ) {      # ignore blank lines
          $global_replaceValues[$nLine]=$line;
          $nLine++;
        }
      }
    }
  }
  
  # inside each <Test>
  
  if($insideTest && !$insideComment && !$blankLine){
    if ($line=~ /\<replace_lines\>/){       # find <replace_lines>
      $nLine=0;

      while (($line=<tstFile>) !~ /\<\/replace_lines\>/){
        chomp($line);
        $replaceLines[$nTest][$nLine]=$line;
        $nLine++;
        #print "$nTest  $line\n"; 
      }
    }
    
    if ($line=~ /\<replace_values\>/){       # find <replace_values>
      $nLine=0;
      while (($line=<tstFile>) !~ /\<\/replace_values\>/){
        chomp($line);
        $replaceValues[$nTest][$nLine]=$line;
        $nLine++;
        #print "$nTest  $line\n";
      }
    } 
  }
}
close(tstFile);

#__________________________________
# Globally, replace lines in the main ups file before each test.
@replacementPatterns = (@global_replaceLines);
foreach $rp (@global_replaceLines){
  system("replace_XML_line", "$rp", "$upsFile") ==0 ||  die("Error replacing_XML_line $rp in file $upsFile \n $@");
  print "\t\treplace_XML_line $rp\n";
}

# replace the values globally
@replacementPatterns = (@global_replaceValues);
foreach $rv (@global_replaceValues){
  @tmp = split(/:/,$rv);
  $xmlPath = $tmp[0];       # you must use a : to separate the xmlPath and value
  $value   = $tmp[1];
  system("replace_XML_value", "$xmlPath", "$value", "$upsFile")==0 ||  die("Error: replace_XML_value $xmlPath $value $upsFile \n $@");
  print "\t\treplace_XML_value $xmlPath $value\n";
}


#__________________________________
# Globally perform substitutions in the main ups
my $substitutions = $data->{AllTests}->[0]->{substitutions};

foreach my $t (@{$substitutions->[0]->{text}}){
  print "Now making the substitution text Find: $t->{find} replace: $t->{replace} in file: $upsFile \n";
  system("findReplace","$t->{find}","$t->{replace}", "$upsFile");
}

open(statsFile,">out.stat");

#__________________________________
# Creating new ups files for each test
for ($i=0;$i<=$num_of_tests;$i++){
  if (! -e $upsFile ){
    print "\n\nERROR(run_tests.pl): $upsFile, File Not Found";
    print " Now exiting\n";
    exit
  }
  
  my $test_ups;
  my $test_output;

  $test_ups     = $ups_basename."_$test_title[$i]".".ups";
  $udaFilename  = $ups_basename."_$test_title[$i]".".uda";
  $test_output  = "out.".$test_title[$i];

  # change the uda filename in each ups file
  print "\n--------------------------------------------------\n";
  print "Now modifying $test_ups\n";
  
  system(" cp $upsFile $test_ups");
  my $fn = "<filebase>".$udaFilename."</filebase>";
  system("replace_XML_line", "$fn", "$test_ups")==0 ||  die("Error replace_XML_line $fn in file $test_ups \n $@");
  print "\t\treplace_XML_line $fn\n";
  
  # replace lines in the ups files
  if( defined $replaceLines[$i] ){
    @replacementPatterns = (@{$replaceLines[$i]});
    foreach $rp (@replacementPatterns){
      chomp($rp);
      system("replace_XML_line", "$rp", "$test_ups")==0 ||  die("Error replacing_XML_line $rp in file $test_ups \n $@");
      print "\t\treplace_XML_line $rp\n";
    }
  }
  
  # replace values in the ups files
  if( defined $replaceValues[$i] ){
    @replacementValues = (@{$replaceValues[$i]});
    foreach $rv (@replacementValues){

      if ($rv !~ /^\s*$/ ) {      # ignore blank lines
        @tmp = split(/:/,$rv);
        $xmlPath = $tmp[0];       # you must use a : to separate the xmlPath and value
        $value   = $tmp[1];
        system("replace_XML_value", "$xmlPath", "$value", "$test_ups")==0 ||  die("Error: replace_XML_value $xmlPath $value $test_ups \n $@");
        print "\t\treplace_XML_value $xmlPath $value\n";
      }
    }
  }
  
  #__________________________________
  print statsFile "Test Name :       "."$test_title[$i]"."\n";
  print statsFile "(ups) :         "."$test_ups"."\n";
  print statsFile "(uda) :         "."$udaFilename"."\n";
  print statsFile "output:         "."$test_output"."\n";
  print statsFile "postProcessCmd: "."$postProc_cmd[$i]"."\n";
  
  print statsFile "Command Used : "."$sus_cmd[$i] $test_ups"."\n";
  print "Launching: $sus_cmd[$i] $test_ups\n";
  $now = time();

  @args = ("$sus_cmd[$i]","$test_ups",">& $test_output");
  system("@args")==0 or die("ERROR(run_tests.pl): @args failed: $?");

  #__________________________________
  # execute comparison
  if($postProc_cmd[$i] ne ''){
    print "\nLaunching: analyze_results.pl $tstFile test $i\n";
    @args = ("analyze_results.pl","$tstFile", "$i");
    system("@args")==0 or die("ERROR(run_tests.pl): \t\tFailed running: (@args)\n");
  }
  $fin = time()-$now;
  print  statsFile "Running Time : ".$fin."\n";
  print statsFile "---------------------------------------------\n";
}  # all tests loop

close(statsFile);

