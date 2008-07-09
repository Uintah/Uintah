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
#     - <pbs section not used>
#     - create a xml file that is read by analyze_results.pl
#     - run the test
#
#     if(comparison Command )
#       -run analyze_results.pl
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
$xml = new XML::Simple(forcearray => 1);

# read XML file
$data = $xml->XMLin("$ARGV[0]");

#__________________________________
# Read in the test data
my $i=0;

my $gnuplotFile = $data->{gnuplotFile}->[0];                # if a user provides a gnuplot file

$upsFile         =$data->{upsFile}->[0];
my $ups_basename = $upsFile;
$ups_basename    =~ s/.ups//;                               # Removing the extension .ups so that we can use this to build our uda file names

foreach $e (@{$data->{Test}}){

  $test_title[$i]     =$e->{Title}->[0];                    # test title
  $sus_cmd[$i]        =$e->{sus_cmd}->[0];                 # sus command

  $study[$i]          =$e->{Study}->[0];                    #Study Name
  $x[$i]              =$e->{x}->[0];
  $compUtil_cmd[$i]   =$e->{compare_cmd}->[0];              #comparison utility command
  
  $i++;     
}
$num_of_tests=$i;

#__________________________________
# make a symbolic link to the compareUtil
my @stripped_cmd = split(/ /,$compUtil_cmd[0]);  # remove command options
my $cmd = `which $stripped_cmd[0]`;
system("ln -s $cmd");


`rm -f compareAll.xml`;
`echo \\<start\\> \| tee -a compareAll.xml`;

#__________________________________
# Read in all of the replacement patterns 
# and store them in arrays.
#   There can be global replacement lines and individual test replacement lines
my $nTest=0;
my $line;
my $insideTest=0;
my $insideAllTest=0;

open(tstFile, "$ARGV[0]") or die("ERROR(run_tests.pl): $ARGV[0], File not found");

while ($line=<tstFile>){
  if($line=~ /\<AllTests\>/){
    $insideAllTest=1;
  }
  if($line=~ /\<\/AllTests\>/){
    $insideAllTest=0;
  }
  if($line=~ /\<Test\>/){
    $insideTest=1;
  }
  if($line=~ /\<\/Test\>/){
    $insideTest=0;
  } 
  
  # inside of <AllTests>
  if($insideAllTest){
    if ($line=~ /\<replace_lines\>/){       # find <replace_lines>
      $nLine=0;
      while (($line=<tstFile>) !~ /\<\/replace_lines\>/){
        $global_replaceLines[$nLine]=$line;
        $nLine++;
      }
    }
  }
  
  # inside each <Test>
  if($insideTest){
    if ($line=~ /\<replace_lines\>/){       # find <replace_lines>
      $nLine=0;
      while (($line=<tstFile>) !~ /\<\/replace_lines\>/){
        $replaceLines[$nTest][$nLine]=$line;
        $nLine++;
      }
      $nTest++;
    }
  }
}
close(tstFile);

#__________________________________
# Globally, replace lines in the main ups file before each test.
@replacementPatterns = (@global_replaceLines);
foreach $rp (@global_replaceLines){
  chomp($rp);
  system("replace_XML_line", "$rp", "$upsFile");
  print "\t\t$rp\n"
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
for ($i=0;$i<$num_of_tests;$i++){
  #open(inpFile, $upsFile) or die("ERROR(run_tests.pl): $upsFile, File Not Found");

  if (! -e $upsFile ){
    print "\n\nERROR(run_tests.pl): $upsFile, File Not Found";
    print " Now exiting\n";
    exit
  }
  
  my $test_ups;
  my $test_output;

  $test_ups     = $ups_basename."_$test_title[$i]".".ups";
  $udaFilename  = $ups_basename."_$test_title[$i]".".uda";
  $compFilename = $test_title[$i]."_comp.xml";
  $test_output  = "out.".$test_title[$i];

  # change the uda filename in each ups file
  print "---------------------\n";
  print "Now modifying $test_ups\n";
  
  system(" cp $upsFile $test_ups");
  my $fn = "<filebase>".$udaFilename."</filebase>";
  system("replace_XML_line", "$fn", "$test_ups");
  print "\t\t$fn\n";
  
  # replace lines in the ups files
  @replacementPatterns = (@{$replaceLines[$i]});
  foreach $rp (@replacementPatterns){
    chomp($rp);
    system("replace_XML_line", "$rp", "$test_ups");
    print "\t\t$rp\n"
  }
  print "---------------------\n";

  #__________________________________
  # Create a comparison config file _if_ the comparison command is specified
  # This is read in by analyze_results.pl

  if($compUtil_cmd[$i]){
    `rm -fr $compFilename`;
    
    `echo \\<start\\> \|  tee -a $compFilename`;
    `echo \\<gnuplotFile\\>$gnuplotFile\\</gnuplotFile\\> \|tee -a $compFilename compareAll.xml`;
    `echo \\<Test\\>  \|  tee -a $compFilename compareAll.xml`;
    `echo \\<Title\\>$study[$i]\\</Title\\>  \| tee -a $compFilename compareAll.xml`;
    `echo \\<compareUtil\\>$compUtil_cmd[$i]\\</compareUtil\\>  \| tee -a $compFilename compareAll.xml`;
    `echo \\<x\\>$x[$i]\\</x\\>  \| tee -a $compFilename compareAll.xml`;
    `echo \\<uda\\>$udaFilename\\</uda\\>  \| tee -a $compFilename compareAll.xml`;
    `echo \\</Test\\>  \| tee -a $compFilename compareAll.xml`;
    `echo \\</start\\> \| tee -a $compFilename`;
  }
  
  #__________________________________
  print statsFile "Test Name :       "."$test_title[$i]"."\n";
  print statsFile "(ups) :     "."$test_ups"."\n";
  print statsFile "(uda) :     "."$udaFilename"."\n";
  print statsFile "output:     "."$test_output"."\n";
  print statsFile "compareCmd: "."$compUtil_cmd[$i]"."\n";
  
  print statsFile "Command Used : "."$sus_cmd[$i] $test_ups"."\n";
  print "Launching $sus_cmd[$i] $test_ups\n";
  $now = time();

  @args = ("$sus_cmd[$i]","$test_ups",">& $test_output");
  system("@args")==0 or die("ERROR(run_tests.pl): @args failed: $?");

  #__________________________________
  # execute comparison
  if($compUtil_cmd[$i]){
    print "\n\nLaunching analyze_results.pl $compFilename\n\n";
    @args = ("analyze_results.pl","$compFilename");
    system("@args")==0 or die("ERROR(run_tests.pl): \t\tFailed running: (@args)\n");

    system("rm $compFilename");
  }
  $fin = time()-$now;
  print  statsFile "Running Time : ".$fin."\n";
  print statsFile "---------------------------------------------\n";
}  # all tests loop

close(statsFile);


`echo \\</start\\> \| tee -a compareAll.xml`;
