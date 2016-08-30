#!/usr/bin/perl -w
#______________________________________________________________________
#  extractGrid.pl:
#
#  input:  timestep.xml
#
#  This script parses the grid section of a timestep.xml file and produces
#  the <level> specs that can be used in a ups file.  The user would 
#  run this script on a timestep.xml file that has a complicated multilevel grid
#  and then import that into a new ups file using.  This is useful for testing
#  and debugging.
#
#  <include href="/tmp/3DcomplicatedGrid.xml"/>  
#
#  Perl Dependencies:  
#    libxml-simple-perl
#    libxml-dumper-perl
#______________________________________________________________________
use strict;
use XML::Simple;
use Data::Dumper;
use Cwd;
#use diagnostics;

#__________________________________

my $simple = XML::Simple->new(ForceArray=>1, suppressempty => "");

if( $#ARGV == -1){
  print "\n\nextractGrid.pl <timestep.xml> \n";
  print "Now exiting\n \n";
  exit;
}

my $inputFile = $ARGV[0];    # timestep.xml file

#__________________________________
# read in input file
my $xml = $simple->XMLin($inputFile);
my $grid = $xml->{Grid}[0];
my @Levels = @{$grid->{Level}};

#print Dumper(@Levels);

print "<?xml version='1.0' encoding='ISO-8859-1' ?>\n";
print "<Uintah_Include>\n";

#__________________________________
# loop over each Level 
my $l=0;
for($l = 0; $l<=$#Levels; $l++){
  my $level = $Levels[$l];
  
  # read in the number of extraCells
  my $extraCells = $level->{extraCells}[0];
  my $spacing    = $level->{cellspacing}[0];
  my @dx = &xmlToVector($spacing);

  print "<Level>\n";
  print "  <spacing> [$spacing] </spacing>\n";

  #__________________________________
  # loop over each patch
  my $p=0;
  my @Patches = @{$level->{Patch}};
  
  for($p = 0; $p<= $#Patches; $p++){
    my $Patch = $Patches[$p];
    
    # subtract off  extra cells from the lower/upper point of each patch
    # if they exist
    
    #__________________
    # Lower Point
    my @numExtraCells = (0.0, 0.0, 0.0);
    my $intLo   = $Patch->{interiorLowIndex}[0];
    
    if( defined $intLo){
      my $lo      = $Patch->{lowIndex}[0];
      my @low     =  &xmlToVector($lo);
      my @intLow  =  &xmlToVector($intLo);
      
      $numExtraCells[0] = ($low[0] - $intLow[0]);
      $numExtraCells[1] = ($low[1] - $intLow[1]);
      $numExtraCells[2] = ($low[2] - $intLow[2]);
    }
    
    my $lowPt = $Patch->{lower}[0];
    my @lowPt =  &xmlToVector($lowPt);
    $lowPt[0] -= $dx[0] * $numExtraCells[0];
    $lowPt[1] -= $dx[1] * $numExtraCells[1];
    $lowPt[2] -= $dx[2] * $numExtraCells[2];
    

    #__________________
    # upper Point
    @numExtraCells = (0.0, 0.0, 0.0);
    my $intHi   = $Patch->{interiorHighIndex}[0];
    
    if( defined $intHi ){
      my $hi       = $Patch->{highIndex}[0];
      my @high     =  &xmlToVector($hi);
      my @intHigh  =  &xmlToVector($intHi);
      
      $numExtraCells[0] = ($high[0] - $intHigh[0]);
      $numExtraCells[1] = ($high[1] - $intHigh[1]);
      $numExtraCells[2] = ($high[2] - $intHigh[2]);
    }
    
    my $highPt = $Patch->{upper}[0];
    my @highPt =  &xmlToVector($highPt);
    $highPt[0] -= $dx[0] * $numExtraCells[0];
    $highPt[1] -= $dx[1] * $numExtraCells[1];
    $highPt[2] -= $dx[2] * $numExtraCells[2];
    
    
    #__________________
    # print out the patch information
    print "  <Box>\n";
    printf "    <lower>    [ %15.16e, %15.16e, %15.16e ]    </lower>\n", $lowPt[0],  $lowPt[1],  $lowPt[2];
    printf "    <upper>    [ %15.16e, %15.16e, %15.16e ]    </upper>\n", $highPt[0], $highPt[1], $highPt[2];
    print "    <patches>    [1, 1, 1]        </patches>\n";
    print "    <extraCells>  $extraCells     </extraCells>\n";
    print "  </Box>\n";
  }
  print "</Level>\n";
}
print "</Uintah_Include>\n";

#__________________________________
# Subroutine that parses  "[x,y,z]" and returns array with elements x, y, z
# use '&' when calling function to surpress warnings
sub xmlToVector{
    $_[0]=~ tr/\[]//d;    # get rid of'[]'
    my @vector = split(/,/, $_[0]);
    #print " xmlToVector: input $_[0]  vector: $vector[0] $vector[1] $vector[2]\n";
    return @vector
}

