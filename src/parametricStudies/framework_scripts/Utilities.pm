#
# The MIT License
#
# Copyright (c) 1997-2023 The University of Utah
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

package Utilities;
use strict;
use warnings;
use diagnostics;
use warnings;
use XML::LibXML;
use Data::Dumper;
use File::Which qw(which where);
use Exporter 'import';

our @ISA = qw(Exporter);
our @EXPORT_OK = qw(cleanStr setPath print_XML_ElementTree get_XML_value modify_xml_file modify_xml_files modify_batchScript read_file write_file runPreProcessCmd runSusCmd submitBatchScript );

#______________________________________________________________________
#
#  Remove any white space or newlines in array elements or scalars
sub cleanStr {

  my @inputs = @_;
  my $n   = scalar @inputs;           # number of array elements
  my $len = length $inputs[0];        # number of characters in first element

  if( ! @inputs  ){
    return undef;
  }

  # if the first element is empty return ""
  if( $len == 0 ){
    return "";
  }

  #__________________________________
  # if there is one array element return a scalar
  if( $n == 1 ){
    $inputs[0] =~ s/\n//g;        # remove newlines
    $inputs[0] =~ s/ //g;         # remove white spaces
    $inputs[0] =~ s/"//g;         # remove quotes
    return $inputs[0];
  }

  #__________________________________
  #  Arrays
  my @result = ();
  my $i = 0;

  foreach $i (@inputs){
    $i =~ s/\n//g;        # remove newlines
    $i =~ s/ //g;         # remove white spaces
    $i =~ s/"//g;         # remove quotes
    my $l = length $i;

    if ($l > 0){
      push( @result, $i );
    }
  }
  return @result;
}

#______________________________________________________________________
#   Test to see if file exist at the user input path or in it's default location(s)
#   The paths can be either a scalar or an array
sub setPath{
  my($input, @paths) = @_;

  # remove quotes from the input
  $input =~ s/"//g;

  if( -e $input ){
    return $input;
  }

  # the input file may have a wildcard
  if($input =~ m/\*/){
    system("file -E $input > /dev/null 2>&1")  == 0 || die("\nInvalid path($input)");
    return $input;
  }

  foreach  my $path (@paths){
    my $modInput  = $path."/".$input;

    if( -e $modInput ){
      return $modInput;
    }
  }

  #  Bulletproofing
  print "\n \nERROR:setPath():\n";
  print "The file: \n ($input) \n";

  foreach my $path (@paths){
    my $modInput  = $path."/".$input;
    print " ($modInput)\n";
  }

  print "does not exist.  Now exiting\n";
  exit
}

#______________________________________________________________________
#   usage:  get_XML_value( elementList, <xmltag>, "defaultValue")
#  This returns either an array or a scalar.  It returns the default value
#  if the xml tag does not exist.

sub get_XML_value{
  my $elementList = $_[0];
  my $xmltag      = $_[1];
  my $defaultVal  = $_[2];

  if( ! $elementList->exists( $xmltag ) ){
    return $defaultVal;
  }

  my @result = ();

  foreach my $element ( $elementList->getElementsByTagName( $xmltag ) ) {
    $element = cleanStr( $element->textContent() );

    if( length $element == 0 && defined $defaultVal){
      $element = $defaultVal;
    }

    push( @result, $element );
    #print "xmlTag: ", $xmltag, " len: ", length $element, " (", $element, ")\n";
  }

  my $len = scalar @result;

  if ( $len == 1) {
    return $result[0];
  }
  else {
    return @result;
  }
}


#______________________________________________________________________
#  Prints the element tree and the values for each element
#  It needs more work.
sub print_XML_ElementTree{
  my $input = $_[0];

  my @elements = grep { $_->nodeType == XML_ELEMENT_NODE } $input->childNodes;
  my $count = @elements;

  print "\nXML tree has $count child elements: \n";

  my $i = 0;
  foreach my $child (@elements) {
    print "  ",$i++, ": is a ", ref($child), ', name = ', $child->nodeName;
    print "     toLiterial: (", $child->to_literal(), ")  ";
    print "textContent: (", $child->textContent(), ")\n";
  }
}


#______________________________________________________________________

sub modify_batchScript{
  my ($filename, @xmlNodes ) = @_;

  my $data = read_file($filename);

  foreach my $X ( @xmlNodes ){
    my $tag   = $X->{tag};
    $tag      =~ s/\[/\\[/g;     # add escape chars to the metachars
    $tag      =~ s/\]/\\]/g;
    my $value = $X->{value};

    print "\tmodifying batch script  Changing ($tag) -> ($value)\n";
    $data =~ s/$tag/$value/g;

    # bulletproofing
    if( ! ($data=~/$value/) ){
      print "\n\tERROR Modyify_batchScript, Could not find the tag ", $tag, "\n\n";
      die "$!";
    }
  }
   write_file($filename, $data);
}


#______________________________________________________________________
#  This subrouting verifies each element of an xmlPath and verifies that
#  it exists in the dom file
#______________________________________________________________________

sub isXPathValid{
  my ( $xmlPath, $upsFile, $ups_dom) = @_;

  my @elements  = split (/\//, $xmlPath);           # extract xpath elements
  my $old_xpath = "";
  my $isValid     = 1;                                # is xpath isValid

  foreach my $x (@elements) {
    my $test_xpath = $old_xpath."/".$x;

    my $exists = $ups_dom->exists($test_xpath);
    $old_xpath = $test_xpath;

    if ( $exists == 0) {
      $isValid =0;
      last;
    }
  }

  if ( $isValid == 0 ){
    print "\n****************************\n";
    print "ERROR:  The Xpath could not be found in the file ($upsFile)\n";

    $old_xpath = "";

    foreach my $x (@elements) {
      my $test_xpath = $old_xpath."/".$x;

      my $exists = $ups_dom->exists($test_xpath);

      my $desc = "GOOD";
      if ( $exists == 0 ){
        $desc = "NOT FOUND";
      }

      print "   $test_xpath   $desc\n";
      $old_xpath = $test_xpath;
    }
    print "****************************\n";
  }
}


#______________________________________________________________________
#   This function will modify an array of ups files
#      $ups_files            -array of ups file names
#      $ups_doms             - array of ups xml Domain Object models
#      $test_nodes           - parent xml node to replace_lines & replace_values
#______________________________________________________________________
sub modify_xml_files{
  my ($ups_files_ref, $ups_doms_ref, $test_node) = @_;

  my @ups_files  = @{ $ups_files_ref };       # dereferencing and copying each array
  my @ups_doms   = @{ $ups_doms_ref };

  my $size = scalar @ups_files;               # number of files in the array
  $size -= 1;
  #__________________________________
  # replace xml line
   foreach my $rpl_node($test_node->findnodes('replace_lines/*')) {

    my $rpl_dom = XML::LibXML->load_xml(string => $rpl_node);       # create document object model from string
    my $rpl_Elements = $rpl_dom->documentElement;
    my $rpl_nodeName = $rpl_Elements->nodeName;
    my $xmlPath      = "//".$rpl_nodeName;                     # "//" means recursively search all nodes
    my $nModifiedFiles = 0;

    for my $i (0 .. $size) {
      my $ups_dom = $ups_doms[$i];
      my $upsFile = $ups_files[$i];

      if ( $ups_dom->exists($xmlPath) ){
        print "\treplace_XML_line $rpl_node \t $upsFile\n";

        system("replace_XML_line", "$rpl_node", "$upsFile");

        # bulletproofing
        system("xmlstarlet val -q $upsFile") == 0 ||  die("\nERROR: $upsFile, contains errors.\n");

        $nModifiedFiles += 1;
      }
    }

    # bulletproofing
    if( $nModifiedFiles != 1){
      print "\n*************************************************\n";
      print "ERROR replacing the line ($rpl_node) failed.\n";
      print "      To find the xpath xmlstarlet el -v  uda/input.xml or uda/checkpoints/<timestep>/timestep.xml\n";
      print "*************************************************\n\n";
      die("$@")
    }
  }

  #__________________________________
  # replace xml Value
  foreach my $rv_node ($test_node->findnodes('replace_values/entry')) {

    my $xmlPath = $rv_node->{path};
    my $value   = $rv_node->{value};
    XML::LibXML::XPathExpression->new( $xmlPath );            # this checks the syntax of the xpath
    my $nModifiedFiles = 0;

    for my $i (0 .. $size) {
      my $ups_dom = $ups_doms[$i];
      my $upsFile = $ups_files[$i];

      if ( $ups_dom->exists($xmlPath) ){
        print "\treplace_XML_value $xmlPath $value \t $upsFile\n";

        system("replace_XML_value", "$xmlPath", "$value", "$upsFile");

        # bulletproofing
        system("xmlstarlet val -q $upsFile") == 0 ||  die("\nERROR: $upsFile, contains errors.\n");

        $nModifiedFiles += 1;
      }
    }

    #__________________________________
    # bulletproofing
    if( $nModifiedFiles != 1 ){
      print "\n*************************************************\n";
      print "ERROR replacing the xml value ($xmlPath) failed.\n";
      print "      To find the xpaths run xmlstarlet el -v uda/input.xml or uda/checkpoints/<timestep>/timestep.xml\n";
      print "      Detailed analysis of the xmlPath is below\n";
      print "*************************************************\n\n";

      for my $i (0 .. 1) {
        my $ups_dom = $ups_doms[$i];
        my $upsFile = $ups_files[$i];
        isXPathValid($xmlPath, $upsFile, $ups_dom);
      }
      die("$@")
    }
  }
}

#______________________________________________________________________
#   This function will modify a ups file
#      $ups_files            - ups file name
#      $test_nodes           - parent xml node to replace_lines & replace_values
#______________________________________________________________________
sub modify_xml_file{
  my ( $upsFile, $test_nodes ) = @_;

  #__________________________________
  # replace lines in test_ups
  foreach my $rpl_node ( $test_nodes->findnodes('replace_lines/*') ) {
    print "\treplace_XML_line $rpl_node\n";
    system("replace_XML_line", "$rpl_node", "$upsFile")==0 ||  die("Error replacing_XML_line $rpl_node in file $upsFile \n $@");
  }

  #__________________________________
  # replace any values in the ups files
  foreach my $rv_node ( $test_nodes->findnodes('replace_values/entry') ) {
    my $xmlPath = $rv_node->{path};
    my $value   = $rv_node->{value};

    print "\treplace_XML_value $xmlPath $value\n";
    system("replace_XML_value", "$xmlPath", "$value", "$upsFile")==0 ||  die("Error: replace_XML_value $xmlPath $value $upsFile \n $@");
  }

  #bulletproofing
  system("xmlstarlet val --err $upsFile") == 0 ||  die("\nERROR: $upsFile, contains errors.\n");
}


#______________________________________________________________________

sub read_file {
    my ($filename) = @_;

    open my $in, '<:encoding(UTF-8)', $filename or die "Could not open '$filename' for reading $!";
    local $/ = undef;
    my $all = <$in>;
    close $in;

    return $all;
}

#______________________________________________________________________

sub write_file {
    my ($filename, $content) = @_;

    open my $out, '>:encoding(UTF-8)', $filename or die "Could not open '$filename' for writing $!";;
    print $out $content;
    close $out;

    return;
}

#______________________________________________________________________

sub runPreProcessCmd {
  my ( $upsFile_base, $upsFile_mod, $test_nodes ) = @_;
  
  if( ! $test_nodes->findnodes('preProcess_cmd') ){
    return;
  }
   
  print "\n";

  foreach my $ppc_node ( $test_nodes->findnodes('preProcess_cmd') ) {
    my $cmd0      = $ppc_node->getAttribute('cmd');
    my $whichUps  = uc( $ppc_node->getAttribute('which_ups') );
    my $add_ups   = uc( $ppc_node->getAttribute('add_ups') );
    my $upsFile   = "";

    my @cmd = split(/ /,$cmd0);
    
    # bulletproofing
    my $test1 = ( ( $add_ups eq "TRUE" )  &&( $whichUps ne "TEST_UPS" ) && ( $whichUps ne "BASE_UPS" ) );
    my $test2 = ( ( $add_ups eq "FALSE" ) &&( $whichUps ne "NONE" ) );

    if ( $test1 || $test2 ){
      print "\n\tERROR:runPreProcessCmd:\n";
      print "\t- The only valid options for 'which_ups' is 'TEST_UPS', 'BASE_UPS', or 'none.'\n";
      print "\t- If 'add_ups' == false then 'which_ups' must be none.\n";
      print "\t $ppc_node\n";
      print "\tNow exiting\n";
      exit 
    }
    if ( ( ! which($cmd[0]) ) ){
      print "\n\tERROR:runPreProcessCmd:\n";
      print "\tThe command specified could not be found:\n";
      print "\t $ppc_node\n";
      print "\tNow exiting\n";
      exit 
    }

    if( $whichUps eq "TEST_UPS" ){
      $upsFile = $upsFile_mod;
    }
    elsif ( $whichUps eq "BASE_UPS" ) {
      $upsFile = $upsFile_base;
    }

    if ( ( ! -e $upsFile ) && ( $whichUps ne "NONE") ){
      die("\nERROR \trunPreProcessCmd Could not find the ups file ($upsFile)\n\n");
    }
    
    my @full_cmd = ( "@cmd", "$upsFile", ">> out.preProcess 2>&1" );
    
    my $outFile;
    open( $outFile,">>", "out.preProcess");    
    print  $outFile "\n______________________________________________________________________\n";
    print  $outFile "   cmd: (@full_cmd) whichUps: ".$whichUps." Ups: ".$upsFile. "\n";
    print  $outFile "______________________________________________________________________\n";
    close($outFile);
    
    print "\tExecuting preProcessCmd (@full_cmd)\n";
    
    my $rc = system( "@full_cmd" );
    
    if( $rc != 0){
      print "ERROR \trunPreProcessCmd, the command (@full_cmd) failed\n";
      die("ERROR");
    }
  }
};

#______________________________________________________________________

sub runSusCmd {
  my( $timeout, $exitOnCrash, $statsFile, @sus_cmd ) = @_;

  print "\tLaunching: (@sus_cmd)\n";
  my @cmd = (" timeout --preserve-status $timeout @sus_cmd ");

  my $rc = -9;
  if ( $exitOnCrash eq "TRUE" ) {

    $rc = system("@cmd");

    if ( $rc != 0 && $rc != 36608 ){
      die("ERROR(run_tests.pl): \t\tFailed running: (@sus_cmd)\n");
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

sub submitBatchScript{
  my( $doRestart, $test_title, $batchCmd, $test_batch, $statsFile, @sus_cmd ) = @_;

  #__________________________________
  # concatenate
  open( SCRIPT, '>>', $test_batch) or die "Could not open file '$test_batch' $!";

  if( $doRestart == 1){
    print SCRIPT "echo 'Now applying patch to input.xml and timestep.xml'\n";
    print SCRIPT "patch -p0 --verbose --input diff.",$test_title,"\n";
  }

  print SCRIPT "\n @sus_cmd \n";
  close SCRIPT;

  #__________________________________
  # edit batch script
  my $data  = read_file($test_batch);

  #  change job name
  my $tag   = "\\[jobName\\]";
  my $value = $test_title;
  $data     =~ s/$tag/"$value"/g;

  # change the job output name
  $tag      = "\\[output\\]";
  $value    = "job-". $test_title . ".out";
  $data     =~ s/$tag/"$value"/g;

  # remove white spaces before and after "="
  # Slurm doesn't like white spaces
  $data     =~ s{\s+=}{=}g;
  $data     =~ s{=\s+}{=}g;

  #print "$data";
  write_file($test_batch, $data);

  #__________________________________
  # concatenate postProcess cmd to batch script  TODO

  #__________________________________
  # submit batch script

  print "\t Submitting batch script: ", $batchCmd, " " , $test_batch, "\n";
  my @cmd = ( "$batchCmd", "$test_batch" );
  system("@cmd")==0 or die("ERROR(run_tests.pl): \t\tFailed running: (@cmd)\n");
};
1;
