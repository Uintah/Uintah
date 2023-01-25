package Utilities;
use strict;
use warnings;
use XML::LibXML;
use Data::Dumper;
use Exporter 'import';

our @ISA = qw(Exporter);
our @EXPORT_OK = qw(cleanStr setPath print_XML_ElementTree get_XML_value modify_batchScript read_file write_file );

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
}

#______________________________________________________________________
#   Test to see if file exist at the user input path or in it's default location(s)
#   The paths can be either a scalar or an array
sub setPath{
  my($input, @paths) = @_;
  
  if( -e $input ){
    return $input;
  } 

  # the input file may have a wildcard
  if($input =~ m/\*/){
    my @files = glob($input);
    foreach  my $file (@files){
      print "The file contains a wild card, ignoring  ($file)\n";
    }
    return $input;
  }

  foreach  my $path (@paths){
    my $modInput  = $path."/".$input;
    
    if( -e $modInput ){
      return $modInput;
    }
  }
  
  #  Bulletproofing
  print "\n \nERROR:setupFrameWork:\n";
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

  print "\$XML tree has $count child elements: \n";

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

1;
