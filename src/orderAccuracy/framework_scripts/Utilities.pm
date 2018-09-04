package Utilities;
use strict;
use warnings;
use Exporter 'import';
our @ISA = qw(Exporter);
our @EXPORT_OK = qw(cleanStr);
#______________________________________________________________________
#  
#  Remove any white space or newlines in array elements or scalars
#  (This belongs in a separate common module to avoid duplication -Todd)

sub cleanStr {

  my @inputs = @_;
  
#  print Dumper (@inputs);
  
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
}

1;
