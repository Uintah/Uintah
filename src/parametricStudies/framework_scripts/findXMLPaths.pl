#!/usr/bin/env perl
#
# The MIT License
#
# Copyright (c) 1997-2026 The University of Utah
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
#  findXMLPaths.pl:
#
#  Analyzes a ups (or any xml) file and prints every xmlPath a user could
#  put into a <replace_values><entry path="..." value="..."/></replace_values>
#  block, ready to copy/paste.
#
#  Unlike "xmlstarlet el -v", which prints each level's disambiguating
#  predicate on its own line and does NOT chain a parent's predicate onto
#  its child's line, this walks the dom itself and builds one complete,
#  already-chained path per element -- the exact string xmlstarlet (and
#  replace_XML_value) expect.
#
#  Usage:  findXMLPaths.pl  [options]  <ups_file>
#
#  Options:
#    -g, --grep <pattern>   only print paths matching <pattern> (case-insensitive)
#        --no-values        don't print element-text entries
#        --no-attrs         don't print attribute entries
#    -h, --help             this message
#______________________________________________________________________
use strict;
use warnings;
use XML::LibXML;
use Getopt::Long;
use File::Basename;

#______________________________________________________________________
sub usage{
  print "\nUsage:  ", basename($0), "  [options]  <ups_file>\n\n";
  print "  Prints every xmlPath a <replace_values><entry path=\"...\"/></replace_values>\n";
  print "  could target in <ups_file>, ready to copy/paste.\n\n";
  print "  Options:\n";
  print "    -g, --grep <pattern>   only print paths matching <pattern> (case-insensitive)\n";
  print "        --no-values        don't print element-text entries\n";
  print "        --no-attrs         don't print attribute entries\n";
  print "    -h, --help             this message\n\n";
  exit(1);
}

#______________________________________________________________________
#  Does $candidate have exactly the attribute name/value pairs in @$attrPairs?
sub elem_has_attrs{
  my ($candidate, $attrPairs) = @_;

  foreach my $p (@$attrPairs){
    my ($name, $value) = @$p;
    return 0 if ! $candidate->hasAttribute($name);
    return 0 if $candidate->getAttribute($name) ne $value;
  }
  return 1;
}

#______________________________________________________________________
#  Build the path segment for $elem (its tag name, plus a predicate if
#  needed to disambiguate it from same-tag siblings under the same parent).
sub path_segment{
  my ($elem) = @_;
  my $tag    = $elem->nodeName;
  my $parent = $elem->parentNode;

  if ( ! $parent || $parent->nodeType != XML::LibXML::XML_ELEMENT_NODE ){
    return $tag;                                     # document root
  }

  my @siblings = grep { $_->nodeType == XML::LibXML::XML_ELEMENT_NODE && $_->nodeName eq $tag }
                      $parent->childNodes;

  if ( scalar(@siblings) <= 1 ){
    return $tag;                                     # already unique
  }

  #__________________________________
  # try this element's own attributes as a disambiguating predicate
  my @attrPairs = map { [ $_->nodeName, $_->getValue ] } $elem->attributes;

  if ( @attrPairs ){
    my $nMatches = grep { elem_has_attrs( $_, \@attrPairs ) } @siblings;

    if ( $nMatches == 1 ){
      my $pred = join( " and ", map { sprintf( "\@%s='%s'", $_->[0], $_->[1] ) } @attrPairs );
      return "$tag\[$pred\]";
    }
  }

  #__________________________________
  # fall back to a 1-based positional predicate among same-tag siblings
  my $idx = 1;
  foreach my $s (@siblings){
    last if $s->isSameNode($elem);
    $idx += 1;
  }
  return "$tag\[$idx\]";
}

#______________________________________________________________________
#  Walk the dom, collecting a ready-to-paste entry per leaf element text
#  and per attribute.
sub walk{
  my ($elem, $path, $valueEntries, $attrEntries) = @_;

  my $myPath;
  if ( length($path) == 0 ){
    $myPath = path_segment($elem);
  }
  else{
    $myPath = $path."/".path_segment($elem);
  }

  my @childElements = grep { $_->nodeType == XML::LibXML::XML_ELEMENT_NODE } $elem->childNodes;

  if ( ! @childElements ){
    my $text = $elem->textContent;
    $text =~ s/\s+/ /g;              # collapse internal whitespace/newlines to one space
    $text =~ s/^\s+|\s+$//g;
    if ( length($text) > 0 ){
      push( @$valueEntries, [ $myPath, $text ] );
    }
  }

  foreach my $attr ( $elem->attributes ){
    push( @$attrEntries, [ $myPath."/\@".$attr->nodeName, $attr->getValue ] );
  }

  foreach my $child (@childElements){
    walk( $child, $myPath, $valueEntries, $attrEntries );
  }
}

#______________________________________________________________________
#                                   main

my $grepPattern = undef;
my $showValues   = 1;
my $showAttrs    = 1;
my $help         = 0;

GetOptions( "g|grep=s"   => \$grepPattern,
            "values!"    => \$showValues,
            "attrs!"     => \$showAttrs,
            "h|help"     => \$help ) || usage();

usage() if $help;
usage() if ( scalar(@ARGV) != 1 );

my $upsFile = $ARGV[0];

if ( ! -e $upsFile ){
  print "\nERROR: findXMLPaths.pl: Could not find the file ($upsFile)\n\n";
  exit(1);
}

my $dom  = XML::LibXML->load_xml( location => $upsFile, no_blanks => 1 );
my $root = $dom->documentElement;

my @valueEntries = ();
my @attrEntries  = ();

walk( $root, "", \@valueEntries, \@attrEntries );

if ( defined($grepPattern) ){
  @valueEntries = grep { $_->[0] =~ /$grepPattern/i } @valueEntries;
  @attrEntries  = grep { $_->[0] =~ /$grepPattern/i } @attrEntries;
}

print "#______________________________________________________________________\n";
print "#  <replace_values> paths for: $upsFile\n";
print "#\n";
print "#  Copy an <entry .../> line below into a <replace_values> block in your\n";
print "#  .tst file, then edit the value as needed.\n";
print "#______________________________________________________________________\n";

if ( $showValues ){
  print "\n# -- element values (", scalar(@valueEntries), ") --\n";
  foreach my $e (@valueEntries){
    printf( "    <entry path=\"%s\" value=\"%s\"/>\n", $e->[0], $e->[1] );
  }
}

if ( $showAttrs ){
  print "\n# -- attributes (", scalar(@attrEntries), ") --\n";
  foreach my $e (@attrEntries){
    printf( "    <entry path=\"%s\" value=\"%s\"/>\n", $e->[0], $e->[1] );
  }
}

print "\n";
