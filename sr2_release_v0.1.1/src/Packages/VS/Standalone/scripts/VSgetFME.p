#!/usr/bin/perl
$_=$ARGV[0];
s/_/ /g;
`/usr/bin/mozilla -remote 'openurl(http://fme.biostr.washington.edu:8089/FME/index.jsp?initialConcept=$_)'`;
