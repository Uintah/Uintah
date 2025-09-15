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
#  analyze.pm
#  Perl module used to call the comparison utility for each test
#     There could be multiple post process commands
#
#
#______________________________________________________________________
package analyze;
use strict;
use warnings;
use XML::LibXML;
use Data::Dumper;
use File::Which;
use File::Basename;
use lib dirname (__FILE__);  # needed to find local Utilities.pm
use Utilities;

use Exporter 'import';
our @ISA = qw(Exporter);
our @EXPORT_OK = qw(analyze);

#______________________________________________________________________

sub analyze{
  my( $test_node, $uda, $statsFile, $exitOnCrash ) = @_;

  my $X  = Utilities::cleanStr( $test_node->findvalue('x') );

  #__________________________________
  #   Loop over the postProcess commands, there could several
  foreach my $node ( $test_node->findnodes('postProcess_cmd') ) {

    my $postProc_cmd = $node->textContent;

    # remove leading spaces from command
    $postProc_cmd=~ s/^\s+//;

    # change command from a scalar to array for easier parsing
    my @cmd_A = ( split(/ /, $postProc_cmd) );

    # prune out the command options
    my $cmd_trim = $cmd_A[0];

    my $cmd = which($cmd_trim);

    # command basename
    my $cmd_basename = basename( $postProc_cmd );

    #__________________________________
    #   bulletproofing
    if ( ! defined $cmd ){
      my $mypath = $ENV{"PATH"};
      print "\n\n__________________________________\n";
      print "ERROR:analyze_results:\n";
      print "The comparison utility: ($cmd_basename)";
      print " doesn't exist in path \n $mypath \n Now exiting\n\n\n";
      die("ERROR(run_tests.pl): \t\tFailed running: ($cmd_basename)\n");
    }

    if ( ! -l $cmd){
      system("ln -fs $cmd > /dev/null 2>&1");
    }

    #__________________________________
    #   Run the post processing scripts

    my $ppOutput = "outPP.$X";

    print "\n\tLaunching: ($cmd_basename -o $ppOutput -uda $uda)\n\n";
    `rm -f $ppOutput`;

    my @cmd = ("$cmd_basename", "-o", "$ppOutput", "-uda", "$uda");

    if ( $exitOnCrash eq "TRUE" ) {
      system( "@cmd" )==0 or die("ERROR(analyze.pm): \t\tFailed running: (@cmd)\n");
    }else{
      system( "@cmd" );
    }

    print $statsFile "postProcessCmd:  "."@cmd"."\n";

    #__________________________________
    #   If the script outputs concatenate results
    if ( -e $ppOutput && -s $ppOutput ){
      my $L2norm = `cat $ppOutput`;
      chomp($L2norm);
      `echo $X $L2norm >> L2norm.dat`;            # HARDCODED
      `rm -f $ppOutput`;
    }
  }
};
1;    # Required for a Perl Module
