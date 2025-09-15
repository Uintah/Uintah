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
#  gnuplot.pm
#  Perl modules that contain all the gnuplot scripts
#
#  The plot script path is relative to the tst file
#  The plot script is copied to the test directory upstream
#______________________________________________________________________
package gnuplot;
use strict;
use warnings;
use XML::LibXML;
use Data::Dumper;
use File::Basename;
use lib dirname (__FILE__);  # needed to find local Utilities.pm and gnuplot.pm
use Utilities;

use Exporter 'import';
our @ISA = qw(Exporter);
our @EXPORT_OK = qw(gnuplot_singleTest gnuplot_allTests);

#______________________________________________________________________
#   This function is run on every test
sub gnuplot_singleTest{
  my( $testNode, $uda, $statsFile, $exitOnCrash ) = @_;  

  my $gpFile = Utilities::cleanStr($testNode->findvalue( '/gnuplot/script' ) );

  if ( -e $gpFile ) {
    # modify the plot script
    my $title = $testNode->findvalue( '/gnuplot/title' );
    my $arg1  = $testNode->findvalue( '/gnuplot/arg1' );
    my $arg2  = $testNode->findvalue( '/gnuplot/arg2' );
    my $xlabel= $testNode->findvalue( '/gnuplot/xlabel' );
    my $ylabel= $testNode->findvalue( '/gnuplot/ylabel' );
    my $label = $testNode->findvalue( '/gnuplot/label' );

    system("sed", "-i", "s/#title/set title   \"$title\"/g",  "$gpFile");
    system("sed", "-i", "s/#xlabel/set xlabel \"$xlabel\"/g", "$gpFile");
    system("sed", "-i", "s/#ylabel/set ylabel \"$ylabel\"/g", "$gpFile");
    system("sed", "-i", "s/#label/set label   \"$label\"/g",  "$gpFile");


    my @gpCmd = ( "gnuplot -c", "$gpFile", "$arg1", "$arg2" );
    print "Now plotting using the modified gnuplot script (@gpCmd) \n";
    
    if ( $exitOnCrash eq "TRUE" ) {
      system("@gpCmd > gp.out 2>&1") ==0 or die("ERROR(gnuplot.pm):\tFailed running: (@gpCmd)) failed: $@");
    
    }else{
      system( "@gpCmd" );
    }

    print $statsFile "gnuplot command:  "."@gpCmd"."\n";
  }
};


#______________________________________________________________________
#  This function is called after all the tests have been completed
sub gnuplot_allTests{
  my( $doc, $statsFile, $exitOnCrash) = @_;  

  my $gpFile = Utilities::cleanStr($doc->findvalue( '/start/gnuplot/script' ) );

  if ( -e $gpFile ) {
    # modify the plot script
    my $title = $doc->findvalue( '/start/gnuplot/title' );
    my $arg1  = $doc->findvalue( '/start/gnuplot/arg1' );
    my $arg2  = $doc->findvalue( '/start/gnuplot/arg2' );
    my $xlabel= $doc->findvalue( '/start/gnuplot/xlabel' );
    my $ylabel= $doc->findvalue( '/start/gnuplot/ylabel' );
    my $label = $doc->findvalue( '/start/gnuplot/label' );

    system("sed", "-i", "s/#title/set title   \"$title\"/g",  "$gpFile");
    system("sed", "-i", "s/#xlabel/set xlabel \"$xlabel\"/g", "$gpFile");
    system("sed", "-i", "s/#ylabel/set ylabel \"$ylabel\"/g", "$gpFile");
    system("sed", "-i", "s/#label/set label   \"$label\"/g",  "$gpFile");


    my @gpCmd = ( "gnuplot -c", "$gpFile", "$arg1", "$arg2" );
    print "Now plotting using the modified gnuplot script (@gpCmd) \n";

    system("@gpCmd > gp.out 2>&1") ==0 or die("ERROR(gnuplot.pm):\tFailed running: (@gpCmd)) failed: $@");
  }
};
1;    # Required for a Perl Module
