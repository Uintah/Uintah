#!/usr/bin/perl
#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

# Print out the "connectedness" of the .h files in the tree.
# This determines how many .o files will get recompiled if you
# touch the .h file.  It is good to minimize these numbers, to
# reduce compile times.
# TODO:
#  - Go through a list of directories from argv, instead of just .

#open(FILES, "find . -name depend.mk -print|")
#	or die "Can't find depend.mk files";
open(FILES, "find . -name *.d -print|")
	or die "Can't find depend.mk files";
$nfiles=0;
foreach $file (<FILES>) {
   open(D, $file)
	or die "Can't open $file";
   foreach $line (<D>) {
      @files = split(/ /, $line);
      foreach $file (@files) {
         #$file =~ s/^(\.\.\/)*//;
	 if(!($file =~ /#|:/) && !($file =~ /^\\$/) && !($file =~ /^$/)) {
	    $counts{$file}++;
	 }
      }
   }
   close(D);
}
close(FILES);

# Sort and print them.
@scounts = sort {$counts{$b} <=> $counts{$a}} keys counts;
$sum=0;
$nfiles=$#scounts;
$lsum=0;
$totallines=0;
foreach $t (@scounts) {
    if( open(F, $t) ){
        $n=0;
        @lines = <F>;
        $n = $#lines;
        $lcount="$n lines";
    } else {
        $lcount="unknown # lines";
	$n = 0;
    }
    $p=100.*$counts{$t}/$nfiles;
    $p=int($p*10+0.5)/10.;
    print "$counts{$t}\t$p%\t$t ($lcount)\n";
    $sum+=$counts{$t};
    $nlcounts{$t}=$n*$counts{$t};
    $lsum+=$n*$counts{$t};
    $totallines+=$n
}
print "$sum total files included ($lsum total lines)\n";
print "$nfiles unique files included ($totallines total lines)\n";
