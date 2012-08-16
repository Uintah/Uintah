#!/usr/bin/perl
#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
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
