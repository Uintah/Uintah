#!/usr/bin/perl
#
# Print out the "connectedness" of the .h files in the tree.
# This determines how many .o files will get recompiled if you
# touch the .h file.  It is good to minimize these numbers, to
# reduce compile times.
#
# TODO:
#  - Go through a list of directories from argv, instead of just .
#

open(FILES, "find . -name \"*.d\" -print|")
	or die "Can't find .d files";
$nfiles=0;
foreach $file (<FILES>) {
   open(D, $file)
	or die "Can't open $file";
   foreach $line (<D>) {
      @files = split(/ /, $line);
      foreach $file (@files) {
         $file =~ s/^(\.\.\/)*//;
	 if(!($file =~ /#|:/)) {
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
foreach $t (@scounts) {
    if( open(F, $t) ){
        $n=0;
        @lines = <F>;
        $n = $#lines;
        $lcount="$n lines";
    } else {
        $lcount="unknown # lines";
    }
    $p=100.*$counts{$t}/$nfiles;
    $p=int($p*10+0.5)/10.;
    print "$counts{$t}\t$p%\t$t ($lcount)\n";
    $sum+=$counts{$t};
    $nlcounts{$t}=$lcount*$counts{$t};
    $lsum+=$lcount*$counts{$t};
}
print "$sum total files included ($lsum total lines)\n";
