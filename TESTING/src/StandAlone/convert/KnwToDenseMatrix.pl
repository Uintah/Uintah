#!/usr/bin/perl
#
# C.Wolters, 15.11.2004
# Perl-script for the transformation of a ca_perm.knw in  tensor form to SCIRun DenseMatrix format
#
# Call: KnwToDenseMatrix.pl tensorcond.knw tensorcond.mat
#
#require "usage.pl";

#&usage('KnwToDenseMatrix.pl',1,'converts conductivity file in tensor valued CAUCHY-knw format into SCIRun DenseMatrix format',
#                      'KnwToDenseMatrix.pl tensorcond.knw ',
#                      'tensorcond.knw   CAUCHY knw conductivity file in tensor form',
#                      '$Revision$','$Author$','$Date$');
$inf=$ARGV[0];
$out=$ARGV[1];

# Use the first run through the knw file in order to get the number of tensor elements $number_of_ele 
open(INF,"<$inf") || die "\n Can't open $inf !\n Wrong filename base?\n ...";
$diagonal=1;
$nondiagonal=0;
$maindata=0;
while (<INF>)	
	{
   	split;
	if ($_=~/EOI\s-\sTENSOR/)
   		{
		$maindata=0;
  		}
	if ($maindata)
   		{
		if ($diagonal)
			{
   			$number_of_ele = $_[0];
			$diagonal = 0;
			$nondiagonal=1;
    			}
		else 
			{
			$diagonal = 1;
			$nondiagonal=0;
    			}
   		}
	if (/BOI\s-\sTENSOR/)
		{
		$maindata=1;
   		}
	if (/BOI\s-\sTENSORVALUEFILE/)
		{
		$maindata=0;
   		}
 	}
close(INF);

# Now write out the SCIRun file within the second run through the knw-file
open(OUT,">$out");
print OUT "$number_of_ele 9\n";

open(INF,"<$inf") || die "\n Can't open $inf !\n Wrong filename base?\n ...";
$diagonal=1;
$maindata=0;
@diag=();
while (<INF>)	
	{
   	split;
	if ($_=~/EOI\s-\sTENSOR/)
   		{
		$maindata=0;
  		}
	if ($maindata)
   		{
		if ($diagonal)
			{
			# diagonal entries xx, yy, zz
			@diag[0]=$_[1];
			@diag[1]=$_[2];
			@diag[2]=$_[3];
			$diagonal = 0;
    			}
		else 
			{
			# nondiagonal entries xy=yx, yz=zy, xz=zx
   			split;
			# DenseMatrix order: xx, xy, xz, yx, yy, yz, zx, zy, zz
			print OUT "@diag[0] $_[0] $_[2] $_[0] @diag[1] $_[1] $_[2] $_[1] @diag[2]\n";
			$diagonal = 1;
    			}
   		}
	if (/BOI\s-\sTENSOR/)
		{
		$maindata=1;
   		}
	if (/BOI\s-\sTENSORVALUEFILE/)
		{
		$maindata=0;
   		}
 	}
close(INF);
close(OUT);
