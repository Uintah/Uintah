/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/


int MPEGIntraQ[] = 
{8, 16, 19, 22, 26, 27, 29, 34,
16, 16, 22, 24, 27, 29, 34, 37,
19, 22, 26, 27, 29, 34, 34, 38,
22, 22, 26, 27, 29, 34, 37, 40,

22, 26, 27, 29, 32, 35, 40, 48,
26, 27, 29, 32, 35, 40, 48, 58,
26, 27, 29, 34, 38, 46, 56, 69,
27, 29, 35, 38, 46, 56, 69, 83};


int MPEGNonIntraQ[] = 
{16, 16, 16, 16, 16, 16, 16, 16,
16, 16, 16, 16, 16, 16, 16, 16,
16, 16, 16, 16, 16, 16, 16, 16,
16, 16, 16, 16, 16, 16, 16, 16,

16, 16, 16, 16, 16, 16, 16, 16,
16, 16, 16, 16, 16, 16, 16, 16,
16, 16, 16, 16, 16, 16, 16, 16,
16, 16, 16, 16, 16, 16, 16, 16};

