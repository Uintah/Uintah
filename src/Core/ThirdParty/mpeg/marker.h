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
/*
************************************************************
marker.h

Generic markers
************************************************************
*/

#define PSC 0x00
#define PSC_LENGTH 8

#define UDSC 0xb2
#define UDSC_LENGTH 8

#define VSSC 0xb3
#define VSSC_LENGTH 8

#define ERRC 0xb4
#define ERRC_LENGTH 8

#define EXSC 0xb5
#define EXSC_LENGTH 8

#define VSEC 0xb7
#define VSEC_LENGTH 8

#define GOPSC 0xb8
#define GOP_LENGTH 8

#define MBSC 1
#define MBSC_LENGTH 24

#define TYPE_FORMAT 0x8
