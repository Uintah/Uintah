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
mem.h

This file contains the structures inherent manipulating the
memory management of image structures.

************************************************************
*/

#ifndef MEM_DONE
#define MEM_DONE

#define MEM struct memory_construct

typedef unsigned char **BLOCK;
typedef MEM * (* Ifunc)();


MEM {
int len;
int width;
int height;
unsigned char *data;
};

#define LBOUND(index,value) (((index) < (value)) ? (value) : (index))
#define UBOUND(index,value) (((index) > (value)) ? (value) : (index))
#define CHARBOUND(value) (((value) > 255) ? 255 :\
			  (((value) < 0) ?  0: (value)))

#define ILBOUND(ptr,index,value) ((index & 1) ? 0 :\
				  (((index) < (value)) ?\
				   ptr[(value)>>1] : ptr[(index)>>1]))
#define IUBOUND(ptr,index,value) ((index & 1) ? 0 :\
				  (((index) > (value)) ?\
				   ptr[(value)>>1] : ptr[(index)>>1]))

#ifndef BEGIN
#define BEGIN(name) static char RoutineName[]= name
#define WHEREAMI() printf("F>%s:R>%s:L>%d: ",\
			  __FILE__,RoutineName,__LINE__)
#define MakeStructure(named_st) ((named_st *) malloc(sizeof(named_st)))
#endif

#ifndef ERROR_NONE
#define ERROR_NONE 0
#define ERROR_BOUNDS 1            /*Input Values out of bounds */
#define ERROR_HUFFMAN_READ 2      /*Huffman Decoder finds bad code */
#define ERROR_HUFFMAN_ENCODE 3    /*Undefined value in encoder */
#define ERROR_MARKER 4            /*Error Found in Marker */
#define ERROR_INIT_FILE 5         /*Cannot initialize files */
#define ERROR_UNRECOVERABLE 6     /*No recovery mode specified */
#define ERROR_PREMATURE_EOF 7     /*End of file unexpected */
#define ERROR_MARKER_STRUCTURE 8  /*Bad Marker Structure */
#define ERROR_WRITE 9             /*Cannot write output */
#define ERROR_READ 10             /*Cannot write input */
#define ERROR_PARAMETER 11        /*System Parameter Error */
#define ERROR_MEMORY 12           /*Memory allocation failure */
#endif

#endif
