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
globals.h

This is where the global definitions are placed.

************************************************************
*/

#ifndef GLOBAL_DONE
#define GLOBAL_DONE

#include <stdio.h>
#include "mem.h"
#include "system.h"
#include "huffman.h"

#define BLOCKSIZE 64
#define BLOCKWIDTH 8
#define BLOCKHEIGHT 8

#define MAXIMUM_SOURCES 3
#define UMASK 0666  /* Octal */
#define BUFFERSIZE 256 

#define MAXIMUM_FGROUP 256

#define sropen mropen
#define srclose mrclose
#define swopen mwopen
#define swclose mwclose

#define sgetb mgetb
#define sgetv mgetv
#define sputv mputv

#define swtell mwtell
#define srtell mrtell

#define swseek mwseek
#define srseek mrseek

#define READ_IOB 1
#define WRITE_IOB 2

#define P_FORBIDDEN 0
#define P_INTRA 1
#define P_PREDICTED 2
#define P_INTERPOLATED 3
#define P_DCINTRA 4

#define M_DECODER 1

/* Image types */

#define IT_NTSC 0
#define IT_CIF 1
#define IT_QCIF 2

#define HUFFMAN_ESCAPE 0x1bff

/* Can be typedef'ed */
#define IMAGE struct Image_Definition
#define FRAME struct Frame_Definition
#define FSTORE struct FStore_Definition
#define STAT struct Statistics_Definition
#define RATE struct Rate_Control_Definition

#define MUTE 0
#define WHISPER 1
#define TALK 2
#define NOISY 3
#define SCREAM 4

/* Memory locations */

#define L_SQUANT 1
#define L_MQUANT 2
#define L_PTYPE 3
#define L_MTYPE 4
#define L_BD 5
#define L_FDBD 6
#define L_BDBD 7
#define L_IDBD 8
#define L_VAROR 9
#define L_FVAR 10
#define L_BVAR 11
#define L_IVAR 12
#define L_DVAR 13
#define L_RATE 14
#define L_BUFFERSIZE 15
#define L_BUFFERCONTENTS 16
#define L_QDFACT 17
#define L_QOFFS 18

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

typedef int iFunc();
typedef void vFunc();

#define GetFlag(value,flag) (((value) & (flag)) ? 1:0)

#define MAX(x,y) ((x > y) ? x:y)
#define MIN(x,y) ((x > y) ? y:x)
#define BEGIN(name) static char RoutineName[]= name
#define WHEREAMI() printf("F>%s:R>%s:L>%d: ",\
			  __FILE__,RoutineName,__LINE__)
/* InBounds is used to test whether a value is in or out of bounds. */
#define InBounds(var,lo,hi,str)\
{if (((var) < (lo)) || ((var) > (hi)))\
{WHEREAMI(); printf("%s in %d\n",(str),(var));ErrorValue=ERROR_BOUNDS;}}
#define BoundValue(var,lo,hi,str)\
{if((var)<(lo)){WHEREAMI();printf("Bounding %s to %d\n",str,lo);var=lo;}\
  else if((var)>(hi)){WHEREAMI();printf("Bounding %s to %d\n",str,hi);var=hi;}}

#define MakeStructure(named_st) ((named_st *) malloc(sizeof(named_st)))

IMAGE {
char *StreamFileName;
int PartialFrame;
int MpegMode;
int Height;
int Width;
};

FRAME {
int NumberComponents;
char ComponentFilePrefix[MAXIMUM_SOURCES][200];
char ComponentFileSuffix[MAXIMUM_SOURCES][200];
char ComponentFileName[MAXIMUM_SOURCES][200];
int PHeight[MAXIMUM_SOURCES];
int PWidth[MAXIMUM_SOURCES];
int Height[MAXIMUM_SOURCES];
int Width[MAXIMUM_SOURCES];
int hf[MAXIMUM_SOURCES];
int vf[MAXIMUM_SOURCES];
};

FSTORE {
int NumberComponents;
IOBUF *Iob[MAXIMUM_SOURCES];
};

STAT {
double mean;
double mse;
double mrsnr;
double snr;
double psnr;
double entropy;
};

RATE {
int position;
int size;
int baseq;
};

#include "prototypes.h"

#endif
