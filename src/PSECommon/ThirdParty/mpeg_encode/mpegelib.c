#line 1 "arklib.c" /* automagicly created from arklib.c*/
/* File: arklib.c		-*- C -*- 				     */
/* Created by: Alex Knowles (alex@ed.ac.uk) Thu Nov 23 22:44:44 1995	     */
/* Last Modified: Time-stamp: <27 Jul 98 1147 Alex Knowles> 		     */
/* RCS $Id$ */
char *arklib_version( void );
static char rcsid[]=
"$Id$";
char *arklib_version( void )
{ return rcsid; }

/* (C)Copyright 1995/1996, Alex Knowles                                 */
/*                                                                      */
/* This program, its source and its documentation is produced by        */
/* Alex Knowles.                                                        */
/*                                                                      */
/* This program has been included for  its instructional value. It has  */
/* been  tested with  care but  is not guaranteed   for any particular  */
/* purpose.  Neither the author, nor anyone  associated with him offer  */
/* any   warranties  or  representations,  nor  do   they accept   any  */
/* liabilities with respect to this program.                            */
/*                                                                      */
/* This program must not be used for commercial gain without the        */
/* written permission of the author(s).                                 */
/*                                                                      */
/* For more information please contact: Alex Knowles (alex@ed.ac.uk)    */

/*  documentation from  http://www.tardis.ed.ac.uk/~ark/mpegelib/	*/
/*  latest version from ftp://ftp.tardis.ed.ac.uk/users/ark/mpegelib/	*/

/*  This file takes a lot of stuff from main.c and mpeg.c	 */

/*==============*
 * HEADER FILES *
 *==============*/

#include <assert.h>
#include "all.h"
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include "mtypes.h"
#include "mpeg.h"
#include "search.h"
#include "prototypes.h"
#include "param.h"
#include "parallel.h"
#include "readframe.h"
#include "combine.h"
#include "frames.h"
#include "jpeg.h"
#include "specifics.h"
#include "opts.h"
#include "fsize.h"
#include "mheaders.h"
#include "rate.h"

#ifdef MIPS
#include <sys/types.h>
#endif
#include <sys/stat.h>

/* the library header file */
#include "mpege.h"

/*============*
 *  VERSIONs  *
 *============*/
/* this is the version of mpeg_encode we're using. it's updated by setversion*/
#define VERSION "1.5b"

/* this is the version of the library we're using */
#define LIB_VERSION "0.2"

/*===========*
 * CONSTANTS *
 *===========*/

#define	FPS_30	    0x5 /* from MPEG standard sect. 2.4.3.2 */
#define ASPECT_1    0x1	/* aspect ratio, from MPEG standard sect. 2.4.3.2 */

/* ark: this is stolen from rate.c as we need if for our options. we should */
/* make sure that with later releases that it is the same! */
#define DEFAULT_BUFFER_SIZE 327680    /* maximun for "constrained" bitstream */

/*==================*
 * GLOBAL VARIABLES *
 *==================*/
extern int  yuvWidth, yuvHeight;

/*==================*
 * STATIC VARIABLES *
 *==================*/

static int	frameStart = -1;

/*==================*
 * GLOBAL VARIABLES *
 *==================*/
/* these are stolen from main.c which isn't included in the library! */
extern int  IOtime;
int	whichGOP 	= -1;
boolean	childProcess 	= FALSE;
boolean	ioServer 	= FALSE;
boolean	outputServer 	= FALSE;
boolean	decodeServer 	= FALSE;
int	quietTime 	= 0;
boolean realQuiet 	= TRUE;
boolean	frameSummary 	= TRUE;
boolean debugSockets 	= FALSE;
boolean debugMachines 	= FALSE;
boolean showBitRatePerFrame 	= FALSE;
boolean	computeMVHist 	= FALSE;
int 	baseFormat;

extern  boolean specificsOn;
extern  FrameSpecList *fsl;
boolean pureDCT		=FALSE;
char    encoder_name[1024]="Berkely encoder library via alex@ed.ac.uk";

extern int	    gopSize;  /* default */
extern int32	    tc_hrs, tc_min, tc_sec, tc_pict, tc_extra;
extern int	    totalFramesSent;
extern int	    realWidth, realHeight;
extern char	    currentPath[MAXPATHLEN];
extern char	    statFileName[256];
extern char	    bitRateFileName[256];
extern time_t	    timeStart, timeEnd;
extern FILE	   *statFile;
extern FILE	   *bitRateFile;
extern char	   *framePattern;
extern int	    framePatternLen;
extern int	    referenceFrame;

static int  framesRead;

extern MpegFrame  *pastRefFrame;
extern MpegFrame  *futureRefFrame;
extern int	    frameRate;
extern int	    frameRateRounded;
extern boolean	    frameRateInteger;
extern int	    aspectRatio;
extern unsigned char userDataFileName[];
extern int mult_seq_headers;

extern int32 bit_rate, buf_size;
extern boolean stdinUsed;

/*==================*
 * Variables from mpeg.c
 *==================*/

extern int32	diffTime;
extern int 	framesOutput;
extern int	realStart, realEnd;	
extern int	currentGOP;
extern int	timeMask;
extern int	numI, numP, numB;

/* my stuff folows below */
/* my global variables! (used to be in function GenMPEGstream)*/
int frameType;
BitBucket *bb=NULL;
MpegFrame *frame = NULL;
MpegFrame *tempFrame;
int     firstFrame, lastFrame;
int     inputFrameBits = 0;
char    inputFileName[1024];
time_t  tempTimeStart, tempTimeEnd;
boolean firstFrameDone = FALSE;
int numBits;
int32 bitstreamMode, res;

/* moved from the header of genmpeg stream */
extern int32   qtable[];
extern int32   niqtable[];
int numFrames;

/* stuff added by ark for the library */
static char itoa_buf[1024];	/* a buffer for my into to ascii function */

/* ARK's function prototypes */
static void ARKReadFrame( MpegFrame *, ImVfb * );
static void MPEGe_init( int width, int height, MPEGe_options * );
static char *itoa( int i );

/*===============================*
 * INTERNAL PROCEDURE prototypes *
 *===============================*/

extern void 	Tune_Init( void );
extern void	ProcessRefFrame _ANSI_ARGS_((MpegFrame *frame,
					      BitBucket *buck, int lastFrame,
					      char *outputFileName));

MPEGe_options *MPEGe_default_options( MPEGe_options *o )
{
  if( !o ){
    o=(MPEGe_options *) malloc( sizeof( MPEGe_options ) );
    assert(o);
  }
  
  /* now fill up the structure with the default options */
  o->gop_size=30;
  o->frame_pattern=(char *)strdup("IBBPBBPBBPBBPBB");
  o->slices_per_frame=1;
  o->search_range[0]=10;
  o->search_range[1]=10;
  o->IQscale=8;
  o->BQscale=10;
  o->PQscale=25;
  o->pixel_search=HALF;
  o->psearchalg=P_LOGARITHMIC;
  o->bsearchalg=B_CROSS2;
  o->bit_rate=-1; /* variable bit rate */
  o->buffer_size=DEFAULT_BUFFER_SIZE;

  o->state=NO_INIT;
  o->FrameNumber=0;
  
  return o;
}

char *itoa( int i )
{
  sprintf(itoa_buf, "%d", i );
  return itoa_buf;
}

Boolean MPEGe_open( FILE *outFile, MPEGe_options *options )
{
  if( !options ){
    fprintf(stderr,"MPEGe_open: NULL options\n");
    return FALSE;
  }
  
  if( !outFile ){
    sprintf(options->error,"MPEGe_open: called with NULL FILE pointer");
    return FALSE;
  }
  
  if( options->state != NO_INIT ){
    sprintf(options->error,"MPEGe_open: Library not in correct state");
    return FALSE;
  }    
  
  SetStatFileName("");
  
  /* all this stuff is from read_param.c */
  stdinUsed=TRUE;
  
  /* now to simulate reading a default param file which will be very simple! */
  SetGOPSize( options->gop_size );
  SetSlicesPerFrame( options->slices_per_frame );
  SetSearchRange(options->search_range[0],
		 options->search_range[1]);
  SetIQScale(options->IQscale);
  SetBQScale(options->BQscale);
  SetPQScale(options->PQscale);
  
  SetFramePattern(options->frame_pattern);				

  switch( options->pixel_search ){
  case HALF:    SetPixelSearch("HALF");    break;
  case FULL:    SetPixelSearch("FULL");    break;
  }
  switch( options->psearchalg ){
  case P_LOGARITHMIC:	SetPSearchAlg("LOGARITHMIC");	break;
  case P_SUBSAMPLE:	SetPSearchAlg("SUBSAMPLE");	break;
  case P_EXHAUSTIVE:	SetPSearchAlg("EXHAUSTIVE");	break;
  case P_TWOLEVEL:	SetPSearchAlg("TWOLEVEL");	break;
  }
  switch( options->bsearchalg ){
  case B_CROSS2:	SetBSearchAlg("CROSS2");	break;
  case B_SIMPLE:	SetBSearchAlg("SIMPLE");	break;
  case B_EXHAUSTIVE:	SetBSearchAlg("EXHAUSTIVE");	break;
  }
  
  setBufferSize(itoa(options->buffer_size));
  if( options->bit_rate >0 ){
    setBitRate(itoa(options->bit_rate));
  }
  
  SetReferenceFrameType("ORIGINAL");	
  
  options->ofp=outFile;
  
  options->state=INIT_DONE;
  return TRUE;
}


void MPEGe_init( int width, int height, MPEGe_options *options )
{
  int i;
  
  /* this stuff is taken from param.c */
  
  /* this needs fixing badly! (well fix it well, but it needs it badly) */
  realWidth = yuvWidth = width;
  realHeight = yuvHeight = height;
  
  /* make sure yuv* is devisable by 16 (round down) */
  Fsize_Validate(&yuvWidth, &yuvHeight);
  
  SetFCode();
  
  time(&timeStart);
  
  numMachines = 0;

  /* back to main.c now */
  
  Tune_Init();
  Frame_Init();
  ComputeFrameTable();
  
  /* now we move into the domain of mpeg.c */
  
  framesRead = 0;
  
  ResetIFrameStats();
  ResetPFrameStats();
  ResetBFrameStats();
  
  Fsize_Reset();
  
  framesOutput = 0;
  
  /* cos ! childprocess  */
  SetFileType(inputConversion);
  
  numFrames=1000;
  
  firstFrame = 0;
  lastFrame = numFrames-1;
  
  realStart = 0;
  realEnd = numFrames-1;

  /* count number of I, P, and B frames */
  numI = 0;        numP = 0;   numB = 0;
  timeMask = 0;
  
  /* we don't really know how many there will be so use this */
  numI = numP = numB = MAXINT/4;
  
  /* open the bit bucket for output */
  bb = Bitio_New(options->ofp);
  
  tc_hrs = 0;        tc_min = 0; tc_sec = 0; tc_pict = 0; tc_extra = 0;
  for ( i = 0; i < firstFrame; i++ ) {
    IncrementTCTime();
  }

  totalFramesSent = firstFrame;
  currentGOP = gopSize;        /* so first I-frame generates GOP Header */

  /* Rate Control Initialization  */
  bitstreamMode = getRateMode();
  if (bitstreamMode == FIXED_RATE) {
    res = initRateControl();
    /* SetFrameRate(); */
  }
  pastRefFrame = NULL;
  futureRefFrame = NULL;
}  

Boolean MPEGe_image( ImVfb  *image, MPEGe_options *options )
{
  int i; /* i used to be used in the big for loop in mpeg.c */
  
  /* check the parameters */
  if( !options ){
    fprintf(stderr,"MPEGe_image: NULL options\n");
    return FALSE;
  }
  
  if( !image ){
    sprintf(options->error,"MPEGe_image: NULL VFB pointer");
    return FALSE;
  }    
  
  if( options->state==INIT_DONE ){
    MPEGe_init( ImVfbQWidth( image ), ImVfbQHeight( image ), options );
    options->state=READING_FRAMES;
  }
  
  if( options->state != READING_FRAMES ){
    sprintf(options->error,
	    "MPEGe_image: Library in wrong state did you call MPEGe_open?");
    return FALSE;
  }
  
  if(ImVfbQWidth(image)  != realWidth  ||
     ImVfbQHeight(image) != realHeight ){
    sprintf(options->error,
	    "MPEGe_image: Image is wrong size expected %dx%d got %dx%d",
	    realWidth,realHeight, ImVfbQWidth( image ), ImVfbQHeight(image));
    return FALSE;
  }    
  
  i=options->FrameNumber++;
  frameType = FType_Type(i);
  
  time(&tempTimeStart);

  /*  read in non-reference frames if interactive	 */
  /* if it's a b frame then it's buffered  it will be used later! */
  if ( frameType == 'b' ) {
    frame = Frame_New(i, frameType);
    ARKReadFrame(frame, image );
    framesRead++;
      
    time(&tempTimeEnd);
    IOtime += (tempTimeEnd-tempTimeStart);
      
    /* Add the B frame to the end of the queue of B-frames 
     * for later encoding */
    
    if (futureRefFrame) {
      tempFrame = futureRefFrame;
      while (tempFrame->next != NULL) {
	tempFrame = tempFrame->next;
      }
      tempFrame->next = frame;
    } else {
      fprintf(stderr, "Yow, something wrong in neverland!"
	      "(hit bad code in mpeg.c\n");
      exit(2);
    }
    return TRUE;
  }

  frame = Frame_New(i, frameType);
  
  pastRefFrame = futureRefFrame;
  futureRefFrame = frame;

  ARKReadFrame(frame, image );
  
  framesRead++;
  
  time(&tempTimeEnd);
  IOtime += (tempTimeEnd-tempTimeStart);
  
  /* this is the bit which creates the header! */
  if ( ! firstFrameDone ) {
    char *userData = (char *)NULL;
    int userDataSize = 0;
    
    inputFrameBits = 24*Fsize_x*Fsize_y;
    SetBlocksPerSlice();
    
    if ( (whichGOP == -1) && (frameStart == -1) ) {
      DBG_PRINT(("Generating sequence header\n"));
      bitstreamMode = getRateMode();
      if (bitstreamMode == FIXED_RATE) {
	bit_rate = getBitRate();
	buf_size = getBufferSize();
      }
      else {
	bit_rate = -1;
	buf_size = -1;
      }
      
      if (strlen((char *)userDataFileName) != 0) {
	struct stat statbuf;
	FILE *fp;
	
	stat((char *)userDataFileName,&statbuf);
	userDataSize = statbuf.st_size;
	userData = malloc(userDataSize);
	if ((fp = fopen((char *)userDataFileName,"rb")) == NULL) {
	  fprintf(stderr,"Could not open userdata file-%s.\n",
		  userDataFileName);
	  userData = NULL;
	  userDataSize = 0;
	  goto write;
	}
	if (fread(userData,1,userDataSize,fp) != userDataSize) {
	  fprintf(stderr,"Could not read %d bytes from userdata file-%s.\n",
		  userDataSize,userDataFileName);
	  userData = NULL;
	  userDataSize = 0;
	  goto write;
	}
      } else { /* Put in our UserData Header */
	time_t now;
	
	time(&now);
	userData = malloc(1024);
	sprintf(userData,
		"MPEG stream encoded by MPEGelib (v%s) using UCB Encoder (v%s) on %s.",
		LIB_VERSION,VERSION, ctime(&now));
	userDataSize = strlen(userData);
      }
    write:
      Mhead_GenSequenceHeader(bb, Fsize_x, Fsize_y,
			      /* pratio */ aspectRatio,
			      /* pict_rate */ frameRate, 
			      /* bit_rate */ bit_rate,
			      /* buf_size */ buf_size,
			      /*c_param_flag */ 1,
			      /* iq_matrix */ qtable,
			      /* niq_matrix */ niqtable,
			      /* ext_data */ NULL,
			      /* ext_data_size */ 0,
			      /* user_data */ (uint8 *) userData,
			      /* user_data_size */ userDataSize);
    }
    
    firstFrameDone = TRUE;
  }
  ProcessRefFrame(frame, bb, lastFrame, outputFileName);
  
  return TRUE;
}

Boolean MPEGe_close( MPEGe_options *options )
{
  if( !options ){
    fprintf(stderr,"MPEGe_close: NULL options\n");
    return FALSE;
  }
  
  if( options->state != READING_FRAMES ){
    sprintf(options->error,"MPEGe_close: Library not read in any frames yet!");
    return FALSE;
  }
  
  /* SEQUENCE END CODE */
  Mhead_GenSequenceEnder(bb);
  numBits = bb->cumulativeBits;
  
  Bitio_Flush(bb);
  bb = NULL;
  fclose(options->ofp);
    
  time(&timeEnd);
  diffTime = (int32)(timeEnd-timeStart);
  
  /* free up the frames */
  Frame_Exit();
  
  /* now reset all the gloabl variables to make me do this again */
  frameStart=-1;
  whichGOP = -1;
  firstFrameDone = FALSE;
  
  options->state=DONE;
  
  return TRUE;
}
  
/*=====================*
 * INTERNAL PROCEDURES *
 *=====================*/

/* gives Y luminance value in range 0->255	*/
#define Yval(p)	(0.2989*((float)ImVfbQRed(  vfb,p))+	\
		 0.5866*((float)ImVfbQGreen(vfb,p))+	\
		 0.1144*((float)ImVfbQBlue( vfb,p)) )

/* gives U -- blue chroma value in range -127->128	*/
#define Uval(p) (0.493*(0.8856*((float)ImVfbQBlue( vfb,p))-	\
		        0.5866*((float)ImVfbQGreen(vfb,p))-	\
		        0.2989*((float)ImVfbQRed(  vfb,p))))

/* gives V -- red chroma value in range -127->128	*/
#define Vval(p)	(0.877*(0.7011*((float)ImVfbQRed(  vfb,p))-	\
			0.5866*((float)ImVfbQGreen(vfb,p))-	\
			0.1144*((float)ImVfbQBlue( vfb,p))))

#define MIN(a,b)	((a)>(b) ? (b) : (a))	/* minimum of a and b        */
#define MAX(a,b)	((a)>(b) ? (a) : (b))	/* maximum of a and b        */
#define CORRECT(x)	MIN( MAX((x),0), 255)	/* limits x to range 0->255  */

/* loads the image into the mpeg frame! */
void ARKReadFrame( MpegFrame *frame, ImVfb *vfb )
{
  int 		x,y;
  float		U,V;
  ImVfbPtr vptr1,vptr2;
  
  Fsize_Note(frame->id, yuvWidth, yuvHeight);
  
  Frame_AllocYCC(frame);
  
  /* fill up the frame here! */
  for( y=0 ; y<yuvHeight ; y++ ){
    vptr1=ImVfbQPtr( vfb, 0,y );
    for( x=0 ; x<yuvWidth ; x++ ){
      frame->orig_y[y][x]=(uint8 )Yval( vptr1 );
      vptr1=ImVfbQNext(vfb, vptr1);	
    }
  }
  
  for( y=0 ; y<yuvHeight ; y+=2 ){
    vptr1=ImVfbQPtr( vfb, 0,y );
    vptr2=ImVfbQPtr( vfb, 0,y+1 );
    for( x=0 ; x<yuvWidth ; x+=2 ){
      U=Uval(vptr1);
      V=Vval(vptr1);  vptr1=ImVfbQNext(vfb, vptr1);
      U+=Uval(vptr1);
      V+=Vval(vptr1); vptr1=ImVfbQNext(vfb, vptr1);
      U+=Uval(vptr2);
      V+=Vval(vptr2); vptr2=ImVfbQNext(vfb, vptr2);
      U+=Uval(vptr2);
      V+=Vval(vptr2); vptr2=ImVfbQNext(vfb, vptr2);
      frame->orig_cb[y/2][x/2]=CORRECT((U/4.0)+128.0);
      frame->orig_cr[y/2][x/2]=CORRECT((V/4.0)+128.0);
    }
  }
  
  MotionSearchPreComputation(frame);
}  

/* these were taken from imvfb.c in SDSC Image Library */
/* sizes of various variable types:					*/
#define IM_SIZEOFBYTE	sizeof( unsigned char )
#define IM_SIZEOFSHORT	sizeof( unsigned short )
#define IM_SIZEOFINT	sizeof( unsigned int )
#define IM_SIZEOFFLOAT	sizeof( float )

/* ways of aligning to various types:					*/
#define IM_ALIGNBYTE(n)
#define IM_ALIGNINT(n)	( n += ( (m=(n%IM_SIZEOFINT) ) == 0 ? 0 : IM_SIZEOFINT-m ) )
#define IM_ALIGNFLOAT(n)	( n += ( (m=(n%IM_SIZEOFFLOAT)) == 0 ? 0 : IM_SIZEOFFLOAT-m ))

/*  This is a hacked up version of ImVfbAlloc provided with SDSC Image 	*/
/*  Library. They have kindly given me permission to provide you with 	*/
/*  this function. however I also recommend that you get a copy of SDSC */
/*  Image Library as it's great!	 				*/
/*  Arguments:-	 							*/
/*  width	the width of the image	 				*/
/*  height	the height of the image	 				*/
/*  fields	what you're storing in the inage (usually IMVFBRGB)	*/
/*  for_image    do we allocate memory for the image	 		*/
ImVfb *MPEGe_ImVfbAlloc( int width, int height, int fields, int for_image )
{
  ImVfb *v;	/* ImVfb allocated			*/
  int nbytes;	/* # bytes per pixel			*/
  ImVfbPtr p;	/* pointer to the frame buffer		*/
  int m;	/* mod( nbytes, sizeof(int) )		*/
  int allbytes;	/* TRUE if only allocating byte data	*/
  
  /* check if have been asked to allocate anything:		*/
  
  if ( fields == 0 ){
    /*  ImErrNo = IMEFIELD;	 */
    return ( IMVFBNULL );
  }

  /* initialize:							*/
  
  allbytes = TRUE;
  nbytes = 0;
  
  v = (ImVfb *) malloc( sizeof(ImVfb) );
  if ( v == NULL )
  {
    /*  ImErrNo = IMEMALLOC;	 */
    return ( IMVFBNULL );
  }


  if ( fields & IMVFBRED )    fields = (fields & ~IMVFBRED)   | IMVFBRGB;
  if ( fields & IMVFBGREEN )  fields = (fields & ~IMVFBGREEN) | IMVFBRGB;
  if ( fields & IMVFBBLUE )   fields = (fields & ~IMVFBBLUE)  | IMVFBRGB;
  
  v->vfb_width  = width;
  v->vfb_height = height;
  v->vfb_fields = fields;
  
  v->vfb_clt      = IMCLTNULL;
  
  v->vfb_roff     = -1;
  v->vfb_goff	= -1;
  v->vfb_boff	= -1;
  v->vfb_i8off    = -1;
  v->vfb_i16off	= -1;
  v->vfb_aoff     = -1;
  v->vfb_wpoff	= -1;
  v->vfb_zoff	= -1;
  v->vfb_moff	= -1;
  v->vfb_fpoff    = -1;
  v->vfb_ioff	= -1;
  
  
  /* check each fields possibility:				*/
  
  if  ( fields & IMVFBRGB ){
    IM_ALIGNBYTE( nbytes );
    v->vfb_roff = nbytes;
    nbytes += IM_SIZEOFBYTE;
    
    IM_ALIGNBYTE( nbytes );
    v->vfb_goff = nbytes;
    nbytes += IM_SIZEOFBYTE;
    
    IM_ALIGNBYTE( nbytes );
    v->vfb_boff = nbytes;
    nbytes += IM_SIZEOFBYTE;
  }
  
  if ( fields & IMVFBALPHA ){
    IM_ALIGNBYTE( nbytes );
    v->vfb_aoff = nbytes;
    nbytes += IM_SIZEOFBYTE;
  }

  if ( fields & IMVFBINDEX8 ){
    IM_ALIGNBYTE( nbytes );
    v->vfb_i8off = nbytes;
    nbytes += IM_SIZEOFBYTE;
  }

  if ( fields & IMVFBMONO ){
    /* Use 1 byte for each mono value.			*/
    IM_ALIGNBYTE( nbytes );
    v->vfb_moff = nbytes;
    nbytes += IM_SIZEOFBYTE;
  }

  if ( fields & IMVFBWPROT ){
    IM_ALIGNBYTE( nbytes );
    v->vfb_wpoff = nbytes;
    nbytes += IM_SIZEOFBYTE;
  }
  
  if ( fields & IMVFBINDEX16 ){
    IM_ALIGNINT( nbytes );
    v->vfb_i16off = nbytes;
    nbytes += IM_SIZEOFSHORT;
    allbytes = FALSE;
  }
  
  if ( fields & IMVFBZ ){
    IM_ALIGNINT( nbytes );
    v->vfb_zoff = nbytes;
    nbytes += IM_SIZEOFINT;
    allbytes = FALSE;
  }

  if ( fields & IMVFBFDATA ){
    IM_ALIGNFLOAT( nbytes );
    v->vfb_fpoff = nbytes;
    nbytes += IM_SIZEOFFLOAT;
    allbytes = FALSE;
  }

  if ( fields & IMVFBIDATA ){
    IM_ALIGNINT( nbytes );
    v->vfb_ioff = nbytes;
    nbytes += IM_SIZEOFINT;
    allbytes = FALSE;
  }
  /* pad so produces an even integer size, if necessary:		*/
  
  if ( ! allbytes )    IM_ALIGNINT( nbytes );
  
  v->vfb_nbytes = nbytes;
  
  if( for_image ){
    /* allocate the necessary frame buffer:				*/
    
    p = (ImVfbPtr) malloc( width*height*nbytes );
    if ( p == (ImVfbPtr)NULL )
    {
      free( (char *) v );
      /*  ImErrNo = IMEMALLOC;	 */
      return ( IMVFBNULL );
    }
    v->vfb_pfirst = p;
    v->vfb_plast  = p + ( nbytes * width * height )  -  nbytes;
  } else {
    v->vfb_pfirst = NULL;
    v->vfb_plast  = NULL;
  }
  
  
  /* done: return the pointer to the ImVfb structure:		*/
  
  return( v );
}
