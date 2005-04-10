/* File: mpege.h		-*- C -*- 				     */
/* Created by: Alex Knowles (alex@ed.ac.uk) Thu Nov 23 16:20:03 1995	     */
/* Last Modified: Time-stamp: <26 Feb 96 1606 Alex Knowles> 		     */
/* RCS $Id$ */
#ifndef _MPEGE_H
#define _MPEGE_H

#ifdef HAVE_SDSC_IM
/* include the sandiego image tool kit library */
#include "im.h"
#else
/* include my cut down version of im.h only if im.h has not been included yet*/
#ifndef __IMH__
#include "mpege_im.h"
#endif
#endif

#ifndef BOOLEAN_TYPE_EXISTS
typedef char Boolean;
#define BOOLEAN_TYPE_EXISTS
#endif

typedef enum {NO_INIT, INIT_DONE, READING_FRAMES, DONE } MPEGe_state;


/* this is where you can change the behaviour of the MPEG encoder */

typedef struct {
  /* Initialisation control varaibles - you can change these only b4 init */
  int 	gop_size;
  char 	*frame_pattern; 	/* order of the I, b & P frames */
  int	slices_per_frame;
  int	search_range[2];
  int	IQscale;
  int	BQscale;
  int	PQscale;
  int 	bit_rate;		/* -1 for variable or specify your own */
  int	buffer_size;
  enum { HALF, FULL } pixel_search;
  enum { P_EXHAUSTIVE, P_SUBSAMPLE, P_LOGARITHMIC, P_TWOLEVEL } psearchalg;
  enum { B_SIMPLE, B_CROSS2, B_EXHAUSTIVE } bsearchalg;
  
  /* Variables set by the library for your use */
  char error[1024];		/* the error string */
  
  /* private stuff that is needed by the library internals - Ignore! */
  /* none yet! */
  MPEGe_state state; 	/* the state of the library */
  int FrameNumber; 	/* which frame are we encoding! */
  FILE *ofp;		/* the output file pointer */
} MPEGe_options;

#ifdef __cplusplus
extern "C" {
#endif
  extern MPEGe_options *MPEGe_default_options( MPEGe_options * );
  
  extern Boolean MPEGe_open( FILE *, MPEGe_options *);
  extern Boolean MPEGe_image( ImVfb *, MPEGe_options * );
  extern Boolean MPEGe_close( MPEGe_options * );
  
  extern ImVfb *MPEGe_ImVfbAlloc(int width, int height, 
				 int fields, int for_image );
#ifdef __cplusplus
}
#endif


#endif /* _MPEGE_H */
