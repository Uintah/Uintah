/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
** Copyright 1991-1993, Silicon Graphics, Inc.
** All Rights Reserved.
** 
** This is UNPUBLISHED PROPRIETARY SOURCE CODE of Silicon Graphics, Inc.;
** the contents of this file may not be disclosed to third parties, copied or
** duplicated in any form, in whole or in part, without the prior written
** permission of Silicon Graphics, Inc.
** 
** RESTRICTED RIGHTS LEGEND:
** Use, duplication or disclosure by the Government is subject to restrictions
** as set forth in subdivision (c)(1)(ii) of the Rights in Technical Data
** and Computer Software clause at DFARS 252.227-7013, and/or in similar or
** successor clauses in the FAR, DOD or NASA FAR Supplement. Unpublished -
** rights reserved under the Copyright Laws of the United States.
*/

#ifndef __GLXTOKENS
#define __GLXTOKENS

#ifdef __cplusplus
extern "C" {
#endif

/* Visual Config Attributes (XVisualInfo or GLXFBConfigSGIX) */
#define GLX_USE_GL		1	/* support GLX rendering */
#define GLX_BUFFER_SIZE		2	/* depth of the color buffer */
#define GLX_LEVEL		3	/* level in plane stacking */
#define GLX_RGBA		4	/* true if RGBA mode */
#define GLX_DOUBLEBUFFER	5	/* double buffering supported */
#define GLX_STEREO		6	/* stereo buffering supported */
#define GLX_AUX_BUFFERS 	7	/* number of aux buffers */
#define GLX_RED_SIZE		8	/* number of red component bits */
#define GLX_GREEN_SIZE		9	/* number of green component bits */
#define GLX_BLUE_SIZE		10	/* number of blue component bits */
#define GLX_ALPHA_SIZE		11	/* number of alpha component bits */
#define GLX_DEPTH_SIZE		12	/* number of depth bits */
#define GLX_STENCIL_SIZE	13	/* number of stencil bits */
#define GLX_ACCUM_RED_SIZE	14	/* number of red accum bits */
#define GLX_ACCUM_GREEN_SIZE	15	/* number of green accum bits */
#define GLX_ACCUM_BLUE_SIZE	16	/* number of blue accum bits */
#define GLX_ACCUM_ALPHA_SIZE	17	/* number of alpha accum bits */
#define GLX_SAMPLES_SGIS		100000	/* number of samples per pixel */
#define GLX_SAMPLE_BUFFERS_SGIS		100001	/* the number of multisample buffers */
#define GLX_VISUAL_CAVEAT_EXT		0x20	/* visual_rating extension type */
#define GLX_X_VISUAL_TYPE_EXT		0x22	/* visual_info extension type */
#define GLX_TRANSPARENT_TYPE_EXT	0x23	/* visual_info extension */
#define GLX_TRANSPARENT_INDEX_VALUE_EXT	0x24	/* visual_info extension */
#define GLX_TRANSPARENT_RED_VALUE_EXT	0x25	/* visual_info extension */
#define GLX_TRANSPARENT_GREEN_VALUE_EXT	0x26	/* visual_info extension */
#define GLX_TRANSPARENT_BLUE_VALUE_EXT	0x27	/* visual_info extension */
#define GLX_TRANSPARENT_ALPHA_VALUE_EXT	0x28	/* visual_info extension */

/* FBConfig Attributes */
#define GLX_DRAWABLE_TYPE_SGIX		0x8010
#define GLX_RENDER_TYPE_SGIX		0x8011
#define GLX_X_RENDERABLE_SGIX		0x8012
#define GLX_FBCONFIG_ID_SGIX		0x8013
#define GLX_MAX_PBUFFER_WIDTH_SGIX	0x8016
#define GLX_MAX_PBUFFER_HEIGHT_SGIX	0x8017
#define GLX_MAX_PBUFFER_PIXELS_SGIX	0x8018
#define GLX_OPTIMAL_PBUFFER_WIDTH_SGIX	0x8019
#define GLX_OPTIMAL_PBUFFER_HEIGHT_SGIX	0x801A

/*
** Error return values from glXGetConfig.  Success is indicated by
** a value of 0.
*/
#define GLX_BAD_SCREEN		1	/* screen # is bad */
#define GLX_BAD_ATTRIBUTE	2	/* attribute to get is bad */
#define GLX_NO_EXTENSION	3	/* no glx extension on server */
#define GLX_BAD_VISUAL		4	/* visual # not known by GLX */
#define GLX_BAD_CONTEXT		5
#define GLX_BAD_VALUE		6
#define GLX_BAD_ENUM		7

/*
** Errors.
*/
#define GLXBadContext           0
#define GLXBadContextState      1
#define GLXBadDrawable          2
#define GLXBadPixmap            3
#define GLXBadContextTag        4
#define GLXBadCurrentWindow     5
#define GLXBadRenderRequest     6
#define GLXBadLargeRequest      7
#define GLXUnsupportedPrivateRequest    8
#define GLXBadFBConfigSGIX      9
#define GLXBadPbufferSGIX       10

#define __GLX_NUMBER_ERRORS 8
#define __GLX_NUMBER_EVENTS 17

/*****************************************************************************/

#define GLX_EXTENSION_NAME      "GLX"

#define GLX_VENDOR		1
#define GLX_VERSION		2
#define GLX_EXTENSIONS		3

/*****************************************************************************/

/*
** GLX Extension Strings
*/
#define GLX_EXT_import_context		1
#define GLX_EXT_visual_info		1
#define GLX_EXT_visual_rating		1
#define GLX_SGI_make_current_read	1
#define GLX_SGI_swap_control		1
#define GLX_SGI_video_sync		1
#define GLX_SGIS_multisample		1
#define GLX_SGIX_fbconfig		1
#define GLX_SGIX_pbuffer		1
#define GLX_SGIX_video_source		1
#define GLX_SGIX_dm_pbuffer		1
#define GLX_SGIX_video_resize		1
#define GLX_SGIX_swap_barrier		1
#define GLX_SGIX_swap_group		1

/* Visual Ratings */
#define GLX_NONE_EXT			0x8000
#define GLX_SLOW_VISUAL_EXT		0x8001
#define GLX_NON_CONFORMANT_VISUAL_EXT	0x800D

/* X Visual Types */
#define GLX_TRUE_COLOR_EXT		0x8002
#define GLX_DIRECT_COLOR_EXT		0x8003
#define GLX_PSEUDO_COLOR_EXT		0x8004
#define GLX_STATIC_COLOR_EXT		0x8005
#define GLX_GRAY_SCALE_EXT		0x8006
#define GLX_STATIC_GRAY_EXT		0x8007

/* Transparent Pixel Types */
/*      GLX_NONE_EXT */
#define GLX_TRANSPARENT_RGB_EXT		0x8008
#define GLX_TRANSPARENT_INDEX_EXT	0x8009

/* Context Info Attributes */
#define GLX_SHARE_CONTEXT_EXT		0x800A	/* id of share context */
#define GLX_VISUAL_ID_EXT		0x800B	/* id of context's visual */
#define GLX_SCREEN_EXT			0x800C	/* screen number */

/* FBConfig Drawable Type Bits */
#define GLX_WINDOW_BIT_SGIX		0x00000001
#define GLX_PIXMAP_BIT_SGIX		0x00000002
#define GLX_PBUFFER_BIT_SGIX		0x00000004

/* FBConfig Render Type Bits */
#define GLX_RGBA_BIT_SGIX		0x00000001
#define GLX_COLOR_INDEX_BIT_SGIX	0x00000002

/* Render Types */
#define GLX_RGBA_TYPE_SGIX		0x8014
#define GLX_COLOR_INDEX_TYPE_SGIX	0x8015

/* Pbuffer Attributes */
#define GLX_PRESERVED_CONTENTS_SGIX	0x801B
#define GLX_LARGEST_PBUFFER_SGIX	0x801C
#define GLX_WIDTH_SGIX			0x801D
#define GLX_HEIGHT_SGIX			0x801E
#define GLX_EVENT_MASK_SGIX		0x801F
#define GLX_DIGITAL_MEDIA_PBUFFER_SGIX	0x8024

/* Event Selection Masks */
#define GLX_BUFFER_CLOBBER_MASK_SGIX	0x08000000

/* Event Classes */
#define GLX_DAMAGED_SGIX 		0x8020
#define GLX_SAVED_SGIX			0x8021

/* Event Drawable Types */
#define GLX_WINDOW_SGIX			0x8022
#define GLX_PBUFFER_SGIX		0x8023

/* Event Buffer Types */
#define GLX_FRONT_LEFT_BUFFER_BIT_SGIX	0x00000001
#define GLX_FRONT_RIGHT_BUFFER_BIT_SGIX	0x00000002
#define GLX_BACK_LEFT_BUFFER_BIT_SGIX	0x00000004
#define GLX_BACK_RIGHT_BUFFER_BIT_SGIX	0x00000008
#define GLX_AUX_BUFFERS_BIT_SGIX	0x00000010
#define GLX_DEPTH_BUFFER_BIT_SGIX	0x00000020
#define GLX_STENCIL_BUFFER_BIT_SGIX	0x00000040
#define GLX_ACCUM_BUFFER_BIT_SGIX	0x00000080
#define GLX_SAMPLE_BUFFERS_BIT_SGIX	0x00000100

/* Event Numbers */
#define GLX_BufferClobberSGIX		16

/* Video Resize */
#define GLX_SYNC_FRAME_SGIX 		0x00000000 
#define GLX_SYNC_SWAP_SGIX 		0x00000001 

#ifdef __cplusplus
}
#endif

#endif /* __GLXTOKENS */
