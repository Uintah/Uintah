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

#ifndef __GLX_glx_h__
#define __GLX_glx_h__

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

#if 0
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xmd.h>
#include <GL/gl.h>
#endif // 0

#include <GL/glxtokens.h>

#if 0

#ifdef __cplusplus
extern "C" {
#endif

#define GLX_VERSION_1_1 1
#define GLX_VERSION_1_2 1
/*
** GLX resources.
*/
typedef XID GLXContextID;
typedef XID GLXPixmap;
typedef XID GLXDrawable;
typedef XID GLXVideoSourceSGIX;
typedef XID GLXFBConfigIDSGIX;
typedef XID GLXPbufferSGIX;

/*
** GLXContext is a pointer to opaque data 
*/
typedef struct __GLXcontextRec *GLXContext;

/*
** GLXFBConfigSGIX is a pointer to opaque data
*/
typedef struct __GLXFBConfigSGIXRec *GLXFBConfigSGIX;

/*
** GLX Events
*/
typedef struct {
    int type;
    unsigned long serial;	/* # of last request processed by server */
    Bool send_event;		/* true if this came for SendEvent request */
    Display *display;		/* display the event was read from */
    GLXDrawable drawable;	/* i.d. of Drawable */
    int event_type;		/* GLX_DAMAGED_SGIX or GLX_SAVED_SGIX */
    int draw_type;		/* GLX_WINDOW_SGIX or GLX_PBUFFER_SGIX */
    unsigned int mask;		/* mask indicating which buffers are affected*/
    int x, y;
    int width, height;
    int count;			/* if nonzero, at least this many more */
} GLXBufferClobberEventSGIX;

typedef union __GLXEvent {
    GLXBufferClobberEventSGIX glxbufferclobber;
    long pad[24];
} GLXEvent;

/************************************************************************/

extern XVisualInfo * glXChooseVisual (Display *dpy, int screen, int *attribList);
extern void glXCopyContext (Display *dpy, GLXContext src, GLXContext dst, GLuint mask);
extern GLXContext glXCreateContext (Display *dpy, XVisualInfo *vis, GLXContext shareList, Bool direct);
extern GLXPixmap glXCreateGLXPixmap (Display *dpy, XVisualInfo *vis, Pixmap pixmap);
extern void glXDestroyContext (Display *dpy, GLXContext ctx);
extern void glXDestroyGLXPixmap (Display *dpy, GLXPixmap pix);
extern int glXGetConfig (Display *dpy, XVisualInfo *vis, int attrib, int *value);
extern GLXContext glXGetCurrentContext (void);
extern GLXDrawable glXGetCurrentDrawable (void);
extern Bool glXIsDirect (Display *dpy, GLXContext ctx);
extern Bool glXMakeCurrent (Display *dpy, GLXDrawable drawable, GLXContext ctx);
extern Bool glXQueryExtension (Display *dpy, int *errorBase, int *eventBase);
extern Bool glXQueryVersion (Display *dpy, int *major, int *minor);
extern void glXSwapBuffers (Display *dpy, GLXDrawable drawable);
extern void glXUseXFont (Font font, int first, int count, int listBase);
extern void glXWaitGL (void);
extern void glXWaitX (void);
extern const char * glXQueryExtensionsString (Display *dpy, int screen);
extern const char * glXGetClientString (Display *dpy, int name);
extern const char * glXQueryServerString (Display *dpy, int screen, int name);

/************************************************************************/

/*
** GLX extensions
*/

/*
** Video Sync extension
*/
extern int glXGetVideoSyncSGI (unsigned int *count);
extern int glXWaitVideoSyncSGI (int divisor, int remainder, unsigned int *count);

/*
** Swap Control extension
*/
extern int glXSwapIntervalSGI (int interval);

/*
** MakeCurrentRead extension
*/
extern Bool glXMakeCurrentReadSGI (Display *dpy, GLXDrawable draw, GLXDrawable read, GLXContext gc);
extern GLXDrawable glXGetCurrentReadDrawableSGI (void);

/*
** Dynamic Channel Resizing extension
**
*/
extern int glXBindChannelToWindowSGIX (Display *dpy, int screen, int channel, Window window);
extern int glXQueryChannelDeltasSGIX (Display *dpy, int screen, int channel, int *dx, int *dy, int *dw, int *dh);
extern int glXChannelRectSGIX (Display *dpy, int screen, int channel, int x, int y, int w, int h);
extern int glXQueryChannelRectSGIX (Display *dpy, int screen, int channel, int *x, int *y, int *w, int *h);
extern int glXChannelRectSyncSGIX (Display *dpy, int screen, int channel, GLenum synctype);

#if defined(_VL_H_)
/*
** Video Source extension
*/
extern GLXVideoSourceSGIX glXCreateGLXVideoSourceSGIX (Display *dpy, int screen, VLServer svr, VLPath path, int nodeClass, VLNode node);
extern void glXDestroyGLXVideoSourceSGIX (Display *dpy, GLXVideoSourceSGIX videosource);
#endif

/* 
** ImportContext extension
*/
extern int glXQueryContextInfoEXT (Display *dpy, GLXContext ctx, int attribute, int *value);
extern Display * glXGetCurrentDisplayEXT (void);
extern Display * glXGetCurrentDisplay (void);
extern GLXContextID glXGetContextIDEXT (const GLXContext gc);
extern GLXContext glXImportContextEXT (Display *dpy, GLXContextID contextID);
extern void glXFreeContextEXT (Display *dpy, GLXContext gc);

/*
** FBConfig (Frame Buffer Configuration) extension
*/
extern int glXGetFBConfigAttribSGIX (Display *dpy, GLXFBConfigSGIX config, int attribute, int *value);
extern GLXFBConfigSGIX * glXChooseFBConfigSGIX (Display *dpy, int screen, int *attrib_list, int *nitems);
extern GLXPixmap glXCreateGLXPixmapWithConfigSGIX (Display *dpy, GLXFBConfigSGIX config, Pixmap pixmap);
extern GLXContext glXCreateContextWithConfigSGIX (Display *dpy,  GLXFBConfigSGIX config, int render_type, GLXContext share_list, Bool direct);
extern XVisualInfo * glXGetVisualFromFBConfigSGIX (Display *dpy, GLXFBConfigSGIX config);
extern GLXFBConfigSGIX glXGetFBConfigFromVisualSGIX (Display *dpy, XVisualInfo *vis);

/*
** Pbuffer (Pixel Buffer) extension
*/
extern GLXPbufferSGIX glXCreateGLXPbufferSGIX (Display *dpy, GLXFBConfigSGIX config, unsigned int width, unsigned int height, int *attrib_list);
extern void glXDestroyGLXPbufferSGIX (Display *dpy, GLXPbufferSGIX pbuf);
extern int glXQueryGLXPbufferSGIX (Display *dpy, GLXPbufferSGIX pbuf, int attribute, unsigned int *value);
extern void glXSelectEventSGIX (Display *dpy, GLXDrawable drawable, unsigned long mask);
extern void glXGetSelectedEventSGIX (Display *dpy, GLXDrawable drawable, unsigned long *mask);

#if defined(_DM_BUFFER_H_)
/*
** Digital Media Pbuffer extension
*/
extern Bool glXAssociateDMPbufferSGIX (Display *dpy, GLXPbufferSGIX pbuffer, DMparams *params, DMbuffer dmbuffer); 
#endif

/*
** Swap Group extension
*/
extern void glXJoinSwapGroupSGIX (Display *dpy, GLXDrawable drawable, GLXDrawable member);

/*
** Swap Barrier extension
*/
extern void glXBindSwapBarrierSGIX (Display *dpy, GLXDrawable drawable, int barrier);
extern Bool glXQueryMaxSwapBarriersSGIX (Display *dpy, int screen, int *max);
/************************************************************************/

#ifdef __cplusplus
}
#endif

#endif // 0

#endif /* !__GLX_glx_h__ */
