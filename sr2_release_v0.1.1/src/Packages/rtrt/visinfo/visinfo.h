/*
 * Copyright (c) 1994 Silicon Graphics, Inc.
 * 
 * Permission to use, copy, modify, distribute, and sell this software and
 * its documentation for any purpose is hereby granted without fee,
 * provided that (i) the above copyright notices and this permission
 * notice appear in all copies of the software and related documentation,
 * and (ii) the name of Silicon Graphics may not be used in any
 * advertising or publicity relating to the software without the specific,
 * prior written permission of Silicon Graphics.
 * 
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * 
 * IN NO EVENT SHALL SILICON GRAPHICS BE LIABLE FOR ANY SPECIAL,
 * INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
 * OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 */

/*****************************************************************************
 * visPixelFormat, visGetGLXVisualInfo - tools for choosing GLX Visuals
 *
 * Usage:
 *
 *	char* criteria;		// e.g. "max rgba, z >= 16, db"
 *	XVisualInfo* vInfo;
 *	int nVInfo;
 *	Display* dpy;
 *	int screen;
 *	int i;
 *
 *	if (!visPixelFormat(criteria)) {
 *		printf("syntax error in pixel format specification\n");
 *		exit(1);
 *		}
 *	vInfo = visGetGLXVisualInfo(dpy, screen, &nVInfo);
 *	for (i = 0; i < nVInfo; ++i)
 *		DoSomethingWith(&vInfo[i]);
 *	free(vInfo);
 *****************************************************************************/

#ifndef __VISINFO_H__
#define __VISINFO_H__


#include <GL/glx.h>


#ifdef __cplusplus
extern "C" {
#endif



XVisualInfo* visGetGLXVisualInfo(Display* dpy, int screen, int* nVInfo);
int visPixelFormat(char* criteria);



#ifdef __cplusplus
}
#endif

#endif /* !__VISINFO_H__ */
