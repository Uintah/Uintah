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



#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "visinfo.h"

char* AppName;


void DescribeVisual(Display* dpy, XVisualInfo* v, int verbose, FILE* f);
void Fatal(char* format, ...);
void Usage(void);


void
main(int argc, char** argv) {
	XVisualInfo* v;
	int i;
	int n;
	Display* dpy;
	char description[512];
	char* displayName;
	int showAll;
	int verbose;

	AppName = argv[0];

	/* Define default option values: */
	displayName = NULL;
	showAll = 1;
	verbose = 1;
	description[0] = '\0';

	/* Process command-line options, if any: */
	for (i = 1; i < argc; ++i)
		if (strcmp(argv[i], "-1") == 0)
			showAll = 0;
		else if (strcmp(argv[i], "-display") == 0) {
			if (i == argc - 1)
				Usage();
			displayName = argv[++i];
			}
		else if (strcmp(argv[i], "-help") == 0)
			Usage();
		else if (strcmp(argv[i], "-id") == 0)
			verbose = 0;
		else {
			/*
			 * Concatenate all non-option arguments to form
			 * a single description string.  Separate args
			 * with commas, which is usually the right thing
			 * to do for visPixelFormat().
			 */
			if (description[0] != '\0')
				strcat(description, ",");
			strcat(description, argv[i]);
			}

	/* By default, display all visuals: */
	if (description[0] == '\0')
		strcat(description, "1");

	/* Parse the description expression and set up for filtering Visuals: */
	if (!visPixelFormat(description))
		Usage();

	/* Open the X11 display and fetch the appropriate Visuals: */
	dpy = XOpenDisplay(displayName);
	if (!dpy)
		Fatal("can't open display %s\n", displayName);
	v = visGetGLXVisualInfo(dpy, 0, &n);
	if (!v)
		Fatal("NULL visual array\n");

	/* Print short descriptions of (one or more) acceptable Visuals: */
	for (i = 0; i < n; ++i) {
		DescribeVisual(dpy, v + i, verbose, stdout);
		if (!showAll)
			break;
		}

	/* Clean up and quit. Exit with 0 if a Visual was found, 1 otherwise. */
	free(v);
	XCloseDisplay(dpy);
	exit(n == 0);
	}



void
DescribeVisual(Display* dpy, XVisualInfo* v, int verbose, FILE* f) {
	int value;
	int value2;
	int value3;
	int value4;

	fprintf(f, "0x%x", v->visualid);

	if (!verbose) {
		fprintf(f, "\n");
		return;
		}

	glXGetConfig(dpy, v, GLX_LEVEL, &value);
	if (value < 0)		fprintf(f, ", underlay");
	else if (value > 0)	fprintf(f, ", overlay");

	glXGetConfig(dpy, v, GLX_RGBA, &value);
	if (value) {
		fprintf(f, ", RGBA ");
		glXGetConfig(dpy, v, GLX_RED_SIZE, &value);
		glXGetConfig(dpy, v, GLX_GREEN_SIZE, &value2);
		glXGetConfig(dpy, v, GLX_BLUE_SIZE, &value3);
		glXGetConfig(dpy, v, GLX_ALPHA_SIZE, &value4);
		fprintf(f, "%d/%d/%d/%d", value, value2, value3, value4);
		}
	else {
		glXGetConfig(dpy, v, GLX_BUFFER_SIZE, &value);
		fprintf(f, ", CI %d", value);
		}

	glXGetConfig(dpy, v, GLX_DOUBLEBUFFER, &value);
	if (value)	fprintf(f, ", db");

	glXGetConfig(dpy, v, GLX_STEREO, &value);
	if (value)	fprintf(f, ", stereo");

	glXGetConfig(dpy, v, GLX_AUX_BUFFERS, &value);
	if (value)	fprintf(f, ", aux %d", value);

	glXGetConfig(dpy, v, GLX_DEPTH_SIZE, &value);
	if (value)	fprintf(f, ", Z %d", value);

	glXGetConfig(dpy, v, GLX_STENCIL_SIZE, &value);
	if (value)	fprintf(f, ", S %d", value);

	glXGetConfig(dpy, v, GLX_ACCUM_RED_SIZE, &value);
	glXGetConfig(dpy, v, GLX_ACCUM_GREEN_SIZE, &value2);
	glXGetConfig(dpy, v, GLX_ACCUM_BLUE_SIZE, &value3);
	glXGetConfig(dpy, v, GLX_ACCUM_ALPHA_SIZE, &value4);
	if (value || value2 || value3 || value4)
		fprintf(f,", accum %d/%d/%d/%d", value, value2, value3, value4);

#if defined(GL_SGIS_multisample) && defined(__sgi)
	glXGetConfig(dpy, v, GLX_SAMPLES_SGIS, &value);
	if (value)	fprintf(f, ", samples %d", value);
#endif

	fprintf(f, "\n");
	}



/******************************************************************************
 * Fatal
 *	Print an error message, then exit with status 2.
 ******************************************************************************/

void
Fatal(char* format, ...) {
	va_list args;

	fprintf(stderr, "%s: ", AppName);

	va_start(args, format);
	vfprintf(stderr, format, args);
	va_end(args);

	exit(2);
	}



/******************************************************************************
 * Usage
 *	Print a command usage message, then exit with status 2.
 ******************************************************************************/

void
Usage(void) {
	fprintf(stderr, "Usage: %s options visual-descriptions\n", AppName);
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "	-display <X11-display-name>\n");
	fprintf(stderr, "	-id	(display visual id only)\n");
	fprintf(stderr, "	-1	(display just the first matching visual)\n");
	fprintf(stderr, "	-help\n");
	fprintf(stderr, "Visual-Descriptions: comma-separated lists of expressions and sort keys.\n");
	fprintf(stderr, "All expressions must evalute to nonzero for a visual to be selected.\n");
	fprintf(stderr, "Selected visuals will be reported in the order specified by the sort keys.\n");
	fprintf(stderr, "Expression: a C expression involving these operators\n");
	fprintf(stderr, "	|| && < <= > >= == != + - * / %% ! ( )\n");
	fprintf(stderr, "also integer constants, and these variables\n");
	fprintf(stderr, "	r g b a rgb rgba ci\n");
	fprintf(stderr, "	accumr accumg accumb accuma accumrgb accumrgba\n");
	fprintf(stderr, "	z s ms\n");
	fprintf(stderr, "	id level overlay main underlay\n");
	fprintf(stderr, "	sb db mono stereo aux\n");
	fprintf(stderr, "Sort Key: min or max followed by one of the above variables\n");
	fprintf(stderr, "Examples:\n");
	fprintf(stderr, "	%s \"max rgb, db, z\"\n", AppName);
	fprintf(stderr, "	%s \"ci == 8 && z >= 16\"\n", AppName);
	fprintf(stderr, "	%s rgb overlay\n", AppName);
	exit(2);
	}
