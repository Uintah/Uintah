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
 * tkBevel.c --
 *
 *	This module implements "Bevel" widgets. 
 *
 * Copyright (c) 1991-1993 The Regents of the University of California.
 * All rights reserved.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 * 
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF
 * CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 */

#include <sci_config.h>

#include "tkPort.h"
#include "tkInt.h"

/*
 * A data structure of the following type is kept for each Bevel
 * widget managed by this file:
 */

typedef struct {
    Tk_Window tkwin;		/* Window that embodies the Bevel.  NULL
				 * means window has been deleted but
				 * widget record hasn't been cleaned up yet. */
    Display *display;		/* X's token for the window's display. */
    Tcl_Interp *interp;		/* Interpreter associated with widget. */
    int x, y;			/* Position of Bevel's upper-left corner
				 * within widget. */
    int width, height;		/* Width and height of Bevel. */
    int pwidth;
    int pto;
    int pborder;
    char* edge;

    /*
     * Information used when displaying widget:
     */

    int borderWidth;		/* Width of 3-D border around whole widget. */
    Tk_3DBorder bgBorder;	/* Used for drawing background. */
    int relief;			/* Indicates whether window as a whole is
				 * raised, sunken, or flat. */
    GC gc;			/* Graphics context for copying from
				 * off-screen pixmap onto screen. */
    int doubleBuffer;		/* Non-zero means double-buffer redisplay
				 * with pixmap;  zero means draw straight
				 * onto the display. */
    int updatePending;		/* Non-zero means a call to BevelDisplay
				 * has already been scheduled. */
} Bevel;

static int opposite(int relief)
{
    switch(relief){
    case TK_RELIEF_RAISED:
	return TK_RELIEF_SUNKEN;
    case TK_RELIEF_FLAT:
	return TK_RELIEF_FLAT;
    case TK_RELIEF_SUNKEN:
	return TK_RELIEF_RAISED;
    case TK_RELIEF_GROOVE:
	return TK_RELIEF_RIDGE;
    case TK_RELIEF_RIDGE:
	return TK_RELIEF_SUNKEN;
    }
    return 0;
}


/*
 * Information used for argv parsing.
 */

static Tk_ConfigSpec configSpecs[] = {
    {TK_CONFIG_BORDER, "-background", "background", "Background",
	"#cdb79e", Tk_Offset(Bevel, bgBorder), TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_BORDER, "-background", "background", "Background",
	"white", Tk_Offset(Bevel, bgBorder), TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_SYNONYM, "-bd", "borderWidth", (char *) NULL,
	(char *) NULL, 0, 0},
    {TK_CONFIG_SYNONYM, "-bg", "background", (char *) NULL,
	(char *) NULL, 0, 0},
    {TK_CONFIG_PIXELS, "-borderwidth", "borderWidth", "BorderWidth",
	"2", Tk_Offset(Bevel, borderWidth), 0},
    {TK_CONFIG_INT, "-dbl", "doubleBuffer", "DoubleBuffer",
	"1", Tk_Offset(Bevel, doubleBuffer), 0},
    {TK_CONFIG_PIXELS, "-width", "width", "Width",
        "10", Tk_Offset(Bevel, width), 0},
    {TK_CONFIG_PIXELS, "-height", "height", "Height",
        "4", Tk_Offset(Bevel, height), 0},
    {TK_CONFIG_PIXELS, "-pwidth", "pwidth", "Pwidth",
        "4", Tk_Offset(Bevel, pwidth), 0},
    {TK_CONFIG_PIXELS, "-pto", "pto", "Pto",
        "4", Tk_Offset(Bevel, pto), 0},
    {TK_CONFIG_PIXELS, "-pborder", "pborder", "Pborder",
        "2", Tk_Offset(Bevel, pborder), 0},
    {TK_CONFIG_RELIEF, "-relief", "relief", "Relief",
	"raised", Tk_Offset(Bevel, relief), 0},
    {TK_CONFIG_STRING, "-edge", "edge", "Edge",
        "top", Tk_Offset(Bevel, edge), 0},
    {TK_CONFIG_END, (char *) NULL, (char *) NULL, (char *) NULL,
	(char *) NULL, 0, 0}
};

/*
 * Forward declarations for procedures defined later in this file:
 */

static int		BevelConfigure _ANSI_ARGS_((Tcl_Interp *interp,
			    Bevel *BevelPtr, int argc, char **argv,
			    int flags));
static void		BevelDestroy _ANSI_ARGS_((ClientData clientData));
static void		BevelDisplay _ANSI_ARGS_((ClientData clientData));
static void		BevelEventProc _ANSI_ARGS_((ClientData clientData,
			    XEvent *eventPtr));
static int		BevelWidgetCmd _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *, int argc, char **argv));

/*
 *--------------------------------------------------------------
 *
 * BevelCmd --
 *
 *	This procedure is invoked to process the "Bevel" Tcl
 *	command.  It creates a new "Bevel" widget.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	A new widget is created and configured.
 *
 *--------------------------------------------------------------
 */

int
BevelCmd(clientData, interp, argc, argv)
    ClientData clientData;	/* Main window associated with
				 * interpreter. */
    Tcl_Interp *interp;		/* Current interpreter. */
    int argc;			/* Number of arguments. */
    char **argv;		/* Argument strings. */
{
    Tk_Window tkwin = (Tk_Window) clientData;
    Bevel *BevelPtr;
    Tk_Window new_widget;

    if (argc < 2) {
	Tcl_AppendResult(interp, "wrong # args:  should be \"",
		argv[0], " pathName ?options?\"", (char *) NULL);
	return TCL_ERROR;
    }

    new_widget = Tk_CreateWindowFromPath(interp, tkwin, argv[1], (char *) NULL);
    if (new_widget == NULL) {
	return TCL_ERROR;
    }

    /*
     * Allocate and initialize the widget record.
     */

    BevelPtr = (Bevel *) ckalloc(sizeof(Bevel));
    BevelPtr->tkwin = new_widget;
    BevelPtr->display = Tk_Display(new_widget);
    BevelPtr->interp = interp;
    BevelPtr->x = 0;
    BevelPtr->y = 0;
    BevelPtr->width = 10;
    BevelPtr->height = 4;
    BevelPtr->pwidth = 4;
    BevelPtr->borderWidth = 2;
    BevelPtr->edge = NULL;
    BevelPtr->bgBorder = NULL;
    BevelPtr->relief = TK_RELIEF_RAISED;
    BevelPtr->gc = None;
    BevelPtr->doubleBuffer = 0;
    BevelPtr->updatePending = 0;

    Tk_SetClass(BevelPtr->tkwin, "Bevel");
    Tk_CreateEventHandler(BevelPtr->tkwin, ExposureMask|StructureNotifyMask,
	    BevelEventProc, (ClientData) BevelPtr);
    Tcl_CreateCommand(interp, Tk_PathName(BevelPtr->tkwin), BevelWidgetCmd,
	    (ClientData) BevelPtr, (Tcl_CmdDeleteProc*)NULL);
    if (BevelConfigure(interp, BevelPtr, argc-2, argv+2, 0) != TCL_OK) {
	Tk_DestroyWindow(BevelPtr->tkwin);
	return TCL_ERROR;
    }

    interp->result = Tk_PathName(BevelPtr->tkwin);
    return TCL_OK;
}

/*
 *--------------------------------------------------------------
 *
 * BevelWidgetCmd --
 *
 *	This procedure is invoked to process the Tcl command
 *	that corresponds to a widget managed by this module.
 *	See the user documentation for details on what it does.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	See the user documentation.
 *
 *--------------------------------------------------------------
 */

static int
BevelWidgetCmd(clientData, interp, argc, argv)
    ClientData clientData;		/* Information about Bevel widget. */
    Tcl_Interp *interp;			/* Current interpreter. */
    int argc;				/* Number of arguments. */
    char **argv;			/* Argument strings. */
{
    Bevel *BevelPtr = (Bevel *) clientData;
    int result = TCL_OK;
    int length;
    char c;

    if (argc < 2) {
	Tcl_AppendResult(interp, "wrong # args: should be \"",
		argv[0], " option ?arg arg ...?\"", (char *) NULL);
	return TCL_ERROR;
    }
    Tk_Preserve((ClientData) BevelPtr);
    c = argv[1][0];
    length = strlen(argv[1]);
    if ((c == 'c') && (strncmp(argv[1], "configure", length) == 0)) {
	if (argc == 2) {
	    result = Tk_ConfigureInfo(interp, BevelPtr->tkwin, configSpecs,
		    (char *) BevelPtr, (char *) NULL, 0);
	} else if (argc == 3) {
	    result = Tk_ConfigureInfo(interp, BevelPtr->tkwin, configSpecs,
		    (char *) BevelPtr, argv[2], 0);
	} else {
	    result = BevelConfigure(interp, BevelPtr, argc-2, argv+2,
		    TK_CONFIG_ARGV_ONLY);
	}
    } else {
	Tcl_AppendResult(interp, "bad option \"", argv[1],
		"\":  must be configure, position, or size", (char *) NULL);
	goto error;
    }
    if (!BevelPtr->updatePending) {
	Tk_DoWhenIdle(BevelDisplay, (ClientData) BevelPtr);
	BevelPtr->updatePending = 1;
    }
    Tk_Release((ClientData) BevelPtr);
    return result;

    error:
    Tk_Release((ClientData) BevelPtr);
    return TCL_ERROR;
}

/*
 *----------------------------------------------------------------------
 *
 * BevelConfigure --
 *
 *	This procedure is called to process an argv/argc list in
 *	conjunction with the Tk option database to configure (or
 *	reconfigure) a Bevel widget.
 *
 * Results:
 *	The return value is a standard Tcl result.  If TCL_ERROR is
 *	returned, then interp->result contains an error message.
 *
 * Side effects:
 *	Configuration information, such as colors, border width,
 *	etc. get set for BevelPtr;  old resources get freed,
 *	if there were any.
 *
 *----------------------------------------------------------------------
 */

static int
BevelConfigure(interp, BevelPtr, argc, argv, flags)
    Tcl_Interp *interp;			/* Used for error reporting. */
    Bevel *BevelPtr;			/* Information about widget. */
    int argc;				/* Number of valid entries in argv. */
    char **argv;			/* Arguments. */
    int flags;				/* Flags to pass to
					 * Tk_ConfigureWidget. */
{
    if (Tk_ConfigureWidget(interp, BevelPtr->tkwin, configSpecs,
	    argc, argv, (char *) BevelPtr, flags) != TCL_OK) {
	return TCL_ERROR;
    }

    /*
     * Set the background for the window and create a graphics context
     * for use during redisplay.
     */

    Tk_SetBackgroundFromBorder(BevelPtr->tkwin, BevelPtr->bgBorder);
    if (BevelPtr->doubleBuffer) {
	XGCValues gcValues;
	gcValues.function = GXcopy;
	gcValues.graphics_exposures = False;
	gcValues.foreground = Tk_3DBorderColor(BevelPtr->bgBorder)->pixel;
	if(BevelPtr->gc != None)
	    Tk_FreeGC(BevelPtr->display, BevelPtr->gc);
	BevelPtr->gc = Tk_GetGC(BevelPtr->tkwin,
				GCForeground|GCFunction|GCGraphicsExposures,
				&gcValues);
    }

    /*
     * Register the desired geometry for the window.  Then arrange for
     * the window to be redisplayed.
     */

    Tk_GeometryRequest(BevelPtr->tkwin, BevelPtr->width, BevelPtr->height);
    if (!BevelPtr->updatePending) {
	Tk_DoWhenIdle(BevelDisplay, (ClientData) BevelPtr);
	BevelPtr->updatePending = 1;
    }
    return TCL_OK;
}

/*
 *--------------------------------------------------------------
 *
 * BevelEventProc --
 *
 *	This procedure is invoked by the Tk dispatcher for various
 *	events on Bevels.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	When the window gets deleted, internal structures get
 *	cleaned up.  When it gets exposed, it is redisplayed.
 *
 *--------------------------------------------------------------
 */

static void
BevelEventProc(clientData, eventPtr)
    ClientData clientData;	/* Information about window. */
    XEvent *eventPtr;		/* Information about event. */
{
    Bevel *BevelPtr = (Bevel *) clientData;

    if (eventPtr->type == Expose) {
	if (!BevelPtr->updatePending) {
	    Tk_DoWhenIdle(BevelDisplay, (ClientData) BevelPtr);
	    BevelPtr->updatePending = 1;
	}
    } else if (eventPtr->type == ConfigureNotify) {
	if (!BevelPtr->updatePending) {
	    Tk_DoWhenIdle(BevelDisplay, (ClientData) BevelPtr);
	    BevelPtr->updatePending = 1;
	}
    } else if (eventPtr->type == DestroyNotify) {
	Tcl_DeleteCommand(BevelPtr->interp, Tk_PathName(BevelPtr->tkwin));
	BevelPtr->tkwin = NULL;
	if (BevelPtr->updatePending) {
	    Tk_CancelIdleCall(BevelDisplay, (ClientData) BevelPtr);
	}
	Tk_EventuallyFree((ClientData) BevelPtr, (Tcl_FreeProc*)BevelDestroy);
    }
}

/*
 *--------------------------------------------------------------
 *
 * BevelDisplay --
 *
 *	This procedure redraws the contents of a Bevel window.
 *	It is invoked as a do-when-idle handler, so it only runs
 *	when there's nothing else for the application to do.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Information appears on the screen.
 *
 *--------------------------------------------------------------
 */

static void
BevelDisplay(clientData)
    ClientData clientData;	/* Information about window. */
{
    Bevel *BevelPtr = (Bevel *) clientData;
    Tk_Window tkwin = BevelPtr->tkwin;
#if 0
    Pixmap pm = None;
#endif
    Drawable d;
    char* edge=BevelPtr->edge;
    int bw=BevelPtr->borderWidth;

    BevelPtr->updatePending = 0;
    if (!Tk_IsMapped(tkwin)) {
	return;
    }

#if 1
    /*
     * Create a pixmap for double-buffering, if necessary.
     */
    /* double buffering is broken... */
    BevelPtr->doubleBuffer=0;
/*
    if (BevelPtr->doubleBuffer) {
	pm = XCreatePixmap(Tk_Display(tkwin), Tk_WindowId(tkwin),
		Tk_Width(tkwin), Tk_Height(tkwin),
		DefaultDepthOfScreen(Tk_Screen(tkwin)));
	d = pm;
	XFillRectangle(Tk_Display(tkwin), d, BevelPtr->gc, 0, 0,
		       Tk_Width(tkwin), Tk_Height(tkwin));
    } else*/ {
	d = Tk_WindowId(tkwin);
	XFillRectangle(Tk_Display(tkwin), d, BevelPtr->gc, 0, 0,
		       Tk_Width(tkwin), Tk_Height(tkwin));
    }

    /*
     * Redraw the widget's background and border.
     */
    if(!strcmp(edge, "left")){
	Tk_Draw3DRectangle(tkwin, d, BevelPtr->bgBorder,
			   0, -bw, Tk_Width(tkwin)+bw, Tk_Height(tkwin)+2*bw,
			   BevelPtr->borderWidth, BevelPtr->relief);

    } else if(!strcmp(edge, "right")){
	Tk_Draw3DRectangle(tkwin, d, BevelPtr->bgBorder,
			   -bw, -bw, Tk_Width(tkwin)+bw, Tk_Height(tkwin)+2*bw,
			   BevelPtr->borderWidth, BevelPtr->relief);

    } else if(!strcmp(edge, "top")){
	Tk_Draw3DRectangle(tkwin, d, BevelPtr->bgBorder,
			   -bw, 0, Tk_Width(tkwin)+2*bw, Tk_Height(tkwin)+bw,
			   BevelPtr->borderWidth, BevelPtr->relief);

    } else if(!strcmp(edge, "bottom")){
	Tk_Draw3DRectangle(tkwin, d, BevelPtr->bgBorder,
			   -bw, -bw, Tk_Width(tkwin)+2*bw, Tk_Height(tkwin)+bw,
			   BevelPtr->borderWidth, BevelPtr->relief);
    } else if(!strcmp(edge, "outtop")){
	int pto=BevelPtr->pto;
	int pwidth=BevelPtr->pwidth;
	int pborder=BevelPtr->pborder;
	int o=(bw+pborder-1)/pborder*pborder;
	o-=pborder;
	for(;o>=0;o-=pborder){
	    Tk_Draw3DRectangle(tkwin, d, BevelPtr->bgBorder,
			       -pborder, -pborder-o,
			       pto+2*pborder+1, pborder+bw,
			       pborder, opposite(BevelPtr->relief));
	    Tk_Draw3DRectangle(tkwin, d, BevelPtr->bgBorder,
			       pto+pwidth-1, -pborder-o,
			       Tk_Width(tkwin)-pto-pwidth+2*pborder,
			       pborder+bw,
			       pborder, opposite(BevelPtr->relief));
	}
    } else if(!strcmp(edge, "outbottom")){
	int pto=BevelPtr->pto;
	int pwidth=BevelPtr->pwidth;
	int pborder=BevelPtr->pborder;
	int o=(bw+pborder-1)/pborder*pborder;
	o-=pborder;
	for(;o>=0;o-=pborder){
	    Tk_Draw3DRectangle(tkwin, d, BevelPtr->bgBorder,
			       -pborder, Tk_Height(tkwin)-pborder+o-1,
			       pto+2*pborder+1, pborder+bw,
			       pborder+40, opposite(BevelPtr->relief));
	    Tk_Draw3DRectangle(tkwin, d, BevelPtr->bgBorder,
			       pto+pwidth-1, Tk_Height(tkwin)-pborder+o-1,
			       Tk_Width(tkwin)-pto-pwidth+2*pborder,
			       pborder+bw,
			       pborder+40, opposite(BevelPtr->relief));
	}
    }

    /*
     * If double-buffered, copy to the screen and release the pixmap.
     */

	/*
    if (BevelPtr->doubleBuffer) {
	XCopyArea(Tk_Display(tkwin), pm, Tk_WindowId(tkwin), BevelPtr->gc,
		0, 0, Tk_Width(tkwin), Tk_Height(tkwin), 0, 0);
	//XFreePixmap(Tk_Display(tkwin), pm);
    }
	*/
#else
	printf("bevel not done\n");
#endif
}

/*
 *----------------------------------------------------------------------
 *
 * BevelDestroy --
 *
 *	This procedure is invoked by Tk_EventuallyFree or Tk_Release
 *	to clean up the internal structure of a Bevel at a safe time
 *	(when no-one is using it anymore).
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Everything associated with the Bevel is freed up.
 *
 *----------------------------------------------------------------------
 */

static void
BevelDestroy(clientData)
    ClientData clientData;	/* Info about Bevel widget. */
{
    Bevel *BevelPtr = (Bevel *) clientData;

    Tk_FreeOptions(configSpecs, (char *) BevelPtr, BevelPtr->display, 0);
    if (BevelPtr->gc != None) {
	Tk_FreeGC(BevelPtr->display, BevelPtr->gc);
    }
    ckfree((char *) BevelPtr);
}
