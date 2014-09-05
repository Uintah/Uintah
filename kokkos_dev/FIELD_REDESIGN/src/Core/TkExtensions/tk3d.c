/* 
 * tk3d.c --
 *
 *	This module provides procedures to draw borders in
 *	the three-dimensional Motif style.
 *
 * Copyright (c) 1990-1994 The Regents of the University of California.
 * Copyright (c) 1994-1997 Sun Microsystems, Inc.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 *
 * RCS: @(#) $Id$
 */

#include <sci_config.h>
#include <tk3d.h>

/*
 * Hash table to map from a border's values (color, etc.) to a
 * Border structure for those values.
 */

static Tcl_HashTable borderTable;
typedef struct {
    Tk_Uid colorName;		/* Color for border. */
    Colormap colormap;		/* Colormap used for allocating border
				 * colors. */
    Screen *screen;		/* Screen on which border will be drawn. */
} BorderKey;

static int initialized = 0;	/* 0 means static structures haven't
				 * been initialized yet. */

/*
 * Forward declarations for procedures defined in this file:
 */

static void		BorderInit _ANSI_ARGS_((void));
static int		Intersect _ANSI_ARGS_((XPoint *a1Ptr, XPoint *a2Ptr,
			    XPoint *b1Ptr, XPoint *b2Ptr, XPoint *iPtr));
static void		ShiftLine _ANSI_ARGS_((XPoint *p1Ptr, XPoint *p2Ptr,
			    int distance, XPoint *p3Ptr));

/*
 *--------------------------------------------------------------
 *
 * Tk_Get3DBorder --
 *
 *	Create a data structure for displaying a 3-D border.
 *
 * Results:
 *	The return value is a token for a data structure
 *	describing a 3-D border.  This token may be passed
 *	to Tk_Draw3DRectangle and Tk_Free3DBorder.  If an
 *	error prevented the border from being created then
 *	NULL is returned and an error message will be left
 *	in interp->result.
 *
 * Side effects:
 *	Data structures, graphics contexts, etc. are allocated.
 *	It is the caller's responsibility to eventually call
 *	Tk_Free3DBorder to release the resources.
 *
 *--------------------------------------------------------------
 */

Tk_3DBorder
Tk_Get3DBorder(interp, tkwin, colorName)
    Tcl_Interp *interp;		/* Place to store an error message. */
    Tk_Window tkwin;		/* Token for window in which border will
				 * be drawn. */
    Tk_Uid colorName;		/* String giving name of color
				 * for window background. */
{
    BorderKey key;
    Tcl_HashEntry *hashPtr;
    register TkBorder *borderPtr;
    int new;
    XGCValues gcValues;

    if (!initialized) {
	BorderInit();
    }

    /*
     * First, check to see if there's already a border that will work
     * for this request.
     */

    key.colorName = colorName;
    key.colormap = Tk_Colormap(tkwin);
    key.screen = Tk_Screen(tkwin);

    hashPtr = Tcl_CreateHashEntry(&borderTable, (char *) &key, &new);
    if (!new) {
	borderPtr = (TkBorder *) Tcl_GetHashValue(hashPtr);
	borderPtr->refCount++;
    } else {
	XColor *bgColorPtr;

	/*
	 * No satisfactory border exists yet.  Initialize a new one.
	 */
    
	bgColorPtr = Tk_GetColor(interp, tkwin, colorName);
	if (bgColorPtr == NULL) {
	    Tcl_DeleteHashEntry(hashPtr);
	    return NULL;
	}

	borderPtr = TkpGetBorder();
	borderPtr->screen = Tk_Screen(tkwin);
	borderPtr->visual = Tk_Visual(tkwin);
	borderPtr->depth = Tk_Depth(tkwin);
	borderPtr->colormap = key.colormap;
	borderPtr->refCount = 1;
	borderPtr->bgColorPtr = bgColorPtr;
	borderPtr->darkColorPtr = NULL;
	borderPtr->lightColorPtr = NULL;
	borderPtr->shadow = None;
	borderPtr->bgGC = None;
	borderPtr->darkGC = None;
	borderPtr->lightGC = None;
	borderPtr->hashPtr = hashPtr;
	Tcl_SetHashValue(hashPtr, borderPtr);
    
	/*
	 * Create the information for displaying the background color,
	 * but delay the allocation of shadows until they are actually
	 * needed for drawing.
	 */
    
	gcValues.foreground = borderPtr->bgColorPtr->pixel;
	borderPtr->bgGC = Tk_GetGC(tkwin, GCForeground, &gcValues);
    }
    return (Tk_3DBorder) borderPtr;
}

/*
 *--------------------------------------------------------------
 *
 * Tk_Draw3DRectangle --
 *
 *	Draw a 3-D border at a given place in a given window.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	A 3-D border will be drawn in the indicated drawable.
 *	The outside edges of the border will be determined by x,
 *	y, width, and height.  The inside edges of the border
 *	will be determined by the borderWidth argument.
 *
 *--------------------------------------------------------------
 */

void
Tk_Draw3DRectangle(tkwin, drawable, border, x, y, width, height,
	borderWidth, relief)
    Tk_Window tkwin;		/* Window for which border was allocated. */
    Drawable drawable;		/* X window or pixmap in which to draw. */
    Tk_3DBorder border;		/* Token for border to draw. */
    int x, y, width, height;	/* Outside area of region in
				 * which border will be drawn. */
    int borderWidth;		/* Desired width for border, in
				 * pixels. */
    int relief;			/* Type of relief: TK_RELIEF_RAISED,
				 * TK_RELIEF_SUNKEN, TK_RELIEF_GROOVE, etc. */
{
    if (width < 2*borderWidth) {
	borderWidth = width/2;
    }
    if (height < 2*borderWidth) {
	borderWidth = height/2;
    }
    Tk_3DVerticalBevel(tkwin, drawable, border, x, y, borderWidth, height,
	    1, relief);
    Tk_3DVerticalBevel(tkwin, drawable, border, x+width-borderWidth, y,
	    borderWidth, height, 0, relief);
    Tk_3DHorizontalBevel(tkwin, drawable, border, x, y, width, borderWidth,
	    1, 1, 1, relief);
    Tk_3DHorizontalBevel(tkwin, drawable, border, x, y+height-borderWidth,
	    width, borderWidth, 0, 0, 0, relief);
}

/*
 *--------------------------------------------------------------
 *
 * Tk_NameOf3DBorder --
 *
 *	Given a border, return a textual string identifying the
 *	border's color.
 *
 * Results:
 *	The return value is the string that was used to create
 *	the border.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

char *
Tk_NameOf3DBorder(border)
    Tk_3DBorder border;		/* Token for border. */
{
    TkBorder *borderPtr = (TkBorder *) border;

    return ((BorderKey *) borderPtr->hashPtr->key.words)->colorName;
}

/*
 *--------------------------------------------------------------------
 *
 * Tk_3DBorderColor --
 *
 *	Given a 3D border, return the X color used for the "flat"
 *	surfaces.
 *
 * Results:
 *	Returns the color used drawing flat surfaces with the border.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------------
 */
XColor *
Tk_3DBorderColor(border)
    Tk_3DBorder border;		/* Border whose color is wanted. */
{
    return(((TkBorder *) border)->bgColorPtr);
}

/*
 *--------------------------------------------------------------------
 *
 * Tk_3DBorderGC --
 *
 *	Given a 3D border, returns one of the graphics contexts used to
 *	draw the border.
 *
 * Results:
 *	Returns the graphics context given by the "which" argument.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------------
 */
GC
Tk_3DBorderGC(tkwin, border, which)
    Tk_Window tkwin;		/* Window for which border was allocated. */
    Tk_3DBorder border;		/* Border whose GC is wanted. */
    int which;			/* Selects one of the border's 3 GC's:
				 * TK_3D_FLAT_GC, TK_3D_LIGHT_GC, or
				 * TK_3D_DARK_GC. */
{
    TkBorder * borderPtr = (TkBorder *) border;

    if ((borderPtr->lightGC == None) && (which != TK_3D_FLAT_GC)) {
	TkpGetShadows(borderPtr, tkwin);
    }
    if (which == TK_3D_FLAT_GC) {
	return borderPtr->bgGC;
    } else if (which == TK_3D_LIGHT_GC) {
	return borderPtr->lightGC;
    } else if (which == TK_3D_DARK_GC){
	return borderPtr->darkGC;
    }
    panic("bogus \"which\" value in Tk_3DBorderGC");

    /*
     * The code below will never be executed, but it's needed to
     * keep compilers happy.
     */

    return (GC) None;
}

/*
 *--------------------------------------------------------------
 *
 * Tk_Free3DBorder --
 *
 *	This procedure is called when a 3D border is no longer
 *	needed.  It frees the resources associated with the
 *	border.  After this call, the caller should never again
 *	use the "border" token.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Resources are freed.
 *
 *--------------------------------------------------------------
 */

void
Tk_Free3DBorder(border)
    Tk_3DBorder border;		/* Token for border to be released. */
{
    register TkBorder *borderPtr = (TkBorder *) border;
    Display *display = DisplayOfScreen(borderPtr->screen);

    borderPtr->refCount--;
    if (borderPtr->refCount == 0) {
	TkpFreeBorder(borderPtr);
	if (borderPtr->bgColorPtr != NULL) {
	    Tk_FreeColor(borderPtr->bgColorPtr);
	}
	if (borderPtr->darkColorPtr != NULL) {
	    Tk_FreeColor(borderPtr->darkColorPtr);
	}
	if (borderPtr->lightColorPtr != NULL) {
	    Tk_FreeColor(borderPtr->lightColorPtr);
	}
	if (borderPtr->shadow != None) {
	    Tk_FreeBitmap(display, borderPtr->shadow);
	}
	if (borderPtr->bgGC != None) {
	    Tk_FreeGC(display, borderPtr->bgGC);
	}
	if (borderPtr->darkGC != None) {
	    Tk_FreeGC(display, borderPtr->darkGC);
	}
	if (borderPtr->lightGC != None) {
	    Tk_FreeGC(display, borderPtr->lightGC);
	}
	Tcl_DeleteHashEntry(borderPtr->hashPtr);
	ckfree((char *) borderPtr);
    }
}

/*
 *----------------------------------------------------------------------
 *
 * Tk_SetBackgroundFromBorder --
 *
 *	Change the background of a window to one appropriate for a given
 *	3-D border.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Tkwin's background gets modified.
 *
 *----------------------------------------------------------------------
 */

void
Tk_SetBackgroundFromBorder(tkwin, border)
    Tk_Window tkwin;		/* Window whose background is to be set. */
    Tk_3DBorder border;		/* Token for border. */
{
    register TkBorder *borderPtr = (TkBorder *) border;

    Tk_SetWindowBackground(tkwin, borderPtr->bgColorPtr->pixel);
}

/*
 *----------------------------------------------------------------------
 *
 * Tk_GetRelief --
 *
 *	Parse a relief description and return the corresponding
 *	relief value, or an error.
 *
 * Results:
 *	A standard Tcl return value.  If all goes well then
 *	*reliefPtr is filled in with one of the values
 *	TK_RELIEF_RAISED, TK_RELIEF_FLAT, or TK_RELIEF_SUNKEN.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

int
Tk_GetRelief(interp, name, reliefPtr)
    Tcl_Interp *interp;		/* For error messages. */
    char *name;			/* Name of a relief type. */
    int *reliefPtr;		/* Where to store converted relief. */
{
    char c;
    size_t length;

    c = name[0];
    length = strlen(name);
    if ((c == 'f') && (strncmp(name, "flat", length) == 0)) {
	*reliefPtr = TK_RELIEF_FLAT;
    } else if ((c == 'g') && (strncmp(name, "groove", length) == 0)
	    && (length >= 2)) {
        *reliefPtr = TK_RELIEF_GROOVE;
    } else if ((c == 'r') && (strncmp(name, "raised", length) == 0)
	    && (length >= 2)) {
	*reliefPtr = TK_RELIEF_RAISED;
    } else if ((c == 'r') && (strncmp(name, "ridge", length) == 0)) {
        *reliefPtr = TK_RELIEF_RIDGE;
    } else if ((c == 's') && (strncmp(name, "solid", length) == 0)) {
	*reliefPtr = TK_RELIEF_SOLID;
    } else if ((c == 's') && (strncmp(name, "sunken", length) == 0)) {
	*reliefPtr = TK_RELIEF_SUNKEN;
    } else {
	sprintf(interp->result, "bad relief type \"%.50s\": must be %s",
		name, "flat, groove, raised, ridge, solid, or sunken");
	return TCL_ERROR;
    }
    return TCL_OK;
}

/*
 *--------------------------------------------------------------
 *
 * Tk_NameOfRelief --
 *
 *	Given a relief value, produce a string describing that
 *	relief value.
 *
 * Results:
 *	The return value is a static string that is equivalent
 *	to relief.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

char *
Tk_NameOfRelief(relief)
    int relief;		/* One of TK_RELIEF_FLAT, TK_RELIEF_RAISED,
			 * or TK_RELIEF_SUNKEN. */
{
    if (relief == TK_RELIEF_FLAT) {
	return "flat";
    } else if (relief == TK_RELIEF_SUNKEN) {
	return "sunken";
    } else if (relief == TK_RELIEF_RAISED) {
	return "raised";
    } else if (relief == TK_RELIEF_GROOVE) {
	return "groove";
    } else if (relief == TK_RELIEF_RIDGE) {
	return "ridge";
    } else if (relief == TK_RELIEF_SOLID) {
	return "solid";
    } else {
	return "unknown relief";
    }
}

/*
 *--------------------------------------------------------------
 *
 * Tk_Draw3DPolygon --
 *
 *	Draw a border with 3-D appearance around the edge of a
 *	given polygon.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Information is drawn in "drawable" in the form of a
 *	3-D border borderWidth units width wide on the left
 *	of the trajectory given by pointPtr and numPoints (or
 *	-borderWidth units wide on the right side, if borderWidth
 *	is negative).
 *
 *--------------------------------------------------------------
 */

void
Tk_Draw3DPolygon(tkwin, drawable, border, pointPtr, numPoints,
	borderWidth, leftRelief)
    Tk_Window tkwin;		/* Window for which border was allocated. */
    Drawable drawable;		/* X window or pixmap in which to draw. */
    Tk_3DBorder border;		/* Token for border to draw. */
    XPoint *pointPtr;		/* Array of points describing
				 * polygon.  All points must be
				 * absolute (CoordModeOrigin). */
    int numPoints;		/* Number of points at *pointPtr. */
    int borderWidth;		/* Width of border, measured in
				 * pixels to the left of the polygon's
				 * trajectory.   May be negative. */
    int leftRelief;		/* TK_RELIEF_RAISED or
				 * TK_RELIEF_SUNKEN: indicates how
				 * stuff to left of trajectory looks
				 * relative to stuff on right. */
{
    XPoint poly[4], b1, b2, newB1, newB2;
    XPoint perp, c, shift1, shift2;	/* Used for handling parallel lines. */
    register XPoint *p1Ptr, *p2Ptr;
    TkBorder *borderPtr = (TkBorder *) border;
    GC gc;
    int i, lightOnLeft, dx, dy, parallel, pointsSeen;
    Display *display = Tk_Display(tkwin);

    if (borderPtr->lightGC == None) {
	TkpGetShadows(borderPtr, tkwin);
    }

    /*
     * Handle grooves and ridges with recursive calls.
     */

    if ((leftRelief == TK_RELIEF_GROOVE) || (leftRelief == TK_RELIEF_RIDGE)) {
	int halfWidth;

	halfWidth = borderWidth/2;
	Tk_Draw3DPolygon(tkwin, drawable, border, pointPtr, numPoints,
		halfWidth, (leftRelief == TK_RELIEF_GROOVE) ? TK_RELIEF_RAISED
		: TK_RELIEF_SUNKEN);
	Tk_Draw3DPolygon(tkwin, drawable, border, pointPtr, numPoints,
		-halfWidth, (leftRelief == TK_RELIEF_GROOVE) ? TK_RELIEF_SUNKEN
		: TK_RELIEF_RAISED);
	return;
    }

    /*
     * If the polygon is already closed, drop the last point from it
     * (we'll close it automatically).
     */

    p1Ptr = &pointPtr[numPoints-1];
    p2Ptr = &pointPtr[0];
    if ((p1Ptr->x == p2Ptr->x) && (p1Ptr->y == p2Ptr->y)) {
	numPoints--;
    }

    /*
     * The loop below is executed once for each vertex in the polgon.
     * At the beginning of each iteration things look like this:
     *
     *          poly[1]       /
     *             *        /
     *             |      /
     *             b1   * poly[0] (pointPtr[i-1])
     *             |    |
     *             |    |
     *             |    |
     *             |    |
     *             |    |
     *             |    | *p1Ptr            *p2Ptr
     *             b2   *--------------------*
     *             |
     *             |
     *             x-------------------------
     *
     * The job of this iteration is to do the following:
     * (a) Compute x (the border corner corresponding to
     *     pointPtr[i]) and put it in poly[2].  As part of
     *	   this, compute a new b1 and b2 value for the next
     *	   side of the polygon.
     * (b) Put pointPtr[i] into poly[3].
     * (c) Draw the polygon given by poly[0..3].
     * (d) Advance poly[0], poly[1], b1, and b2 for the
     *     next side of the polygon.
     */

    /*
     * The above situation doesn't first come into existence until
     * two points have been processed;  the first two points are
     * used to "prime the pump", so some parts of the processing
     * are ommitted for these points.  The variable "pointsSeen"
     * keeps track of the priming process;  it has to be separate
     * from i in order to be able to ignore duplicate points in the
     * polygon.
     */

    pointsSeen = 0;
    for (i = -2, p1Ptr = &pointPtr[numPoints-2], p2Ptr = p1Ptr+1;
	    i < numPoints; i++, p1Ptr = p2Ptr, p2Ptr++) {
	if ((i == -1) || (i == numPoints-1)) {
	    p2Ptr = pointPtr;
	}
	if ((p2Ptr->x == p1Ptr->x) && (p2Ptr->y == p1Ptr->y)) {
	    /*
	     * Ignore duplicate points (they'd cause core dumps in
	     * ShiftLine calls below).
	     */
	    continue;
	}
	ShiftLine(p1Ptr, p2Ptr, borderWidth, &newB1);
	newB2.x = newB1.x + (p2Ptr->x - p1Ptr->x);
	newB2.y = newB1.y + (p2Ptr->y - p1Ptr->y);
	poly[3] = *p1Ptr;
	parallel = 0;
	if (pointsSeen >= 1) {
	    parallel = Intersect(&newB1, &newB2, &b1, &b2, &poly[2]);

	    /*
	     * If two consecutive segments of the polygon are parallel,
	     * then things get more complex.  Consider the following
	     * diagram:
	     *
	     * poly[1]
	     *    *----b1-----------b2------a
	     *                                \
	     *                                  \
	     *         *---------*----------*    b
	     *        poly[0]  *p2Ptr   *p1Ptr  /
	     *                                /
	     *              --*--------*----c
	     *              newB1    newB2
	     *
	     * Instead of using x and *p1Ptr for poly[2] and poly[3], as
	     * in the original diagram, use a and b as above.  Then instead
	     * of using x and *p1Ptr for the new poly[0] and poly[1], use
	     * b and c as above.
	     *
	     * Do the computation in three stages:
	     * 1. Compute a point "perp" such that the line p1Ptr-perp
	     *    is perpendicular to p1Ptr-p2Ptr.
	     * 2. Compute the points a and c by intersecting the lines
	     *    b1-b2 and newB1-newB2 with p1Ptr-perp.
	     * 3. Compute b by shifting p1Ptr-perp to the right and
	     *    intersecting it with p1Ptr-p2Ptr.
	     */

	    if (parallel) {
		perp.x = p1Ptr->x + (p2Ptr->y - p1Ptr->y);
		perp.y = p1Ptr->y - (p2Ptr->x - p1Ptr->x);
		(void) Intersect(p1Ptr, &perp, &b1, &b2, &poly[2]);
		(void) Intersect(p1Ptr, &perp, &newB1, &newB2, &c);
		ShiftLine(p1Ptr, &perp, borderWidth, &shift1);
		shift2.x = shift1.x + (perp.x - p1Ptr->x);
		shift2.y = shift1.y + (perp.y - p1Ptr->y);
		(void) Intersect(p1Ptr, p2Ptr, &shift1, &shift2, &poly[3]);
	    }
	}
	if (pointsSeen >= 2) {
	    dx = poly[3].x - poly[0].x;
	    dy = poly[3].y - poly[0].y;
	    if (dx > 0) {
		lightOnLeft = (dy <= dx);
	    } else {
		lightOnLeft = (dy < dx);
	    }
	    if (lightOnLeft ^ (leftRelief == TK_RELIEF_RAISED)) {
		gc = borderPtr->lightGC;
	    } else {
		gc = borderPtr->darkGC;
	    }
	    XFillPolygon(display, drawable, gc, poly, 4, Convex,
		    CoordModeOrigin);
	}
	b1.x = newB1.x;
	b1.y = newB1.y;
	b2.x = newB2.x;
	b2.y = newB2.y;
	poly[0].x = poly[3].x;
	poly[0].y = poly[3].y;
	if (parallel) {
	    poly[1].x = c.x;
	    poly[1].y = c.y;
	} else if (pointsSeen >= 1) {
	    poly[1].x = poly[2].x;
	    poly[1].y = poly[2].y;
	}
	pointsSeen++;
    }
}

/*
 *----------------------------------------------------------------------
 *
 * Tk_Fill3DRectangle --
 *
 *	Fill a rectangular area, supplying a 3D border if desired.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Information gets drawn on the screen.
 *
 *----------------------------------------------------------------------
 */

void
Tk_Fill3DRectangle(tkwin, drawable, border, x, y, width,
	height, borderWidth, relief)
    Tk_Window tkwin;		/* Window for which border was allocated. */
    Drawable drawable;		/* X window or pixmap in which to draw. */
    Tk_3DBorder border;		/* Token for border to draw. */
    int x, y, width, height;	/* Outside area of rectangular region. */
    int borderWidth;		/* Desired width for border, in
				 * pixels. Border will be *inside* region. */
    int relief;			/* Indicates 3D effect: TK_RELIEF_FLAT,
				 * TK_RELIEF_RAISED, or TK_RELIEF_SUNKEN. */
{
    register TkBorder *borderPtr = (TkBorder *) border;
    int doubleBorder;

    /*
     * This code is slightly tricky because it only draws the background
     * in areas not covered by the 3D border. This avoids flashing
     * effects on the screen for the border region.
     */
  
    if (relief == TK_RELIEF_FLAT) {
	borderWidth = 0;
    }
    doubleBorder = 2*borderWidth;

    if ((width > doubleBorder) && (height > doubleBorder)) {
	XFillRectangle(Tk_Display(tkwin), drawable, borderPtr->bgGC,
		x + borderWidth, y + borderWidth,
		(unsigned int) (width - doubleBorder),
		(unsigned int) (height - doubleBorder));
    }
    if (borderWidth) {
	Tk_Draw3DRectangle(tkwin, drawable, border, x, y, width,
		height, borderWidth, relief);
    }
}

/*
 *----------------------------------------------------------------------
 *
 * Tk_Fill3DPolygon --
 *
 *	Fill a polygonal area, supplying a 3D border if desired.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Information gets drawn on the screen.
 *
 *----------------------------------------------------------------------
 */

void
Tk_Fill3DPolygon(tkwin, drawable, border, pointPtr, numPoints,
	borderWidth, leftRelief)
    Tk_Window tkwin;		/* Window for which border was allocated. */
    Drawable drawable;		/* X window or pixmap in which to draw. */
    Tk_3DBorder border;		/* Token for border to draw. */
    XPoint *pointPtr;		/* Array of points describing
				 * polygon.  All points must be
				 * absolute (CoordModeOrigin). */
    int numPoints;		/* Number of points at *pointPtr. */
    int borderWidth;		/* Width of border, measured in
				 * pixels to the left of the polygon's
				 * trajectory.   May be negative. */
    int leftRelief;			/* Indicates 3D effect of left side of
				 * trajectory relative to right:
				 * TK_RELIEF_FLAT, TK_RELIEF_RAISED,
				 * or TK_RELIEF_SUNKEN. */
{
    register TkBorder *borderPtr = (TkBorder *) border;

    XFillPolygon(Tk_Display(tkwin), drawable, borderPtr->bgGC,
	    pointPtr, numPoints, Complex, CoordModeOrigin);
    if (leftRelief != TK_RELIEF_FLAT) {
	Tk_Draw3DPolygon(tkwin, drawable, border, pointPtr, numPoints,
		borderWidth, leftRelief);
    }
}

/*
 *--------------------------------------------------------------
 *
 * BorderInit --
 *
 *	Initialize the structures used for border management.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Read the code.
 *
 *-------------------------------------------------------------
 */

static void
BorderInit()
{
    initialized = 1;
    Tcl_InitHashTable(&borderTable, sizeof(BorderKey)/sizeof(int));
}

/*
 *--------------------------------------------------------------
 *
 * ShiftLine --
 *
 *	Given two points on a line, compute a point on a
 *	new line that is parallel to the given line and
 *	a given distance away from it.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

static void
ShiftLine(p1Ptr, p2Ptr, distance, p3Ptr)
    XPoint *p1Ptr;		/* First point on line. */
    XPoint *p2Ptr;		/* Second point on line. */
    int distance;		/* New line is to be this many
				 * units to the left of original
				 * line, when looking from p1 to
				 * p2.  May be negative. */
    XPoint *p3Ptr;		/* Store coords of point on new
				 * line here. */
{
    int dx, dy, dxNeg, dyNeg;

    /*
     * The table below is used for a quick approximation in
     * computing the new point.  An index into the table
     * is 128 times the slope of the original line (the slope
     * must always be between 0 and 1).  The value of the table
     * entry is 128 times the amount to displace the new line
     * in y for each unit of perpendicular distance.  In other
     * words, the table maps from the tangent of an angle to
     * the inverse of its cosine.  If the slope of the original
     * line is greater than 1, then the displacement is done in
     * x rather than in y.
     */

    static int shiftTable[129];

    /*
     * Initialize the table if this is the first time it is
     * used.
     */

    if (shiftTable[0] == 0) {
	int i;
	double tangent, cosine;

	for (i = 0; i <= 128; i++) {
	    tangent = i/128.0;
	    cosine = 128/cos(atan(tangent)) + .5;
	    shiftTable[i] = (int) cosine;
	}
    }

    *p3Ptr = *p1Ptr;
    dx = p2Ptr->x - p1Ptr->x;
    dy = p2Ptr->y - p1Ptr->y;
    if (dy < 0) {
	dyNeg = 1;
	dy = -dy;
    } else {
	dyNeg = 0;
    }
    if (dx < 0) {
	dxNeg = 1;
	dx = -dx;
    } else {
	dxNeg = 0;
    }
    if (dy <= dx) {
	dy = ((distance * shiftTable[(dy<<7)/dx]) + 64) >> 7;
	if (!dxNeg) {
	    dy = -dy;
	}
	p3Ptr->y += dy;
    } else {
	dx = ((distance * shiftTable[(dx<<7)/dy]) + 64) >> 7;
	if (dyNeg) {
	    dx = -dx;
	}
	p3Ptr->x += dx;
    }
}

/*
 *--------------------------------------------------------------
 *
 * Intersect --
 *
 *	Find the intersection point between two lines.
 *
 * Results:
 *	Under normal conditions 0 is returned and the point
 *	at *iPtr is filled in with the intersection between
 *	the two lines.  If the two lines are parallel, then
 *	-1 is returned and *iPtr isn't modified.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

static int
Intersect(a1Ptr, a2Ptr, b1Ptr, b2Ptr, iPtr)
    XPoint *a1Ptr;		/* First point of first line. */
    XPoint *a2Ptr;		/* Second point of first line. */
    XPoint *b1Ptr;		/* First point of second line. */
    XPoint *b2Ptr;		/* Second point of second line. */
    XPoint *iPtr;		/* Filled in with intersection point. */
{
    int dxadyb, dxbdya, dxadxb, dyadyb, p, q;

    /*
     * The code below is just a straightforward manipulation of two
     * equations of the form y = (x-x1)*(y2-y1)/(x2-x1) + y1 to solve
     * for the x-coordinate of intersection, then the y-coordinate.
     */

    dxadyb = (a2Ptr->x - a1Ptr->x)*(b2Ptr->y - b1Ptr->y);
    dxbdya = (b2Ptr->x - b1Ptr->x)*(a2Ptr->y - a1Ptr->y);
    dxadxb = (a2Ptr->x - a1Ptr->x)*(b2Ptr->x - b1Ptr->x);
    dyadyb = (a2Ptr->y - a1Ptr->y)*(b2Ptr->y - b1Ptr->y);

    if (dxadyb == dxbdya) {
	return -1;
    }
    p = (a1Ptr->x*dxbdya - b1Ptr->x*dxadyb + (b1Ptr->y - a1Ptr->y)*dxadxb);
    q = dxbdya - dxadyb;
    if (q < 0) {
	p = -p;
	q = -q;
    }
    if (p < 0) {
	iPtr->x = - ((-p + q/2)/q);
    } else {
	iPtr->x = (p + q/2)/q;
    }
    p = (a1Ptr->y*dxadyb - b1Ptr->y*dxbdya + (b1Ptr->x - a1Ptr->x)*dyadyb);
    q = dxadyb - dxbdya;
    if (q < 0) {
	p = -p;
	q = -q;
    }
    if (p < 0) {
	iPtr->y = - ((-p + q/2)/q);
    } else {
	iPtr->y = (p + q/2)/q;
    }
    return 0;
}


/*
 * Added by Steve Parker to support tk3d2.c
 */

GC Tk_BorderDarkGC(border)
    Tk_3DBorder border;
{
    TkBorder* b=(TkBorder*)border;
    return b->darkGC;
}

GC Tk_BorderLightGC(border)
    Tk_3DBorder border;
{
    TkBorder* b=(TkBorder*)border;
    return b->lightGC;
}

GC Tk_BorderBgGC(border)
    Tk_3DBorder border;
{
    TkBorder* b=(TkBorder*)border;
    return b->bgGC;
}

int Tk_BorderIntersect (a1Ptr, a2Ptr, b1Ptr, b2Ptr, iPtr)
    XPoint* a1Ptr;
    XPoint* a2Ptr;
    XPoint* b1Ptr;
    XPoint* b2Ptr;
    XPoint* iPtr;
{
    return Intersect(a1Ptr, a2Ptr, b1Ptr, b2Ptr, iPtr);
}

void Tk_BorderShiftLine (p1Ptr, p2Ptr, distance, p3Ptr)
    XPoint* p1Ptr;
    XPoint* p2Ptr;
    int distance;
    XPoint* p3Ptr;
{
    ShiftLine(p1Ptr, p2Ptr, distance, p3Ptr);
}
