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
 * tkCanvBLine.c --
 *
 *	This file implements beveled line items for canvas widgets.
 *
 */

#include <sci_config.h>

#include <stdio.h>
#include "tkPort.h"
#include "tkInt.h"
#include "tcl.h"

void Tk_DrawBeveledLine(Display* display, Drawable drawable,
			Tk_3DBorder border, XPoint *pointPtr,
			int numPoints, int width, int borderWidth,
			int relief);

/*
 * The structure below defines the record for each line item.
 */

typedef struct BLineItem  {
    Tk_Item header;		/* Generic stuff that's the same for all
				 * types.  MUST BE FIRST IN STRUCTURE. */
    int numPoints;		/* Number of points in line (always >= 2). */
    double *coordPtr;		/* Pointer to malloc-ed array containing
				 * x- and y-coords of all points in line.
				 * X-coords are even-valued indices, y-coords
				 * are corresponding odd-valued indices. If
				 * the line has arrowheads then the first
				 * and last points have been adjusted to refer
				 * to the necks of the arrowheads rather than
				 * their tips.  The actual endpoints are
				 * stored in the *firstArrowPtr and
				 * *lastArrowPtr, if they exist. */
    int width;			/* Width of line. */
    int capStyle;		/* Cap style for line. */
    int joinStyle;		/* Join style for line. */
    Tk_3DBorder border;		/* Graphics context for filling line. */
    int borderWidth;
    int relief;
} BLineItem;

/*
 * Prototypes for procedures defined in this file:
 */

static void		ComputeBLineBbox _ANSI_ARGS_((Tk_Canvas canvas,
			    BLineItem *linePtr));
static int		ConfigureBLine _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Canvas canvas, Tk_Item *itemPtr, int argc,
			    char **argv, int flags));
static int		CreateBLine _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Canvas canvas,
			    struct Tk_Item *itemPtr, int argc, char **argv));
static void		DeleteBLine _ANSI_ARGS_((Tk_Canvas canvas,
			    Tk_Item *itemPtr, Display *display));
static void		DisplayBLine _ANSI_ARGS_((Tk_Canvas canvas,
			    Tk_Item *itemPtr, Display *display, Drawable dst,
			    int x, int y, int width, int height));
static int		BLineCoords _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Canvas canvas, Tk_Item *itemPtr,
			    int argc, char **argv));
static int		BLineToArea _ANSI_ARGS_((Tk_Canvas canvas,
			    Tk_Item *itemPtr, double *rectPtr));
static double		BLineToPoint _ANSI_ARGS_((Tk_Canvas canvas,
			    Tk_Item *itemPtr, double *coordPtr));
static int		BLineToPostscript _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Canvas canvas, Tk_Item *itemPtr, int prepass));
static void		ScaleBLine _ANSI_ARGS_((Tk_Canvas canvas,
			    Tk_Item *itemPtr, double originX, double originY,
			    double scaleX, double scaleY));
static void		TranslateBLine _ANSI_ARGS_((Tk_Canvas canvas,
			    Tk_Item *itemPtr, double deltaX, double deltaY));

/*
 * Information used for parsing configuration specs.  If you change any
 * of the default strings, be sure to change the corresponding default
 * values in CreateBLine.
 */

static Tk_CustomOption tagsOption = {Tk_CanvasTagsParseProc,
                                     Tk_CanvasTagsPrintProc, (ClientData) NULL};

static Tk_ConfigSpec configSpecs[] = {
    {TK_CONFIG_CAP_STYLE, "-capstyle", (char *) NULL, (char *) NULL,
	"butt", Tk_Offset(BLineItem, capStyle), TK_CONFIG_DONT_SET_DEFAULT},
    {TK_CONFIG_BORDER, "-fill", (char *) NULL, (char *) NULL,
	"black", Tk_Offset(BLineItem, border), TK_CONFIG_NULL_OK},
    {TK_CONFIG_JOIN_STYLE, "-joinstyle", (char *) NULL, (char *) NULL,
	"round", Tk_Offset(BLineItem, joinStyle), TK_CONFIG_DONT_SET_DEFAULT},
    {TK_CONFIG_CUSTOM, "-tags", (char *) NULL, (char *) NULL,
	(char *) NULL, 0, TK_CONFIG_NULL_OK, &tagsOption},
    {TK_CONFIG_PIXELS, "-width", (char *) NULL, (char *) NULL,
	"1", Tk_Offset(BLineItem, width), TK_CONFIG_DONT_SET_DEFAULT},
    {TK_CONFIG_PIXELS, "-borderwidth", (char *) NULL, (char *) NULL,
        "2", Tk_Offset(BLineItem, borderWidth), TK_CONFIG_DONT_SET_DEFAULT},
    {TK_CONFIG_RELIEF, "-relief", (char*) NULL, (char *) NULL,
     "raised", Tk_Offset(BLineItem, relief), TK_CONFIG_DONT_SET_DEFAULT},
    {TK_CONFIG_END, (char *) NULL, (char *) NULL, (char *) NULL,
	(char *) NULL, 0, 0}
};

/*
 * The structures below defines the line item type by means
 * of procedures that can be invoked by generic item code.
 */

Tk_ItemType TkBLineType = {
    "bline",				/* name */
    sizeof(BLineItem),			/* itemSize */
    CreateBLine,				/* createProc */
    configSpecs,			/* configSpecs */
    ConfigureBLine,			/* configureProc */
    BLineCoords,				/* coordProc */
    DeleteBLine,				/* deleteProc */
    DisplayBLine,			/* displayProc */
    0,					/* alwaysRedraw */
    BLineToPoint,			/* pointProc */
    BLineToArea,				/* areaProc */
    BLineToPostscript,			/* postscriptProc */
    ScaleBLine,				/* scaleProc */
    TranslateBLine,			/* translateProc */
    (Tk_ItemIndexProc *) NULL,		/* indexProc */
    (Tk_ItemCursorProc *) NULL,		/* icursorProc */
    (Tk_ItemSelectionProc *) NULL,	/* selectionProc */
    (Tk_ItemInsertProc *) NULL,		/* insertProc */
    (Tk_ItemDCharsProc *) NULL,		/* dTextProc */
    (Tk_ItemType *) NULL		/* nextPtr */
};

/*
 * The definition below determines how large are static arrays
 * used to hold spline points (splines larger than this have to
 * have their arrays malloc-ed).
 */

#define MAX_STATIC_POINTS 200

/*
 *--------------------------------------------------------------
 *
 * CreateBLine --
 *
 *	This procedure is invoked to create a new line item in
 *	a canvas.
 *
 * Results:
 *	A standard Tcl return value.  If an error occurred in
 *	creating the item, then an error message is left in
 *	canvasPtr->interp->result;  in this case itemPtr is
 *	left uninitialized, so it can be safely freed by the
 *	caller.
 *
 * Side effects:
 *	A new line item is created.
 *
 *--------------------------------------------------------------
 */

static int
CreateBLine(interp, canvas, itemPtr, argc, argv)
    Tcl_Interp* interp ;		/* Interpreter for error reporting. */
    Tk_Canvas canvas;			/* Canvas to hold new item. */
    Tk_Item *itemPtr;			/* Record to hold new item;  header
					 * has been initialized by caller. */
    int argc;				/* Number of arguments in argv. */
    char **argv;			/* Arguments describing line. */
{
    BLineItem *linePtr = (BLineItem *) itemPtr;
    int i;

    if (argc < 4) {
	Tcl_AppendResult(interp, "wrong # args:  should be \"",
		Tk_PathName(Tk_CanvasTkwin(canvas)), "\" create ",
		itemPtr->typePtr->name, " x1 y1 x2 y2 ?x3 y3 ...? ?options?",
		(char *) NULL);
	return TCL_ERROR;
    }

    /*
     * Carry out initialization that is needed to set defaults and to
     * allow proper cleanup after errors during the the remainder of
     * this procedure.
     */

    linePtr->numPoints = 0;
    linePtr->coordPtr = NULL;
    linePtr->width = 1;
    linePtr->capStyle = CapButt;
    linePtr->joinStyle = JoinRound;
    linePtr->borderWidth = 2;
    linePtr->relief = TK_RELIEF_RAISED;
    linePtr->border = NULL;

    /*
     * Count the number of points and then parse them into a point
     * array.  Leading arguments are assumed to be points if they
     * start with a digit or a minus sign followed by a digit.
     */

    for (i = 4; i < (argc-1); i+=2) {
	if ((!isdigit(UCHAR(argv[i][0]))) &&
		((argv[i][0] != '-') || (!isdigit(UCHAR(argv[i][1]))))) {
	    break;
	}
    }
    if (BLineCoords(interp, canvas, itemPtr, i, argv) != TCL_OK) {
	goto error;
    }
    if (ConfigureBLine(interp, canvas, itemPtr, argc-i, argv+i, 0) == TCL_OK) {
	return TCL_OK;
    }

    error:
    DeleteBLine(canvas, itemPtr, Tk_Display(Tk_CanvasTkwin(canvas)));
    return TCL_ERROR;
}

/*
 *--------------------------------------------------------------
 *
 * BLineCoords --
 *
 *	This procedure is invoked to process the "coords" widget
 *	command on lines.  See the user documentation for details
 *	on what it does.
 *
 * Results:
 *	Returns TCL_OK or TCL_ERROR, and sets canvasPtr->interp->result.
 *
 * Side effects:
 *	The coordinates for the given item may be changed.
 *
 *--------------------------------------------------------------
 */

static int
BLineCoords(interp, canvas, itemPtr, argc, argv)
    Tcl_Interp *interp;			/* The interpreter. */
    Tk_Canvas canvas;			/* Canvas containing item. */
    Tk_Item *itemPtr;			/* Item whose coordinates are to be
					 * read or modified. */
    int argc;				/* Number of coordinates supplied in
					 * argv. */
    char **argv;			/* Array of coordinates: x1, y1,
					 * x2, y2, ... */
{
    register BLineItem *linePtr = (BLineItem *) itemPtr;
    char buffer[TCL_DOUBLE_SPACE];
    int i, numPoints;

    if (argc == 0) {
	double *coordPtr;
	int numCoords;

	numCoords = 2*linePtr->numPoints;
	coordPtr = linePtr->coordPtr;
	for (i = 0; i < numCoords; i++, coordPtr++) {
	    if (i == 2) {
		coordPtr = linePtr->coordPtr+2;
	    }
	    Tcl_PrintDouble(interp, *coordPtr, buffer);
	    Tcl_AppendElement(interp, buffer);
	}
    } else if (argc < 4) {
	Tcl_AppendResult(interp,
		"too few coordinates for line:  must have at least 4",
		(char *) NULL);
	return TCL_ERROR;
    } else if (argc & 1) {
	Tcl_AppendResult(interp,
		"odd number of coordinates specified for line",
		(char *) NULL);
	return TCL_ERROR;
    } else {
	numPoints = argc/2;
	if (linePtr->numPoints != numPoints) {
	    if (linePtr->coordPtr != NULL) {
		ckfree((char *) linePtr->coordPtr);
	    }
	    linePtr->coordPtr = (double *) ckalloc((unsigned)
		    (sizeof(double) * argc));
	    linePtr->numPoints = numPoints;
	}
	for (i = argc-1; i >= 0; i--) {
	    if (Tk_CanvasGetCoord(interp, canvas, argv[i],
		    &linePtr->coordPtr[i]) != TCL_OK) {
		return TCL_ERROR;
	    }
	}

	ComputeBLineBbox(canvas, linePtr);
    }
    return TCL_OK;
}

/*
 *--------------------------------------------------------------
 *
 * ConfigureBLine --
 *
 *	This procedure is invoked to configure various aspects
 *	of a line item such as its background color.
 *
 * Results:
 *	A standard Tcl result code.  If an error occurs, then
 *	an error message is left in canvasPtr->interp->result.
 *
 * Side effects:
 *	Configuration information, such as colors and stipple
 *	patterns, may be set for itemPtr.
 *
 *--------------------------------------------------------------
 */

static int
ConfigureBLine(interp, canvas, itemPtr, argc, argv, flags)
    Tcl_Interp *interp;		/* The intepreter. */
    Tk_Canvas canvas;	/* Canvas containing itemPtr. */
    Tk_Item *itemPtr;		/* BLine item to reconfigure. */
    int argc;			/* Number of elements in argv.  */
    char **argv;		/* Arguments describing things to configure. */
    int flags;			/* Flags to pass to Tk_ConfigureWidget. */
{
    register BLineItem *linePtr = (BLineItem *) itemPtr;
    Tk_Window tkwin;
    
    tkwin = Tk_CanvasTkwin(canvas);
    if (Tk_ConfigureWidget(interp, tkwin, configSpecs, argc, argv,
	     (char *) linePtr, flags) != TCL_OK) {
	return TCL_ERROR;
    }

    /*
     * Recompute bounding box for line.
     */

    ComputeBLineBbox(canvas, linePtr);

    return TCL_OK;
}

/*
 *--------------------------------------------------------------
 *
 * DeleteBLine --
 *
 *	This procedure is called to clean up the data structure
 *	associated with a line item.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Resources associated with itemPtr are released.
 *
 *--------------------------------------------------------------
 */

static void
DeleteBLine(canvas, itemPtr, display)
    Tk_Canvas canvas;		/* Info about overall canvas widget. */
    Tk_Item *itemPtr;			/* Item that is being deleted. */
    Display *display;			/* Display containing window for
					 * canvas. */
{
    BLineItem *linePtr = (BLineItem *) itemPtr;

    if (linePtr->coordPtr != NULL) {
	ckfree((char *) linePtr->coordPtr);
    }
    if (linePtr->border != NULL) {
	Tk_Free3DBorder(linePtr->border);
    }
}

/*
 *--------------------------------------------------------------
 *
 * ComputeBLineBbox --
 *
 *	This procedure is invoked to compute the bounding box of
 *	all the pixels that may be drawn as part of a line.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The fields x1, y1, x2, and y2 are updated in the header
 *	for itemPtr.
 *
 *--------------------------------------------------------------
 */

static void
ComputeBLineBbox(canvas, linePtr)
    Tk_Canvas canvas;			/* Canvas that contains item. */
    BLineItem *linePtr;			/* Item whose bbos is to be
					 * recomputed. */
{
    register double *coordPtr;
    int i;

    coordPtr = linePtr->coordPtr;
    linePtr->header.x1 = linePtr->header.x2 = *coordPtr;
    linePtr->header.y1 = linePtr->header.y2 = coordPtr[1];

    /*
     * Compute the bounding box of all the points in the line,
     * then expand in all directions by the line's width to take
     * care of butting or rounded corners and projecting or
     * rounded caps.  This expansion is an overestimate (worst-case
     * is square root of two over two) but it's simple.  Don't do
     * anything special for curves.  This causes an additional
     * overestimate in the bounding box, but is faster.
     */

    for (i = 1, coordPtr = linePtr->coordPtr+2; i < linePtr->numPoints;
	    i++, coordPtr += 2) {
	TkIncludePoint((Tk_Item *) linePtr, coordPtr);
    }
    linePtr->header.x1 -= linePtr->width;
    linePtr->header.x2 += linePtr->width;
    linePtr->header.y1 -= linePtr->width;
    linePtr->header.y2 += linePtr->width;

    /*
     * For mitered lines, make a second pass through all the points.
     * Compute the locations of the two miter vertex points and add
     * those into the bounding box.
     */

    if (linePtr->joinStyle == JoinMiter) {
	for (i = linePtr->numPoints, coordPtr = linePtr->coordPtr; i >= 3;
		i--, coordPtr += 2) {
	    double miter[4];
	    int j;
    
	    if (TkGetMiterPoints(coordPtr, coordPtr+2, coordPtr+4,
		    (double) linePtr->width, miter, miter+2)) {
		for (j = 0; j < 4; j += 2) {
		    TkIncludePoint((Tk_Item *) linePtr, miter+j);
		}
	    }
	}
    }

    /*
     * Add one more pixel of fudge factor just to be safe (e.g.
     * X may round differently than we do).
     */

    linePtr->header.x1 -= 1;
    linePtr->header.x2 += 1;
    linePtr->header.y1 -= 1;
    linePtr->header.y2 += 1;
}

/*
 *--------------------------------------------------------------
 *
 * DisplayBLine --
 *
 *	This procedure is invoked to draw a line item in a given
 *	drawable.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	ItemPtr is drawn in drawable using the transformation
 *	information in canvasPtr.
 *
 *--------------------------------------------------------------
 */

static void
DisplayBLine(canvas, itemPtr, display, drawable, x, y, width, height)
    Tk_Canvas canvas;			/* Canvas that contains item. */
    Tk_Item *itemPtr;			/* Item to be displayed. */
    Display *display; 			/* Display on which to draw item. */
    Drawable drawable;			/* Pixmap or window in which to draw
					 * item. */
    int x, y, width, height;		/* Describes region of canvas that
					   must be redispalyed (not used). */
{
    register BLineItem *linePtr = (BLineItem *) itemPtr;
    XPoint staticPoints[MAX_STATIC_POINTS];
    XPoint *pointPtr;
    register XPoint *pPtr;
    register double *coordPtr;
    int i, numPoints;

    /*
     * Build up an array of points in screen coordinates.  Use a
     * static array unless the line has an enormous number of points;
     * in this case, dynamically allocate an array.  For smoothed lines,
     * generate the curve points on each redisplay.
     */

    numPoints = linePtr->numPoints;

    if (numPoints <= MAX_STATIC_POINTS) {
	pointPtr = staticPoints;
    } else {
	pointPtr = (XPoint *) ckalloc((unsigned) (numPoints * sizeof(XPoint)));
    }

    for (i = 0, coordPtr = linePtr->coordPtr, pPtr = pointPtr;
	 i < linePtr->numPoints;  i += 1, coordPtr += 2, pPtr++) {
	Tk_CanvasDrawableCoords(canvas, coordPtr[0], coordPtr[1],
				&pPtr->x, &pPtr->y);
    }

    Tk_DrawBeveledLine(display, drawable, linePtr->border,
		       pointPtr, numPoints, linePtr->width,
		       linePtr->borderWidth, linePtr->relief);
    
    if (pointPtr != staticPoints) {
	ckfree((char *) pointPtr);
    }
}

/*
 *--------------------------------------------------------------
 *
 * BLineToPoint --
 *
 *	Computes the distance from a given point to a given
 *	line, in canvas units.
 *
 * Results:
 *	The return value is 0 if the point whose x and y coordinates
 *	are pointPtr[0] and pointPtr[1] is inside the line.  If the
 *	point isn't inside the line then the return value is the
 *	distance from the point to the line.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

	/* ARGSUSED */
static double
BLineToPoint(canvas, itemPtr, pointPtr)
    Tk_Canvas canvas;	/* Canvas containing item. */
    Tk_Item *itemPtr;		/* Item to check against point. */
    double *pointPtr;		/* Pointer to x and y coordinates. */
{
    register BLineItem *linePtr = (BLineItem *) itemPtr;
    register double *coordPtr, *linePoints;
    double staticSpace[2*MAX_STATIC_POINTS];
    double poly[10];
    double bestDist, dist;
    int numPoints, count;
    int changedMiterToBevel;	/* Non-zero means that a mitered corner
				 * had to be treated as beveled after all
				 * because the angle was < 11 degrees. */

    bestDist = 1.0e40;

    /*
     * Handle smoothed lines by generating an expanded set of points
     * against which to do the check.
     */

    numPoints = linePtr->numPoints;
    linePoints = linePtr->coordPtr;

    /*
     * The overall idea is to iterate through all of the edges of
     * the line, computing a polygon for each edge and testing the
     * point against that polygon.  In addition, there are additional
     * tests to deal with rounded joints and caps.
     */

    changedMiterToBevel = 0;
    for (count = numPoints, coordPtr = linePoints; count >= 2;
	    count--, coordPtr += 2) {

	/*
	 * If rounding is done around the first point then compute
	 * the distance between the point and the point.
	 */

	if (((linePtr->capStyle == CapRound) && (count == numPoints))
		|| ((linePtr->joinStyle == JoinRound)
			&& (count != numPoints))) {
	    dist = hypot(coordPtr[0] - pointPtr[0], coordPtr[1] - pointPtr[1])
		    - linePtr->width/2.0;
	    if (dist <= 0.0) {
		bestDist = 0.0;
		goto done;
	    } else if (dist < bestDist) {
		bestDist = dist;
	    }
	}

	/*
	 * Compute the polygonal shape corresponding to this edge,
	 * consisting of two points for the first point of the edge
	 * and two points for the last point of the edge.
	 */

	if (count == numPoints) {
	    TkGetButtPoints(coordPtr+2, coordPtr, (double) linePtr->width,
		    linePtr->capStyle == CapProjecting, poly, poly+2);
	} else if ((linePtr->joinStyle == JoinMiter) && !changedMiterToBevel) {
	    poly[0] = poly[6];
	    poly[1] = poly[7];
	    poly[2] = poly[4];
	    poly[3] = poly[5];
	} else {
	    TkGetButtPoints(coordPtr+2, coordPtr, (double) linePtr->width, 0,
		    poly, poly+2);

	    /*
	     * If this line uses beveled joints, then check the distance
	     * to a polygon comprising the last two points of the previous
	     * polygon and the first two from this polygon;  this checks
	     * the wedges that fill the mitered joint.
	     */

	    if ((linePtr->joinStyle == JoinBevel) || changedMiterToBevel) {
		poly[8] = poly[0];
		poly[9] = poly[1];
		dist = TkPolygonToPoint(poly, 5, pointPtr);
		if (dist <= 0.0) {
		    bestDist = 0.0;
		    goto done;
		} else if (dist < bestDist) {
		    bestDist = dist;
		}
		changedMiterToBevel = 0;
	    }
	}
	if (count == 2) {
	    TkGetButtPoints(coordPtr, coordPtr+2, (double) linePtr->width,
		    linePtr->capStyle == CapProjecting, poly+4, poly+6);
	} else if (linePtr->joinStyle == JoinMiter) {
	    if (TkGetMiterPoints(coordPtr, coordPtr+2, coordPtr+4,
		    (double) linePtr->width, poly+4, poly+6) == 0) {
		changedMiterToBevel = 1;
		TkGetButtPoints(coordPtr, coordPtr+2, (double) linePtr->width,
			0, poly+4, poly+6);
	    }
	} else {
	    TkGetButtPoints(coordPtr, coordPtr+2, (double) linePtr->width, 0,
		    poly+4, poly+6);
	}
	poly[8] = poly[0];
	poly[9] = poly[1];
	dist = TkPolygonToPoint(poly, 5, pointPtr);
	if (dist <= 0.0) {
	    bestDist = 0.0;
	    goto done;
	} else if (dist < bestDist) {
	    bestDist = dist;
	}
    }

    /*
     * If caps are rounded, check the distance to the cap around the
     * final end point of the line.
     */

    if (linePtr->capStyle == CapRound) {
	dist = hypot(coordPtr[0] - pointPtr[0], coordPtr[1] - pointPtr[1])
		- linePtr->width/2.0;
	if (dist <= 0.0) {
	    bestDist = 0.0;
	    goto done;
	} else if (dist < bestDist) {
	    bestDist = dist;
	}
    }

    done:
    if ((linePoints != staticSpace) && (linePoints != linePtr->coordPtr)) {
	ckfree((char *) linePoints);
    }
    return bestDist;
}

/*
 *--------------------------------------------------------------
 *
 * BLineToArea --
 *
 *	This procedure is called to determine whether an item
 *	lies entirely inside, entirely outside, or overlapping
 *	a given rectangular area.
 *
 * Results:
 *	-1 is returned if the item is entirely outside the
 *	area, 0 if it overlaps, and 1 if it is entirely
 *	inside the given area.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

	/* ARGSUSED */
static int
BLineToArea(canvas, itemPtr, rectPtr)
    Tk_Canvas canvas;	/* Canvas containing item. */
    Tk_Item *itemPtr;		/* Item to check against line. */
    double *rectPtr;
{
    register BLineItem *linePtr = (BLineItem *) itemPtr;
    register double *coordPtr;
    double staticSpace[2*MAX_STATIC_POINTS];
    double *linePoints, poly[10];
    double radius;
    int numPoints, count;
    int changedMiterToBevel;	/* Non-zero means that a mitered corner
				 * had to be treated as beveled after all
				 * because the angle was < 11 degrees. */
    int inside;			/* Tentative guess about what to return,
				 * based on all points seen so far:  one
				 * means everything seen so far was
				 * inside the area;  -1 means everything
				 * was outside the area.  0 means overlap
				 * has been found. */ 

    radius = linePtr->width/2.0;
    inside = -1;

    numPoints = linePtr->numPoints;
    linePoints = linePtr->coordPtr;

    coordPtr = linePoints;
    if ((coordPtr[0] >= rectPtr[0]) && (coordPtr[0] <= rectPtr[2])
	    && (coordPtr[1] >= rectPtr[1]) && (coordPtr[1] <= rectPtr[3])) {
	inside = 1;
    }

    /*
     * Iterate through all of the edges of the line, computing a polygon
     * for each edge and testing the area against that polygon.  In
     * addition, there are additional tests to deal with rounded joints
     * and caps.
     */

    changedMiterToBevel = 0;
    for (count = numPoints; count >= 2; count--, coordPtr += 2) {

	/*
	 * If rounding is done around the first point of the edge
	 * then test a circular region around the point with the
	 * area.
	 */

	if (((linePtr->capStyle == CapRound) && (count == numPoints))
		|| ((linePtr->joinStyle == JoinRound)
		&& (count != numPoints))) {
	    poly[0] = coordPtr[0] - radius;
	    poly[1] = coordPtr[1] - radius;
	    poly[2] = coordPtr[0] + radius;
	    poly[3] = coordPtr[1] + radius;
	    if (TkOvalToArea(poly, rectPtr) != inside) {
		inside = 0;
		goto done;
	    }
	}

	/*
	 * Compute the polygonal shape corresponding to this edge,
	 * consisting of two points for the first point of the edge
	 * and two points for the last point of the edge.
	 */

	if (count == numPoints) {
	    TkGetButtPoints(coordPtr+2, coordPtr, (double) linePtr->width,
		    linePtr->capStyle == CapProjecting, poly, poly+2);
	} else if ((linePtr->joinStyle == JoinMiter) && !changedMiterToBevel) {
	    poly[0] = poly[6];
	    poly[1] = poly[7];
	    poly[2] = poly[4];
	    poly[3] = poly[5];
	} else {
	    TkGetButtPoints(coordPtr+2, coordPtr, (double) linePtr->width, 0,
		    poly, poly+2);

	    /*
	     * If the last joint was beveled, then also check a
	     * polygon comprising the last two points of the previous
	     * polygon and the first two from this polygon;  this checks
	     * the wedges that fill the beveled joint.
	     */

	    if ((linePtr->joinStyle == JoinBevel) || changedMiterToBevel) {
		poly[8] = poly[0];
		poly[9] = poly[1];
		if (TkPolygonToArea(poly, 5, rectPtr) != inside) {
		    inside = 0;
		    goto done;
		}
		changedMiterToBevel = 0;
	    }
	}
	if (count == 2) {
	    TkGetButtPoints(coordPtr, coordPtr+2, (double) linePtr->width,
		    linePtr->capStyle == CapProjecting, poly+4, poly+6);
	} else if (linePtr->joinStyle == JoinMiter) {
	    if (TkGetMiterPoints(coordPtr, coordPtr+2, coordPtr+4,
		    (double) linePtr->width, poly+4, poly+6) == 0) {
		changedMiterToBevel = 1;
		TkGetButtPoints(coordPtr, coordPtr+2, (double) linePtr->width,
			0, poly+4, poly+6);
	    }
	} else {
	    TkGetButtPoints(coordPtr, coordPtr+2, (double) linePtr->width, 0,
		    poly+4, poly+6);
	}
	poly[8] = poly[0];
	poly[9] = poly[1];
	if (TkPolygonToArea(poly, 5, rectPtr) != inside) {
	    inside = 0;
	    goto done;
	}
    }

    /*
     * If caps are rounded, check the cap around the final point
     * of the line.
     */

    if (linePtr->capStyle == CapRound) {
	poly[0] = coordPtr[0] - radius;
	poly[1] = coordPtr[1] - radius;
	poly[2] = coordPtr[0] + radius;
	poly[3] = coordPtr[1] + radius;
	if (TkOvalToArea(poly, rectPtr) != inside) {
	    inside = 0;
	    goto done;
	}
    }

    done:
    if ((linePoints != staticSpace) && (linePoints != linePtr->coordPtr)) {
	ckfree((char *) linePoints);
    }
    return inside;
}

/*
 *--------------------------------------------------------------
 *
 * ScaleBLine --
 *
 *	This procedure is invoked to rescale a line item.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The BLine referred to by itemPtr is rescaled so that the
 *	following transformation is applied to all point
 *	coordinates:
 *		x' = originX + scaleX*(x-originX)
 *		y' = originY + scaleY*(y-originY)
 *
 *--------------------------------------------------------------
 */

static void
ScaleBLine(canvas, itemPtr, originX, originY, scaleX, scaleY)
    Tk_Canvas canvas;		/* Canvas containing line. */
    Tk_Item *itemPtr;			/* Line to be scaled. */
    double originX, originY;		/* Origin about which to scale rect. */
    double scaleX;			/* Amount to scale in X direction. */
    double scaleY;			/* Amount to scale in Y direction. */
{
    BLineItem *linePtr = (BLineItem *) itemPtr;
    register double *coordPtr;
    int i;

    for (i = 0, coordPtr = linePtr->coordPtr; i < linePtr->numPoints;
	    i++, coordPtr += 2) {
	coordPtr[0] = originX + scaleX*(*coordPtr - originX);
	coordPtr[1] = originY + scaleY*(coordPtr[1] - originY);
    }
    ComputeBLineBbox(canvas, linePtr);
}

/*
 *--------------------------------------------------------------
 *
 * TranslateBLine --
 *
 *	This procedure is called to move a line by a given amount.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The position of the line is offset by (xDelta, yDelta), and
 *	the bounding box is updated in the generic part of the item
 *	structure.
 *
 *--------------------------------------------------------------
 */

static void
TranslateBLine(canvas, itemPtr, deltaX, deltaY)
    Tk_Canvas canvas;		/* Canvas containing item. */
    Tk_Item *itemPtr;			/* Item that is being moved. */
    double deltaX, deltaY;		/* Amount by which item is to be
					 * moved. */
{
    BLineItem *linePtr = (BLineItem *) itemPtr;
    register double *coordPtr;
    int i;

    for (i = 0, coordPtr = linePtr->coordPtr; i < linePtr->numPoints;
	    i++, coordPtr += 2) {
	coordPtr[0] += deltaX;
	coordPtr[1] += deltaY;
    }
    ComputeBLineBbox(canvas, linePtr);
}

/*
 *--------------------------------------------------------------
 *
 * LineToPostscript --
 *
 *	This procedure is called to generate Postscript for
 *	line items.
 *
 * Results:
 *	The return value is a standard Tcl result.  If an error
 *	occurs in generating Postscript then an error message is
 *	left in canvasPtr->interp->result, replacing whatever used
 *	to be there.  If no error occurs, then Postscript for the
 *	item is appended to the result.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

static int
BLineToPostscript(interp, canvas, itemPtr, prepass)
    Tcl_Interp *interp;			/* Leave Postscript or error message
					 * here. */
    Tk_Canvas canvas;			/* Information about overall canvas. */
    Tk_Item *itemPtr;			/* Item for which Postscript is
					 * wanted. */
    int prepass;			/* 1 means this is a prepass to
					 * collect font information;  0 means
					 * final Postscript is being created. */
{
    register BLineItem *linePtr = (BLineItem *) itemPtr;
    char buffer[200];
    char *style;

    /*
     * Generate a path for the line's center-line (do this differently
     * for straight lines and smoothed lines).
     */

    Tk_CanvasPsPath(interp, canvas, linePtr->coordPtr, linePtr->numPoints);

    /*
     * Set other line-drawing parameters and stroke out the line.
     */

    sprintf(buffer, "%d setlinewidth\n", linePtr->width);
    Tcl_AppendResult(interp, buffer, (char *) NULL);
    style = "0 setlinecap\n";
    if (linePtr->capStyle == CapRound) {
	style = "1 setlinecap\n";
    } else if (linePtr->capStyle == CapProjecting) {
	style = "2 setlinecap\n";
    }
    Tcl_AppendResult(interp, style, (char *) NULL);
    style = "0 setlinejoin\n";
    if (linePtr->joinStyle == JoinRound) {
	style = "1 setlinejoin\n";
    } else if (linePtr->joinStyle == JoinBevel) {
	style = "2 setlinejoin\n";
    }
    Tcl_AppendResult(interp, style, (char *) NULL);
    if (Tk_CanvasPsColor(interp, canvas,
		      Tk_3DBorderColor(linePtr->border)) != TCL_OK) {
	return TCL_ERROR;
    };
    Tcl_AppendResult(interp, "stroke\n", (char *) NULL);

    return TCL_OK;
}

/*
 * Register the item...
 */
void BLineInit()
{
    Tk_CreateItemType(&TkBLineType);
}
