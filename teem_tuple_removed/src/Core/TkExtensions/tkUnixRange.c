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
 * tkUnixRange.c --
 *
 *	This file implements the X specific portion of the scrollbar
 *	widget.
 *
 * Copyright (c) 1996 by Sun Microsystems, Inc.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 *
 */

#include <sci_config.h>
#include "tkRange.h"
#include "tkInt.h"

/*
 * Forward declarations for procedures defined later in this file:
 */

static void		DisplayHorizontalRange _ANSI_ARGS_((TkRange *rangePtr,
			    Drawable drawable, XRectangle *drawnAreaPtr));
static void		DisplayHorizontalValue _ANSI_ARGS_((TkRange *rangePtr,
			    Drawable drawable, double value, int top));
static void		DisplayVerticalRange _ANSI_ARGS_((TkRange *rangePtr,
			    Drawable drawable, XRectangle *drawnAreaPtr));
static void		DisplayVerticalValue _ANSI_ARGS_((TkRange *rangePtr,
			    Drawable drawable, double value, int rightEdge));

/*
 *----------------------------------------------------------------------
 *
 * TkpCreateRange --
 *
 *	Allocate a new TkRange structure.
 *
 * Results:
 *	Returns a newly allocated TkRange structure.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

TkRange *
TkpCreateRange(tkwin)
    Tk_Window tkwin;
{
    return (TkRange *) ckalloc(sizeof(TkRange));
}

/*
 *----------------------------------------------------------------------
 *
 * TkpDestroyRange --
 *
 *	Destroy a TkRange structure.
 *
 * Results:
 *	None
 *
 * Side effects:
 *	Memory is freed.
 *
 *----------------------------------------------------------------------
 */

void
TkpDestroyRange(rangePtr)
    TkRange *rangePtr;
{
    ckfree((char *) rangePtr);
}

/*
 *--------------------------------------------------------------
 *
 * DisplayVerticalRange --
 *
 *	This procedure redraws the contents of a vertical range
 *	window.  It is invoked as a do-when-idle handler, so it only
 *	runs when there's nothing else for the application to do.
 *
 * Results:
 *	There is no return value.  If only a part of the range needs
 *	to be redrawn, then drawnAreaPtr is modified to reflect the
 *	area that was actually modified.
 *
 * Side effects:
 *	Information appears on the screen.
 *
 *--------------------------------------------------------------
 */

static void
DisplayVerticalRange(rangePtr, drawable, drawnAreaPtr)
    TkRange *rangePtr;			/* Widget record for range. */
    Drawable drawable;			/* Where to display range (window
					 * or pixmap). */
    XRectangle *drawnAreaPtr;		/* Initally contains area of window;
					 * if only a part of the range is
					 * redrawn, gets modified to reflect
					 * the part of the window that was
					 * redrawn. */
{
    Tk_Window tkwin = rangePtr->tkwin;
    int x, y1, y2, width, height, shadowWidth;
    double tickValue;
    Tk_3DBorder sliderBorder;
    int rangeFillHeight;

    /*
     * Display the information from left to right across the window.
     */

    if (!(rangePtr->flags & REDRAW_OTHER)) {
	drawnAreaPtr->x = rangePtr->vertTickRightX;
	drawnAreaPtr->y = rangePtr->inset;
	drawnAreaPtr->width = rangePtr->vertTroughX + rangePtr->width
		+ 2*rangePtr->borderWidth - rangePtr->vertTickRightX;
	drawnAreaPtr->height -= 2*rangePtr->inset;
    }
    Tk_Fill3DRectangle(tkwin, drawable, rangePtr->bgBorder,
	    drawnAreaPtr->x, drawnAreaPtr->y, drawnAreaPtr->width,
	    drawnAreaPtr->height, 0, TK_RELIEF_FLAT);
    if (rangePtr->flags & REDRAW_OTHER) {
	/*
	 * Display the tick marks.
	 */

	if (rangePtr->tickInterval != 0) {
	    for (tickValue = rangePtr->fromValue; ;
		    tickValue += rangePtr->tickInterval) {
		/*
		 * The TkRoundToResolution call gets rid of accumulated
		 * round-off errors, if any.
		 */

		tickValue = TkRoundToResolution(rangePtr, tickValue);
		if (rangePtr->toValue >= rangePtr->fromValue) {
		    if (tickValue > rangePtr->toValue) {
			break;
		    }
		} else {
		    if (tickValue < rangePtr->toValue) {
			break;
		    }
		}
		DisplayVerticalValue(rangePtr, drawable, tickValue,
			rangePtr->vertTickRightX);
	    }
	}
    }

    /*
     * Display the value, if it is desired.
     */

    if (rangePtr->showValue) {
	DisplayVerticalValue(rangePtr, drawable, rangePtr->min_value,
		rangePtr->vertValueRightX);
	DisplayVerticalValue(rangePtr, drawable, rangePtr->max_value,
		rangePtr->vertValueRightX);
    }

    /*
     * Display the trough and the slider.
     */

    Tk_Draw3DRectangle(tkwin, drawable,
	    rangePtr->bgBorder, rangePtr->vertTroughX, rangePtr->inset,
	    rangePtr->width + 2*rangePtr->borderWidth,
	    Tk_Height(tkwin) - 2*rangePtr->inset, rangePtr->borderWidth,
	    TK_RELIEF_SUNKEN);
    XFillRectangle(rangePtr->display, drawable, rangePtr->troughGC,
	    rangePtr->vertTroughX + rangePtr->borderWidth,
	    rangePtr->inset + rangePtr->borderWidth,
	    (unsigned) rangePtr->width,
	    (unsigned) (Tk_Height(tkwin) - 2*rangePtr->inset
		- 2*rangePtr->borderWidth));
    if (rangePtr->state == tkActiveUid) {
	sliderBorder = rangePtr->activeBorder;
    } else {
	sliderBorder = rangePtr->bgBorder;
    }
    width = rangePtr->width;
    height = rangePtr->sliderLength/2;
    x = rangePtr->vertTroughX + rangePtr->borderWidth;
    y1 = TkpRangeValueToPixel(rangePtr, rangePtr->min_value) - height;
    y2 = TkpRangeValueToPixel(rangePtr, rangePtr->max_value);
    shadowWidth = rangePtr->borderWidth/2;
    if (shadowWidth == 0) {
	shadowWidth = 1;
    }
    rangeFillHeight = y2 - y1 - height - 2 * shadowWidth;
    if (rangeFillHeight>0) {		/* enough space to draw range fill */
	XFillRectangle(rangePtr->display, drawable, rangePtr->rangeGC,
		       rangePtr->vertTroughX + rangePtr->borderWidth,
		       y1 + height + shadowWidth,
		       (unsigned) rangePtr->width,
		       (unsigned) rangeFillHeight);
    }
    Tk_Draw3DRectangle(tkwin, drawable, sliderBorder, x, y1, width,
	    height, shadowWidth, rangePtr->sliderRelief);
    Tk_Draw3DRectangle(tkwin, drawable, sliderBorder, x, y2, width,
	    height, shadowWidth, rangePtr->sliderRelief);
    x += shadowWidth;
    y1 += shadowWidth;
    y2 += shadowWidth;
    width -= 2*shadowWidth;
    height -= shadowWidth;
    Tk_Fill3DRectangle(tkwin, drawable, sliderBorder, 
		       x, y1, width, height, shadowWidth, 
		       rangePtr->sliderRelief);
    Tk_Fill3DRectangle(tkwin, drawable, sliderBorder, 
		       x, y2, width, height, shadowWidth, 
		       rangePtr->sliderRelief);

    /*
     * Draw the label to the right of the range.
     */

    if ((rangePtr->flags & REDRAW_OTHER) && (rangePtr->labelLength != 0)) {
	Tk_FontMetrics fm;

	Tk_GetFontMetrics(rangePtr->tkfont, &fm);
	Tk_DrawChars(rangePtr->display, drawable, rangePtr->textGC,
		rangePtr->tkfont, rangePtr->label, rangePtr->labelLength,
		rangePtr->vertLabelX, rangePtr->inset + (3*fm.ascent)/2);
    }
}

/*
 *----------------------------------------------------------------------
 *
 * DisplayVerticalValue --
 *
 *	This procedure is called to display values (range readings)
 *	for vertically-oriented ranges.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The numerical value corresponding to value is displayed with
 *	its right edge at "rightEdge", and at a vertical position in
 *	the range that corresponds to "value".
 *
 *----------------------------------------------------------------------
 */

static void
DisplayVerticalValue(rangePtr, drawable, value, rightEdge)
    register TkRange *rangePtr;	/* Information about widget in which to
				 * display value. */
    Drawable drawable;		/* Pixmap or window in which to draw
				 * the value. */
    double value;		/* Y-coordinate of number to display,
				 * specified in application coords, not
				 * in pixels (we'll compute pixels). */
    int rightEdge;		/* X-coordinate of right edge of text,
				 * specified in pixels. */
{
    register Tk_Window tkwin = rangePtr->tkwin;
    int y, width, length;
    char valueString[PRINT_CHARS];
    Tk_FontMetrics fm;

    Tk_GetFontMetrics(rangePtr->tkfont, &fm);
    y = TkpRangeValueToPixel(rangePtr, value) + fm.ascent/2;
    sprintf(valueString, rangePtr->format, value);
    length = strlen(valueString);
    width = Tk_TextWidth(rangePtr->tkfont, valueString, length);

    /*
     * Adjust the y-coordinate if necessary to keep the text entirely
     * inside the window.
     */

    if ((y - fm.ascent) < (rangePtr->inset + SPACING)) {
	y = rangePtr->inset + SPACING + fm.ascent;
    }
    if ((y + fm.descent) > (Tk_Height(tkwin) - rangePtr->inset - SPACING)) {
	y = Tk_Height(tkwin) - rangePtr->inset - SPACING - fm.descent;
    }
    Tk_DrawChars(rangePtr->display, drawable, rangePtr->textGC,
	    rangePtr->tkfont, valueString, length, rightEdge - width, y);
}

/*
 *--------------------------------------------------------------
 *
 * DisplayHorizontalRange --
 *
 *	This procedure redraws the contents of a horizontal range
 *	window.  It is invoked as a do-when-idle handler, so it only
 *	runs when there's nothing else for the application to do.
 *
 * Results:
 *	There is no return value.  If only a part of the range needs
 *	to be redrawn, then drawnAreaPtr is modified to reflect the
 *	area that was actually modified.
 *
 * Side effects:
 *	Information appears on the screen.
 *
 *--------------------------------------------------------------
 */

static void
DisplayHorizontalRange(rangePtr, drawable, drawnAreaPtr)
    TkRange *rangePtr;			/* Widget record for range. */
    Drawable drawable;			/* Where to display range (window
					 * or pixmap). */
    XRectangle *drawnAreaPtr;		/* Initally contains area of window;
					 * if only a part of the range is
					 * redrawn, gets modified to reflect
					 * the part of the window that was
					 * redrawn. */
{
    register Tk_Window tkwin = rangePtr->tkwin;
    int x1, x2, y, width, height, shadowWidth;
    double tickValue;
    Tk_3DBorder sliderBorder;
    int rangeFillWidth;

    /*
     * Display the information from bottom to top across the window.
     */

    if (!(rangePtr->flags & REDRAW_OTHER)) {
	drawnAreaPtr->x = rangePtr->inset;
	drawnAreaPtr->y = rangePtr->horizValueY;
	drawnAreaPtr->width -= 2*rangePtr->inset;
	drawnAreaPtr->height = rangePtr->horizTroughY + rangePtr->width
		+ 2*rangePtr->borderWidth - rangePtr->horizValueY;
    }
    Tk_Fill3DRectangle(tkwin, drawable, rangePtr->bgBorder,
	    drawnAreaPtr->x, drawnAreaPtr->y, drawnAreaPtr->width,
	    drawnAreaPtr->height, 0, TK_RELIEF_FLAT);
    if (rangePtr->flags & REDRAW_OTHER) {
	/*
	 * Display the tick marks.
	 */

	if (rangePtr->tickInterval != 0) {
	    for (tickValue = rangePtr->fromValue; ;
		    tickValue += rangePtr->tickInterval) {
		/*
		 * The TkRoundToResolution call gets rid of accumulated
		 * round-off errors, if any.
		 */

		tickValue = TkRoundToResolution(rangePtr, tickValue);
		if (rangePtr->toValue >= rangePtr->fromValue) {
		    if (tickValue > rangePtr->toValue) {
			break;
		    }
		} else {
		    if (tickValue < rangePtr->toValue) {
			break;
		    }
		}
		DisplayHorizontalValue(rangePtr, drawable, tickValue,
			rangePtr->horizTickY);
	    }
	}
    }

    /*
     * Display the value, if it is desired.
     */

    if (rangePtr->showValue) {
	DisplayHorizontalValue(rangePtr, drawable, rangePtr->min_value,
		rangePtr->horizValueY);
	DisplayHorizontalValue(rangePtr, drawable, rangePtr->max_value,
		rangePtr->horizValueY);
    }

    /*
     * Display the trough and the slider.
     */

    y = rangePtr->horizTroughY;
    Tk_Draw3DRectangle(tkwin, drawable,
	    rangePtr->bgBorder, rangePtr->inset, y,
	    Tk_Width(tkwin) - 2*rangePtr->inset,
	    rangePtr->width + 2*rangePtr->borderWidth,
	    rangePtr->borderWidth, TK_RELIEF_SUNKEN);
    XFillRectangle(rangePtr->display, drawable, rangePtr->troughGC,
	    rangePtr->inset + rangePtr->borderWidth,
	    y + rangePtr->borderWidth,
	    (unsigned) (Tk_Width(tkwin) - 2*rangePtr->inset
		- 2*rangePtr->borderWidth),
	    (unsigned) rangePtr->width);
    if (rangePtr->state == tkActiveUid) {
	sliderBorder = rangePtr->activeBorder;
    } else {
	sliderBorder = rangePtr->bgBorder;
    }
    width = rangePtr->sliderLength/2;
    height = rangePtr->width;
    x1 = TkpRangeValueToPixel(rangePtr, rangePtr->min_value) - width;
    x2 = TkpRangeValueToPixel(rangePtr, rangePtr->max_value);
    y += rangePtr->borderWidth;
    shadowWidth = rangePtr->borderWidth/2;
    if (shadowWidth == 0) {
	shadowWidth = 1;
    }
    rangeFillWidth = x2 - x1 - width - 2*shadowWidth;
    if (rangeFillWidth>0) {		/* enough space to draw range fill */
	XFillRectangle(rangePtr->display, drawable, rangePtr->rangeGC,
		       x1 + width + shadowWidth,
		       y,
		       (unsigned) rangeFillWidth,
		       (unsigned) rangePtr->width);
    }
    Tk_Draw3DRectangle(tkwin, drawable, sliderBorder,
	    x1, y, width, height, shadowWidth, rangePtr->sliderRelief);
    Tk_Draw3DRectangle(tkwin, drawable, sliderBorder,
	    x2, y, width, height, shadowWidth, rangePtr->sliderRelief);
    x1 += shadowWidth;
    x2 += shadowWidth;
    y += shadowWidth;
    width -= shadowWidth;
    height -= 2*shadowWidth;
    Tk_Fill3DRectangle(tkwin, drawable, sliderBorder, x1, y, width, height,
	    shadowWidth, rangePtr->sliderRelief);
    Tk_Fill3DRectangle(tkwin, drawable, sliderBorder, x2, y,
	    width, height, shadowWidth, rangePtr->sliderRelief);

    /*
     * Draw the label at the top of the range.
     */

    if ((rangePtr->flags & REDRAW_OTHER) && (rangePtr->labelLength != 0)) {
	Tk_FontMetrics fm;

	Tk_GetFontMetrics(rangePtr->tkfont, &fm);
	Tk_DrawChars(rangePtr->display, drawable, rangePtr->textGC,
		rangePtr->tkfont, rangePtr->label, rangePtr->labelLength,
		rangePtr->inset + fm.ascent/2, rangePtr->horizLabelY + fm.ascent);
    }
}

/*
 *----------------------------------------------------------------------
 *
 * DisplayHorizontalValue --
 *
 *	This procedure is called to display values (range readings)
 *	for horizontally-oriented ranges.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The numerical value corresponding to value is displayed with
 *	its bottom edge at "bottom", and at a horizontal position in
 *	the range that corresponds to "value".
 *
 *----------------------------------------------------------------------
 */

static void
DisplayHorizontalValue(rangePtr, drawable, value, top)
    register TkRange *rangePtr;	/* Information about widget in which to
				 * display value. */
    Drawable drawable;		/* Pixmap or window in which to draw
				 * the value. */
    double value;		/* X-coordinate of number to display,
				 * specified in application coords, not
				 * in pixels (we'll compute pixels). */
    int top;			/* Y-coordinate of top edge of text,
				 * specified in pixels. */
{
    register Tk_Window tkwin = rangePtr->tkwin;
    int x, y, length, width;
    char valueString[PRINT_CHARS];
    Tk_FontMetrics fm;

    x = TkpRangeValueToPixel(rangePtr, value);
    Tk_GetFontMetrics(rangePtr->tkfont, &fm);
    y = top + fm.ascent;
    sprintf(valueString, rangePtr->format, value);
    length = strlen(valueString);
    width = Tk_TextWidth(rangePtr->tkfont, valueString, length);

    /*
     * Adjust the x-coordinate if necessary to keep the text entirely
     * inside the window.
     */

    x -= (width)/2;
    if (x < (rangePtr->inset + SPACING)) {
	x = rangePtr->inset + SPACING;
    }
    if (x > (Tk_Width(tkwin) - rangePtr->inset)) {
	x = Tk_Width(tkwin) - rangePtr->inset - SPACING - width;
    }
    Tk_DrawChars(rangePtr->display, drawable, rangePtr->textGC,
	    rangePtr->tkfont, valueString, length, x, y);
}

/*
 *----------------------------------------------------------------------
 *
 * TkpDisplayRange --
 *
 *	This procedure is invoked as an idle handler to redisplay
 *	the contents of a range widget.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The range gets redisplayed.
 *
 *----------------------------------------------------------------------
 */

void
TkpDisplayRange(clientData)
    ClientData clientData;	/* Widget record for range. */
{
    TkRange *rangePtr = (TkRange *) clientData;
    Tk_Window tkwin = rangePtr->tkwin;
    Tcl_Interp *interp = rangePtr->interp;
    Pixmap pixmap;
    int result;
    char string[PRINT_CHARS];
    XRectangle drawnArea;

    if ((rangePtr->tkwin == NULL) || !Tk_IsMapped(rangePtr->tkwin)) {
	goto done;
    }

    /*
     * Invoke the range's command if needed.
     */

    Tcl_Preserve((ClientData) rangePtr);
    Tcl_Preserve((ClientData) interp);
    if ((rangePtr->flags & INVOKE_COMMAND) && (rangePtr->command != NULL)) {
	sprintf(string, rangePtr->format, rangePtr->min_value);
	result = Tcl_VarEval(interp, rangePtr->command,	" ", string,
                             (char *) NULL);
	if (result != TCL_OK) {
	    Tcl_AddErrorInfo(interp, "\n    (command executed by range)");
	    Tcl_BackgroundError(interp);
	}
	sprintf(string, rangePtr->format, rangePtr->max_value);
	result = Tcl_VarEval(interp, rangePtr->command,	" ", string,
                             (char *) NULL);
	if (result != TCL_OK) {
	    Tcl_AddErrorInfo(interp, "\n    (command executed by range)");
	    Tcl_BackgroundError(interp);
	}
    }
    Tcl_Release((ClientData) interp);
    rangePtr->flags &= ~INVOKE_COMMAND;
    if (rangePtr->tkwin == NULL) {
	Tcl_Release((ClientData) rangePtr);
	return;
    }
    Tcl_Release((ClientData) rangePtr);

    /*
     * In order to avoid screen flashes, this procedure redraws
     * the range in a pixmap, then copies the pixmap to the
     * screen in a single operation.  This means that there's no
     * point in time where the on-sreen image has been cleared.
     */

    pixmap = Tk_GetPixmap(rangePtr->display, Tk_WindowId(tkwin),
	    Tk_Width(tkwin), Tk_Height(tkwin), Tk_Depth(tkwin));
    drawnArea.x = 0;
    drawnArea.y = 0;
    drawnArea.width = Tk_Width(tkwin);
    drawnArea.height = Tk_Height(tkwin);

    /*
     * Much of the redisplay is done totally differently for
     * horizontal and vertical ranges.  Handle the part that's
     * different.
     */

    if (rangePtr->vertical) {
	DisplayVerticalRange(rangePtr, pixmap, &drawnArea);
    } else {
	DisplayHorizontalRange(rangePtr, pixmap, &drawnArea);
    }

    /*
     * Now handle the part of redisplay that is the same for
     * horizontal and vertical ranges:  border and traversal
     * highlight.
     */

    if (rangePtr->flags & REDRAW_OTHER) {
	if (rangePtr->relief != TK_RELIEF_FLAT) {
	    Tk_Draw3DRectangle(tkwin, pixmap, rangePtr->bgBorder,
		    rangePtr->highlightWidth, rangePtr->highlightWidth,
		    Tk_Width(tkwin) - 2*rangePtr->highlightWidth,
		    Tk_Height(tkwin) - 2*rangePtr->highlightWidth,
		    rangePtr->borderWidth, rangePtr->relief);
	}
	if (rangePtr->highlightWidth != 0) {
	    GC gc;
    
	    if (rangePtr->flags & GOT_FOCUS) {
		gc = Tk_GCForColor(rangePtr->highlightColorPtr, pixmap);
	    } else {
		gc = Tk_GCForColor(rangePtr->highlightBgColorPtr, pixmap);
	    }
	    Tk_DrawFocusHighlight(tkwin, gc, rangePtr->highlightWidth, pixmap);
	}
    }

    /*
     * Copy the information from the off-screen pixmap onto the screen,
     * then delete the pixmap.
     */

    XCopyArea(rangePtr->display, pixmap, Tk_WindowId(tkwin),
	    rangePtr->copyGC, drawnArea.x, drawnArea.y, drawnArea.width,
	    drawnArea.height, drawnArea.x, drawnArea.y);
    Tk_FreePixmap(rangePtr->display, pixmap);

    done:
    rangePtr->flags &= ~REDRAW_ALL;
}

/*
 *----------------------------------------------------------------------
 *
 * TkpRangeElement --
 *
 *	Determine which part of a range widget lies under a given
 *	point.
 *
 * Results:
 *	The return value is either TROUGH1, SLIDER, TROUGH2, or
 *	OTHER, depending on which of the range's active elements
 *	(if any) is under the point at (x,y).
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

int
TkpRangeElement(rangePtr, x, y)
    TkRange *rangePtr;		/* Widget record for range. */
    int x, y;			/* Coordinates within rangePtr's window. */
{
    int sliderMin;
    int sliderMax;
    if (rangePtr->vertical) {
	if ((x < rangePtr->vertTroughX)
		|| (x >= (rangePtr->vertTroughX + 2*rangePtr->borderWidth +
		rangePtr->width))) {
	    return OTHER;
	}
	if ((y < rangePtr->inset)
		|| (y >= (Tk_Height(rangePtr->tkwin) - rangePtr->inset))) {
	    return OTHER;
	}
	sliderMin = TkpRangeValueToPixel(rangePtr, rangePtr->min_value)
		- rangePtr->sliderLength/2;
	sliderMax = TkpRangeValueToPixel(rangePtr, rangePtr->max_value)
		- rangePtr->sliderLength/2;
	if (y < sliderMin) {
	    return TROUGH1;
	}
	if (y > sliderMax) {
	    return TROUGH2;
	}
	sliderMin += rangePtr->sliderLength/2;
	sliderMax -= rangePtr->sliderLength/2;
	if (y < sliderMin) {
	    return MIN_SLIDER;
	}
	if (y > sliderMax) {
	    return MAX_SLIDER;
	}
	return RANGE;
    }

    if ((y < rangePtr->horizTroughY)
	    || (y >= (rangePtr->horizTroughY + 2*rangePtr->borderWidth +
	    rangePtr->width))) {
	return OTHER;
    }
    if ((x < rangePtr->inset)
	    || (x >= (Tk_Width(rangePtr->tkwin) - rangePtr->inset))) {
	return OTHER;
    }
    sliderMin = TkpRangeValueToPixel(rangePtr, rangePtr->min_value)-
	rangePtr->sliderLength/2;
    sliderMax = TkpRangeValueToPixel(rangePtr, rangePtr->max_value)+
	rangePtr->sliderLength/2;
    if (x < sliderMin) {
	return TROUGH1;
    }
    if (x > sliderMax) {
	return TROUGH2;
    }
    sliderMin += rangePtr->sliderLength/2;
    sliderMax -= rangePtr->sliderLength/2;
    if (x < sliderMin) {
	return MIN_SLIDER;
    }
    if (x > sliderMax) {
	return MAX_SLIDER;
    }
    return RANGE;
}

/*
 *--------------------------------------------------------------
 *
 * TkpSetRangeValue --
 *
 *	This procedure changes the value of a range and invokes
 *	a Tcl command to reflect the current position of a range
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	A Tcl command is invoked, and an additional error-processing
 *	command may also be invoked.  The range's slider is redrawn.
 *
 *--------------------------------------------------------------
 */

void
TkpSetRangeMinValue(rangePtr, value, setVar, invokeCommand)
    register TkRange *rangePtr;	/* Info about widget. */
    double value;		/* New value for range.  Gets adjusted
				 * if it's off the range. */
    int setVar;			/* Non-zero means reflect new value through
				 * to associated variable, if any. */
    int invokeCommand;		/* Non-zero means invoked -command option
				 * to notify of new value, 0 means don't. */
{
    char string[PRINT_CHARS];

    value = TkRoundToResolution(rangePtr, value);
    if ((value < rangePtr->fromValue)
	    ^ (rangePtr->toValue < rangePtr->fromValue)) {
	value = rangePtr->fromValue;
    }
    if ((value > rangePtr->toValue)
	    ^ (rangePtr->toValue < rangePtr->fromValue)) {
	value = rangePtr->toValue;
    }
    if (rangePtr->flags & NEVER_SET) {
	rangePtr->flags &= ~NEVER_SET;
    } else if (rangePtr->min_value == value) {
	return;
    }
    rangePtr->min_value = value;
    if (invokeCommand) {
	rangePtr->flags |= INVOKE_COMMAND;
    }
    TkEventuallyRedrawRange(rangePtr, REDRAW_SLIDER);

    if (setVar && (rangePtr->min_varName != NULL)) {
	sprintf(string, rangePtr->format, rangePtr->min_value);
	rangePtr->flags |= SETTING_VAR;
	Tcl_SetVar(rangePtr->interp, rangePtr->min_varName, string,
	       TCL_GLOBAL_ONLY);
	rangePtr->flags &= ~SETTING_VAR;
    }
}


/*
 *--------------------------------------------------------------
 *
 * TkpSetRangeValue --
 *
 *	This procedure changes the value of a range and invokes
 *	a Tcl command to reflect the current position of a range
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	A Tcl command is invoked, and an additional error-processing
 *	command may also be invoked.  The range's slider is redrawn.
 *
 *--------------------------------------------------------------
 */

void
TkpSetRangeMaxValue(rangePtr, value, setVar, invokeCommand)
    register TkRange *rangePtr;	/* Info about widget. */
    double value;		/* New value for range.  Gets adjusted
				 * if it's off the range. */
    int setVar;			/* Non-zero means reflect new value through
				 * to associated variable, if any. */
    int invokeCommand;		/* Non-zero means invoked -command option
				 * to notify of new value, 0 means don't. */
{
    char string[PRINT_CHARS];

    value = TkRoundToResolution(rangePtr, value);
    if ((value < rangePtr->fromValue)
	    ^ (rangePtr->toValue < rangePtr->fromValue)) {
	value = rangePtr->fromValue;
    }
    if ((value > rangePtr->toValue)
	    ^ (rangePtr->toValue < rangePtr->fromValue)) {
	value = rangePtr->toValue;
    }
    if (rangePtr->flags & NEVER_SET) {
	rangePtr->flags &= ~NEVER_SET;
    } else if (rangePtr->max_value == value) {
	return;
    }
    rangePtr->max_value = value;
    if (invokeCommand) {
	rangePtr->flags |= INVOKE_COMMAND;
    }
    TkEventuallyRedrawRange(rangePtr, REDRAW_SLIDER);

    if (setVar && (rangePtr->max_varName != NULL)) {
	sprintf(string, rangePtr->format, rangePtr->max_value);
	rangePtr->flags |= SETTING_VAR;
	Tcl_SetVar(rangePtr->interp, rangePtr->max_varName, string,
	       TCL_GLOBAL_ONLY);
	rangePtr->flags &= ~SETTING_VAR;
    }
}

/*
 *----------------------------------------------------------------------
 *
 * TkRangePixelToValue --
 *
 *	Given a pixel within a range window, return the range
 *	reading corresponding to that pixel.
 *
 * Results:
 *	A double-precision range reading.  If the value is outside
 *	the legal range for the range then it's rounded to the nearest
 *	end of the range.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

/* Dd: Does this really need to be static? */
double
TkRangePixelToValue(rangePtr, x, y)
    register TkRange *rangePtr;		/* Information about widget. */
    int x, y;				/* Coordinates of point within
					 * window. */
{
    double value, pixelRange;

    if (rangePtr->vertical) {
	pixelRange = Tk_Height(rangePtr->tkwin) - rangePtr->sliderLength
		- 2*rangePtr->inset - 2*rangePtr->borderWidth;
	value = y;
    } else {
	pixelRange = Tk_Width(rangePtr->tkwin) - rangePtr->sliderLength
		- 2*rangePtr->inset - 2*rangePtr->borderWidth;
	value = x;
    }

    if (pixelRange <= 0) {
	/*
	 * Not enough room for the slider to actually slide:  just return
	 * the range's current value.
	 */

	return rangePtr->min_value;
    }
    value -= rangePtr->sliderLength/2 + rangePtr->inset
		+ rangePtr->borderWidth;
    value /= pixelRange;
    if (value < 0) {
	value = 0;
    }
    if (value > 1) {
	value = 1;
    }
    value = rangePtr->fromValue +
		value * (rangePtr->toValue - rangePtr->fromValue);
    return TkRoundToResolution(rangePtr, value);
}

/*
 *----------------------------------------------------------------------
 *
 * TkpRangeValueToPixel --
 *
 *	Given a reading of the range, return the x-coordinate or
 *	y-coordinate corresponding to that reading, depending on
 *	whether the range is vertical or horizontal, respectively.
 *
 * Results:
 *	An integer value giving the pixel location corresponding
 *	to reading.  The value is restricted to lie within the
 *	defined range for the range.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

int
TkpRangeValueToPixel(rangePtr, value)
    register TkRange *rangePtr;		/* Information about widget. */
    double value;			/* Reading of the widget. */
{
    int y, pixelRange;
    double valueRange;

    valueRange = rangePtr->toValue - rangePtr->fromValue;
    pixelRange = (rangePtr->vertical ? Tk_Height(rangePtr->tkwin)
	    : Tk_Width(rangePtr->tkwin)) - rangePtr->sliderLength
	    - 2*rangePtr->inset - 2*rangePtr->borderWidth;
    if (valueRange == 0) {
	y = 0;
    } else {
	y = (int) ((value - rangePtr->fromValue) * pixelRange
		  / valueRange + 0.5);
	if (y < 0) {
	    y = 0;
	} else if (y > pixelRange) {
	    y = pixelRange;
	}
    }
    y += rangePtr->sliderLength/2 + rangePtr->inset + rangePtr->borderWidth;
    return y;
}
