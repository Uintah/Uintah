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

#include <sci_config.h> /* for HAVE_LIMITS etc, for tcl files */

#include <tk3d.h>
#include <math.h>


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
