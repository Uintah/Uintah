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


#include <sci_config.h>
#include <Core/share/share.h>

#include <tk.h>
#include <tk3d.h>

SCICORESHARE GC Tk_BorderDarkGC(Tk_3DBorder border);
SCICORESHARE GC Tk_BorderLightGC(Tk_3DBorder border);
SCICORESHARE GC Tk_BorderBgGC(Tk_3DBorder border);
SCICORESHARE void Tk_BorderShiftLine(XPoint *p1Ptr, XPoint *p2Ptr,
			int distance, XPoint *p3Ptr);
SCICORESHARE int Tk_BorderIntersect(XPoint *a1Ptr, XPoint *a2Ptr, XPoint *b1Ptr,
		       XPoint *b2Ptr, XPoint *iPtr);


/*
 *--------------------------------------------------------------
 *
 * Tk_DrawBeveledLine -- 
 *
 *	Draw a line with 3-D appearance on both sides
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
Tk_DrawBeveledLine(display, drawable, border, pointPtr, numPoints,
		   width, borderWidth, relief)
    Display *display;		/* X display in which to draw polygon. */
    Drawable drawable;		/* X window or pixmap in which to draw. */
    Tk_3DBorder border;		/* Token for border to draw. */
    XPoint *pointPtr;		/* Array of points describing
				 * polygon.  All points must be
				 * absolute (CoordModeOrigin). */
    int numPoints;		/* Number of points at *pointPtr. */
    int width;
    int borderWidth;		/* Width of border, measured in
				 * pixels to the left of the polygon's
				 * trajectory.   May be negative. */
    int relief;			/* TK_RELIEF_RAISED or
				 * TK_RELIEF_SUNKEN: indicates how
				 * stuff to left of trajectory looks
				 * relative to stuff on right. */
{
    XPoint x1, x2, x3, x4;
    XPoint b1, b2, b3, b4;
    XPoint c1, c2, c3, c4;
    XPoint* p1Ptr, *p2Ptr, *p3Ptr=0;
    XPoint poly[4];
    int lightOnLeft;
    int dx, dy;
    GC gc;
    int i;
    int dist1=(width+1)/2;
    int dist2=dist1-borderWidth;
    int dist4=dist1-width;
    int dist3=dist4+borderWidth;

    /* struct TkBorder_struct *b = (struct TkBorder_struct*)border; */
    /* used for debugging */

    GC gcDark = Tk_BorderDarkGC(border);
    GC gcLight = Tk_BorderLightGC(border);
    GC gcBg = Tk_BorderBgGC(border);

    if ( (gcDark==0) || (gcLight==0) || (gcBg==0) ) {
      if ( (gcDark==0) && (gcLight==0) && (gcBg==0) ) {
	printf("The GC's sent in the border to Tk_DrawBeveledLine() are all = 0\n");
	return;
      }
      
      if (gcDark==0) 
	gcDark = (gcLight!=0?gcLight:gcBg);
      
      if (gcLight==0)
	gcLight = (gcDark!=0?gcDark:gcBg);
      
      if (gcBg==0)
	gcBg = (gcLight!=0?gcLight:gcDark);
    }




    /*
     * Don't Handle grooves and ridges
     */

    if ((relief == TK_RELIEF_GROOVE) || (relief == TK_RELIEF_RIDGE)) {
	return;
    }
    if(numPoints < 2)
	return;

    p1Ptr=&pointPtr[0];
    p2Ptr=&pointPtr[1];
    i=1;
    while ((p2Ptr->x == p1Ptr->x) && (p2Ptr->y == p1Ptr->y)) {
	/*
	 * Ignore duplicate points (they'd cause core dumps in
	 * Tk_BorderShiftLine calls below).
	 */
	i++;
	p2Ptr=&pointPtr[i];
    }
    Tk_BorderShiftLine(p1Ptr, p2Ptr, dist1, &x1);
    Tk_BorderShiftLine(p1Ptr, p2Ptr, dist2, &x2);
    Tk_BorderShiftLine(p1Ptr, p2Ptr, dist3, &x3);
    Tk_BorderShiftLine(p1Ptr, p2Ptr, dist4, &x4);
    i++;
    for (; i < numPoints+1 ; i++){
	if(i<numPoints){
	    p3Ptr=&pointPtr[i];
	    if ((p2Ptr->x == p3Ptr->x) && (p2Ptr->y == p3Ptr->y)) {
		/*
		 * Ignore duplicate points (they'd cause core dumps in
		 * Tk_BorderShiftLine calls below).
		 */
		continue;
	    }
	    Tk_BorderShiftLine(p1Ptr, p2Ptr, dist1, &b1);
	    b2.x = b1.x + (p2Ptr->x - p1Ptr->x);
	    b2.y = b1.y + (p2Ptr->y - p1Ptr->y);
	    Tk_BorderShiftLine(p2Ptr, p3Ptr, dist1, &b3);
	    b4.x = b3.x + (p3Ptr->x - p2Ptr->x);
	    b4.y = b3.y + (p3Ptr->y - p2Ptr->y);
	    Tk_BorderIntersect(&b1, &b2, &b3, &b4, &c1);
	    Tk_BorderShiftLine(p1Ptr, p2Ptr, dist2, &b1);
	    b2.x = b1.x + (p2Ptr->x - p1Ptr->x);
	    b2.y = b1.y + (p2Ptr->y - p1Ptr->y);
	    Tk_BorderShiftLine(p2Ptr, p3Ptr, dist2, &b3);
	    b4.x = b3.x + (p3Ptr->x - p2Ptr->x);
	    b4.y = b3.y + (p3Ptr->y - p2Ptr->y);
	    Tk_BorderIntersect(&b1, &b2, &b3, &b4, &c2);
	    Tk_BorderShiftLine(p1Ptr, p2Ptr, dist3, &b1);
	    b2.x = b1.x + (p2Ptr->x - p1Ptr->x);
	    b2.y = b1.y + (p2Ptr->y - p1Ptr->y);
	    Tk_BorderShiftLine(p2Ptr, p3Ptr, dist3, &b3);
	    b4.x = b3.x + (p3Ptr->x - p2Ptr->x);
	    b4.y = b3.y + (p3Ptr->y - p2Ptr->y);
	    Tk_BorderIntersect(&b1, &b2, &b3, &b4, &c3);
	    Tk_BorderShiftLine(p1Ptr, p2Ptr, dist4, &b1);
	    b2.x = b1.x + (p2Ptr->x - p1Ptr->x);
	    b2.y = b1.y + (p2Ptr->y - p1Ptr->y);
	    Tk_BorderShiftLine(p2Ptr, p3Ptr, dist4, &b3);
	    b4.x = b3.x + (p3Ptr->x - p2Ptr->x);
	    b4.y = b3.y + (p3Ptr->y - p2Ptr->y);
	    Tk_BorderIntersect(&b1, &b2, &b3, &b4, &c4);
	} else {
	    Tk_BorderShiftLine(p2Ptr, p1Ptr, -dist1, &c1);
	    Tk_BorderShiftLine(p2Ptr, p1Ptr, -dist2, &c2);
	    Tk_BorderShiftLine(p2Ptr, p1Ptr, -dist3, &c3);
	    Tk_BorderShiftLine(p2Ptr, p1Ptr, -dist4, &c4);
	}

	dx = p2Ptr->x - p1Ptr->x;
	dy = p2Ptr->y - p1Ptr->y;
	if (dx > 0) {
	    lightOnLeft = (dy <= dx);
	} else {
	    lightOnLeft = (dy < dx);
	}	
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
	/* Mipspro doesn't like these expressions... */
#pragma set woff 3496
#endif
	if ( lightOnLeft ^ (relief == TK_RELIEF_RAISED)) {
	    gc = gcDark; /* Tk_BorderDarkGC(border); */
	} else {
	    gc = gcLight; /* Tk_BorderLightGC(border); */
	}
	poly[0]=x1;
	poly[1]=x2;
	poly[2]=c2;
	poly[3]=c1;
	XFillPolygon(display, drawable, gc, poly, 4, Convex,
		     CoordModeOrigin);
	gc = gcBg; /* Tk_BorderBgGC(border); */
	poly[0]=x2;
	poly[1]=x3;
	poly[2]=c3;
	poly[3]=c2;
	XFillPolygon(display, drawable, gc, poly, 4, Convex,
		     CoordModeOrigin);
	if ( lightOnLeft ^ (relief == TK_RELIEF_RAISED)) {
	    gc = gcLight; /* Tk_BorderLightGC(border); */
	} else {
	    gc = gcDark; /* Tk_BorderDarkGC(border); */
	}
	poly[0]=x3;
	poly[1]=x4;
	poly[2]=c4;
	poly[3]=c3;
	XFillPolygon(display, drawable, gc, poly, 4, Convex,
		     CoordModeOrigin);
	x1=c1;
	x2=c2;
	x3=c3;
	x4=c4;
	p1Ptr=p2Ptr;
	p2Ptr=p3Ptr;
    }
}
