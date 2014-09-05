#ifndef NAUTILUS_VISQUADRICS_INCLUDED // -*- C++ -*-
#define NAUTILUS_VISQUADRICS_INCLUDED

/************************************************************************

  Ellipsoid drawing stuff.
  $Id$

 ************************************************************************/

#define ELLIPSOID_RED 1
#define ELLIPSOID_GREEN 2

extern GLUquadricObj *newEllipsoidContext();
extern void lightEllipsoids(int color=ELLIPSOID_GREEN);
extern void drawEllipsoid(GLUquadricObj *,const Mat4&,
			  real radius, Vec3 *v=NULL);

// NAUTILUS_VISQUADRICS_INCLUDED
#endif
