/*
 *  Tube.cc: Tube object
 *
 *  Written by:
 *   Han-Wei Shen
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Tube.h>
#include <Geometry/BBox.h>
#include <Geom/Line.h>
#include <Classlib/NotFinished.h>
#include <math.h>


GeomTube::GeomTube()
: GeomObj(0)
{

}

GeomTube::GeomTube(const GeomTube& copy)
:GeomObj(0), pts(copy.pts), rad(copy.rad),
 normal(copy.normal)
{
}

GeomTube::~GeomTube()
{
}

GeomObj* GeomTube::clone() 
{
  return new GeomTube(*this); 
}

void  GeomTube::get_bounds(BBox& bb)
{
    for(int i=0;i<pts.size();i++)
	bb.extend(pts[i]);
}

// the function to extend the length of the tube geometry
int GeomTube::add(Point pt, double radius, Vector norm)
{
pts.add(pt);         // add one more point to tube's path
rad.add(radius);     // specify the radius of that point
normal.add(norm);    // and the direction 
return(1); 
}

// Given a center point and its normal, compute those points on the 
// circle, this is a private member function, called  by 
// objdraw function
Array1<Point> GeomTube::make_circle(Point pt, double radius, Vector norm)
{

double Pi; 
Vector dir; 
double mat[9]; 
double u, v, w; 
Point circle_pt; 
double cx, cy, cz; 
int i; 
Array1<Point> cir; 

dir = norm.normal();        // the normal of tube circle
Pi = 3.1415927; 
u = dir.x(); v = dir.y(); w = dir.z(); 
mat[0] = v*v*(1-w) + w;     // transformation matrix
mat[1] = -1 * u*v*(1-w); 
mat[2] = sqrt(u*u + v*v)*u; 
mat[3] = mat[1]; 
mat[4] = u*u*(1-w) + w; 
mat[5] = sqrt(u*u + v*v)*v; 
mat[6] = -1 * mat[2]; 
mat[7] = -1 * mat[5]; 
mat[8] = w; 

for (i=0; i<20; i++) { // temporarily set the number of grids as 20
  cx = radius * sin(Pi*i/20.0); 
  cy = radius * cos(Pi*i/20.0); 
  cz = 0; 
  circle_pt.x(cx * mat[0] + cy*mat[3] + pt.x()); 
  circle_pt.y(cx * mat[1] + cy*mat[4] + pt.y()); 
  circle_pt.z(cx * mat[2] + cy*mat[5] + pt.z()); 
  cir.add(circle_pt); 
}
return(cir); 
}

void GeomTube::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>&)
{
  NOT_FINISHED("GeomTube::make_prims");
}

void GeomTube::intersect(const Ray&, const MaterialHandle&,
			 Hit&)
{
    NOT_FINISHED("GeomTube::intersect");
}
