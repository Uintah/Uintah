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
 *  GeomTube.cc: Tube object
 *
 *  Written by:
 *   Han-Wei Shen
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomTube.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/TrigTable.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent* make_GeomTube()
{
    return scinew GeomTube;
}

PersistentTypeID GeomTube::type_id("GeomTube", "GeomObj", make_GeomTube);

GeomTube::GeomTube(int nu)
: nu(nu)
{
}

GeomTube::GeomTube(const GeomTube& copy)
: GeomVertexPrim(copy), nu(copy.nu),
  directions(copy.directions), radii(copy.radii)
{
}

GeomTube::~GeomTube()
{
}

GeomObj* GeomTube::clone() 
{
    return scinew GeomTube(*this); 
}

void  GeomTube::get_bounds(BBox& bb)
{
    for(int i=0;i<verts.size();i++)
    {
	bb.extend_disc(verts[i]->p, directions[i], radii[i]);
    }
}

// the function to extend the length of the tube geometry
void GeomTube::add(GeomVertex* vtx, double radius, const Vector& dir)
{
    GeomVertexPrim::add(vtx);  // Add the vertex - point and maybe color
    radii.add(radius);     // specify the radius of that point
    directions.add(dir.normal());    // and the direction 
}

// Given a center point and its normal, compute those points on the 
// circle, this is a private member function, called  by 
// objdraw function
void GeomTube::make_circle(int which, Array1<Point>& circle_pts,
			   const SinCosTable& tab)
{
    Vector dir (directions[which]);
    double u = dir.x();
    double v = dir.y();
    double w = dir.z(); 
    double mat[6];
    if(w < -.999999){
	mat[0]=-1;
	mat[1]=0;
	mat[2]=0;
	mat[3]=0;
	mat[4]=-1;
	mat[5]=0;
    } else {
	double w1=1+w;
	mat[0]=v*v/w1 + w;
	mat[1]=-u*v/w1;
	mat[2]=-u;
	mat[3]=mat[1];
	mat[4]=u*u/w1 + w;
	mat[5]=-v;
    }

    circle_pts.remove_all();
    Point pt(verts[which]->p);
    double radius=radii[which];
    for (int i=0; i<=nu; i++) { // temporarily set the number of grids as 20
	double cx = tab.sin(i)*radius;
	double cy = tab.cos(i)*radius;
	Point circle_pt(cx * mat[0] + cy*mat[3] + pt.x(),
			cx * mat[1] + cy*mat[4] + pt.y(),
			cx * mat[2] + cy*mat[5] + pt.z()); 
	circle_pts.add(circle_pt); 
    }
}

#define GEOMTUBE_VERSION 1

void GeomTube::io(Piostream& stream)
{

    stream.begin_class("GeomTube", GEOMTUBE_VERSION);
    GeomVertexPrim::io(stream);
    Pio(stream, nu);
    Pio(stream, directions);
    Pio(stream, radii);
    stream.end_class();
}

} // End namespace SCIRun


