//static char *id="@(#) $Id$";

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

#include <SCICore/Geom/GeomTube.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/Math/TrigTable.h>

namespace SCICore {
namespace GeomSpace {

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
	bb.extend_cyl(verts[i]->p, directions[i], radii[i]);
}

void GeomTube::get_bounds(BSphere&)
{
    NOT_FINISHED("GeomTube::get_bounds");
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

void GeomTube::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomTube::make_prims");
}

void GeomTube::preprocess()
{
    NOT_FINISHED("GeomTube::preprocess");
}

void GeomTube::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomTube::intersect");
}

#define GEOMTUBE_VERSION 1

void GeomTube::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    stream.begin_class("GeomTube", GEOMTUBE_VERSION);
    GeomVertexPrim::io(stream);
    Pio(stream, nu);
    Pio(stream, directions);
    Pio(stream, radii);
    stream.end_class();
}

bool GeomTube::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomTube::saveobj");
    return false;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:17  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:47  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:54  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

