//static char *id="@(#) $Id$";

/*
 *  ContoursToSurf.cc:  Merge multiple contour sets into a Surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Containers/Array1.h>
#include <SCICore/Util/Assert.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/CoreDatatypes/ContourSet.h>
#include <PSECore/CommonDatatypes/ContourSetPort.h>
#include <SCICore/CoreDatatypes/Surface.h>
#include <PSECore/CommonDatatypes/SurfacePort.h>
#include <SCICore/CoreDatatypes/TriSurface.h>
#include <SCICore/Geometry/Grid.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/Expon.h>

#include <iostream.h>

namespace PSECommon {
namespace Modules {

using PSECore::Dataflow::Module;
using PSECore::CommonDatatypes::ContourSetIPort;
using PSECore::CommonDatatypes::SurfaceOPort;
using PSECore::CommonDatatypes::SurfaceIPort;
using PSECore::CommonDatatypes::SurfaceHandle;
using PSECore::CommonDatatypes::TriSurface;
using PSECore::CommonDatatypes::ContourSetHandle;

using SCICore::Containers::Array1;
using SCICore::Containers::clString;
using SCICore::Geometry::BBox;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;

class ContoursToSurf : public Module {
    Array1<ContourSetIPort*> incontours;
    SurfaceOPort* osurface;
    BBox bbox;
    void break_triangle(int tri_id, int pt_id, const Point& p, TriSurface*);
    void break_edge(int tri1,int tri2,int e1,int e2,int pt_id,const Point &p,
		    TriSurface*);
    void break_edge2(int tri1, int e1, int pt_id,const Point &p, TriSurface*);
    void lace_contours(const ContourSetHandle& contour, TriSurface* surf);
    void add_point(const Point& p, TriSurface* surf);
    void contours_to_surf(const Array1<ContourSetHandle> &contours, TriSurface*);
public:
    ContoursToSurf(const clString&);
    virtual ~ContoursToSurf();
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
};

Module* make_ContoursToSurf(const clString& id) {
  return new ContoursToSurf(id);
}

ContoursToSurf::ContoursToSurf(const clString& id)
: Module("ContoursToSurf", id, Filter)
{
    // Create the input port
    incontours.add(scinew ContourSetIPort(this, "ContourSet", 
				       ContourSetIPort::Atomic));

    add_iport(incontours[0]);
    osurface=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurface);
}

void ContoursToSurf::connection(ConnectionMode mode, int which_port,
			       int output)
{
    if (output) return;
    if (mode==Disconnected) {
	remove_iport(which_port);
	delete incontours[which_port];
	incontours.remove(which_port);
    } else {
	ContourSetIPort* ci=scinew ContourSetIPort(this, "ContourSet", 
						ContourSetIPort::Atomic);
	add_iport(ci);
	incontours.add(ci);
    }
}
	
ContoursToSurf::~ContoursToSurf()
{
}

void ContoursToSurf::execute()
{
    Array1<ContourSetHandle> contours(incontours.size()-1);
    int flag;
    int i;
    for (flag=0, i=0; i<incontours.size()-1; i++)
	if (!incontours[i]->get(contours[i]))
	    flag=1;
    if (flag) return;
    TriSurface* surf=scinew TriSurface;
    contours_to_surf(contours, surf);
    surf->remove_empty_index();		// just in case
    osurface->send(SurfaceHandle(surf));
}


// We go through the contours pairwise and lace them together.
// For each pair we take a starting point, and then find the corresponding
// closest point on the other contour.  Then we choose which contour
// to advance along by taking the one that creates the shortest diagonal.
// This continues until all of the points are laced, and then we move up
// to the next contour.

void ContoursToSurf::lace_contours(const ContourSetHandle& contour, 
				   TriSurface* surf) {
    Array1<int> row;	
    int i, curr;
    for (i=curr=0; i<contour->contours.size(); i++) {
	row.add(curr);	
	curr+=contour->contours[i].size();
	for (int j=0; j<contour->contours[i].size(); j++) {
	    Point p(contour->contours[i][j]-contour->origin);
	    p=Point(0,0,0)+contour->basis[0]*p.x()+contour->basis[1]*p.y()+
		contour->basis[2]*(p.z()*contour->space);
	    surf->add_point(p);
	}
    }
   // i will be the index of the top contour being laced, i-1 being the other
   for (i=1; i<contour->contours.size(); i++) {
       int top=0;
       double dtemp;
       int sz_top=contour->contours[i].size();
       int sz_bot=contour->contours[i-1].size();
       if ((sz_top < 2) && (sz_bot < 2)) {
	   cerr << "Not enough points to lace!\n";
	   return;
       }
       // 0 will be the index of our first bottom point, set top to be the 
       // index of the closest top point to it
       double dist=Sqr(contour->contours[i][0].x()-
		       contour->contours[i-1][0].x())+
		   Sqr(contour->contours[i][0].y()-
		       contour->contours[i-1][0].y());
       for (int start=1; start<sz_top; start++) {
	   if ((dtemp=(Sqr(contour->contours[i][start].x()-
			   contour->contours[i-1][0].x())+
		       Sqr(contour->contours[i][start].y()-
			   contour->contours[i-1][0].y())))<dist) {
	       top=start;
	       dist=dtemp;
	   }
       }
       int bot=0;
       // lets start lacing...  top and bottom will always store the indices
       // of the first matched points so we know when to stop
       int jdone=(sz_top==1); // does this val have to change for us to
       int kdone=(sz_bot==1); // be done lacing
       for (int j=top,k=bot; !jdone || !kdone;) {
	   double d1=Sqr(contour->contours[i][j].x()-
			 contour->contours[i-1][(k+1)%sz_bot].x())+
	       Sqr(contour->contours[i][j].y()-
		   contour->contours[i-1][(k+1)%sz_bot].y());
	   double d2=Sqr(contour->contours[i][(j+1)%sz_top].x()-
			 contour->contours[i-1][k].x())+
	             Sqr(contour->contours[i][(j+1)%sz_top].y()-
			 contour->contours[i-1][k].y());
	   if ((d1<d2 || jdone) && !kdone){ 	// bottom moves
	       surf->add_triangle(row[i]+j,row[i-1]+k,row[i-1]+((k+1)%sz_bot),1);
	       k=(k+1)%sz_bot;
	       if (k==bot) kdone=1;
	   } else {     			// top moves
	       surf->add_triangle(row[i]+j,row[i-1]+k,row[i]+((j+1)%sz_top),1);
	       j=(j+1)%sz_top;
	       if (j==top) jdone=1;
	   }
       }
   }
}


// if the point we're adding is closest to a face, we break that face
// and add the three new traingles.

void ContoursToSurf::break_triangle(int tri_id, int pt_id, const Point&,
				    TriSurface* surf) {
    surf->remove_triangle(tri_id);
    int v0=surf->elements[tri_id]->i1;
    int v1=surf->elements[tri_id]->i2;
    int v2=surf->elements[tri_id]->i3;
    surf->add_triangle(v0, v1, pt_id, 1);
    surf->add_triangle(v1, v2, pt_id, 1);
    surf->add_triangle(v2, v0, pt_id, 1);
}


// if the point we're closest to is nearest to a boundary edge, we just
// add a new triangle hanging over the old boundary.

void ContoursToSurf::break_edge2(int tri1, int e1, int pt_id,
				const Point&, TriSurface *surf) {
    ASSERT(e1<3 && e1>=0);
    int v[3];
    v[0]=surf->elements[tri1]->i1;
    v[1]=surf->elements[tri1]->i2;
    v[2]=surf->elements[tri1]->i3;
    surf->add_triangle(pt_id, v[(e1+2)%3], v[(e1+1)%3], 1);
}


// if the point we're closest to is on an interior edge, we break the 
// two triangles that share that face, and create four new triangles.

void ContoursToSurf::break_edge(int tri1, int tri2, int e1, int e2, int pt_id,
				const Point&, TriSurface *surf) {
    ASSERT(e1<3 && e1>=0);
    ASSERT(e2<3 && e2>=0);

    // first break tri1
    int v[3];
    v[0]=surf->elements[tri1]->i1;
    v[1]=surf->elements[tri1]->i2;
    v[2]=surf->elements[tri1]->i3;
    surf->remove_triangle(tri1);
    surf->add_triangle(pt_id, v[e1], v[(e1+1)%3], 1);
    surf->add_triangle(pt_id, v[(e1+2)%3], v[e1], 1);

    // now break tri2
    v[0]=surf->elements[tri2]->i1;
    v[1]=surf->elements[tri2]->i2;
    v[2]=surf->elements[tri2]->i3;
    surf->remove_triangle(tri2);
    surf->add_triangle(pt_id, v[e2], v[(e2+1)%3], 1);
    surf->add_triangle(pt_id, v[(e2+2)%3], v[e2], 1);
}

void ContoursToSurf::add_point(const Point& p, TriSurface* surf) {
    Array1<int> res;

    /*double dist=*/surf->distance(p, res);
    if (res.size() > 4) {	// we were closest to a vertex
	return;
    } 
    surf->add_point(p);
    if (res.size() == 4) {	// we were closest to an edge 
	    break_edge(res[0], res[2], res[1], res[3], surf->points.size()-1,
		       p, surf);
	    return;
    }
    if (res.size() == 2) {	// we were closest to a boundary edge
	    break_edge2(res[0], res[1], surf->points.size()-1, p, surf);
	    return;
    }				// we were closest to a face
    break_triangle(res[0], surf->points.size()-1, p, surf);
}


// Given a set of contour sets, we lace up the first one, and then
// add points from the rest one at a time to that surface until
// we're done.

void ContoursToSurf::contours_to_surf(const Array1<ContourSetHandle> &contours,
				   TriSurface *surf) {
    // have to make sure all of the contour-sets have valid bboxes...
    // ...the build_bbox() method does just that.

    surf->name=contours[0]->name[0];
    BBox bb;
    int i;
    for (i=0; i<contours.size(); i++) {
	contours[i]->build_bbox();
	BBox bb0(contours[i]->bbox);
	for (int ti=0; ti<=1; ti++) 
	    for (int tj=0; tj<=1; tj++) 
		for (int tk=0; tk<=1; tk++) {
		    Point p1(ti?bb0.min().x():bb0.max().x(),
			     tj?bb0.min().y():bb0.max().y(),
			     tk?bb0.min().z():bb0.max().z());
		    p1-=contours[i]->origin;
		    bb.extend(Point(0,0,0)+contours[i]->basis[0]*p1.x()+
			      contours[i]->basis[1]*p1.y()+
			      contours[i]->basis[2]*
			                   (p1.z()*contours[i]->space));
		}
    }

    // the first contour has to be the most complete, so we'll use it for
    // the number of points, and we're assuming that the middle slice of
    // that set has an average number of points
    double num_pts=contours[0]->contours.size()*
	contours[0]->contours[contours[0]->contours.size()/2].size();
    Vector sides=bb.max()-bb.min();
    double spacing=Cbrt(sides.x()*sides.y()*sides.z()/(num_pts*2.5)); 
    
    Vector parts(sides/spacing);
    surf->construct_grid((int)parts.x()+1, (int)parts.y()+1, (int)parts.z()+1, 
			 bb.min(), spacing);
    lace_contours(contours[0], surf);

    for (i=1;i<contours.size();i++) {
	for (int j=0; j<contours[i]->contours.size(); j++) {
	    for (int k=0; k<contours[i]->contours[j].size(); k++) {
		Point p(contours[i]->contours[j][k]-contours[i]->origin);
		p=Point(0,0,0)+contours[i]->basis[0]*p.x()+
		    contours[i]->basis[1]*p.y()+
		    contours[i]->basis[2]*(p.z()*contours[i]->space);
		    add_point(p, surf);
	    }	
	}
    }
    surf->destroy_grid();
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.4  1999/08/19 23:17:53  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:55  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:42  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:56  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:26  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
