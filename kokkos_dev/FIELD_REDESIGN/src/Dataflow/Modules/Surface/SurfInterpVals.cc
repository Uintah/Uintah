//static char *id="@(#) $Id$";

/*
 *  SurfInterpVals.cc:  Rescale a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

//  Take in a surface and output that same surface, but with augmented data
//  values for at least some of the nodes.
//  Different interpolation techniques can be used to interpolate that data
//  to the surface -- weighted n-nearest neighbor averaging is the easiest 
//  and can be done as a surface-based method (need surface connectivity)
//  or as a volume-based method. 
//  It should be possible to project the second surface's data values onto
//  the first surface using a nearest point algorithm.
//  If either of the surfaces is a SurfTree, we must be able to restrict
//  the algorithm to just use the nodes on that surface.

//  Possible applications:
//  1) Project data from 2nd surface onto a specific surface of a SurfTree
//  2) Interpolate data using a weighted n-nearest volume neighbors method
//  3) Blur data values across surface using voronoi n-neaest surface neighbors

#include <SCICore/Containers/Array2.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/BasicSurfaces.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MusilRNG.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <stdio.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;

#define NBRHD 5

namespace PSECommon {
namespace Modules {

using namespace PSECore::Datatypes;
using namespace PSECore::Dataflow;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;

class SurfInterpVals : public Module {
    SurfaceIPort* isurface1;
    SurfaceIPort* isurface2;
    SurfaceOPort* osurface;
    TCLstring method;
    TCLint numnbrs;
    TCLstring surfid;
    TCLint cache;
    Array2<int> contribIdx;
    Array2<double> contribAmt;
    
public:
    SurfInterpVals(const clString& id);
    virtual ~SurfInterpVals();
    virtual void execute();
};

extern "C" Module* make_SurfInterpVals(const clString& id)
{
    return new SurfInterpVals(id);
}

static clString module_name("SurfInterpVals");

SurfInterpVals::SurfInterpVals(const clString& id)
: Module("SurfInterpVals", id, Filter), method("method", id, this),
  numnbrs("numnbrs", id, this), surfid("surfid", id, this),
  cache("cache", id, this)
{
    isurface1=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface1);
    isurface2=scinew SurfaceIPort(this, "Surface2", SurfaceIPort::Atomic);
    add_iport(isurface2);
    // Create the output port
    osurface=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurface);
}

SurfInterpVals::~SurfInterpVals()
{
}

void SurfInterpVals::execute()
{
    SurfaceHandle isurf1;
    if(!isurface1->get(isurf1))
	return;
    int haveSurf2;
    SurfaceHandle isurf2;
    haveSurf2=(isurface2->get(isurf2) && isurf2.get_rep());

    clString m(method.get());
    int nn(numnbrs.get());
    clString sid(surfid.get());

    // now we know we have isurf1, we might have isurf2, we know what
    // interpolation method to use, how many neighbors to use, and which
    // surfId to use from isurf1 (if it's a SurfTree)

    if (m == "surfblur") {
	cerr << "Sorry, we don't know how to blur across a surface yet...\n";
	return;
    }

    if ((m == "project" || m== "projectNormal") && !haveSurf2) {
	cerr << "Can't project because I don't have a second surface...\n";
	return;
    }
    if (m == "project" || m == "volblur" || m == "projectNormal") {
	TriSurface *ts=scinew TriSurface;
	
	// first, set up the data point locations and values in an array
	Array1<Point> p;
	Array1<Vector> n;
	Array1<double> v;
	SurfaceHandle sh(isurf1);
	if (haveSurf2) sh=isurf2;
	// get the right trisurface and grab those vals
	if(!(ts=sh->getTriSurface())) {
	    cerr << "I only know how to deal with TriSurfaces!\n";
	    return;
	}
	// ok, now TS has the nodes and values of the data - copy them
	int i;
	for (i=0; i<ts->bcIdx.size(); i++) {
	    p.add(ts->points[ts->bcIdx[i]]);
	    v.add(ts->bcVal[i]);
	}
	if (m=="projectNormal" && ts->normType==TriSurface::PointType &&
	    ts->normals.size() == ts->bcIdx.size()) {
	    for (i=0; i<ts->bcIdx.size(); i++) {
		n.add(ts->normals[ts->bcIdx[i]]);
	    }
	}
	if (!sh->getTriSurface()) delete ts; // made it in extract

	if (m=="volblur" && v.size()<nn) {
	    cerr << "Need at least "<<NBRHD<<" bc nodes to interpolate...\n";
	    return;
	}

	// ok, now we have the data in v and p, time to apply it to isurf1
	Array1<int> map;
	Array1<int> imap;
	sh=isurf1;
//	cerr << "ATTEMPT...\n";
	if(!(ts=sh->getTriSurface())) {
	    cerr << "I only know how to deal with TriSurfaces!\n";
	    return;
	}
	// now the data locations and values are in v and p
	// and the surface that needs to be interpolated is a TriSurface ts
	// time to do volume nearest neighbor averaging...

	if (m == "project") {
	    if (ts->bcVal.size()) {
		cerr << "Error -- surf1 already has "<<ts->bcVal.size()<<"boundary values!\n";
		return;
	    }
	    if (p.size() > ts->points.size()) {
		cerr << "Too many points to project ("<<p.size()<<" to "<<ts->points.size()<<")\n";
		return;
	    }

//	    cerr << "HERE ARE ALL THE PTS:\n";
//	    for (int ii=0; ii<ts->points.size(); ii++)
//		cerr << "  "<<ts->points[ii]<<"\n";

	    Array1<int> selected(ts->points.size());
	    selected.initialize(0);
	    for (int aa=0; aa<p.size(); aa++) {
		if (!(aa%500)) update_progress(aa,p.size());
		double dt;
		int si=-1;
		double d;
		for (int bb=0; bb<ts->points.size(); bb++) {
		    if (selected[bb]) continue;
		    dt=Vector(p[aa]-ts->points[bb]).length2();
		    if (si==-1 || dt<d) {
			si=bb;
			d=dt;
		    }
		}
		selected[si]=1;
//		cerr << "("<<aa<<") closest to "<<p[aa]<<"="<<ts->points[si]<<"\n";
		ts->bcIdx.add(si);
		ts->bcVal.add(v[aa]);
	    }
	} else if (m == "projectNormal") {
	    if (ts->bcVal.size()) {
		cerr << "Error -- surf1 already has "<<ts->bcVal.size()<<"boundary values!\n";
		return;
	    }
	    if (p.size() > ts->points.size()) {
		cerr << "Too many points to project ("<<p.size()<<" to "<<ts->points.size()<<")\n";
		return;
	    }

//	    cerr << "HERE ARE ALL THE PTS:\n";
//	    for (int ii=0; ii<ts->points.size(); ii++)
//		cerr << "  "<<ts->points[ii]<<"\n";

	    Array1<Point> newpts;
	    Array1<double> newvals;
	    Array1<int> newidx;
	    Array1<int> selected(ts->points.size());
	    selected.initialize(0);
	    for (int aa=0; aa<p.size(); aa++) {
		int vert=-1;
		double d=-1;
		for (int bb=0; bb<ts->elements.size(); bb++) {
		    ts->intersect(p[aa], n[aa], d, vert, bb);
		}
		if (vert==-1) continue;
		if (selected[vert]) continue;
		selected[vert]=1;
		newpts.add(ts->points[vert]);
		newvals.add(v[aa]);
		newidx.add(newidx.size());
	    }
	    ts->elements.resize(0);
	    ts->points=newpts;
	    ts->bcVal=newvals;
	    ts->bcIdx=newidx;
	} else {
	    Array1<int> selected(v.size());
	    
	    int cacheok=cache.get();

	    if (cacheok && contribIdx.dim1()==ts->points.size() &&
		contribIdx.dim2()==nn && contribAmt.dim1()==ts->points.size() 
		&& contribAmt.dim2()==nn) {
		cacheok=1;
	    } else {
		cacheok=0;
		contribIdx.newsize(ts->points.size(), nn);
		contribAmt.newsize(ts->points.size(), nn);
		contribIdx.initialize(-1);
		contribAmt.initialize(0);
	    }

	    ts->bcIdx.resize(ts->points.size());
	    ts->bcVal.resize(ts->points.size());
	    for (int aa=0; aa<ts->points.size(); aa++) {
		if (!(aa%500)) update_progress(aa,ts->points.size());

		if (!cacheok) {
		    Point a(ts->points[aa]);
		    selected.initialize(0);
		    int nb;
		    for (nb=0; nb<nn; nb++) {
			double dt;
			for (int bb=0; bb<v.size(); bb++) {
			    if (selected[bb]) continue;
			    dt=Vector(a-p[bb]).length2();
//			    dt=dt*dt;
			    if ((contribIdx(aa,nb) == -1) || (contribAmt(aa,nb)>dt)) {
				contribIdx(aa,nb)=bb;
				contribAmt(aa,nb)=dt;
			    }
			}
			selected[contribIdx(aa,nb)]=1;
		    }
		
		    double ratio=0;
		    for (nb=0; nb<nn; nb++) { 
			if (contribAmt(aa,nb)==0) 
			    contribAmt(aa,nb)=0.0000001; 
			contribAmt(aa,nb)=1./Sqrt(contribAmt(aa,nb)); 
			ratio+=contribAmt(aa,nb); 
		    }
		    ratio=1./ratio;
		    for (nb=0; nb<nn; nb++) {
			contribAmt(aa,nb)*=ratio; 
		    }
		} 
		    
		double data=0;
		for (int nb=0; nb<nn; nb++) {
		    data+=v[contribIdx(aa,nb)]*contribAmt(aa,nb);
		}
		ts->bcIdx[aa]=aa;
		ts->bcVal[aa]=data;
	    }
	}
    }
    osurface->send(isurf1);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.8  2000/03/17 09:27:22  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.7  1999/11/09 08:32:59  dmw
// added SurfInterpVals to index
//
// Revision 1.6  1999/10/07 02:07:00  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/05 05:32:26  dmw
// updated and added Modules from old tree to new
//
// Revision 1.4  1999/08/25 03:48:01  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/18 20:19:59  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:44  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:59  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:29  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
