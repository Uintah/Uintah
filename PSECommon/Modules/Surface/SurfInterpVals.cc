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

#include <Util/NotFinished.h>
#include <Containers/Array2.h>
#include <Dataflow/Module.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/ScalarTriSurface.h>
#include <Datatypes/SurfTree.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/BBox.h>
#include <Math/Expon.h>
#include <Math/MusilRNG.h>
#include <TclInterface/TCLvar.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

#define NBRHD 5

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;

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

Module* make_SurfInterpVals(const clString& id) {
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

SurfInterpVals::SurfInterpVals(const SurfInterpVals& copy, int deep)
: Module(copy, deep), method("method", id, this),
  numnbrs("numnbrs", id, this), surfid("surfid", id, this),
  cache("cache", id, this)
{
    NOT_FINISHED("SurfInterpVals::SurfInterpVals");
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
	    SurfTree *st=sh->getSurfTree();
	    if (!st) {
		cerr << "Error -- second surface wasn't a trisurface or a surftree...\n";
		return;
	    }
	    Array1<int> map;	// not used
	    Array1<int> imap;	// not used
	    int comp;
	    int ok;
	    ok = sid.get_int(comp);
	    if (!ok) {
		for (comp=0; comp<st->surfI.size(); comp++) {
		    if (st->surfI[comp].name == sid) {
			break;
		    }
		}
		if (comp == st->surfI.size()) {
		    cerr << "Error: bad surface name "<<sid<<"\n";
		    return;
		}
	    }
	    
	    ts = new TriSurface;
	    if (!st->extractTriSurface(ts, map, imap, comp)) {
		cerr << "Error, couldn't extract triSurface.\n";
		return;
	    }
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
//	    cerr << "I'm in!\n";
	    SurfTree *st=sh->getSurfTree();
	    if (!st) {
		cerr << "Error -- first surface wasn't a trisurface or a surftree...\n";
		return;
	    }
	    int comp;
	    int ok;
	    ok = sid.get_int(comp);
	    if (!ok) {
		for (comp=0; comp<st->surfI.size(); comp++) {
		    if (st->surfI[comp].name == sid) {
			break;
		    }
		}
		if (comp == st->surfI.size()) {
		    cerr << "Error: bad surface name "<<sid<<"\n";
		    return;
		}
	    }
	    ts = new TriSurface;
//	    cerr << "HIYA!\n";
	    if (!st->extractTriSurface(ts, map, imap, comp)) {
		cerr << "Error, couldn't extract triSurface.\n";
		return;
	    } else {
		if (st->data.size() == ts->bcIdx.size()) {
		    st->data.resize(0);
		    st->idx.resize(0);
		}
//		cerr << "Got surface "<<comp<<" out of st.  ST had "<<st->bcVal.size()<<" bcs.  TS has "<<ts->bcVal.size()<<"\n";
	    }
	}
	
	// now the data locations and values are in v and p
	// and the surface that needs to be interpolated is a TriSurface ts
	// time to do volume nearest neighbor averaging...

	if (m == "project") {
	    if (ts->bcVal.size()) {
		cerr << "Error -- surf1 already has "<<ts->bcVal.size()<<"boundary values!\n";
		if (isurf1->getSurfTree()) {
		    delete ts;
		}
		return;
	    }
	    if (p.size() > ts->points.size()) {
		cerr << "Too many points to project ("<<p.size()<<" to "<<ts->points.size()<<")\n";
		if (isurf1->getSurfTree()) {
		    delete ts;
		}
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
		if (isurf1->getSurfTree()) {
		    delete ts;
		}
		return;
	    }
	    if (p.size() > ts->points.size()) {
		cerr << "Too many points to project ("<<p.size()<<" to "<<ts->points.size()<<")\n";
		if (isurf1->getSurfTree()) {
		    delete ts;
		}
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
	    Array2<int> cmap(ts->points.size(), nn);
	    Array2<double> contrib(ts->points.size(), nn);
	    Array1<int> idx(nn);
	    Array1<double> d(nn);
	    Array1<int> selected(v.size());
	    
	    ts->bcIdx.resize(ts->points.size());
	    ts->bcVal.resize(ts->points.size());
	    for (int aa=0; aa<ts->points.size(); aa++) {
		if (!(aa%500)) update_progress(aa,ts->points.size());
		ts->bcIdx[aa]=aa;
		Point a(ts->points[aa]);
		selected.initialize(0);
		idx.initialize(-1);
		int nb;
		for (nb=0; nb<nn; nb++) {
		    double dt;
		    for (int bb=0; bb<v.size(); bb++) {
			if (selected[bb]) continue;
			dt=Vector(a-p[bb]).length2();
			if ((idx[nb] == -1) || (d[nb]>dt)) {
			    idx[nb]=bb;
			    d[nb]=dt;
			}
		    }
		    selected[idx[nb]]=1;
		}
		
		double ratio=0;
		for (nb=0; nb<nn; nb++) { 
		    if (d[nb]==0) 
			d[nb]=0.0000001; d[nb]=1./Sqrt(d[nb]); ratio+=d[nb]; 
		}
		ratio=1./ratio;
		double data=0;
		for (nb=0; nb<nn; nb++) {
		    d[nb]*=ratio; 
		    cmap(aa,nb)=idx[nb]; 
		    contrib(aa,nb)=d[nb]; 
		    data+=v[idx[nb]]*d[nb];
		}
		ts->bcVal[aa]=data;
	    }
	}

	// now if this data came from a SurfTree, using imap and map, copy 
	// values back
	// first have to correct any old values -- make an array of which
	// ones we've fixed

	SurfTree *st;
	if (st=sh->getSurfTree()) {
	    Array1<int> seen(ts->bcVal.size());
	    seen.initialize(0);
	    int aa;
	    for (aa=0; aa<st->idx.size(); aa++) {
		if (map[st->idx[aa]] != -1) {
		    // this node is already known to have a value -- set it
		    st->data[aa]=ts->bcVal[map[st->idx[aa]]];
		    seen[aa]=1;
		}
	    }
	    for (aa=0; aa<ts->bcVal.size(); aa++) {
		if (!seen[aa]) { // this one didn't have a value yet -- add it
		    st->idx.add(imap[ts->bcIdx[aa]]);
		    st->data.add(ts->bcVal[aa]);
		}
	    }
	    delete ts;
	}
    }
    osurface->send(isurf1);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
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
