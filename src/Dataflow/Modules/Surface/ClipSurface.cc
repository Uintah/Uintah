//static char *id="@(#) $Id$";

/*
 *  ClipSurface.cc:  Clip a suface to a plane
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/ScalarTriSurface.h>
#include <Datatypes/SurfTree.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/BBox.h>
#include <Math/Expon.h>
#include <Math/MusilRNG.h>
#include <Math/Trig.h>
#include <TclInterface/TCLvar.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

namespace PSECommon {
namespace Modules {

using PSECommon::Dataflow::Module;
using PSECommon::Datatypes::SurfaceIPort;
using PSECommon::Datatypes::SurfaceOPort;
using PSECommon::Datatypes::SurfaceHandle;
using PSECommon::Datatypes::ScalarTriSurface;
using PSECommon::Datatypes::TriSurface;
using PSECommon::Datatypes::SurfTree;
using PSECommon::Datatypes::PointsSurface;
using PSECommon::Datatypes::NodeHandle;
using SCICore::Containers::clString;
using SCICore::Containers::Array1;
using SCICore::Geometry::Point;
using SCICore::TclInterface::TCLdouble;

class ClipSurface : public Module {
    SurfaceIPort* isurface;
    SurfaceOPort* osurface;
    TCLdouble pa, pb, pc, pd;
    SurfaceHandle osh;
    double last_pa;
    double last_pb;
    double last_pc;
    double last_pd;
    int generation;
public:
    ClipSurface(const clString& id);
    virtual ~ClipSurface();
    virtual void execute();
};

Module* make_ClipSurface(const clString& id) {
   return new ClipSurface(id);
}

static clString module_name("ClipSurface");

ClipSurface::ClipSurface(const clString& id)
: Module("ClipSurface", id, Filter),
  pa("pa", id, this), pb("pb", id, this), pc("pc", id, this), 
  pd("pd", id, this), last_pa(0), last_pb(0), last_pc(0), last_pd(0),
  generation(-1)
{
    isurface=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    // Create the output port
    osurface=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurface);
}

ClipSurface::ClipSurface(const ClipSurface& copy, int deep)
: Module(copy, deep),
  pa("pa", id, this), pb("pb", id, this), pc("pc", id, this), 
  pd("pd", id, this), last_pa(0), last_pb(0), last_pc(0), last_pd(0)
{
    NOT_FINISHED("ClipSurface::ClipSurface");
}

ClipSurface::~ClipSurface()
{
}

void ClipSurface::execute()
{
    SurfaceHandle isurf;
    if(!isurface->get(isurf))
	return;

    PointsSurface *ps=isurf->getPointsSurface();
    TriSurface *ts=isurf->getTriSurface();
    SurfTree *st=isurf->getSurfTree();
    ScalarTriSurface *ss=isurf->getScalarTriSurface();
    if (!ps && !ts && !st && !ss) {
	cerr << "ClipSurface: unknown surface type.\n";
	return;
    }

    double paa=pa.get();
    double pbb=pb.get();
    double pcc=pc.get();
    double pdd=pd.get();

    Array1<NodeHandle> nodes;
    isurf->get_surfnodes(nodes);
    Array1<Point> pts;
	
    int i;
    if (generation == isurf->generation && 
	paa==last_pa && pbb==last_pb && pcc==last_pc && pdd==last_pd) {
	osurface->send(osh);
	return;
    }
    
    last_pa=paa;
    last_pb=pbb;
    last_pc=pcc;
    last_pd=pdd;

    for (i=0; i<nodes.size(); i++) {
	if ((nodes[i]->p.x()*paa + nodes[i]->p.y()*pbb +
	    nodes[i]->p.z()*pcc + pdd) >= 0)
	    pts.add(nodes[i]->p);
    }
    cerr << "Started with "<<nodes.size()<<" nodes -- kept "<<pts.size()<<" after clipping to ("<<paa<<","<<pbb<<","<<pcc<<","<<pdd<<")\n";
    if (ps) {
	PointsSurface* nps=new PointsSurface(*ps);
	*nps=*ps;
	nps->pos=pts;
	osh=nps;
    } else if (ts) {
	TriSurface* nts=new TriSurface;
	*nts=*ts;
	nts->points=pts;
	osh=nts;
    } else if (st) {
	SurfTree* nst=new SurfTree;
	*nst=*st;
	nst->nodes=pts;
	osh=nst;
    } else if (ss) {
	ScalarTriSurface* nss=new ScalarTriSurface;
	*nss=*ss;
	nss->points=pts;
	osh=nss;
    }
    osurface->send(osh);
    return;
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.3  1999/08/25 03:47:59  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/18 20:19:55  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.1  1999/07/27 16:57:56  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
