//static char *id="@(#) $Id$";

/*
 *  RescaleSurface.cc:  Rescale a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
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
#include <TclInterface/TCLvar.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

namespace PSECommon {
namespace Modules {

using PSECommon::Dataflow::Module;
using PSECommon::Datatypes::SurfTree;
using PSECommon::Datatypes::SurfaceIPort;
using PSECommon::Datatypes::SurfaceOPort;
using PSECommon::Datatypes::SurfaceHandle;
using PSECommon::Datatypes::ScalarTriSurface;
using PSECommon::Datatypes::PointsSurface;
using PSECommon::Datatypes::TriSurface;
using PSECommon::Datatypes::NodeHandle;

using namespace SCICore::TclInterface;
using SCICore::Containers::Array1;
using SCICore::Geometry::Point;
using SCICore::Geometry::BBox;

class RescaleSurface : public Module {
    SurfaceIPort* isurface;
    SurfaceOPort* osurface;
    TCLint coreg;
    TCLdouble scale;
    SurfaceHandle osh;
    double last_scale;
    int generation;
public:
    RescaleSurface(const clString& id);
    virtual ~RescaleSurface();
    virtual void execute();
};

Module* make_RescaleSurface(const clString& id) {
  return new RescaleSurface(id);
}

//static clString module_name("RescaleSurface");

RescaleSurface::RescaleSurface(const clString& id)
: Module("RescaleSurface", id, Filter),
  scale("scale", id, this), coreg("coreg", id, this),
  generation(-1), last_scale(0)
{
    isurface=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    // Create the output port
    osurface=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurface);
}

RescaleSurface::RescaleSurface(const RescaleSurface& copy, int deep)
: Module(copy, deep),
  scale("scale", id, this), coreg("coreg", id, this),
  generation(-1), last_scale(0)
{
    NOT_FINISHED("RescaleSurface::RescaleSurface");
}

RescaleSurface::~RescaleSurface()
{
}

void RescaleSurface::execute()
{
    SurfaceHandle isurf;
    if(!isurface->get(isurf))
	return;

    PointsSurface *ps=isurf->getPointsSurface();
    TriSurface *ts=isurf->getTriSurface();
    SurfTree *st=isurf->getSurfTree();
    ScalarTriSurface *ss=isurf->getScalarTriSurface();
    if (!ps && !ts && !st && !ss) {
	cerr << "RescaleSurface: unknown surface type.\n";
	return;
    }

    double new_scale=scale.get();
    double s=pow(10.,new_scale);
	
    Array1<NodeHandle> nodes;
    isurf->get_surfnodes(nodes);
    Array1<Point> pts(nodes.size());
	
    int i;
    if (coreg.get()) {
	BBox b;
	for (i=0; i<nodes.size(); i++) b.extend(nodes[i]->p);
	s=1./b.longest_edge()*1.7;
	cerr << "b.min="<<b.min()<<" b.max="<<b.max()<<"   s="<<s<<"\n";
	scale.set(log10(s));
	reset_vars();
	new_scale=scale.get();
    }

    if (generation == isurf->generation && new_scale == last_scale) {
	osurface->send(osh);
	return;
    }
    
    last_scale = new_scale;

    // scale about the center!

    Point c(0,0,0);
    for (i=0; i<nodes.size(); i++)
	c+=nodes[i]->p.vector();
    c.x(c.x()/nodes.size());
    c.y(c.y()/nodes.size());
    c.z(c.z()/nodes.size());

    for (i=0; i<nodes.size(); i++) {
	pts[i].x(c.x()+(nodes[i]->p.x()-c.x())*s);
	pts[i].y(c.y()+(nodes[i]->p.y()-c.y())*s);
	pts[i].z(c.z()+(nodes[i]->p.z()-c.z())*s);
    }
	
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
// Revision 1.3  1999/08/25 03:48:00  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/18 20:19:56  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.1  1999/07/27 16:57:57  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
