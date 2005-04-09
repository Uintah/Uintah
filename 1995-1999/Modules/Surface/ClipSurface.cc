
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/ScalarTriSurface.h>
#include <Datatypes/SurfTree.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/BBox.h>
#include <Math/Expon.h>
#include <Math/MusilRNG.h>
#include <Math/Trig.h>
#include <TCL/TCLvar.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

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
    ClipSurface(const ClipSurface&, int deep);
    virtual ~ClipSurface();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_ClipSurface(const clString& id)
{
    return new ClipSurface(id);
}
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

Module* ClipSurface::clone(int deep)
{
    return new ClipSurface(*this, deep);
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
