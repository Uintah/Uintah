
/*
 *  TranslateSurface.cc:  Translate a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
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
#include <TCL/TCLvar.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

class TranslateSurface : public Module {
    SurfaceIPort* isurface;
    SurfaceOPort* osurface;
    TCLdouble tx, ty, tz;
    SurfaceHandle osh;
    double last_tx;
    double last_ty;
    double last_tz;
    int generation;
public:
    TranslateSurface(const clString& id);
    TranslateSurface(const TranslateSurface&, int deep);
    virtual ~TranslateSurface();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_TranslateSurface(const clString& id)
{
    return new TranslateSurface(id);
}
};

static clString module_name("TranslateSurface");

TranslateSurface::TranslateSurface(const clString& id)
: Module("TranslateSurface", id, Filter),
  tx("tx", id, this), ty("ty", id, this), tz("tz", id, this),
  generation(-1), last_tx(0), last_ty(0), last_tz(0)
{
    isurface=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    // Create the output port
    osurface=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurface);
}

TranslateSurface::TranslateSurface(const TranslateSurface& copy, int deep)
: Module(copy, deep),
  tx("tx", id, this), ty("ty", id, this), tz("tz", id, this),
  generation(-1), last_tx(0), last_ty(0), last_tz(0)
{
    NOT_FINISHED("TranslateSurface::TranslateSurface");
}

TranslateSurface::~TranslateSurface()
{
}

Module* TranslateSurface::clone(int deep)
{
    return new TranslateSurface(*this, deep);
}

void TranslateSurface::execute()
{
    SurfaceHandle isurf;
    if(!isurface->get(isurf))
	return;

    PointsSurface *ps=isurf->getPointsSurface();
    TriSurface *ts=isurf->getTriSurface();
    SurfTree *st=isurf->getSurfTree();
    ScalarTriSurface *ss=isurf->getScalarTriSurface();
    if (!ps && !ts && !st && !ss) {
	cerr << "TranslateSurface: unknown surface type.\n";
	return;
    }

    double xx=tx.get();
    double yy=ty.get();
    double zz=tz.get();

    Array1<NodeHandle> nodes;
    isurf->get_surfnodes(nodes);
    Array1<Point> pts(nodes.size());
	
    int i;
    if (generation == isurf->generation && xx==last_tx && yy==last_ty && zz==last_tz) {
	osurface->send(osh);
	return;
    }
    
    last_tx = xx;
    last_ty = yy;
    last_tz = zz;

    for (i=0; i<nodes.size(); i++) {
	pts[i].x(xx+nodes[i]->p.x());
	pts[i].y(yy+nodes[i]->p.y());
	pts[i].z(zz+nodes[i]->p.z());
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
	nst->points=pts;
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
