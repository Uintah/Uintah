//static char *id="@(#) $Id$";

/*
 *  RotateSurface.cc:  Rotate a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Util/NotFinished.h>
#include <PSECommon/Dataflow/Module.h>
#include <PSECommon/Dataflow/ModuleList.h>
#include <PSECommon/Datatypes/SurfacePort.h>
#include <PSECommon/Datatypes/BasicSurfaces.h>
#include <PSECommon/Datatypes/ScalarTriSurface.h>
#include <PSECommon/Datatypes/SurfTree.h>
#include <PSECommon/Datatypes/TriSurface.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MusilRNG.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <stdio.h>
#include <SCICore/Malloc/Allocator.h>

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

using SCICore::TclInterface::TCLdouble;
using SCICore::Containers::clString;
using SCICore::Containers::Array1;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

void buildRotateMatrix(double rm[][3], double angle, const Vector& axis) {
    // From Foley and Van Dam, Pg 227
    // NOTE: Element 0,1 is wrong in the text!
    double sintheta=Sin(angle);
    double costheta=Cos(angle);
    double ux=axis.x();
    double uy=axis.y();
    double uz=axis.z();
    rm[0][0]=ux*ux+costheta*(1-ux*ux);
    rm[0][1]=ux*uy*(1-costheta)-uz*sintheta;
    rm[0][2]=uz*ux*(1-costheta)+uy*sintheta;
    rm[1][0]=ux*uy*(1-costheta)+uz*sintheta;
    rm[1][1]=uy*uy+costheta*(1-uy*uy);
    rm[1][2]=uy*uz*(1-costheta)-ux*sintheta;
    rm[2][0]=uz*ux*(1-costheta)-uy*sintheta;
    rm[2][1]=uy*uz*(1-costheta)+ux*sintheta;
    rm[2][2]=uz*uz+costheta*(1-uz*uz);
}

Vector rotateVector(const Vector& v_r, double rm[][3]) {
    return Vector(v_r.x()*rm[0][0]+v_r.y()*rm[0][1]+v_r.z()*rm[0][2],
		  v_r.x()*rm[1][0]+v_r.y()*rm[1][1]+v_r.z()*rm[1][2],
		  v_r.x()*rm[2][0]+v_r.y()*rm[2][1]+v_r.z()*rm[2][2]);
}

// transform a pt by pushing it though a rotation matrix (M) and adding a
// displacement Vector (v)
Point transformPt(double m[3][3], const Point& p, const Vector& v) {
    return Point(p.x()*m[0][0]+p.y()*m[0][1]+p.z()*m[0][2]+v.x(),
		 p.x()*m[1][0]+p.y()*m[1][1]+p.z()*m[1][2]+v.y(),
		 p.x()*m[2][0]+p.y()*m[2][1]+p.z()*m[2][2]+v.z());
}

class RotateSurface : public Module {
    SurfaceIPort* isurface;
    SurfaceOPort* osurface;
    TCLdouble rx, ry, rz, th;
    SurfaceHandle osh;
    double last_rx;
    double last_ry;
    double last_rz;
    double last_th;
    int generation;
public:
    RotateSurface(const clString& id);
    virtual ~RotateSurface();
    virtual void execute();
};

Module* make_RotateSurface(const clString& id) {
  return new RotateSurface(id);
}

static clString module_name("RotateSurface");

RotateSurface::RotateSurface(const clString& id)
: Module("RotateSurface", id, Filter),
  rx("rx", id, this), ry("ry", id, this), rz("rz", id, this), 
  th("th", id, this),
  generation(-1), last_rx(0), last_ry(0), last_rz(0), last_th(0)
{
    isurface=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    // Create the output port
    osurface=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurface);
}

RotateSurface::RotateSurface(const RotateSurface& copy, int deep)
: Module(copy, deep),
  rx("rx", id, this), ry("ry", id, this), rz("rz", id, this),
  th("th", id, this),
  generation(-1), last_rx(0), last_ry(0), last_rz(0), last_th(0)
{
    NOT_FINISHED("RotateSurface::RotateSurface");
}

RotateSurface::~RotateSurface()
{
}

void RotateSurface::execute()
{
    SurfaceHandle isurf;
    if(!isurface->get(isurf))
	return;

    PointsSurface *ps=isurf->getPointsSurface();
    TriSurface *ts=isurf->getTriSurface();
    SurfTree *st=isurf->getSurfTree();
    ScalarTriSurface *ss=isurf->getScalarTriSurface();
    if (!ps && !ts && !st && !ss) {
	cerr << "RotateSurface: unknown surface type.\n";
	return;
    }

    double xx=rx.get();
    double yy=ry.get();
    double zz=rz.get();
    double tt=th.get();
    Array1<NodeHandle> nodes;
    isurf->get_surfnodes(nodes);
    Array1<Point> pts(nodes.size());
	
    int i;
    if (generation == isurf->generation && xx==last_rx && yy==last_ry && zz==last_rz && tt==last_th) {
	osurface->send(osh);
	return;
    }
    
    last_rx = xx;
    last_ry = yy;
    last_rz = zz;
    last_th = tt;
    Vector axis(xx,yy,zz);
    if (!axis.length2()) axis.x(1);
    axis.normalize();

    // rotate about the center!

    Point c(0,0,0);
    for (i=0; i<nodes.size(); i++)
	c+=nodes[i]->p.vector();
    c.x(c.x()/nodes.size());
    c.y(c.y()/nodes.size());
    c.z(c.z()/nodes.size());
    Vector v(c.vector());

    double m[3][3];
    buildRotateMatrix(m, tt, axis);
    for (i=0; i<nodes.size(); i++) {
	Point newp(transformPt(m, nodes[i]->p-v, v));
	pts[i].x(newp.x());
	pts[i].y(newp.y());
	pts[i].z(newp.z());
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
// Revision 1.4  1999/08/25 03:48:00  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/18 20:19:57  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:43  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:58  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
