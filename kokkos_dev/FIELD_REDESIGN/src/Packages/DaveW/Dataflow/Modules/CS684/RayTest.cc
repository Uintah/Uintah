//static char *id="@(#) $Id$";

/*
 *  RayTest.cc:  Project parallel rays at a sphere and see where they go
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <DaveW/Datatypes/CS684/RTPrims.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geom/GeomArrows.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
#include <iostream>
using std::cerr;
#define Colormap XColormap
#include <tcl.h>
#include <tk.h>
#undef Colormap

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
using namespace SCICore::GeomSpace;
using namespace SCICore::TclInterface;

class RayTest : public Module {
    GeometryOPort* ogeom;
    int init;

    TCLint density;
    TCLdouble nu;
    TCLdouble energy;

    Array1<int> ray_id;
    Color RT_trace(RTRay& R, const RTSphere& sph, Array1<GeomArrows*>&, int);
    Color RT_shade(RTRay& R, const RTSphere& sph, Array1<GeomArrows*>&, int,
		   int);

public:
    RayTest(const clString& id);
    virtual ~RayTest();
    virtual void execute();
};

extern "C" Module* make_RayTest(const clString& id)
{
    return new RayTest(id);
}

static clString module_name("RayTest");

RayTest::RayTest(const clString& id)
: Module("RayTest", id, Source), density("density", id, this), 
  nu("nu",id,this), energy("energy",id,this), init(0)
{
    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

RayTest::~RayTest()
{
}

Color RayTest::RT_shade(RTRay& R, const RTSphere &sph, Array1<GeomArrows*> &ga,
			int depth, int leavingSph) {
    Vector N;

// GOTTA FIX THIS!
#if 0

//    N=sph.normal(R.origin);
    if (Dot(R.dir,N)>0) N=-N;	// have to flip the normal if we're inside
    RTRay refl=Reflect(R, N);	// copy everything, then fix direction
    RTRay refr(R);		// copy everything...
    if (leavingSph) {
	RTIOR *rtior=nefr.nu;
	refr.nu=*(refr.nu.prev); // pop off an IOR
	delete(rtior);
    } else {
	RTIOR *rtiort=new RTIOR;
	rtiort->val=sph.matl->refraction_index;
	rtiort->prev=refr.nu;
	refr.nu=rtiort;
    }
    int TIR=!Snell(R, N, refr);  	// ... fix direction

    if (TIR || (sph.matl->transparency == 0)) {
	refr.energy=0;
    } else {
	double fresRefl=Fres(R, N, refr.nu.prev->val);
	refr.energy*=(1-fresRefl);
	refl.energy*=fresRefl;
    }

    Color crefl, crefr;
    if (refr.energy>0.02 && depth < 5) {
	crefr=RT_trace(refr, sph, ga, depth+1);	// trace refraction ray
    }
    if (refl.energy>0.02 && depth < 5) {
	crefl=RT_trace(refl, sph, ga, depth+1); // trace reflection ray
    }
    return crefl+crefr;
#endif
    return Color(0,0,0);
}
	

Color RayTest::RT_trace(RTRay& R, const RTSphere &sph, Array1<GeomArrows*> &ga,
			int depth) {
    double t1, t2;
    Point p1, p2;
    int inters;
//    inters = sph.intersect(R, t1, p1, t2, p2);
    inters=0;

    if (inters == 0) {	// missed everything
	if (ga.size()==depth) ga.add(new GeomArrows(.02,.95));
#if 0
	ga[depth]->add(R.origin, R.dir*sph.radius, 
		       new Material(R.color*R.energy));
#endif
	return Color(0,0,0);
    }
    if (ga.size()==depth) ga.add(new GeomArrows(.02,.95));
#if 0
    ga[depth]->add(R.origin, R.dir*t1, 
		   new Material(R.color*R.energy));
#endif
    R.origin=p1;
    int leavingSph;
    if (inters == 1) leavingSph=1; else leavingSph=0;
    return RT_shade(R, sph, ga, depth, leavingSph);
}

void RayTest::execute()
{
    if (!init) {
	Material *blueMatl=new 
	    Material(Color(0,0,.6));
	GeomSphere *sph=new
	    GeomSphere(Point(0,0,0), 10, 30, 30);
	ogeom->addObj(new GeomMaterial(sph, MaterialHandle(blueMatl)),
		      "Sphere");
	init=1;
    }
    int i;
    for (i=0; i<ray_id.size(); i++) {
	ogeom->delObj(ray_id[i]);
	ray_id[i]=0;
    }

    int d=density.get();
    double dd=10./d;
    double x=-10+dd;
    double y;
    d=d*2-1;

    RTSphere sph;
    sph.center=Point(0,0,0);
    sph.radius=10;
#if 0
    sph.matl->refraction_index=nu.get();
    sph.matl->transparency=1;
#endif

    RTRay I;

//  GOTTA FIX THIS!

//    RTRay I(Point(0,0,0), Vector(0,0,-1), Color(0,0,0), 1);
    I.energy=energy.get();
    Array1<GeomArrows*> ga;
    for (i=0; i<d; i++, x+=dd) {
	y=-10+dd;
	for (int j=0; j<d; j++, y+=dd) {
	    if (x*x+y*y<100) {  // if we hit the sphere
#if 0
		I.color=Color((10+x)/20, (10-x)/20, (10+y)/20);
#endif
		I.origin=Point(x,y,15);
		RT_trace(I, sph, ga, 0);
	    }
	}
    }

    ray_id.resize(ga.size());
    for (i=0; i<ray_id.size(); i++) {
	ray_id[i]=ogeom->addObj(ga[i], clString("Rays "+to_string(i)));
    }

    ogeom->flushViews();
}
} // End namespace Modules
} // End namespace DaveW
//
// $Log$
// Revision 1.7  2000/03/17 09:25:30  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:06:25  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/22 19:00:34  dmw
// using Makefile variable to specify itcl includes
//
// Revision 1.4  1999/09/22 18:43:24  dmw
// added new GUI
//
// Revision 1.3  1999/09/08 02:26:20  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/25 03:47:37  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/24 06:23:00  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:12  dmw
// Added and updated DaveW Datatypes/Modules
//
//
