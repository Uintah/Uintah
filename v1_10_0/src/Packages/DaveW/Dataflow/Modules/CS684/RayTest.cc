
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

#include <Packages/DaveW/Core/Datatypes/CS684/RTPrims.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/String.h>
#include <Core/Geom/GeomArrows.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <iostream>
using std::cerr;
#define Colormap XColormap
#include <tcl.h>
#include <tk.h>
#undef Colormap

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class RayTest : public Module {
    GeometryOPort* ogeom;
    int init;

    GuiInt density;
    GuiDouble nu;
    GuiDouble energy;

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
} // End namespace DaveW
}
