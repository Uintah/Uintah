/*
 *  GenSurface.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/SurfacePort.h>
#include <Geom/Material.h>
#include <Geom/Pick.h>
#include <Geom/Sphere.h>
#include <Geom/TCLGeom.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

static clString widget_name("GenSurface Widget");

class GenSurface : public Module {
    TCLstring surfacetype;
    TCLPoint cyl_p1;
    TCLPoint cyl_p2;
    TCLdouble cyl_rad;
    TCLint cyl_nu;
    TCLint cyl_nv;
    TCLint cyl_ndiscu;
    TCLPoint point_pos;
    TCLdouble point_rad;
    TCLColor widget_color;

    SurfaceOPort* outport;
    GeometryOPort* ogeom;

    GeomSphere* sphere;

    clString oldst;
public:
    GenSurface(const clString& id);
    GenSurface(const GenSurface&, int deep);
    virtual ~GenSurface();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void geom_release(GeomPick*, void*);
    virtual void geom_moved(GeomPick*, int, double, const Vector&, void*);
};

extern "C" {
Module* make_GenSurface(const clString& id)
{
    return scinew GenSurface(id);
}
};

GenSurface::GenSurface(const clString& id)
: Module("GenSurface", id, Source), surfacetype("surfacetype", id, this),
  cyl_p1("cyl_p1", id, this), cyl_p2("cyl_p2", id, this),
  cyl_rad("cyl_rad", id, this), cyl_nu("cyl_nu", id, this),
  cyl_nv("cyl_nv", id, this), cyl_ndiscu("cyl_ndiscu", id, this),
  point_pos("point_pos", id, this), point_rad("point_rad", id, this),
  widget_color("widget_color", id, this)
{
    // Create the output port
    outport=scinew SurfaceOPort(this, "Geometry", SurfaceIPort::Atomic);
    add_oport(outport);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

GenSurface::GenSurface(const GenSurface& copy, int deep)
: Module(copy, deep), surfacetype("surfacetype", id, this),
  cyl_p1("cyl_p1", id, this), cyl_p2("cyl_p2", id, this),
  cyl_rad("cyl_rad", id, this), cyl_nu("cyl_nu", id, this),
  cyl_nv("cyl_nv", id, this), cyl_ndiscu("cyl_ndiscu", id, this),
  point_pos("point_pos", id, this), point_rad("point_rad", id, this),
  widget_color("widget_color", id, this)
{
    NOT_FINISHED("GenSurface::GenSurface");
}

GenSurface::~GenSurface()
{
}

Module* GenSurface::clone(int deep)
{
    return scinew GenSurface(*this, deep);
}

void GenSurface::execute()
{
    Surface* surf=0;
    clString st(surfacetype.get());
    // Handle 3D widget
    if(st != oldst){
	ogeom->delAll();
	if(st=="point"){
	    GeomSphere* widget=scinew GeomSphere(point_pos.get(), point_rad.get());
	    MaterialHandle widget_matl(scinew Material(Color(0,0,0),
						    widget_color.get(),
						    Color(.6, .6, .6), 10));
	    GeomMaterial* matl=scinew GeomMaterial(widget, widget_matl);
	    GeomPick* pick=scinew GeomPick(matl, this,
					Vector(1,0,0), Vector(0,1,0),
					Vector(0,0,1));
	    ogeom->addObj(pick, widget_name);
	    sphere=widget;
	} else {
	    NOT_FINISHED("Other surfaces");
	}
	oldst=st;
    }

    // Spit out the surface
    if(st=="cylinder"){
	surf=scinew CylinderSurface(cyl_p1.get(), cyl_p2.get(), cyl_rad.get(),
				 cyl_nu.get(), cyl_nv.get(), cyl_ndiscu.get());
    } else if(st=="point"){
	surf=scinew PointSurface(point_pos.get());
    } else {
	error("Unknown surfacetype: "+st);
    }
    if(surf)
        outport->send(SurfaceHandle(surf));
}

void GenSurface::geom_moved(GeomPick*, int, double, const Vector& delta, void*)
{
    sphere->cen+=delta*6; // SC94 ONLY (*10000)
    point_pos.set(sphere->cen);
    sphere->adjust();
}

void GenSurface::geom_release(GeomPick*, void*)
{
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}
    
