//static char *id="@(#) $Id$";

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

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/BasicSurfaces.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomPick.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Geom/TCLGeom.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <PSECore/Widgets/GaugeWidget.h>
#include <PSECore/Widgets/PointWidget.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace PSECommon {
namespace Modules {

using PSECore::Dataflow::Module;
using PSECore::Datatypes::Surface;
using PSECore::Datatypes::SurfaceHandle;
using PSECore::Datatypes::MaterialHandle;
using PSECore::Datatypes::CylinderSurface;
using PSECore::Datatypes::SphereSurface;
using PSECore::Datatypes::PointSurface;
using PSECore::Datatypes::ColorMapIPort;
using PSECore::Datatypes::ColorMapHandle;
using PSECore::Datatypes::SurfaceIPort;
using PSECore::Datatypes::SurfaceOPort;
using PSECore::Datatypes::GeometryIPort;
using PSECore::Datatypes::GeometryOPort;
using PSECore::Widgets::GaugeWidget;
using PSECore::Widgets::PointWidget;

using namespace SCICore::TclInterface;
using SCICore::GeomSpace::GeomObj;
using SCICore::GeomSpace::GeomSphere;
using SCICore::GeomSpace::GeomMaterial;
using SCICore::GeomSpace::Material;
using SCICore::GeomSpace::Color;
using SCICore::Geometry::Interpolate;
using SCICore::Math::Abs;
using SCICore::Containers::to_string;
using SCICore::Thread::CrowdMonitor;

class GenSurface : public Module {
    TCLstring surfacetype;
    TCLPoint cyl_p1;
    TCLPoint cyl_p2;
    TCLdouble cyl_rad;
    TCLint cyl_nu;
    TCLint cyl_nv;
    TCLint cyl_ndiscu;
    TCLPoint sph_cen;
    TCLVector sph_axis;
    TCLdouble sph_rad;
    TCLint sph_nu;
    TCLint sph_nv;
    TCLPoint point_pos;
    TCLdouble point_rad;
    TCLdouble point_val;

    TCLstring cyl_boundary_expr;
    TCLstring sph_boundary_expr;

    ColorMapIPort* ColorMapport;
    SurfaceOPort* outport;
    GeometryOPort* ogeom;

    GeomSphere* sphere;

    clString oldst;
    int widget_id;
    GaugeWidget* cyl_widget;
    GaugeWidget* sph_widget;
    PointWidget* pt_widget;
    CrowdMonitor widget_lock;

    int surf_id;
    int last_generation;
    Point last_cyl_p1;
    Point last_cyl_p2;
    double last_cyl_rad;
    int last_cyl_nu, last_cyl_nv, last_cyl_ndiscu;
    Point last_sph_cen;
    double last_sph_rad;
    Vector last_sph_axis;
    int last_sph_nu, last_sph_nv;
    int fudge_widget;
public:
    GenSurface(const clString& id);
    virtual ~GenSurface();
    virtual void execute();
    virtual void widget_moved(int last);
};

extern "C" Module* make_GenSurface(const clString& id) {
  return new GenSurface(id);
}

GenSurface::GenSurface(const clString& id)
: Module("GenSurface", id, Source), surfacetype("surfacetype", id, this),
  cyl_p1("cyl_p1", id, this), cyl_p2("cyl_p2", id, this),
  cyl_rad("cyl_rad", id, this), cyl_nu("cyl_nu", id, this),
  cyl_nv("cyl_nv", id, this), cyl_ndiscu("cyl_ndiscu", id, this),
  sph_cen("sph_cen", id, this), sph_axis("sph_axis", id, this),
  sph_rad("sph_rad", id, this),
  sph_nu("sph_nu", id, this), sph_nv("sph_nv", id, this),
  point_pos("point_pos", id, this), point_rad("point_rad", id, this),
  cyl_boundary_expr("cyl_boundary_expr", id, this),
  sph_boundary_expr("sph_boundary_expr", id, this),
  point_val("point_val", id, this), widget_lock("GenSurface widget lock")
{
    // Create the input port
    ColorMapport=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(ColorMapport);

    // Create the output port
    outport=scinew SurfaceOPort(this, "Geometry", SurfaceIPort::Atomic);
    add_oport(outport);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    widget_id=0;
    surf_id=0;

    cyl_widget=new GaugeWidget(this, &widget_lock, .1);
    sph_widget=new GaugeWidget(this, &widget_lock, .1);
    pt_widget=new PointWidget(this, &widget_lock, .1);
    last_generation=0;

    fudge_widget=0;
}

GenSurface::~GenSurface()
{
}

void GenSurface::execute()
{
    ColorMapHandle cmap;
    if(!ColorMapport->get(cmap))
	return;
    Surface* surf=0;
    clString st(surfacetype.get());
    // Handle 3D widget
    if(st != oldst){
	fudge_widget=1;
	if(widget_id)
	    ogeom->delObj(widget_id, 0); // Don't actually delete the geom...
	if(st=="point"){
	    widget_id=ogeom->addObj(pt_widget->GetWidget(), "Point source widget", &widget_lock);
	    pt_widget->Connect(ogeom);
	    pt_widget->SetPosition(point_pos.get());
	    pt_widget->SetScale(point_rad.get());
	} else if(st=="cylinder"){
	    widget_id=ogeom->addObj(cyl_widget->GetWidget(), "Cylinder source widget",
				    &widget_lock);
	    cyl_widget->Connect(ogeom);
	    if(oldst == "sphere"){
		Point cen(sph_cen.get());
		Vector axis(sph_axis.get());
		axis.normalize();
		double rad=sph_rad.get();
		axis*=2*rad;
		Point p1(cen-axis*0.5);
		Point p2(cen+axis*0.5);
		cyl_p1.set(p1);
		cyl_p2.set(p2);
		cyl_rad.set(rad);
	    }
	    Point p1(cyl_p1.get());
	    Point p2(cyl_p2.get());
	    double dist=(p2-p1).length();
	    double ratio=cyl_rad.get()/(2*dist);
	    cyl_widget->SetEndpoints(p1, p2);
	    cyl_widget->SetRatio(ratio);
	    cyl_widget->SetScale(dist/10);
	} else if(st=="sphere"){
	    widget_id=ogeom->addObj(sph_widget->GetWidget(), "Sphere source widget",
				    &widget_lock);
	    if(oldst == "cylinder"){
		Point p1=cyl_p1.get();
		Point p2=cyl_p2.get();
		Point cen=Interpolate(p1, p2, 0.5);
		sph_cen.set(cen);
		Vector axis(p2-p1);
		double rad=axis.normalize()/2;
		sph_axis.set(axis);
		sph_rad.set(rad);
	    }
	    sph_widget->Connect(ogeom);
	    Point cen(sph_cen.get());
	    double rad=sph_rad.get();
	    Vector axis(sph_axis.get());
	    axis.normalize();
	    Point p1(cen-axis*rad);
	    Point p2(cen+axis*rad);
	    sph_widget->SetEndpoints(p1, p2);
	    sph_widget->SetScale(rad/5);
	}
	oldst=st;
	last_generation=0;
	fudge_widget=0;
    }

    // Spit out the surface
    clString be;
    if(st=="cylinder"){
	surf=scinew CylinderSurface(cyl_p1.get(), cyl_p2.get(), cyl_rad.get(),
				    cyl_nu.get(), cyl_nv.get(), cyl_ndiscu.get());
	if(last_generation && cyl_p1.get() == last_cyl_p1 
	   && cyl_p2.get() == last_cyl_p2 && Abs(cyl_rad.get()-last_cyl_rad) < 1.e-8
	   && cyl_nu.get() == last_cyl_nu && cyl_nv.get() == last_cyl_nv 
	   && cyl_ndiscu.get() == last_cyl_ndiscu){
	    surf->generation=last_generation;
	}
	last_cyl_p1=cyl_p1.get();
	last_cyl_p2=cyl_p2.get();
	last_cyl_rad=cyl_rad.get();
	last_cyl_nu=cyl_nu.get();
	last_cyl_nv=cyl_nv.get();
	last_cyl_ndiscu=cyl_ndiscu.get();
	be=cyl_boundary_expr.get();
    } else if(st=="sphere"){
	surf=scinew SphereSurface(sph_cen.get(), sph_rad.get(), sph_axis.get(),
				  sph_nu.get(), sph_nv.get());
	if(last_generation && sph_cen.get() == last_sph_cen
	   && (sph_axis.get()-last_sph_axis).length2() < 1.e-8
	   && Abs(sph_rad.get()-last_sph_rad) < 1.e-8
	   && sph_nu.get() == last_sph_nu && sph_nv.get() == last_sph_nv ){
	    surf->generation=last_generation;
	}
	last_sph_cen=sph_cen.get();
	last_sph_rad=sph_rad.get();
	last_sph_axis=sph_axis.get();
	last_sph_nu=sph_nu.get();
	last_sph_nv=sph_nv.get();
	be=sph_boundary_expr.get();

	fudge_widget=1;
	Point cen(sph_cen.get());
	double rad=sph_rad.get();
	Vector axis(sph_axis.get());
	axis.normalize();
	Point p1(cen-axis*rad);
	Point p2(cen+axis*rad);
	sph_widget->SetEndpoints(p1, p2);
	sph_widget->SetScale(rad/5);
	fudge_widget=0;
    } else if(st=="point"){
	surf=scinew PointSurface(point_pos.get());
	double val=point_val.get();
	cerr << "val=" << val << '\n';
	Color c(cmap->lookup(val)->diffuse);
	cerr << "diffuse=" << c.r() << ", " << c.g() << " " << c.b() << '\n';
	pt_widget->SetMaterial(PointWidget::PointMatl, cmap->lookup(val));
	be=to_string(val);
    } else {
	error("Unknown surfacetype: "+st);
    }
    if(surf_id)
	ogeom->delObj(surf_id);
    if(surf){
	if(be != ""){
	    surf->set_bc(be);
	}
        outport->send(SurfaceHandle(surf));
	last_generation=surf->generation;
    }

    GeomObj* surfobj=surf?surf->get_obj(cmap):0;
    if(surfobj){
	MaterialHandle surf_matl(scinew Material(Color(0,0,0),
						 Color(.5,.5,.5),
						 Color(.6, .6, .6), 10));
	GeomMaterial* matl=scinew GeomMaterial(surfobj, surf_matl);
	surf_id=ogeom->addObj(matl, st+" source");
    }
}

void GenSurface::widget_moved(int last)
{
    if(fudge_widget)
	return;
    clString st(surfacetype.get());
    if(st=="point"){
	Point p1(pt_widget->GetPosition());
	point_pos.set(p1);
    } else if(st=="cylinder"){
	Point p1, p2;
	cyl_widget->GetEndpoints(p1, p2);
	cyl_p1.set(p1);
	cyl_p2.set(p2);
	//double ratio=cyl_widget->GetRatio();
	//double dist=(p2-p1).length();
	//	double radius=2*dist*ratio;
//	cerr << "Setting rad to: " << radius < endl;
//	cyl_rad.set(radius);
    } else if(st=="sphere"){
	Point p1, p2;
	sph_widget->GetEndpoints(p1, p2);
	sph_cen.set(Interpolate(p1, p2, 0.5));
	Vector axis(p2-p1);
	double rad=axis.normalize()/2;
	sph_rad.set(rad);
	sph_axis.set(axis);
    } else {
	cerr << "Unknown st: " << st << endl;
    }
    if(last && !abort_flag)
    {
	abort_flag=1;
	want_to_execute();
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.9  2000/03/17 09:27:20  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.8  1999/10/07 02:06:59  sparker
// use standard iostreams and complex type
//
// Revision 1.7  1999/09/04 06:01:39  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.6  1999/08/29 00:46:44  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/25 03:47:59  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:53  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:56  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:42  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:57  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:27  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
