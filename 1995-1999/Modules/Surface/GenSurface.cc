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
#include <Datatypes/ColorMap.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/SurfacePort.h>
#include <Geom/Material.h>
#include <Geom/Pick.h>
#include <Geom/Sphere.h>
#include <Geom/TCLGeom.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <Widgets/GaugeWidget.h>
#include <Widgets/PointWidget.h>

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
    GenSurface(const GenSurface&, int deep);
    virtual ~GenSurface();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void widget_moved(int last);
};

extern "C" {
Module* make_GenSurface(const clString& id)
{
    return scinew GenSurface(id);
}
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
  point_val("point_val", id, this)
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

GenSurface::GenSurface(const GenSurface& copy, int deep)
: Module(copy, deep), surfacetype("surfacetype", id, this),
  cyl_p1("cyl_p1", id, this), cyl_p2("cyl_p2", id, this),
  cyl_rad("cyl_rad", id, this), cyl_nu("cyl_nu", id, this),
  cyl_nv("cyl_nv", id, this), cyl_ndiscu("cyl_ndiscu", id, this),
  sph_cen("sph_cen", id, this), sph_axis("sph_axis", id, this),
  sph_rad("sph_rad", id, this),
  sph_nu("sph_nu", id, this), sph_nv("sph_nv", id, this),
  point_pos("point_pos", id, this), point_rad("point_rad", id, this),
  cyl_boundary_expr("cyl_boundary_expr", id, this),
  sph_boundary_expr("sph_boundary_expr", id, this),
  point_val("point_val", id, this)
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
