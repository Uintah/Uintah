
/*
 *  IsoSurface.cc:  The first module!
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/BitArray1.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Queue.h>
#include <Classlib/Stack.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColormapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Triangles.h>
#include <Geom/TriStrip.h>
#include <Geometry/Point.h>
#include <Geometry/Plane.h>
#include <Malloc/Allocator.h>
#include <Math/Expon.h>
#include <Math/MiscMath.h>
#include <TCL/TCLvar.h>
#include <Widgets/ArrowWidget.h>
#include <iostream.h>
#include <strstream.h>

// just so I can see the proccess id...

#include <sys/types.h>
#include <unistd.h>

class IsoSurface : public Module {
    ScalarFieldIPort* infield;
    ScalarFieldIPort* incolorfield;
    ColormapIPort* incolormap;

    GeometryOPort* ogeom;
    SurfaceOPort* osurf;

    TCLPoint seed_point;
    TCLint have_seedpoint;
    TCLdouble isoval;
    TCLint do_3dwidget;
    TCLint emit_surface;

    TriSurface* surf;

    int isosurface_id;
    int need_seed;

    double old_min;
    double old_max;
    Point old_bmin;
    Point old_bmax;
    int sp;
    TCLint show_progress;

    MaterialHandle matl;

    int iso_cube(int, int, int, double, GeomTrianglesP*, ScalarFieldRG*);
    int iso_tetra(Element*, Mesh*, ScalarFieldUG*, double, GeomTrianglesP*);
    int iso_tetra_s(int,Element*, Mesh*, ScalarFieldUG*, double, 
		    GeomTriStripList*);
    void iso_tetra_strip(int, Mesh*, ScalarFieldUG*, double, 
			 GeomGroup*, BitArray1&);

    void iso_reg_grid(ScalarFieldRG*, const Point&, GeomTrianglesP*);
    void iso_reg_grid(ScalarFieldRG*, double, GeomTrianglesP*);
    void iso_tetrahedra(ScalarFieldUG*, const Point&, GeomTrianglesP*);
    void iso_tetrahedra(ScalarFieldUG*, double, GeomTrianglesP*);

    // extract an iso-surface into a tri-strip

    void iso_tetrahedra_strip(ScalarFieldUG*, const Point&, GeomGroup*);
    void iso_tetrahedra_strip(ScalarFieldUG*, double, GeomGroup*);
    
    int iso_strip_enter(int,Element*, Mesh*, ScalarFieldUG*, double, 
			GeomTriStripList*);
    void remap_element(int& rval, Element *src, Element *dst);

    void find_seed_from_value(const ScalarFieldHandle&);
    void order_and_add_points(const Point &p1, const Point &p2, 
			      const Point &p3, const Point &v1, double val);

    virtual void widget_moved(int last);
    CrowdMonitor widget_lock;
    int widget_id;
    ArrowWidget* widget;

    int need_find;

    int init;
    Point ov[9];
    Point v[9];
public:
    IsoSurface(const clString& id);
    IsoSurface(const IsoSurface&, int deep);
    virtual ~IsoSurface();
    virtual Module* clone(int deep);
    virtual void execute();
};

#define FACE4 8
#define FACE3 4
#define FACE2 2
#define FACE1 1
#define ALLFACES (FACE1|FACE2|FACE3|FACE4)

struct MCubeTable {
    int which_case;
    int permute[8];
    int nbrs;
};

#include "mcube.h"

extern "C" {
Module* make_IsoSurface(const clString& id)
{
    return scinew IsoSurface(id);
}
};

static clString module_name("IsoSurface");
static clString surface_name("IsoSurface");
static clString widget_name("IsoSurface widget");

IsoSurface::IsoSurface(const clString& id)
: Module("IsoSurface", id, Filter), seed_point("seed_point", id, this),
  have_seedpoint("have_seedpoint", id, this), isoval("isoval", id, this),
  do_3dwidget("do_3dwidget", id, this), emit_surface("emit_surface", id, this),
  show_progress("show_progress", id, this)
{
    // Create the input ports
    infield=scinew ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
    add_iport(infield);
    incolorfield=scinew ScalarFieldIPort(this, "Color Field", ScalarFieldIPort::Atomic);
    add_iport(incolorfield);
    incolormap=scinew ColormapIPort(this, "Color Map", ColormapIPort::Atomic);
    add_iport(incolormap);
    

    // Create the output port
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    osurf=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurf);

    isoval.set(1);
    need_seed=1;

    matl=scinew Material(Color(0,0,0), Color(0,.8,0),
		      Color(.7,.7,.7), 20);
    isosurface_id=0;

    old_min=old_max=0;
    old_bmin=old_bmax=Point(0,0,0);

    float INIT(.1);
    widget = scinew ArrowWidget(this, &widget_lock, INIT);
    need_find=1;
    init=1;
}

IsoSurface::IsoSurface(const IsoSurface& copy, int deep)
: Module(copy, deep), seed_point("seed_point", id, this),
  have_seedpoint("have_seedpoint", id, this), isoval("isoval", id, this),
  do_3dwidget("do_3dwidget", id, this), emit_surface("emit_surface", id, this),
  show_progress("show_progress", id, this)
{
    NOT_FINISHED("IsoSurface::IsoSurface");
}

IsoSurface::~IsoSurface()
{
}

Module* IsoSurface::clone(int deep)
{
    return scinew IsoSurface(*this, deep);
}

void IsoSurface::execute()
{
    if(isosurface_id){
	ogeom->delObj(isosurface_id);
    }
    ScalarFieldHandle field;
    if(!infield->get(field))
	return;
    ScalarFieldHandle colorfield;
    int have_colorfield=incolorfield->get(colorfield);
    ColormapHandle cmap;
    int have_colormap=incolormap->get(cmap);

    if(init == 1){
	init=0;
	widget_id = ogeom->addObj(widget->GetWidget(), widget_name, &widget_lock);
	widget->Connect(ogeom);
    }
	
    double min, max;
    field->get_minmax(min, max);
    if(min != old_min || max != old_max){
	char buf[1000];
	ostrstream str(buf, 1000);
	str << id << " set_minmax " << min << " " << max << '\0';
	TCL::execute(str.str());
	old_min=min;
	old_max=max;
    }
    Point bmin, bmax;
    field->get_bounds(bmin, bmax);
    if(bmin != old_bmin || bmax != old_bmax){
	char buf[1000];
	ostrstream str(buf, 1000);
	str << id << " set_bounds " << bmin.x() << " " << bmin.y() << " " << bmin.z() << " " << bmax.x() << " " << bmax.y() << " " << bmax.z() << '\0';
	TCL::execute(str.str());
	old_bmin=bmin;
	old_bmax=bmax;	
    }
    if(need_seed){
	find_seed_from_value(field);
	need_seed=0;
    }
    sp=show_progress.get();
    if(do_3dwidget.get() && have_seedpoint.get()){
	double widget_scale=0.05*field->longest_dimension();
	

//	Point sp(seed_point.get());
//	widget_sphere->cen=sp;
//	widget_sphere->rad=1*widget_scale;
	if(need_find != 0){
	    Point sp(Interpolate(bmin, bmax, 0.5));
	    widget->SetPosition(sp);
	    widget->SetScale(widget_scale);
	    need_find=0;
	}
	Point sp(widget->GetPosition());
	Vector grad(-field->gradient(sp));
	if(grad.length2() > 0)
	    grad.normalize();
	widget->SetDirection(grad);
	widget->SetState(1);
    } else {
	widget->SetState(0);
    }
    double iv=isoval.get();
    Point sp;
    if(have_seedpoint.get()){
	if(do_3dwidget.get()){
	    sp=widget->GetPosition();
	} else {
	    sp=seed_point.get();
	}
	if(!field->interpolate(sp, iv)){
	    iv=min;
	}
	cerr << "at p=" << sp << ", iv=" << iv << endl;
    }
    GeomTrianglesP* group = scinew GeomTrianglesP;
    GeomGroup* tgroup=scinew GeomGroup;
    GeomObj* topobj=tgroup;
    if(have_colormap && !have_colorfield){
	// Paint entire surface based on colormap
	topobj=scinew GeomMaterial(tgroup, cmap->lookup(iv));
    } else if(have_colormap && have_colorfield){
	// Nothing - done per vertex
    } else {
	// Default material
	topobj=scinew GeomMaterial(tgroup, matl);
    }
    ScalarFieldRG* regular_grid=field->getRG();
    ScalarFieldUG* unstructured_grid=field->getUG();

    Point minPt, maxPt;
    double spacing=0;
    Vector diff;

    if (emit_surface.get()) {
        field->get_bounds(minPt, maxPt);
        diff=maxPt-minPt;
        spacing=Max(diff.x(), diff.y(), diff.z());
    }   

    if(regular_grid){
	if (emit_surface.get()) {
	    surf=scinew TriSurface;
	    int nx=regular_grid->nx;
	    int ny=regular_grid->ny;
	    int nz=regular_grid->nz;
	    spacing=Max(diff.x()/nx, diff.y()/ny, diff.z()/nz);
	    surf->construct_grid(nx+2, ny+2, nz+2, 
				 minPt+(Vector(1.001,1.029,0.917)*(-.001329)),
				 spacing);
	}	
	if(have_seedpoint.get()){
	    iso_reg_grid(regular_grid, sp, group);
	} else {
	    iso_reg_grid(regular_grid, iv, group);
	}
	tgroup->add(group);
    } else if(unstructured_grid){
	if (emit_surface.get()) {
	    surf=scinew TriSurface;
	    int pts_per_side=(int) Cbrt(unstructured_grid->mesh->nodes.size());
	    spacing/=pts_per_side;
	    surf->construct_grid(pts_per_side+2,pts_per_side+2,pts_per_side+2, 
				 minPt+(Vector(1.001,1.029,0.917)*(-.001329)),
				 spacing);
	}	
	if(have_seedpoint.get()){
	    Point sp(seed_point.get());

// the _strip version extracts into tri-strips PPS

//	    iso_tetrahedra(unstructured_grid, sp, group);
	    iso_tetrahedra_strip(unstructured_grid,sp,tgroup);
//	    tgroup->add(group);
	} else {
	    iso_tetrahedra(unstructured_grid, iv, group);
	    tgroup->add(group);
//	    iso_tetrahedra_strip(unstructured_grid,iv,tgroup);
	}
    } else {
	error("I can't IsoSurface this type of field...");
    }

    if(tgroup->size() == 0){
	delete tgroup;
	if (emit_surface.get())
	    delete surf;
	isosurface_id=0;
    } else {
	isosurface_id=ogeom->addObj(topobj, surface_name);
	if (emit_surface.get()) {
	    osurf->send(SurfaceHandle(surf));
	}
    }
}


// Given the points p1,p2,p3 and a point v1 that lies in front of the plane
// and it's implicit value, we calculate the whether the point is in front
// of or beind the plane that p1,p2,p3 define, and make sure that val has
// the apropriate sign (i.e. the points are cw --> negative val is outside.
// finally we "cautiously-add" the correctly directed triangle to surf.
 
void IsoSurface::order_and_add_points(const Point &p1, const Point &p2, 
				      const Point &p3, const Point &v1,
				      double val) 
{

    if (Plane(p1,p2,p3).eval_point(v1)*val >= 0)	// is the order right?
	surf->cautious_add_triangle(p1,p2,p3,1);
    else	
	surf->cautious_add_triangle(p2,p1,p3,1);
}


//The isosurface code needs to be able to create clockwise ordered triangles for it to be useful to me.  Clearly we have the information to judge which side of any triangle we generate is inside -- the side which has positive values.   
 
//My first thought was to generate a point off the centroid, just a tad away from the face in the direction of the triangle's normal.  Then, get the "value" of the point (in the field), if it's negative, switch the ordering.


int IsoSurface::iso_cube(int i, int j, int k, double isoval,
			 GeomTrianglesP* group, ScalarFieldRG* field)
{
    double oval[9];
    oval[1]=field->grid(i, j, k)-isoval;
    oval[2]=field->grid(i+1, j, k)-isoval;
    oval[3]=field->grid(i+1, j+1, k)-isoval;
    oval[4]=field->grid(i, j+1, k)-isoval;
    oval[5]=field->grid(i, j, k+1)-isoval;
    oval[6]=field->grid(i+1, j, k+1)-isoval;
    oval[7]=field->grid(i+1, j+1, k+1)-isoval;
    oval[8]=field->grid(i, j+1, k+1)-isoval;
    ov[1]=field->get_point(i,j,k);
    ov[2]=field->get_point(i+1, j, k);
    ov[3]=field->get_point(i+1, j+1, k);
    ov[4]=field->get_point(i, j+1, k);
    ov[5]=field->get_point(i, j, k+1);
    ov[6]=field->get_point(i+1, j, k+1);
    ov[7]=field->get_point(i+1, j+1, k+1);
    ov[8]=field->get_point(i, j+1, k+1);
    int mask=0;
    int idx;
    for(idx=1;idx<=8;idx++){
	if(oval[idx]<0)
	    mask|=1<<(idx-1);
    }
    MCubeTable* tab=&mcube_table[mask];
    double val[9];
    for(idx=1;idx<=8;idx++){
	val[idx]=oval[tab->permute[idx-1]];
	v[idx]=ov[tab->permute[idx-1]];
    }
    int wcase=tab->which_case;
    switch(wcase){
    case 0:
	break;
    case 1:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[1],val[1]);
	}
	break;
    case 2:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    if (group->add(p3, p4, p1))
		if (emit_surface.get())
		    order_and_add_points(p3,p4,p1,v[1],val[1]);
	}
	break;
    case 3:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    if (group->add(p4, p5, p6))
		if (emit_surface.get())
		    order_and_add_points(p4,p5,p6,v[3],val[3]);
	}
	break;
    case 4:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p5(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p6(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    if (group->add(p4, p5, p6))
		if (emit_surface.get())
		    order_and_add_points(p4,p5,p6,v[7],val[7]);
	}
	break;
    case 5:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[2],val[2]);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    if (group->add(p4, p3, p2))
		if (emit_surface.get())
		    order_and_add_points(p4,p3,p2,v[2],val[2]);
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    if (group->add(p5, p4, p2))
		if (emit_surface.get())
		    order_and_add_points(p5,p4,p2,v[2],val[2]);
	}
	break;
    case 6:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    if (group->add(p3, p4, p1))
		if (emit_surface.get())
		    order_and_add_points(p3,p4,p1,v[1],val[1]);
	    Point p5(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p7(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    if (group->add(p5, p6, p7))
		if (emit_surface.get())
		    order_and_add_points(p5,p6,p7,v[7],val[7]);
	}
	break;
    case 7:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[2],val[2]);
	    Point p4(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p5(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p6(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    if (group->add(p4, p5, p6))
		if (emit_surface.get())
		    order_and_add_points(p4,p5,p6,v[4],val[4]);
	    Point p7(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p8(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    Point p9(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    if (group->add(p7, p8, p9))
		if (emit_surface.get())
		    order_and_add_points(p7,p8,p9,v[7],val[7]);
	}
	break;
    case 8:
	{
	    Point p1(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    if (group->add(p4, p1, p3))
		if (emit_surface.get())
		    order_and_add_points(p4,p1,p3,v[1],val[1]);
	}
	break;
    case 9:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[5],val[5]);
	    Point p4(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    if (group->add(p1, p3, p4))
		if (emit_surface.get())
		    order_and_add_points(p1,p3,p4,v[5],val[5]);
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    if (group->add(p1, p4, p5))
		if (emit_surface.get())	
		    order_and_add_points(p1,p4,p5,v[5],val[5]);
	    Point p6(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    if (group->add(p5, p4, p6))
		if (emit_surface.get())
		    order_and_add_points(p5,p4,p6,v[5],val[5]);
	}
	break;
    case 10:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p3(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    if (group->add(p2, p4, p3))
		if (emit_surface.get())
		    order_and_add_points(p2,p4,p3,v[1],val[1]);
	    Point p5(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p6(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    Point p7(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    if (group->add(p5, p6, p7))
		if (emit_surface.get())
		    order_and_add_points(p5,p6,p7,v[6],val[6]);
	    Point p8(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    if (group->add(p2, p8, p3))
		if (emit_surface.get())
		    order_and_add_points(p2,p8,p3,v[6],val[6]);
	}
	break;
    case 11:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[5],val[5]);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    if (group->add(p1, p3, p4))
		if (emit_surface.get())
		    order_and_add_points(p1,p3,p4,v[5],val[5]);
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    if (group->add(p1, p4, p5))
		if (emit_surface.get())
		    order_and_add_points(p1,p4,p5,v[5],val[5]);
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    if (group->add(p4, p3, p6))
		if (emit_surface.get())	
		    order_and_add_points(p4,p3,p6,v[5],val[5]);
	}
	break;
    case 12:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[5],val[5]);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    if (group->add(p3, p2, p4))
		if (emit_surface.get())	
		    order_and_add_points(p3,p2,p4,v[5],val[5]);
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    if (group->add(p4, p2, p5))
		if (emit_surface.get())
		    order_and_add_points(p4,p2,p5,v[5],val[5]);
	    Point p6(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p7(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p8(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    if (group->add(p6, p7, p8))
		if (emit_surface.get())
		    order_and_add_points(p6,p7,p8,v[4],val[4]);
	}
	break;
    case 13:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    if (group->add(p4, p5, p6))
		if (emit_surface.get())
		    order_and_add_points(p4,p5,p6,v[3],val[3]);
	    Point p7(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p8(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    Point p9(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    if (group->add(p7, p8, p9))
		if (emit_surface.get())
		    order_and_add_points(p7,p8,p9,v[6],val[6]);
	    Point p10(Interpolate(v[8], v[5], val[8]/(val[8]-val[5])));
	    Point p11(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    Point p12(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    if (group->add(p10, p11, p12))
		if (emit_surface.get())
		    order_and_add_points(p10,p11,p12,v[8],val[8]);
	}
	break;
    case 14:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    if (group->add(p1, p2, p3))
		if (emit_surface.get())
		    order_and_add_points(p1,p2,p3,v[5],val[5]);
	    Point p4(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    if (group->add(p1, p3, p4))
		if (emit_surface.get())
		    order_and_add_points(p1,p3,p4,v[5],val[5]);
	    Point p5(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    if (group->add(p1, p4, p5))
		if (emit_surface.get())
		    order_and_add_points(p1,p4,p5,v[5],val[5]);
	    Point p6(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    if (group->add(p3, p6, p4))
		if (emit_surface.get())
		    order_and_add_points(p3,p6,p4,v[5],val[5]);
	}
	break;
    default:
	error("Bad case in marching cubes!\n");
	break;
    }
    return(tab->nbrs);
}

void IsoSurface::iso_reg_grid(ScalarFieldRG* field, double isoval,
			      GeomTrianglesP* group)
{
    int nx=field->nx;
    int ny=field->ny;
    int nz=field->nz;
    for(int i=0;i<nx-1;i++){
	//update_progress(i, nx);
	for(int j=0;j<ny-1;j++){
	    for(int k=0;k<nz-1;k++){
		iso_cube(i,j,k, isoval, group, field);
	    }
	    if(sp && abort_flag)
		return;
	}
    }
}

void IsoSurface::iso_reg_grid(ScalarFieldRG* field, const Point& p,
			      GeomTrianglesP* group)
{
    int nx=field->nx;
    int ny=field->ny;
    int nz=field->nz;
    double iv;
    if(!field->interpolate(p, iv)){
	error("Seed point not in field boundary");
	return;
    }
    isoval.set(iv);
    cerr << "Isoval = " << iv << "\n";
    HashTable<int, int> visitedPts;
    Queue<int> surfQ;
    int px, py, pz;
    field->locate(p, px, py, pz);
    int pLoc=(((pz*ny)+py)*nx)+px;
    int dummy;
    visitedPts.insert(pLoc, 0);
    surfQ.append(pLoc);
    int counter=1;
    GeomID groupid=0;
    while(!surfQ.is_empty()) {
#if 0
	if (sp && counter%400 == 0) {
	    if(!ogeom->busy()){
		if (groupid)
		    ogeom->delObj(groupid);
		groupid=ogeom->addObj(group->clone(), surface_name);
		ogeom->flushViews();
	    }
	}
#endif
	if(sp && abort_flag){
	    if(groupid)
		ogeom->delObj(groupid);
	    return;
	}
	pLoc=surfQ.pop();
	pz=pLoc/(nx*ny);
	dummy=pLoc%(nx*ny);
	py=dummy/nx;
	px=dummy%nx;
	int nbrs=iso_cube(px, py, pz, iv, group, field);
	if ((nbrs & 1) && (px!=0)) {
	    pLoc-=1;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc+=1;
	}
	if ((nbrs & 2) && (px!=nx-2)) {
	    pLoc+=1;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc-=1;
	}
	if ((nbrs & 8) && (py!=0)) {
	    pLoc-=nx;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc+=nx;
	}
	if ((nbrs & 4) && (py!=ny-2)) {
	    pLoc+=nx;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc-=nx;
	}
	if ((nbrs & 16) && (pz!=0)) {
	    pLoc-=nx*ny;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc+=nx*ny;
	}
	if ((nbrs & 32) && (pz!=nz-2)) {
	    pLoc+=nx*ny;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc-=nx*ny;
	}
	counter++;
    }
    if (counter > 400)
	if (groupid)
	    ogeom->delObj(groupid);
}

int IsoSurface::iso_tetra(Element* element, Mesh* mesh,
			  ScalarFieldUG* field, double isoval,
			  GeomTrianglesP* group)
{
    double v1=field->data[element->n[0]]-isoval;
    double v2=field->data[element->n[1]]-isoval;
    double v3=field->data[element->n[2]]-isoval;
    double v4=field->data[element->n[3]]-isoval;
    NodeHandle n1=mesh->nodes[element->n[0]];
    NodeHandle n2=mesh->nodes[element->n[1]];
    NodeHandle n3=mesh->nodes[element->n[2]];
    NodeHandle n4=mesh->nodes[element->n[3]];
    if(v1 == v2 && v3 == v4 && v1 == v4)
	return 0;
    int f1=v1<0;
    int f2=v2<0;
    int f3=v3<0;
    int f4=v4<0;
    int mask=(f1<<3)|(f2<<2)|(f3<<1)|f4;
    int faces=0;
    switch(mask){
    case 0:
    case 15:
	// Nothing to do....
	break;
    case 1:
    case 14:
	// Point 4 is inside
 	if(v4 != 0){
	    Point p1(Interpolate(n4->p, n1->p, v4/(v4-v1)));
	    Point p2(Interpolate(n4->p, n2->p, v4/(v4-v2)));
	    Point p3(Interpolate(n4->p, n3->p, v4/(v4-v3)));
	    group->add(p1, p2, p3);
	    faces=FACE1|FACE2|FACE3;
	}
	break;
    case 2:
    case 13:
	// Point 3 is inside
 	if(v3 != 0){
	    Point p1(Interpolate(n3->p, n1->p, v3/(v3-v1)));
	    Point p2(Interpolate(n3->p, n2->p, v3/(v3-v2)));
	    Point p3(Interpolate(n3->p, n4->p, v3/(v3-v4)));
	    group->add(p1, p2, p3);
	    faces=FACE1|FACE2|FACE4;
	}
	break;
    case 3:
    case 12:
	// Point 3 and 4 are inside
 	{
	    Point p1(Interpolate(n3->p, n1->p, v3/(v3-v1)));
	    Point p2(Interpolate(n3->p, n2->p, v3/(v3-v2)));
	    Point p3(Interpolate(n4->p, n1->p, v4/(v4-v1)));
	    Point p4(Interpolate(n4->p, n2->p, v4/(v4-v2)));
	    if(v3 != v4){
		if(v3 != 0 && v1 != 0)
		    group->add(p1, p2, p3);
		if(v4 != 0 && v2 != 0)
		    group->add(p2, p3, p4);
	    }
	    faces=ALLFACES;
	}
	break;
    case 4:
    case 11:
	// Point 2 is inside
 	if(v2 != 0){
	    Point p1(Interpolate(n2->p, n1->p, v2/(v2-v1)));
	    Point p2(Interpolate(n2->p, n3->p, v2/(v2-v3)));
	    Point p3(Interpolate(n2->p, n4->p, v2/(v2-v4)));
	    group->add(p1, p2, p3);
	    faces=FACE1|FACE3|FACE4;
	}
	break;
    case 5:
    case 10:
	// Point 2 and 4 are inside
 	{
	    Point p1(Interpolate(n2->p, n1->p, v2/(v2-v1)));
	    Point p2(Interpolate(n2->p, n3->p, v2/(v2-v3)));
	    Point p3(Interpolate(n4->p, n1->p, v4/(v4-v1)));
	    Point p4(Interpolate(n4->p, n3->p, v4/(v4-v3)));
	    if(v2 != v4){
		if(v2 != 0 && v1 != 0)
		    group->add(p1, p2, p3);
		if(v4 != 0 && v3 != 0)
		    group->add(p2, p3, p4);
	    }
	    faces=ALLFACES;
	}
	break;
    case 6:
    case 9:
	// Point 2 and 3 are inside
 	{
	    Point p1(Interpolate(n2->p, n1->p, v2/(v2-v1)));
	    Point p2(Interpolate(n2->p, n4->p, v2/(v2-v4)));
	    Point p3(Interpolate(n3->p, n1->p, v3/(v3-v1)));
	    Point p4(Interpolate(n3->p, n4->p, v3/(v3-v4)));
	    if(v2 != v3){
		if(v2 != 0 && v1 != 0)
		    group->add(p1, p2, p3);
		if(v3 != 0 && v4 != 0)
		    group->add(p2, p3, p4);
	    }
	    faces=ALLFACES;
	}
	break;
    case 7:
    case 8:
	// Point 1 is inside
 	if(v1 != 0){
	    Point p1(Interpolate(n1->p, n2->p, v1/(v1-v2)));
	    Point p2(Interpolate(n1->p, n3->p, v1/(v1-v3)));
	    Point p3(Interpolate(n1->p, n4->p, v1/(v1-v4)));
	    group->add(p1, p2, p3);
	    faces=FACE2|FACE3|FACE4;
	}
	break;
    }
    return faces;
}

void IsoSurface::iso_tetrahedra(ScalarFieldUG* field, double isoval,
				GeomTrianglesP* group)
{
    Mesh* mesh=field->mesh.get_rep();
    int nelems=mesh->elems.size();
    for(int i=0;i<nelems;i++){
	//update_progress(i, nelems);
	Element* element=mesh->elems[i];
	iso_tetra(element, mesh, field, isoval, group);
	if(sp && abort_flag)
	    return;
    }
}

// build tri-strips from iso-value


void IsoSurface::iso_tetrahedra_strip(ScalarFieldUG* field, double isoval,
				      GeomGroup* group)
{
    Mesh* mesh=field->mesh.get_rep();
    int nelems=mesh->elems.size();
    BitArray1 visited(nelems, 0);

    for(int i=0;i<nelems;i++){
	if (!visited.is_set(i)) {
	    visited.set(i);
	    iso_tetra_strip(i, mesh, field, isoval, group,visited);
	}
	if(sp && abort_flag)
	    return;
    }
}

// this structure could be used if you want to "start" from a different nod
// you could permute the vertices in the tri strip and try and exit from another face

struct strip_pair {
    int index;
    int flag;
};

// these flags are for each edge

const int E12 = 1;
const int E21 = 1;
const int E13 = 2; const int E31 = 2;
const int E14 = 3; const int E41 = 3;
const int E23 = 4; const int E32 = 4;
const int E24 = 5; const int E42 = 5;
const int E34 = 6; const int E43 = 6;

// this table was generate by a program, it represents how any edge combination
// (3bits 3bits) maps into a given face...

const int EDGE_TO_FACE[64] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, 8, 4, 8, 4, -1, -1, 
        -1, 8, -1, 2, 8, -1, 2, -1, 
        -1, 4, 2, -1, -1, 4, 2, -1, 
        -1, 8, 8, -1, -1, 1, 1, -1, 
        -1, 4, -1, 4, 1, -1, 1, -1, 
        -1, -1, 2, 2, 1, 1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1 
};

const int EDGE_TO_NFACE[9] = { 0, 0,1,3,2,5,6,7,3 };
inline int FREMAP(int val) { return EDGE_TO_NFACE[val]; }

// this maps each edge to the edge exactly opposite from it
// just pad with a zero so indexing is simpler...

const int EDGE_TO_EDGE[7] = {
    0,
    E34, E24, E23, E14, E13, E12 
}; 

inline int GET_EDGE_OPP(int edge) { return EDGE_TO_EDGE[edge]; }

// this table maps values for edge with one triangle
// intersection - it must be the edge that shares the common
// node and also connects the node in either edge:
// 12 13 -> 14

const int EDGE_MAP_ONE[64] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, 3, 2, 5, 4, -1, -1, 
        -1, 3, -1, 1, 6, -1, 4, -1, 
        -1, 2, 1, -1, -1, 6, 5, -1, 
        -1, 5, 6, -1, -1, 1, 2, -1, 
        -1, 4, -1, 6, 1, -1, 3, -1, 
        -1, -1, 4, 5, 2, 3, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1
	};

inline int GET_EDGE_ONE(int val) { return EDGE_MAP_ONE[val&63]; }

// these will mask of the correct bits of a faceted edge map
// LOWE is the latest edge, HIGHE is the oldest...

inline int LOWE(int e) { return e&(7); }
inline int HIGHE(int e) { return (e&(7<<3))>>3; }


// these are macros for finding/encoding faces...

inline int F_FACE(int e1, int e2) { return EDGE_TO_FACE[(e1<<3)|e2]; }
inline int E_FACE(int e1, int e2) { return ((e1<<3)|e2); }

inline void add_strip_point(const int nvert, GeomTriStripList* group,
			    const NodeHandle& n1,
			    const NodeHandle& n2,
			    const NodeHandle& n3,
			    const NodeHandle& n4,
			    const double v1,
			    const double v2,
			    const double v3,
			    const double v4)
{
    int i = group->num_since();
    Point pm1(group->get_pm1());
    Point pm2(group->get_pm2());
    Point p1;

	switch (nvert) {
	case E12:
	    p1 = (Interpolate(n1->p,n2->p,v1/(v1-v2)));
	    break;
	case E13:
	    p1 =  (Interpolate(n1->p,n3->p,v1/(v1-v3)));
	    break;
	case E14:
	    p1 = (Interpolate(n1->p,n4->p,v1/(v1-v4)));
	    break;
	case E23:
	    p1 = (Interpolate(n2->p,n3->p,v2/(v2-v3)));
	    break;
	case E24:
	    p1 = (Interpolate(n2->p,n4->p,v2/(v2-v4)));
	    break;
	case E34:
	    p1 = (Interpolate(n3->p,n4->p,v3/(v3-v4)));
	    break;
	default:
	    cerr << "Major error, unnkown code: " << nvert << "\n";
	}

    Vector n(Cross(p1-pm2,pm1-pm2));
    if (n.length2() > 0){
	n.normalize();
    }
    if (!(i&1)) // this implies a different sign convention
	n *=-1.0;
    group->add(p1,n);
}

// an "entrance" contains 6 bits - 3 bits for 1st edge, 3 for second
// you encode an "exit" with the required "entrance" bits and the following
// other high order bits:  4 bits of what faces must be checked for pushing
// and 4 bits for the face to recures (only 1 can be set at most)

int IsoSurface::iso_strip_enter(int inc, 
				Element *element, Mesh* mesh, 
				ScalarFieldUG* field, double isoval, 
				GeomTriStripList* group)
{
    double v1=field->data[element->n[0]]-isoval;
    double v2=field->data[element->n[1]]-isoval;
    double v3=field->data[element->n[2]]-isoval;
    double v4=field->data[element->n[3]]-isoval;
    NodeHandle n1=mesh->nodes[element->n[0]];
    NodeHandle n2=mesh->nodes[element->n[1]];
    NodeHandle n3=mesh->nodes[element->n[2]];
    NodeHandle n4=mesh->nodes[element->n[3]];
    int f1=v1<0;
    int f2=v2<0;
    int f3=v3<0;
    int f4=v4<0;
//    int mask=(f1<<3)|(f2<<2)|(f3<<1)|f4;
    int pfaces=0;
    int rfaces=0;
    // 1st of all, there must be at least 1 edge,
    // since you can only get here if you are following
    // a tri-strip - if there
    int two_exit = (f1+f2+f3+f4)&(1);

    // remember LOW is the most recently emmited
    if (two_exit) { // exit on only one face
	int nvert = GET_EDGE_ONE(inc);
	add_strip_point(nvert,group,n1,n2,n3,n4,v1,v2,v3,v4);
	rfaces = E_FACE(LOWE(inc),nvert);
	pfaces = F_FACE(HIGHE(inc),nvert);
    }
    else { // you have to add to primitives!
	// this is simple, you must emit the vertex
	// corresponding to the edge opposite the last vertex first
	int nhigh = GET_EDGE_OPP(LOWE(inc));
	add_strip_point(nhigh,group,n1,n2,n3,n4,v1,v2,v3,v4);
	int nlow = GET_EDGE_OPP(HIGHE(inc));
	add_strip_point(nlow,group,n1,n2,n3,n4,v1,v2,v3,v4);
	rfaces = E_FACE(nhigh,nlow);
	pfaces = F_FACE(HIGHE(inc),nhigh)|F_FACE(LOWE(inc),nlow);
    }
    return (pfaces<<6)|rfaces;
}

const int LO_RMAP_T[] = {0,0,0,1,1,2};
const int HI_RMAP_T[] = {1,2,3,2,3,3};

inline int LO_RMAP(int val) { return LO_RMAP_T[val-1]; }
inline int HI_RMAP(int val) { return HI_RMAP_T[val-1]; }

const int FULL_RMAP_TABLE[16] =
{
    0,0,0,1,
    0,2,4,0,
    0,3,5,0,
    6,0,0,0
};

inline int REMAP_FULL(int val) { return FULL_RMAP_TABLE[val];}

void IsoSurface::remap_element(int& rval, Element *src, Element *dst)
{
    // you only need to remap the 3 nodes for the incoming edge
    // but I'm being lazy for now...

    int i,j;
    int mapping[4] = { -1,-2,-3,-4 };

    for(i=0;i<4;i++) {
	for(j=0;j<4;j++) {
	    if (src->n[i] == dst->n[j]) {
		mapping[i] = 1<<j;
		j = 4; // break out for this node
	    }
	}
    }

    // now that we have our mapping vector,
    // remap the two edges

    int older,newer;

    older = (rval>>3)&7;
    newer = (rval&7);

    older = REMAP_FULL(mapping[LO_RMAP(older)]|mapping[HI_RMAP(older)]);
    newer = REMAP_FULL(mapping[LO_RMAP(newer)]|mapping[HI_RMAP(newer)]);

    rval = (rval&(~63))|newer|(older<<3);
}

// this is a function that takes in the edges and faces
// ABCD and generates the correct primitives/codes

void inline emit_in_2(int& rf, int& pf,
		      int& eA,int& eB,int& eC,int& eD,
		      Point& pA,Point& pB,Point& pC,Point& pD,
		      GeomTriStripList* group)
{
    rf = E_FACE(eC,eD);
    pf = ALLFACES&(~F_FACE(eC,eD));

    Vector n(Cross(pB-pA,pC-pA));
    if (n.length2() > 0)
	n.normalize();
    Vector n2(Cross(pB-pC,pD-pC));
    if (n2.length2() > 0)
	n2.normalize();
    
    group->add(pA);
    group->add(pB);
    group->add(pC,n);
    group->add(pD,n2);    
}

int IsoSurface::iso_tetra_s(int nbr_status,Element *element, Mesh* mesh, 
			    ScalarFieldUG* field, double isoval, 
			    GeomTriStripList* group)
{
    double v1=field->data[element->n[0]]-isoval;
    double v2=field->data[element->n[1]]-isoval;
    double v3=field->data[element->n[2]]-isoval;
    double v4=field->data[element->n[3]]-isoval;
    NodeHandle n1=mesh->nodes[element->n[0]];
    NodeHandle n2=mesh->nodes[element->n[1]];
    NodeHandle n3=mesh->nodes[element->n[2]];
    NodeHandle n4=mesh->nodes[element->n[3]];
    int f1=v1<0;
    int f2=v2<0;
    int f3=v3<0;
    int f4=v4<0;
    int mask=(f1<<3)|(f2<<2)|(f3<<1)|f4;
    int pfaces=0;
    int rfaces=0;
    int use2 = !((f1+f2+f3+f4)&1);
    int rfc2,rfc3,rfc4;

    // these points/codes are just for 2 triangle mode!

    int eA,eB,eC,eD; // edges of tetrahedra...
    Point pA,pB,pC,pD; // actual points

//    cerr << "Doing initial face!\n";
    switch(mask){
    case 0:
    case 15:
	// Nothing to do...
	return 0;
    case 1:
    case 14:
	// Point 4 is inside
 	{
	    Point p1(Interpolate(n4->p, n1->p, v4/(v4-v1)));
	    Point p2(Interpolate(n4->p, n2->p, v4/(v4-v2)));
	    Point p3(Interpolate(n4->p, n3->p, v4/(v4-v3)));
	    Vector n(Cross(p2-p1,p3-p1));
	    if (n.length2() > 0){
		n.normalize();
	    }
	    group->add(p1);
	    group->add(p2);
	    group->add(p3,n);  // always add normals from this point on...
	    
	    rfaces=E_FACE(E42,E43);
	    pfaces= F_FACE(E41,E42)|F_FACE(E41,E43);
	    rfc2 = E_FACE(E41,E42);
	    rfc3 = E_FACE(E43,E41);
//		FACE3|FACE2;
//	    cerr <<  (F_FACE(E41,E42)|F_FACE(E41,E43))<<  " -> ";
//	    cerr << (FACE3|FACE2) << "\n";
	}
	break;
    case 2:
    case 13:
	// Point 3 is inside
 	{
	    Point p1(Interpolate(n3->p, n1->p, v3/(v3-v1)));
	    Point p2(Interpolate(n3->p, n2->p, v3/(v3-v2)));
	    Point p3(Interpolate(n3->p, n4->p, v3/(v3-v4)));
	    Vector n(Cross(p2-p1,p3-p1));
	    if (n.length2() > 0){
		n.normalize();
	    }
	    group->add(p1);
	    group->add(p2);
	    group->add(p3,n);

	    rfaces=E_FACE(E32,E34);
	    pfaces= F_FACE(E31,E32)|F_FACE(E31,E34);
	    rfc2 = E_FACE(E31,E32);
	    rfc3 = E_FACE(E34,E31);
//		FACE4|FACE2;

//	    cerr <<  (F_FACE(E31,E32)|F_FACE(E31,E43))<<  " 3-> ";
//	    cerr << (FACE4|FACE2) << "\n";
	}
	break;
    case 3:
    case 12:
	// Point 3 and 4 are inside
 	{
	    pA=Interpolate(n3->p, n1->p, v3/(v3-v1));
	    pB=Interpolate(n3->p, n2->p, v3/(v3-v2));
	    pC=Interpolate(n4->p, n1->p, v4/(v4-v1));
	    pD=Interpolate(n4->p, n2->p, v4/(v4-v2));

	    eA = E31;
	    eB = E32;
	    eC = E41;
	    eD = E42;

	}
	break;
    case 4:
    case 11:
	// Point 2 is inside
 	{
	    Point p1(Interpolate(n2->p, n1->p, v2/(v2-v1)));
	    Point p2(Interpolate(n2->p, n3->p, v2/(v2-v3)));
	    Point p3(Interpolate(n2->p, n4->p, v2/(v2-v4)));
	    Vector n(Cross(p2-p1,p3-p1));
	    if (n.length2() > 0){
		n.normalize();
	    }
	    group->add(p1);
	    group->add(p2);
	    group->add(p3,n);
	    rfaces=E_FACE(E23,E24);
	    pfaces=F_FACE(E21,E23)|F_FACE(E21,E24);
	    rfc2 = E_FACE(E21,E23);
	    rfc3 = E_FACE(E24,E21);

//		FACE4|FACE3;
	}
	break;
    case 5:
    case 10:
	// Point 2 and 4 are inside
 	{
	    pA=Interpolate(n2->p, n1->p, v2/(v2-v1));
	    pB=Interpolate(n2->p, n3->p, v2/(v2-v3));
	    pC=Interpolate(n4->p, n1->p, v4/(v4-v1));
	    pD=Interpolate(n4->p, n3->p, v4/(v4-v3));

	    eA = E21;
	    eB = E23;
	    eC = E41;
	    eD = E43;

	}
	break;
    case 6:
    case 9:
	// Point 2 and 3 are inside
 	{
	    pA=Interpolate(n2->p, n1->p, v2/(v2-v1));
	    pB=Interpolate(n2->p, n4->p, v2/(v2-v4));
	    pC=Interpolate(n3->p, n1->p, v3/(v3-v1));
	    pD=Interpolate(n3->p, n4->p, v3/(v3-v4));

	    eA = E21;
	    eB = E24;
	    eC = E31;
	    eD = E34;
	}
	break;
    case 7:
    case 8:
	// Point 1 is inside
 	{
	    Point p1(Interpolate(n1->p, n2->p, v1/(v1-v2)));
	    Point p2(Interpolate(n1->p, n3->p, v1/(v1-v3)));
	    Point p3(Interpolate(n1->p, n4->p, v1/(v1-v4)));
	    Vector n(Cross(p2-p1,p3-p1));
	    if (n.length2() > 0){
		n.normalize();
	    }
	    group->add(p1);
	    group->add(p2);
	    group->add(p3,n);
	    rfaces = E_FACE(E13,E14);
	    pfaces = F_FACE(E12,E13)|F_FACE(E12,E14);
	    rfc2 = E_FACE(E12,E13);
	    rfc3 = E_FACE(E14,E12);
//		FACE4|FACE3;
	}
	break;
    } 

    if (!use2) {
	if ((nbr_status&EDGE_TO_FACE[rfaces]))
	    return rfaces | (pfaces<<6);
	else if ((nbr_status&EDGE_TO_FACE[rfc2])) {
//	    GeomVertex *tmp = group->verts[2];
//	    group->verts[2] = group->verts[1];
//	    group->verts[1] = group->verts[0];
//	    group->verts[0] = tmp;
	    group->permute(2,0,1);
	    rfaces = rfc2;
	    pfaces = EDGE_TO_FACE[rfaces]|EDGE_TO_FACE[rfc3];
	}
	else if ((nbr_status&EDGE_TO_FACE[rfc3])) {
//	    GeomVertex *tmp = group->verts[2];
//	    group->verts[2] = group->verts[0];
//	    group->verts[0] = group->verts[1];
//	    group->verts[1] = tmp;
	    group->permute(1,2,0);
	    rfaces = rfc3;
	    pfaces = EDGE_TO_FACE[rfaces]|EDGE_TO_FACE[rfc2];
	}
    }
    else {
	if (nbr_status&EDGE_TO_FACE[E_FACE(eC,eD)]) {
	    emit_in_2(rfaces,pfaces,eA,eB,eC,eD,
		      pA,pB,pC,pD,
		      group);
	}
	else if (nbr_status&EDGE_TO_FACE[E_FACE(eB,eD)]) {
	    emit_in_2(rfaces,pfaces,eC,eA,eD,eB,
		      pC,pA,pD,pB,
		      group);
	}
	else if (nbr_status&EDGE_TO_FACE[E_FACE(eA,eC)]) {
	    emit_in_2(rfaces,pfaces,eB,eD,eA,eC,
		      pB,pD,pA,pC,
		      group);
	}
	else { // last option A B -> DCBA
	    emit_in_2(rfaces,pfaces,eD,eC,eB,eA,
		      pD,pC,pB,pA,
		      group);
	}
    }

    return rfaces|(pfaces<<6);
}

void IsoSurface::iso_tetra_strip(int ix, Mesh* mesh, 
				 ScalarFieldUG* field, double iv, 
				 GeomGroup* group, BitArray1& visited)
{
    GeomTriStripList* nstrip = scinew GeomTriStripList;
    Element* element=mesh->elems[ix];
    int strip_done=0;

    visited.set(ix); // mark it as really done...

    int f0=element->face(0);
    int f1=element->face(1);
    int f2=element->face(2);
    int f3=element->face(3);
    int nbrmask = (((f0!=-1) && !visited.is_set(f0))?FACE1:0) |
	(((f1!=-1) && !visited.is_set(f1))?FACE2:0) |
	    (((f2!=-1) && !visited.is_set(f2))?FACE3:0) |
		(((f3!=-1) && !visited.is_set(f3))?FACE4:0);

    int rval = iso_tetra_s(nbrmask,element, mesh, field, iv, nstrip);

    if (!rval) {
	if (!nstrip->size())
	    delete nstrip;
	return;
    }
    int nface = FREMAP(EDGE_TO_FACE[rval&(63)]);
    
    int nf = element->face(nface);
    // 1st try and push if we can

    while(!strip_done) {
	if (visited.is_set(nf))
	    nf = -1;
	if (nf > -1) {
	    visited.set(nf);
	    Element *nelement = mesh->elems[nf];
	    // you have to remap the edges into the
	    // face you are recursing into to match the
	    // current element...

	    remap_element(rval,element,nelement);
	    rval = iso_strip_enter(rval&63,nelement,mesh,field,iv,nstrip);
	    nf = nelement->face(FREMAP(EDGE_TO_FACE[rval&(63)]));
	    element = nelement; // so later code works...
	}
	else
	    strip_done = 1;
    }
    
    group->add(nstrip);
}

// this uses a depth-first search, and builds tri-strips...
// this could easily be multi-threaded, and I am sure
// that it is extremely inefficient (as far as stack space goes)

void IsoSurface::iso_tetrahedra_strip(ScalarFieldUG* field, const Point& p,
				      GeomGroup* group)
{
    Mesh* mesh=field->mesh.get_rep();
    int nelems=mesh->elems.size();
    double iv;
    if(!field->interpolate(p, iv)){
	error("Seed point not in field boundary");
	return;
    }

    int total_prims = 0;
    int max_prim=0;
    int strip_prims=0;
    int grp_prims=0;
    GeomTrianglesP *bgroup = 0;
    isoval.set(iv);
    // 1st bit array is for ones that are done
    // second is for ones we need to check...
    BitArray1 visited(nelems, 0);
    BitArray1 tocheck(nelems,0);
    Queue<int> surfQ;

    int ix;

    mesh->locate(p,ix);

    if (ix == -1)
	return;

    GeomTriStripList* ts = scinew GeomTriStripList;

    tocheck.set(ix);
    surfQ.append(ix);
    int groupid=0;
    int counter=1;
    while(!surfQ.is_empty()) {
	if (sp && abort_flag)
	    break;
#if 0
	if (sp&&counter%400 == 0) {
	    if(!ogeom->busy()){
		if(groupid)
		    ogeom->delObj(groupid);
		groupid=ogeom->addObj(group->clone(), surface_name);
		ogeom->flushViews();
	    }	
	}
#endif

	ix = surfQ.pop();
	if (!visited.is_set(ix)) {
	    Element* element=mesh->elems[ix];
//	    GeomTriStripList* ts = scinew GeomTriStripList;
	    int f0=element->face(0);
	    int f1=element->face(1);
	    int f2=element->face(2);
	    int f3=element->face(3);
	    int nbrmask = (((f0!=-1) && !visited.is_set(f0))?FACE1:0) |
		(((f1!=-1) && !visited.is_set(f1))?FACE2:0) |
		(((f2!=-1) && !visited.is_set(f2))?FACE3:0) |
		(((f3!=-1) && !visited.is_set(f3))?FACE4:0);

	    int rval = iso_tetra_s(nbrmask,element, mesh, field, iv, ts);
	    visited.set(ix); // mark it as really done...

	    int nface = FREMAP(EDGE_TO_FACE[rval&(63)]);

	    int nf = element->face(nface);
	    // 1st try and push if we can

	    do {
		int nbrs = rval>>6;

		if(nbrs & FACE1){
		    if(f0 != -1 && !tocheck.is_set(f0)){
			tocheck.set(f0);
			surfQ.append(f0);
		    }
		}
		if(nbrs & FACE2){
		    if(f1 != -1 && !tocheck.is_set(f1)){
			tocheck.set(f1);
			surfQ.append(f1);
		    }
		}
		if(nbrs & FACE3){
		    if(f2 != -1 && !tocheck.is_set(f2)){
			tocheck.set(f2);
			surfQ.append(f2);
		    }
		}
		if(nbrs & FACE4){
		    if(f3 != -1 && !tocheck.is_set(f3)){
			tocheck.set(f3);
			surfQ.append(f3);
		    }
		}
		    
		if (nf > -1) {  
		    // this means we can still build this tri-strip...
		    if (visited.is_set(nf))
			nf = -1;
		    else {

			visited.set(nf);
			Element *nelement = mesh->elems[nf];
			// you have to remap the edges into the
			// face you are recursing into to match the
			// current element...

			remap_element(rval,element,nelement);
			rval = iso_strip_enter(rval&63,nelement,mesh,field,iv,ts);
			nf = nelement->face(FREMAP(EDGE_TO_FACE[rval&(63)]));
			element = nelement; // so later code works...

			f0=element->face(0);
			f1=element->face(1);
			f2=element->face(2);
			f3=element->face(3);
			if (nf == -1)
			    nf = -2;  // make sure we don't miss anyone...
		    }
		}
		else
		    nf = -1;
	    } while (nf != -1);
#if 0
	    if (ts->size() == 3) { // to small for a complete tri-strip
		if (!bgroup) 
		    bgroup = scinew GeomTrianglesP;
		bgroup->add(ts->verts[0]->p,ts->verts[1]->p,ts->verts[2]->p);
		grp_prims++;
		delete ts;
	    }
	    else
#endif
		{
		strip_prims += ts->size()-2;
//		group->add(ts);
		if (ts->size()-2 > max_prim)
		    max_prim = ts->size()-2;
		ts->end_strip();
	    }
	}
    }

    group->add(ts);

    if (bgroup) {
	group->add(bgroup);
    }
}

// this is the version that just generates standard triangles

void IsoSurface::iso_tetrahedra(ScalarFieldUG* field, const Point& p,
				GeomTrianglesP* group)
{
    Mesh* mesh=field->mesh.get_rep();
    int nelems=mesh->elems.size();
    double iv;
    if(!field->interpolate(p, iv)){
	error("Seed point not in field boundary");
	return;
    }
    cerr << "In iso_tetrahedra!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    isoval.set(iv);
    BitArray1 visited(nelems, 0);
    Queue<int> surfQ;
    int ix;
    mesh->locate(p, ix);
    visited.set(ix);
    surfQ.append(ix);
    int groupid=0;
    int counter=1;
    while(!surfQ.is_empty()){
	if(sp && abort_flag)
	    break;
#if 0
	if(sp && counter%400 == 0){
	    if(!ogeom->busy()){
		if(groupid)
		    ogeom->delObj(groupid);
		groupid=ogeom->addObj(group->clone(), surface_name);
		ogeom->flushViews();
	    }
	}
#endif
	ix=surfQ.pop();
	Element* element=mesh->elems[ix];
	int nbrs=iso_tetra(element, mesh, field, iv, group);
	if(nbrs & FACE1){
	    int f0=element->face(0);
	    if(f0 != -1 && !visited.is_set(f0)){
		visited.set(f0);
		surfQ.append(f0);
	    }
	}
	if(nbrs & FACE2){
	    int f1=element->face(1);
	    if(f1 != -1 && !visited.is_set(f1)){
		visited.set(f1);
		surfQ.append(f1);
	    }
	}
	if(nbrs & FACE3){
	    int f2=element->face(2);
	    if(f2 != -1 && !visited.is_set(f2)){
		visited.set(f2);
		surfQ.append(f2);
	    }
	}
	if(nbrs & FACE4){
	    int f3=element->face(3);
	    if(f3 != -1 && !visited.is_set(f3)){
		visited.set(f3);
		surfQ.append(f3);
	    }
	}
	counter++;
    }
    if(groupid)
	ogeom->delObj(groupid);
}

void IsoSurface::find_seed_from_value(const ScalarFieldHandle& /*field*/)
{
    NOT_FINISHED("IsoSurface::find_seed_from_value");
#if 0
    int nx=field->get_nx();
    int ny=field->get_ny();
    int nz=field->get_nz();
    GeomGroup group;
    for (int i=0; i<nx-1;i++) {
	for (int j=0; j<ny-1; j++) {
	    for (int k=0; k<nz-1; k++) {
		if(iso_cube(i,j,k,isoval, &group, field)) {
		    seed_point=Point(i,j,k);
		    cerr << "New seed=" << seed_point.string() << endl;
		    return;
		}
	    }
	}
    }
#endif
}

void IsoSurface::widget_moved(int last)
{
    if(last && !abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

#ifdef __GNUG__

#include <Classlib/Queue.cc>

template class Queue<int>;

#endif
