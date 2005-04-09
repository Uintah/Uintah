
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
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
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
#include <Geom/Tri.h>
#include <Geometry/Point.h>
#include <Geometry/Plane.h>
#include <Malloc/Allocator.h>
#include <Math/Expon.h>
#include <Math/MiscMath.h>
#include <TCL/TCLvar.h>
#include <Widgets/ArrowWidget.h>
#include <iostream.h>
#include <strstream.h>

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

    int iso_cube(int, int, int, double, GeomGroup*, ScalarFieldRG*);
    int iso_tetra(Element*, Mesh*, ScalarFieldUG*, double, GeomGroup*);

    void iso_reg_grid(ScalarFieldRG*, const Point&, GeomGroup*);
    void iso_reg_grid(ScalarFieldRG*, double, GeomGroup*);
    void iso_tetrahedra(ScalarFieldUG*, const Point&, GeomGroup*);
    void iso_tetrahedra(ScalarFieldUG*, double, GeomGroup*);

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

#define FACE1 8
#define FACE2 4
#define FACE3 2
#define FACE4 1
#define ALLFACES (FACE1|FACE2|FACE3|FACE4)

struct MCubeTable {
    int which_case;
    int permute[8];
    int nbrs;
};

#include "mcube.h"

static Module* make_IsoSurface(const clString& id)
{
    return scinew IsoSurface(id);
}

static RegisterModule db1("Fields", "IsoSurface", make_IsoSurface);
static RegisterModule db2("Visualization", "IsoSurface", make_IsoSurface);

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
    if(do_3dwidget.get()){
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

    GeomGroup* group=scinew GeomGroup;
    GeomObj* topobj=group;
    if(have_colormap && !have_colorfield){
	// Paint entire surface based on colormap
	topobj=scinew GeomMaterial(group, cmap->lookup(iv));
    } else if(have_colormap && have_colorfield){
	// Nothing - done per vertex
    } else {
	// Default material
	topobj=scinew GeomMaterial(group, matl);
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
	    iso_tetrahedra(unstructured_grid, sp, group);
	} else {
	    iso_tetrahedra(unstructured_grid, iv, group);
	}
    } else {
	error("I can't IsoSurface this type of field...");
    }
    cerr << "Finished isosurfacing!  Got " << group->size() << " objects\n";

    if(group->size() == 0){
	delete group;
	if (emit_surface.get())
	    delete surf;
	isosurface_id=0;
    } else {
	isosurface_id=ogeom->addObj(topobj, surface_name);
	if (emit_surface.get()) {
	    osurf->send(surf);
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
				      double val) {

    if (Plane(p1,p2,p3).eval_point(v1)*val >= 0)	// is the order right?
	surf->cautious_add_triangle(p1,p2,p3,1);
    else	
	surf->cautious_add_triangle(p2,p1,p3,1);
}


//The isosurface code needs to be able to create clockwise ordered triangles for it to be useful to me.  Clearly we have the information to judge which side of any triangle we generate is inside -- the side which has positive values.   
 
//My first thought was to generate a point off the centroid, just a tad away from the face in the direction of the triangle's normal.  Then, get the "value" of the point (in the field), if it's negative, switch the ordering.


int IsoSurface::iso_cube(int i, int j, int k, double isoval,
			 GeomGroup* group, ScalarFieldRG* field)
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
    for(int idx=1;idx<=8;idx++){
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
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[1],val[1]);
	}
	break;
    case 2:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p3, p4, p1));
	    if (emit_surface.get())
		order_and_add_points(p3,p4,p1,v[1],val[1]);
	}
	break;
    case 3:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    group->add(scinew GeomTri(p4, p5, p6));
	    if (emit_surface.get())
		order_and_add_points(p4,p5,p6,v[3],val[3]);
	}
	break;
    case 4:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p5(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p6(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    group->add(scinew GeomTri(p4, p5, p6));
	    if (emit_surface.get())
		order_and_add_points(p4,p5,p6,v[7],val[7]);
	}
	break;
    case 5:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[2],val[2]);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(scinew GeomTri(p4, p3, p2));
	    if (emit_surface.get())
		order_and_add_points(p4,p3,p2,v[2],val[2]);
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(scinew GeomTri(p5, p4, p2));
	    if (emit_surface.get())
		order_and_add_points(p5,p4,p2,v[2],val[2]);
	}
	break;
    case 6:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p3, p4, p1));
	    if (emit_surface.get())
		order_and_add_points(p3,p4,p1,v[1],val[1]);
	    Point p5(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p7(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    group->add(scinew GeomTri(p5, p6, p7));
	    if (emit_surface.get())
		order_and_add_points(p5,p6,p7,v[7],val[7]);
	}
	break;
    case 7:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[2],val[2]);
	    Point p4(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p5(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p6(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    group->add(scinew GeomTri(p4, p5, p6));
	    if (emit_surface.get())
		order_and_add_points(p4,p5,p6,v[4],val[4]);
	    Point p7(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p8(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    Point p9(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    group->add(scinew GeomTri(p7, p8, p9));
	    if (emit_surface.get())
		order_and_add_points(p7,p8,p9,v[7],val[7]);
	}
	break;
    case 8:
	{
	    Point p1(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(scinew GeomTri(p4, p1, p3));
	    if (emit_surface.get())
		order_and_add_points(p4,p1,p3,v[1],val[1]);
	}
	break;
    case 9:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[5],val[5]);
	    Point p4(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    group->add(scinew GeomTri(p1, p3, p4));
	    if (emit_surface.get())
		order_and_add_points(p1,p3,p4,v[5],val[5]);
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p1, p4, p5));
	    if (emit_surface.get())
		order_and_add_points(p1,p4,p5,v[5],val[5]);
	    Point p6(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    group->add(scinew GeomTri(p5, p4, p6));
	    if (emit_surface.get())
		order_and_add_points(p5,p4,p6,v[5],val[5]);
	}
	break;
    case 10:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p3(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    group->add(scinew GeomTri(p2, p4, p3));
	    if (emit_surface.get())
		order_and_add_points(p2,p4,p3,v[1],val[1]);
	    Point p5(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p6(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    Point p7(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    group->add(scinew GeomTri(p5, p6, p7));
	    if (emit_surface.get())
		order_and_add_points(p5,p6,p7,v[6],val[6]);
	    Point p8(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    group->add(scinew GeomTri(p2, p8, p3));
	    if (emit_surface.get())
		order_and_add_points(p2,p8,p3,v[6],val[6]);
	}
	break;
    case 11:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[5],val[5]);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(scinew GeomTri(p1, p3, p4));
	    if (emit_surface.get())
		order_and_add_points(p1,p3,p4,v[5],val[5]);
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p1, p4, p5));
	    if (emit_surface.get())
		order_and_add_points(p1,p4,p5,v[5],val[5]);
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    group->add(scinew GeomTri(p4, p3, p6));
	    if (emit_surface.get())
		order_and_add_points(p4,p3,p6,v[5],val[5]);
	}
	break;
    case 12:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[5],val[5]);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(scinew GeomTri(p3, p2, p4));
	    if (emit_surface.get())
		order_and_add_points(p3,p2,p4,v[5],val[5]);
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(scinew GeomTri(p4, p2, p5));
	    if (emit_surface.get())
		order_and_add_points(p4,p2,p5,v[5],val[5]);
	    Point p6(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p7(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p8(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    group->add(scinew GeomTri(p6, p7, p8));
	    if (emit_surface.get())
		order_and_add_points(p6,p7,p8,v[4],val[4]);
	}
	break;
    case 13:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[1],val[1]);
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    group->add(scinew GeomTri(p4, p5, p6));
	    if (emit_surface.get())
		order_and_add_points(p4,p5,p6,v[3],val[3]);
	    Point p7(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p8(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    Point p9(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    group->add(scinew GeomTri(p7, p8, p9));
	    if (emit_surface.get())
		order_and_add_points(p7,p8,p9,v[6],val[6]);
	    Point p10(Interpolate(v[8], v[5], val[8]/(val[8]-val[5])));
	    Point p11(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    Point p12(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    group->add(scinew GeomTri(p10, p11, p12));
	    if (emit_surface.get())
		order_and_add_points(p10,p11,p12,v[8],val[8]);
	}
	break;
    case 14:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    if (emit_surface.get())
		order_and_add_points(p1,p2,p3,v[5],val[5]);
	    Point p4(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    group->add(scinew GeomTri(p1, p3, p4));
	    if (emit_surface.get())
		order_and_add_points(p1,p3,p4,v[5],val[5]);
	    Point p5(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    group->add(scinew GeomTri(p1, p4, p5));
	    if (emit_surface.get())
		order_and_add_points(p1,p4,p5,v[5],val[5]);
	    Point p6(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    group->add(scinew GeomTri(p3, p6, p4));
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
			      GeomGroup* group)
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
			      GeomGroup* group)
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
	if (sp && counter%400 == 0) {
	    if(!ogeom->busy()){
		if (groupid)
		    ogeom->delObj(groupid);
		groupid=ogeom->addObj(group->clone(), surface_name);
		ogeom->flushViews();
	    }
	}
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
	ogeom->delObj(groupid);
}

int IsoSurface::iso_tetra(Element* element, Mesh* mesh,
			  ScalarFieldUG* field, double isoval,
			  GeomGroup* group)
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
    int faces=0;
    switch(mask){
    case 0:
    case 15:
	// Nothing to do....
	break;
    case 1:
    case 14:
	// Point 4 is inside
 	{
	    Point p1(Interpolate(n4->p, n1->p, v4/(v4-v1)));
	    Point p2(Interpolate(n4->p, n2->p, v4/(v4-v2)));
	    Point p3(Interpolate(n4->p, n3->p, v4/(v4-v3)));
	    group->add(scinew GeomTri(p1, p2, p3));
	    faces=FACE1|FACE2|FACE3;
	}
	break;
    case 2:
    case 13:
	// Point 3 is inside
 	{
	    Point p1(Interpolate(n3->p, n1->p, v3/(v3-v1)));
	    Point p2(Interpolate(n3->p, n2->p, v3/(v3-v2)));
	    Point p3(Interpolate(n3->p, n4->p, v3/(v3-v4)));
	    group->add(scinew GeomTri(p1, p2, p3));
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
	    group->add(scinew GeomTri(p1, p2, p3));
	    group->add(scinew GeomTri(p2, p3, p4));
	    faces=ALLFACES;
	}
	break;
    case 4:
    case 11:
	// Point 2 is inside
 	{
	    Point p1(Interpolate(n2->p, n1->p, v2/(v2-v1)));
	    Point p2(Interpolate(n2->p, n3->p, v2/(v2-v3)));
	    Point p3(Interpolate(n2->p, n4->p, v2/(v2-v4)));
	    group->add(scinew GeomTri(p1, p2, p3));
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
	    group->add(scinew GeomTri(p1, p2, p3));
	    group->add(scinew GeomTri(p2, p3, p4));
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
	    group->add(scinew GeomTri(p1, p2, p3));
	    group->add(scinew GeomTri(p2, p3, p4));
	    faces=ALLFACES;
	}
	break;
    case 7:
    case 8:
	// Point 1 is inside
 	{
	    Point p1(Interpolate(n1->p, n2->p, v1/(v1-v2)));
	    Point p2(Interpolate(n1->p, n3->p, v1/(v1-v3)));
	    Point p3(Interpolate(n1->p, n4->p, v1/(v1-v4)));
	    group->add(scinew GeomTri(p1, p2, p3));
	    faces=FACE2|FACE3|FACE4;
	}
	break;
    }
    return faces;
}

void IsoSurface::iso_tetrahedra(ScalarFieldUG* field, double isoval,
				GeomGroup* group)
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

void IsoSurface::iso_tetrahedra(ScalarFieldUG* field, const Point& p,
				GeomGroup* group)
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
	if(sp && counter%400 == 0){
	    if(!ogeom->busy()){
		if(groupid)
		    ogeom->delObj(groupid);
		groupid=ogeom->addObj(group->clone(), surface_name);
		ogeom->flushViews();
	    }
	}
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
