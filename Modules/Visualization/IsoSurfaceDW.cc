
/*
 *  IsoSurfaceDW.cc:  Generate isosurfaces via cache rings... and in parallel
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


// Esc-x replace-string uchar [double | float | int | ushort]

#include <Classlib/BitArray1.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Queue.h>
#include <Classlib/Ring.h>
#include <Classlib/Stack.h>
#include <Classlib/Timer.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRGBase.h>
#include <Datatypes/ScalarFieldRGshort.h>
#include <Datatypes/ScalarFieldRGuchar.h>
#include <Datatypes/ScalarFieldRGchar.h>
#include <Datatypes/ScalarFieldRGint.h>
#include <Datatypes/ScalarFieldRGfloat.h>
#include <Datatypes/ScalarFieldRGdouble.h>
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
#include <Multitask/ITC.h>
#include <Multitask/Task.h>
#include <TCL/TCLvar.h>
#include <Widgets/ArrowWidget.h>
#include <iostream.h>
#include <strstream.h>

// just so I can see the proccess id...

#include <sys/types.h>
#include <unistd.h>

using sci::Element;
using sci::Mesh;
using sci::Node;
using sci::NodeHandle;

#define INTERP(i1, i2) (Interpolate(v[i1],v[i2],val[i1]*1./(val[i1]-val[i2])))

class IsoSurfaceDW : public Module {
    ScalarFieldIPort* infield;
    ScalarFieldIPort* incolorfield;
    ColorMapIPort* inColorMap;

    GeometryOPort* ogeom;
    SurfaceOPort* osurf;

    TCLPoint seed_point;
    TCLint have_seedpoint;
    TCLdouble isoval;
    TCLint do_3dwidget;
    TCLint emit_surface;
    TCLint single;
    TCLdouble clr_r;
    TCLdouble clr_g;
    TCLdouble clr_b;
    int emit;
    int sing;

    TCLstring method;
    TriSurface* surf;

    int IsoSurfaceDW_id;
    int need_seed;

    TriSurface *composite_surf;
    Array1<int> start_pts;
    Array1<int> start_elems;
    Array1<Array1<TSElement*> >all_elems;
    Array1<Array1<Point> > all_pts;
    Array1<GeomTrianglesP*> all_tris;
    Array1<Semaphore*> all_sems;
    Array1<Ring<int>* > all_bdryRings;
    Array1<HashTable<int,int>* > all_bdryHashes;
    double old_min;
    double old_max;
    Point old_bmin;
    Point old_bmax;
    int sp;
    TCLint show_progress;
    Mutex io;

    CPUTimer outerExtract;
    CPUTimer innerExtract;
    CPUTimer lace;
    CPUTimer barrierLace;
    HashTable<int,int> hash;
    Mutex hashing;

    void printIt(clString s, int i, int j, int k, int c, int idx);

    int iso_cubeRing(int, int, int, double, GeomTrianglesP*,
		     Array1<TSElement*>&, Array1<Point>&,
		     int, int, int, int, Ring<int>*, Ring<int>*, 
		     Ring<int>*, Ring<int>*, int &, int &, int &, int &);
    int iso_cubeRingFast(int, int, int, double, GeomTrianglesP*,
			 Array1<TSElement*>&, Array1<Point>&, 
			 Ring<int>*, Ring<int>*, int &, int &);

    int iso_cubeHash(int,int,int,double, GeomTrianglesP*,
		     Array1<TSElement*>&, Array1<Point>&,
		     HashTable<int,int>*, HashTable<int,int>*, int, int);

    int iso_cubeTS(int, int, int, double, GeomTrianglesP*, TriSurface *);
    int iso_cube(int, int, int, double, GeomTrianglesP*);
    int iso_tetra(Element*, Mesh*, ScalarFieldUG*, double, GeomTrianglesP*);
    int iso_tetra_s(int,Element*, Mesh*, ScalarFieldUG*, double, 
		    GeomTriStripList*);
    void iso_tetra_strip(int, Mesh*, ScalarFieldUG*, double, 
			 GeomGroup*, BitArray1&);

    void iso_reg_grid(const Point&, GeomTrianglesP*);
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

    ScalarFieldRGBase* rgbase;
    ScalarFieldRGuchar* rgfield_uc;
    ScalarFieldRGchar* rgfield_c;
    ScalarFieldRGshort* rgfield_s;
    ScalarFieldRGint* rgfield_i;
    ScalarFieldRGfloat* rgfield_f;
    ScalarFieldRGdouble* rgfield_d;
    double the_isoval;
    GeomGroup* maingroup;
    Mutex grouplock;
    Barrier barrier;
    int blockSize;
    TCLint tclBlockSize;
    int np;
    void iso_reg_grid();
    void iso_reg_grid_hash();
    void iso_reg_grid_rings();
public:
    void parallel_reg_grid(int proc);
    void parallel_reg_grid_hash(int proc);
    void parallel_reg_grid_rings(int proc);
    IsoSurfaceDW(const clString& id);
    IsoSurfaceDW(const IsoSurfaceDW&, int deep);
    virtual ~IsoSurfaceDW();
    virtual Module* clone(int deep);
    virtual void execute();
};

#define FACE4 8
#define FACE3 4
#define FACE2 2
#define FACE1 1
#define ALLFACES (FACE1|FACE2|FACE3|FACE4)

// below determines wether to normalzie normals or not

#ifdef SCI_NORM_OGL
#define NORMALIZE_NORMALS 0
#else
#define NORMALIZE_NORMALS 1
#endif
 
struct MCubeTable {
    int which_case;
    int permute[8];
    int nbrs;
};

#include "mcube2.h"

extern "C" {
Module* make_IsoSurfaceDW(const clString& id)
{
    return scinew IsoSurfaceDW(id);
}
};

static clString module_name("IsoSurfaceDW");
static clString surface_name("IsoSurfaceDW");
static clString widget_name("IsoSurfaceDW widget");

static void do_parallel_reg_grid_hash(void* obj, int proc)
{
  IsoSurfaceDW* module=(IsoSurfaceDW*)obj;
  module->parallel_reg_grid_hash(proc);
}

static void do_parallel_reg_grid_rings(void* obj, int proc)
{
  IsoSurfaceDW* module=(IsoSurfaceDW*)obj;
  module->parallel_reg_grid_rings(proc);
}

static void do_parallel_reg_grid(void* obj, int proc)
{
  IsoSurfaceDW* module=(IsoSurfaceDW*)obj;
  module->parallel_reg_grid(proc);
}

IsoSurfaceDW::IsoSurfaceDW(const clString& id)
: Module("IsoSurfaceDW", id, Filter), seed_point("seed_point", id, this),
  have_seedpoint("have_seedpoint", id, this), isoval("isoval", id, this),
  do_3dwidget("do_3dwidget", id, this), emit_surface("emit_surface", id, this),
  single("single", id, this), show_progress("show_progress", id, this), 
  tclBlockSize("tclBlockSize", id, this),
  method("method", id, this), clr_r("clr-r", id, this), 
  clr_g("clr-g", id, this), clr_b("clr-b", id, this)
{
    // Create the input ports
    infield=scinew ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
    add_iport(infield);
    incolorfield=scinew ScalarFieldIPort(this, "Color Field", ScalarFieldIPort::Atomic);
    add_iport(incolorfield);
    inColorMap=scinew ColorMapIPort(this, "Color Map", ColorMapIPort::Atomic);
    add_iport(inColorMap);
    

    // Create the output port
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    osurf=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurf);

    isoval.set(1);
    need_seed=1;

    IsoSurfaceDW_id=0;

    old_min=old_max=0;
    old_bmin=old_bmax=Point(0,0,0);

    float INIT(.1);
    widget = scinew ArrowWidget(this, &widget_lock, INIT);
    need_find=1;
    init=1;
}

IsoSurfaceDW::IsoSurfaceDW(const IsoSurfaceDW& copy, int deep)
: Module(copy, deep), seed_point("seed_point", id, this),
  have_seedpoint("have_seedpoint", id, this), isoval("isoval", id, this),
  do_3dwidget("do_3dwidget", id, this), emit_surface("emit_surface", id, this),
  single("single", id, this), show_progress("show_progress", id, this),
  tclBlockSize("tclBlockSize", id, this),
  method("method", id, this), clr_r("clr-r", id, this), 
  clr_g("clr-g", id, this), clr_b("clr-b", id, this)
{
    NOT_FINISHED("IsoSurfaceDW::IsoSurfaceDW");
}

IsoSurfaceDW::~IsoSurfaceDW()
{
}

Module* IsoSurfaceDW::clone(int deep)
{
    return scinew IsoSurfaceDW(*this, deep);
}

void IsoSurfaceDW::execute()
{
    update_state(NeedData);
    if(IsoSurfaceDW_id){
	ogeom->delObj(IsoSurfaceDW_id);
    }
    ScalarFieldHandle field;
    if(!infield->get(field))
	return;
    ScalarFieldHandle colorfield;
    int have_colorfield=incolorfield->get(colorfield);
    ColorMapHandle cmap;
    int have_ColorMap=inColorMap->get(cmap);
    update_state(JustStarted);

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
    if (method.get() == clString("None")) return;
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
//	cerr << "at p=" << sp << ", iv=" << iv << endl;
    }
    GeomTrianglesP* group = scinew GeomTrianglesP;
    GeomGroup* tgroup=scinew GeomGroup;
    GeomObj* topobj=tgroup;
    if(have_ColorMap && !have_colorfield){
	// Paint entire surface based on ColorMap
	topobj=scinew GeomMaterial(tgroup, cmap->lookup(iv));
    } else if(have_ColorMap && have_colorfield){
	// Nothing - done per vertex
    } else {
	// Default material
	topobj=tgroup;
    }
    rgbase=field->getRGBase();
    ScalarFieldUG* unstructured_grid=field->getUG();
//cerr << "regular_grid="<<regular_grid<<"\n";
//cerr << "unstructured_grid="<<unstructured_grid<<"\n";
    Point minPt, maxPt;
    double spacing=0;
    Vector diff;

    emit=emit_surface.get();
    sing=single.get();

    if (emit) {
        field->get_bounds(minPt, maxPt);
        diff=maxPt-minPt;
        spacing=Max(diff.x(), diff.y(), diff.z());
    }   

    MaterialHandle mpick(new Material(Color(.2,.2,.2), 
	            Color(clr_r.get(), clr_g.get(), clr_b.get()),
		    Color(.2,.2,.2), 20));
    if(rgbase){
	rgfield_uc = rgbase->getRGUchar();
	rgfield_c = rgbase->getRGChar();
	rgfield_s = rgbase->getRGShort();
	rgfield_i = rgbase->getRGInt();
	rgfield_f = rgbase->getRGFloat();
	rgfield_d = rgbase->getRGDouble();
	surf=scinew TriSurface;
	if(have_seedpoint.get()){
	    iso_reg_grid(sp, group);
	    tgroup->add(group);
	} else {
	    the_isoval=iv;
	    maingroup=tgroup;
	    if (method.get() == clString("MC")) {
		iso_reg_grid();
	    } else if (method.get() == clString("Rings")) {
		iso_reg_grid_rings();
	    } else if (method.get() == clString("Hash")) {
		iso_reg_grid_hash();
	    } else {
		cerr << "Error:  unknown method: "<<method.get()<<"\n";
	    }
	    surf=composite_surf;
	}
    } else if(unstructured_grid){
	if (emit) {
	    surf=scinew TriSurface;
	    int pts_per_side=(int) Cbrt(unstructured_grid->mesh->nodes.size());
	    spacing/=pts_per_side;
	    surf->construct_grid(pts_per_side+2,pts_per_side+2,pts_per_side+2, 
				 minPt+(Vector(1.001,1.029,0.917)*(-.001329)),
				 spacing);
	}	
	if(have_seedpoint.get()){
	    Point sp(seed_point.get());
	    iso_tetrahedra_strip(unstructured_grid,sp,tgroup);
	} else {
	    iso_tetrahedra(unstructured_grid, iv, group);
	    tgroup->add(group);
	}
    } else {
	error("I can't IsoSurfaceDW this type of field...");
    }

    if(tgroup->size() == 0){
	delete tgroup;
	if (emit) {
	    delete composite_surf;
	}
	IsoSurfaceDW_id=0;
    } else {
	IsoSurfaceDW_id=ogeom->addObj(scinew GeomMaterial(topobj, mpick), surface_name);
	if (emit) {
	    osurf->send(SurfaceHandle(composite_surf));
	}
    }
}

void IsoSurfaceDW::printIt(clString s, int i, int j, int k, int c, int idx) {
    grouplock.lock();
    cerr << s<<": ("<<i<<", "<<j<<", "<<k<<")  case: "<<c<<"  idx: "<<idx<<"\n";
    grouplock.unlock();
}

int IsoSurfaceDW::iso_cubeTS(int i, int j, int k, double isoval,
			     GeomTrianglesP* group, 
			     TriSurface *surf) {
    double val[8];
    if (rgfield_uc) {
	val[0]=rgfield_uc->grid(i, j, k)-isoval;
	val[1]=rgfield_uc->grid(i+1, j, k)-isoval;
	val[2]=rgfield_uc->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_uc->grid(i, j+1, k)-isoval;
	val[4]=rgfield_uc->grid(i, j, k+1)-isoval;
	val[5]=rgfield_uc->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_uc->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_uc->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_c) {
	val[0]=rgfield_c->grid(i, j, k)-isoval;
	val[1]=rgfield_c->grid(i+1, j, k)-isoval;
	val[2]=rgfield_c->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_c->grid(i, j+1, k)-isoval;
	val[4]=rgfield_c->grid(i, j, k+1)-isoval;
	val[5]=rgfield_c->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_c->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_c->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_s) {
	val[0]=rgfield_s->grid(i, j, k)-isoval;
	val[1]=rgfield_s->grid(i+1, j, k)-isoval;
	val[2]=rgfield_s->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_s->grid(i, j+1, k)-isoval;
	val[4]=rgfield_s->grid(i, j, k+1)-isoval;
	val[5]=rgfield_s->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_s->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_s->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_i) {
	val[0]=rgfield_i->grid(i, j, k)-isoval;
	val[1]=rgfield_i->grid(i+1, j, k)-isoval;
	val[2]=rgfield_i->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_i->grid(i, j+1, k)-isoval;
	val[4]=rgfield_i->grid(i, j, k+1)-isoval;
	val[5]=rgfield_i->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_i->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_i->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_f) {
	val[0]=rgfield_f->grid(i, j, k)-isoval;
	val[1]=rgfield_f->grid(i+1, j, k)-isoval;
	val[2]=rgfield_f->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_f->grid(i, j+1, k)-isoval;
	val[4]=rgfield_f->grid(i, j, k+1)-isoval;
	val[5]=rgfield_f->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_f->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_f->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_d) {
	val[0]=rgfield_d->grid(i, j, k)-isoval;
	val[1]=rgfield_d->grid(i+1, j, k)-isoval;
	val[2]=rgfield_d->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_d->grid(i, j+1, k)-isoval;
	val[4]=rgfield_d->grid(i, j, k+1)-isoval;
	val[5]=rgfield_d->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_d->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_d->grid(i, j+1, k+1)-isoval;
    }
    int mask=0;
    int idx;
    for(idx=0;idx<8;idx++){
	if(val[idx]<0)
	    mask|=1<<idx;
    }
    if (mask==0 || mask==255) return 0;
    Point v[8];
    v[0]=rgbase->get_point(i, j, k);
    v[1]=rgbase->get_point(i+1, j, k);
    v[2]=rgbase->get_point(i+1, j+1, k);
    v[3]=rgbase->get_point(i, j+1, k);
    v[4]=rgbase->get_point(i, j, k+1);
    v[5]=rgbase->get_point(i+1, j, k+1);
    v[6]=rgbase->get_point(i+1, j+1, k+1);
    v[7]=rgbase->get_point(i, j+1, k+1);
    TRIANGLE_CASES *tcase=triCases+mask;
    EDGE_LIST *edges=tcase->edges;
    Point p[3];
    
    for (; edges[0]>-1; edges+=3) {
	int idx[3];
	for (i=0; i<3; i++) {
	    int v1 = edge_table[edges[i]][0];
	    int v2 = edge_table[edges[i]][1];
	    p[i]=Interpolate(v[v1], v[v2], val[v1]/(val[v1]-
						    val[v2]));
	    idx[i]=surf->points.size()+i;
	}
	if (group->add(p[0], p[1], p[2])) {
	    surf->points.add(p[0]);
	    surf->points.add(p[1]);
	    surf->points.add(p[2]);
	    surf->add_triangle(idx[0], idx[1], idx[2]);
	}
    }
    return(tcase->nbrs);
}

int IsoSurfaceDW::iso_cube(int i, int j, int k, double isoval,
			   GeomTrianglesP* group) {

    double val[8];
    if (rgfield_uc) {
	val[0]=rgfield_uc->grid(i, j, k)-isoval;
	val[1]=rgfield_uc->grid(i+1, j, k)-isoval;
	val[2]=rgfield_uc->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_uc->grid(i, j+1, k)-isoval;
	val[4]=rgfield_uc->grid(i, j, k+1)-isoval;
	val[5]=rgfield_uc->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_uc->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_uc->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_c) {
	val[0]=rgfield_c->grid(i, j, k)-isoval;
	val[1]=rgfield_c->grid(i+1, j, k)-isoval;
	val[2]=rgfield_c->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_c->grid(i, j+1, k)-isoval;
	val[4]=rgfield_c->grid(i, j, k+1)-isoval;
	val[5]=rgfield_c->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_c->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_c->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_s) {
	val[0]=rgfield_s->grid(i, j, k)-isoval;
	val[1]=rgfield_s->grid(i+1, j, k)-isoval;
	val[2]=rgfield_s->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_s->grid(i, j+1, k)-isoval;
	val[4]=rgfield_s->grid(i, j, k+1)-isoval;
	val[5]=rgfield_s->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_s->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_s->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_i) {
	val[0]=rgfield_i->grid(i, j, k)-isoval;
	val[1]=rgfield_i->grid(i+1, j, k)-isoval;
	val[2]=rgfield_i->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_i->grid(i, j+1, k)-isoval;
	val[4]=rgfield_i->grid(i, j, k+1)-isoval;
	val[5]=rgfield_i->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_i->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_i->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_f) {
	val[0]=rgfield_f->grid(i, j, k)-isoval;
	val[1]=rgfield_f->grid(i+1, j, k)-isoval;
	val[2]=rgfield_f->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_f->grid(i, j+1, k)-isoval;
	val[4]=rgfield_f->grid(i, j, k+1)-isoval;
	val[5]=rgfield_f->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_f->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_f->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_d) {
	val[0]=rgfield_d->grid(i, j, k)-isoval;
	val[1]=rgfield_d->grid(i+1, j, k)-isoval;
	val[2]=rgfield_d->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_d->grid(i, j+1, k)-isoval;
	val[4]=rgfield_d->grid(i, j, k+1)-isoval;
	val[5]=rgfield_d->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_d->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_d->grid(i, j+1, k+1)-isoval;
    }
    int mask=0;
    int idx;
    for(idx=0;idx<8;idx++){
	if(val[idx]<0)
	    mask|=1<<idx;
    }
    if (mask==0 || mask==255) return 0;
    Point v[8];
    v[0]=rgbase->get_point(i, j, k);
    v[1]=rgbase->get_point(i+1, j, k);
    v[2]=rgbase->get_point(i+1, j+1, k);
    v[3]=rgbase->get_point(i, j+1, k);
    v[4]=rgbase->get_point(i, j, k+1);
    v[5]=rgbase->get_point(i+1, j, k+1);
    v[6]=rgbase->get_point(i+1, j+1, k+1);
    v[7]=rgbase->get_point(i, j+1, k+1);

    TRIANGLE_CASES *tcase=triCases+mask;
    EDGE_LIST *edges=tcase->edges;
    Point p[3];
//    cerr << "Min="<<v[0]<<"  Max="<<v[6]<<"\n";
    for (; edges[0]>-1; edges+=3) {
	for (i=0; i<3; i++) {
	    int v1 = edge_table[edges[i]][0];
	    int v2 = edge_table[edges[i]][1];
	    p[i]=Interpolate(v[v1], v[v2], val[v1]/(val[v1]-
						    val[v2]));
	}
//	cerr << "Adding... "<<p[0]<<" "<<p[1]<<" "<<p[2]<<"\n";
	
        group->add(p[0], p[1], p[2]);
    }
    return(tcase->nbrs);
}

void IsoSurfaceDW::parallel_reg_grid(int proc)
{
    GeomTrianglesP* tris=all_tris[proc];

    int nx=rgbase->nx;
    int ny=rgbase->ny;
    int nz=rgbase->nz;
    int sx;
    int ex;

    blockSize=tclBlockSize.get();

    if (nx <= np*blockSize) { 	// too many processors!
	if ((proc+1)*blockSize < nx) {  // full block
	    sx=proc*blockSize;
	    ex=(proc+1)*blockSize;
	} else if (proc*blockSize < (nx-1)) { // last block is a partial
	    sx=proc*blockSize;	
	    ex=nx-1;
	} else {			// empty block -- unused processor!
	    cerr << "ERROR -- SHOULDN'T BE HERE!!!!\n";
	    return;
	}
    } else {
	sx=proc*(nx-1)/np;
	ex=(proc+1)*(nx-1)/np;
    }

    CPUTimer myThreadTime;
    CPUTimer myOuterExtract;

    myThreadTime.start();
    myOuterExtract.start();
    for(int i=sx;i<ex;i++){
	for(int j=0;j<ny-1;j++){
	    for(int k=0;k<nz-1;k++){
		iso_cube(i, j, k, the_isoval, tris);
	    }
	    if(sp && abort_flag)
		return;
	}	
    }
    myOuterExtract.stop();
    myThreadTime.stop();     
    outerExtract.add(myOuterExtract.time());
    timer.add(myThreadTime.time());
}

int IsoSurfaceDW::iso_cubeHash(int i, int j, int k, double isoval, 
			       GeomTrianglesP* group, 
			       Array1<TSElement*>& elems, 
			       Array1<Point>& pts, 
			       HashTable<int,int>* hash,
			       HashTable<int,int>* Bdry, int first, int last) {
    double val[8];
    if (rgfield_uc) {
	val[0]=rgfield_uc->grid(i, j, k)-isoval;
	val[1]=rgfield_uc->grid(i+1, j, k)-isoval;
	val[2]=rgfield_uc->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_uc->grid(i, j+1, k)-isoval;
	val[4]=rgfield_uc->grid(i, j, k+1)-isoval;
	val[5]=rgfield_uc->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_uc->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_uc->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_c) {
	val[0]=rgfield_c->grid(i, j, k)-isoval;
	val[1]=rgfield_c->grid(i+1, j, k)-isoval;
	val[2]=rgfield_c->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_c->grid(i, j+1, k)-isoval;
	val[4]=rgfield_c->grid(i, j, k+1)-isoval;
	val[5]=rgfield_c->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_c->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_c->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_s) {
	val[0]=rgfield_s->grid(i, j, k)-isoval;
	val[1]=rgfield_s->grid(i+1, j, k)-isoval;
	val[2]=rgfield_s->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_s->grid(i, j+1, k)-isoval;
	val[4]=rgfield_s->grid(i, j, k+1)-isoval;
	val[5]=rgfield_s->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_s->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_s->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_i) {
	val[0]=rgfield_i->grid(i, j, k)-isoval;
	val[1]=rgfield_i->grid(i+1, j, k)-isoval;
	val[2]=rgfield_i->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_i->grid(i, j+1, k)-isoval;
	val[4]=rgfield_i->grid(i, j, k+1)-isoval;
	val[5]=rgfield_i->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_i->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_i->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_f) {
	val[0]=rgfield_f->grid(i, j, k)-isoval;
	val[1]=rgfield_f->grid(i+1, j, k)-isoval;
	val[2]=rgfield_f->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_f->grid(i, j+1, k)-isoval;
	val[4]=rgfield_f->grid(i, j, k+1)-isoval;
	val[5]=rgfield_f->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_f->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_f->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_d) {
	val[0]=rgfield_d->grid(i, j, k)-isoval;
	val[1]=rgfield_d->grid(i+1, j, k)-isoval;
	val[2]=rgfield_d->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_d->grid(i, j+1, k)-isoval;
	val[4]=rgfield_d->grid(i, j, k+1)-isoval;
	val[5]=rgfield_d->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_d->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_d->grid(i, j+1, k+1)-isoval;
    }
    int mask=0;
    int idx;
    for(idx=0;idx<8;idx++){
	if(val[idx]<0)
	    mask|=1<<idx;
    }
    if (mask==0 || mask==255) return 0;
    Point v[8];
    v[0]=rgbase->get_point(i, j, k);
    v[1]=rgbase->get_point(i+1, j, k);
    v[2]=rgbase->get_point(i+1, j+1, k);
    v[3]=rgbase->get_point(i, j+1, k);
    v[4]=rgbase->get_point(i, j, k+1);
    v[5]=rgbase->get_point(i+1, j, k+1);
    v[6]=rgbase->get_point(i+1, j+1, k+1);
    v[7]=rgbase->get_point(i, j+1, k+1);
    TRIANGLE_CASES *tcase=triCases+mask;
    EDGE_LIST *edges=tcase->edges;
    int pidx[3];
    int edgesVisited[12];
    for (int ev=0; ev<12; ev++) edgesVisited[ev]=-1;
    for(; edges[0]>-1; edges+=3) {
	for (int ii=0; ii<3; ii++) {
	    int v1=edges[ii];
	    if (edgesVisited[v1] != -1) {
		pidx[ii]=edgesVisited[v1];
	    } else {
		int i1,j1,k1,d1;
		i1=j1=k1=d1=0;
		if ((((v1-1)%4) == 0) || (v1==11)) i1=1;
		if ((((v1-2)%4) == 0) || (v1==11)) j1=1;
		if (v1<8 && v1>3) k1=1;
		if (v1>7) d1=2;
		else if ((v1%2)==1) d1=1; 
		int e=((i+i1)<<22)+((j+j1)<<12)+((k+k1)<<2)+d1;
		if (last && i1) {
//		    Bdry->dave_hack(e,pidx[ii]);
		    Bdry->lookup(e,pidx[ii]);
		    if (pidx[ii]%4 != 2) {
			Bdry->remove(e);
			Bdry->insert(e,pidx[ii]+1);
//			Bdry->swap(e,pidx[ii]+1);
		    } else
		        Bdry->remove(e);
		    pidx[ii]=-2-(pidx[ii]>>2);
		} else if (first && (v1==3 || v1==7 || v1==8 || v1==10)) {
//		    if (Bdry->dave_hack(e,pidx[ii])) {
		    if (Bdry->lookup(e,pidx[ii])) {
			Bdry->remove(e);
			Bdry->insert(e,pidx[ii]+1);
//			Bdry->swap(e,pidx[ii]+1);
			pidx[ii]=pidx[ii]>>2;
		    } else {
			pidx[ii]=pts.size();
			Bdry->insert(e,pidx[ii]<<2);
			int p0=edge_table[v1][0];
			int p1=edge_table[v1][1];
			pts.add(INTERP(p0,p1));
		    }
//		} else if (hash->dave_hack(e,pidx[ii])) {
		} else if (hash->lookup(e,pidx[ii])) {
		    if (pidx[ii]%4 != 2) {
//			hash->swap(e,pidx[ii]+1);
			hash->remove(e);
			hash->insert(e,pidx[ii]+1);
		    } else
			hash->remove(e);
		    pidx[ii]=pidx[ii]>>2;
		} else {
		    pidx[ii]=pts.size();
		    hash->insert(e,pidx[ii]<<2);
		    int p0=edge_table[v1][0];
		    int p1=edge_table[v1][1];
		    pts.add(INTERP(p0,p1));
		}
		edgesVisited[v1]=pidx[ii];
	    }
	}
	if (pidx[0]>=0 && pidx[1]>=0 && pidx[2]>=0) {
	    group->add(pts[pidx[0]], pts[pidx[1]], pts[pidx[2]]);
	    if (emit) 
		elems.add(new TSElement(pidx[0], pidx[1], pidx[2]));
	} else {
	    elems.add(new TSElement(pidx[0], pidx[1], pidx[2]));
	}
    }
    return(tcase->nbrs);
}

void IsoSurfaceDW::parallel_reg_grid_hash(int proc)
{
    GeomTrianglesP* tris=all_tris[proc];

    int nx=rgbase->nx;
    int ny=rgbase->ny;
    int nz=rgbase->nz;
    int sx;
    int ex;

    blockSize=tclBlockSize.get();

    if (nx <= np*blockSize) { 	// too many processors!
	if ((proc+1)*blockSize < nx) {  // full block
	    sx=proc*blockSize;
	    ex=(proc+1)*blockSize;
	} else if (proc*blockSize < (nx-1)) { // last block is a partial
	    sx=proc*blockSize;	
	    ex=nx-1;
	} else {			// empty block -- unused processor!
	    cerr << "ERROR -- SHOULDN'T BE HERE!!!!\n";
	    return;
	}
    } else {
	sx=proc*(nx-1)/np;
	ex=(proc+1)*(nx-1)/np;
    }

    CPUTimer myThreadTime;
    CPUTimer myOuterExtract;
    CPUTimer myLace;
    CPUTimer myInnerExtract;

    myThreadTime.start();
    myOuterExtract.start();
    Array1<TSElement *>* elems=&all_elems[proc];
    Array1<Point>* pts=&all_pts[proc];
    HashTable<int,int>* hash = new HashTable<int,int>;
    HashTable<int,int>* BdryFirst=all_bdryHashes[proc];
    HashTable<int,int>* BdryLast;
    if (proc != np-1) {
	BdryLast=all_bdryHashes[proc+1];    
    }

    int proc0=0;	
    if (proc==0) proc0=1;
    int procN=0;
    if (proc==(np-1)) procN=1;

    for(int j=0;j<ny-1;j++){
	for(int k=0; k<nz-1; k++){
	    iso_cubeHash(sx, j, k, the_isoval, tris, *elems, *pts, hash,
			 BdryFirst, !proc0, 0);
	}
    }
    if (!proc0) {
	all_sems[proc-1]->up();
    }
    for(int i=sx+1;i<ex-1;i++){
	for(int j=0;j<ny-1;j++){
	    for(int k=0;k<nz-1;k++){
		iso_cubeHash(i, j, k, the_isoval, tris, *elems, *pts, hash,
			     0, 0, 0);
	    }
	    if(sp && abort_flag)
		return;
	}	
    }
    if (!procN) {
	all_sems[proc]->down();
    }
    if (sx<ex-1) {
	for(j=0;j<ny-1;j++){
	    for(int k=0; k<nz-1; k++){
		iso_cubeHash(ex-1, j, k, the_isoval, tris, *elems, *pts, 
			     hash, BdryLast, 0, !procN);
	    }
	}
    }

    // now merge the seperate pts and elems arrays into a sungle surface...

    myOuterExtract.stop();
    barrier.wait(np);
    myLace.start();

    if(proc==0){
	composite_surf = new TriSurface();
	int npts=0, nelems=0;
	for(int i=0;i<np;i++){
	    start_pts[i]=npts; start_elems[i]=nelems;
	    npts+=all_pts[i].size(); nelems+=all_elems[i].size();
	}
	composite_surf->points.resize(npts);
	composite_surf->elements.resize(nelems);
    }

    barrier.wait(np);

    int start_elem=start_elems[proc];
    int start_pt=start_pts[proc];
    int next_start_pt;
    if (proc != np-1)
	next_start_pt=start_pts[proc+1];
    for(i=0;i<all_pts[proc].size();i++)
	composite_surf->points[i+start_pt]=all_pts[proc][i];
	
    barrier.wait(np);

    for(i=0;i<all_elems[proc].size();i++){
	TSElement* e=all_elems[proc][i];
	int flag=0;
	int ii1, ii2, ii3;
	ii1=e->i1; ii2=e->i2; ii3=e->i3;
	if (e->i1<0) {
	    flag=1;
	    e->i1 = -2 - e->i1 + next_start_pt;
	} else {
	    e->i1 += start_pt; 
	}
	if (e->i2 < 0) {
	    flag=1;
	    e->i2 = -2 - e->i2 + next_start_pt;
	} else {
	    e->i2 += start_pt;
	}
	if (e->i3 < 0) {
	    flag=1;
	    e->i3 = -2 - e->i3 + next_start_pt;
	} else {
	    e->i3 += start_pt;
	}
	composite_surf->elements[i+start_elem]=e;
	if (flag) {	// need to add this boundary GeomTri
	    all_tris[proc]->add(composite_surf->points[e->i1],
				composite_surf->points[e->i2],
				composite_surf->points[e->i3]);
	}	
    }
    myLace.stop();
    myThreadTime.stop();     
    lace.add(myLace.time());
    outerExtract.add(myOuterExtract.time());
    innerExtract.add(myInnerExtract.time());
    timer.add(myThreadTime.time());
}

void IsoSurfaceDW::iso_reg_grid_hash()
{
    WallClockTimer wct;
    wct.start();
    blockSize=tclBlockSize.get();
    if (sing) np=1;	
    else np=Min(Task::nprocessors(), ((rgbase->nx-2)/blockSize)+1);

    cerr << "Parallel extraction with Hashing -- using "<<np<<" processors, blocksize="<<blockSize<<"  emit="<<emit<<"\n";

    // build them as separate surfaces...
    for (int i=0; i<all_elems.size(); i++) 
	all_elems[i].resize(0);
    for (i=0; i<all_pts.size(); i++)
	all_pts[i].resize(0);
    for (i=0; i<all_sems.size(); i++)
	delete all_sems[i];
    for (i=0; i<all_bdryHashes.size(); i++)
	delete all_bdryHashes[i];
    all_elems.resize(np);
    all_pts.resize(np);
    all_tris.resize(np);
    all_sems.resize(np);
    all_bdryHashes.resize(np);
    start_pts.resize(np);
    start_elems.resize(np);
    for (i=0; i<np; i++) {
	all_tris[i] = new GeomTrianglesP();
	all_sems[i] = new Semaphore(0);
	all_bdryHashes[i] = new HashTable<int,int>;
    }
    
    outerExtract.clear();
    innerExtract.clear();
    lace.clear();

    Task::multiprocess(np, do_parallel_reg_grid_hash, this);

    for (i=0; i<np; i++) {
	if (all_tris[i]->size())
	    maingroup->add(all_tris[i]);
    }

    cerr << "Total outerExtract: "<<outerExtract.time()<<"\n";
    cerr << "Total innerExtract: "<<innerExtract.time()<<"\n";
    cerr << "Total lace: "<<lace.time()<<"\n";
    cerr << "TOTAL: "<<outerExtract.time()+lace.time()<<"\n";

    if (emit) {
	int total_pts=composite_surf->points.size();
	int total_tris=composite_surf->elements.size();
	if ((total_pts-2)*2 != total_tris) {
	    cerr << "NOT A SINGLE SURFACE --  # pts: "<<total_pts<<"   # tris: "<<total_tris<<"\n";
	    } else {
		cerr << "SINGLE SURFACE FOUND\n";
	    }
    }

    wct.stop();
    cerr << "TOTAL WALL CLOCK TIME: "<<wct.time()<<"\n\n\n";
}

void IsoSurfaceDW::iso_reg_grid()
{
    if (emit) {
	cerr << "Can't build topological surface with this method.\n";
    }
    WallClockTimer wct;
    wct.start();
    blockSize=tclBlockSize.get();
    if (sing) np=1;	
    else np=Min(Task::nprocessors(), ((rgbase->nx-2)/blockSize)+1);
    cerr << "Parallel extraction -- using "<<np<<" processors, blocksize="<<blockSize<<"  emit="<<emit<<"\n";

    all_tris.resize(np);
    for (int i=0; i<np; i++) {
	all_tris[i] = new GeomTrianglesP();
    }
    
    outerExtract.clear();

    Task::multiprocess(np, do_parallel_reg_grid, this);

    for (i=0; i<np; i++) {
	if (all_tris[i]->size())
	    maingroup->add(all_tris[i]);
    }

    wct.stop();
    cerr << "Total outerExtract: "<<outerExtract.time()<<"\n";
    cerr << "TOTAL: "<<outerExtract.time()<<"\n";
    cerr << "TOTAL WALL CLOCK TIME: "<<wct.time()<<"\n\n\n";
}

int IsoSurfaceDW::iso_cubeRing(int i, int j, int k, double isoval,
			     GeomTrianglesP* group, 
			     Array1<TSElement*>& elems, Array1<Point>& pts,
			     int first, int last, int lastY, int proc0,
			     Ring<int>* RX, Ring<int>* RY,
			     Ring<int>* BdryFirst, Ring<int>* BdryLast,
			     int &Z0, int &Z1, int &Z2, int &Z3) {
    double val[8];
    int twoPushYFlag=0;
    if (rgfield_uc) {
	val[0]=rgfield_uc->grid(i, j, k)-isoval;
	val[1]=rgfield_uc->grid(i+1, j, k)-isoval;
	val[2]=rgfield_uc->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_uc->grid(i, j+1, k)-isoval;
	val[4]=rgfield_uc->grid(i, j, k+1)-isoval;
	val[5]=rgfield_uc->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_uc->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_uc->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_c) {
	val[0]=rgfield_c->grid(i, j, k)-isoval;
	val[1]=rgfield_c->grid(i+1, j, k)-isoval;
	val[2]=rgfield_c->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_c->grid(i, j+1, k)-isoval;
	val[4]=rgfield_c->grid(i, j, k+1)-isoval;
	val[5]=rgfield_c->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_c->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_c->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_s) {
	val[0]=rgfield_s->grid(i, j, k)-isoval;
	val[1]=rgfield_s->grid(i+1, j, k)-isoval;
	val[2]=rgfield_s->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_s->grid(i, j+1, k)-isoval;
	val[4]=rgfield_s->grid(i, j, k+1)-isoval;
	val[5]=rgfield_s->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_s->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_s->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_i) {
	val[0]=rgfield_i->grid(i, j, k)-isoval;
	val[1]=rgfield_i->grid(i+1, j, k)-isoval;
	val[2]=rgfield_i->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_i->grid(i, j+1, k)-isoval;
	val[4]=rgfield_i->grid(i, j, k+1)-isoval;
	val[5]=rgfield_i->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_i->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_i->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_f) {
	val[0]=rgfield_f->grid(i, j, k)-isoval;
	val[1]=rgfield_f->grid(i+1, j, k)-isoval;
	val[2]=rgfield_f->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_f->grid(i, j+1, k)-isoval;
	val[4]=rgfield_f->grid(i, j, k+1)-isoval;
	val[5]=rgfield_f->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_f->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_f->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_d) {
	val[0]=rgfield_d->grid(i, j, k)-isoval;
	val[1]=rgfield_d->grid(i+1, j, k)-isoval;
	val[2]=rgfield_d->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_d->grid(i, j+1, k)-isoval;
	val[4]=rgfield_d->grid(i, j, k+1)-isoval;
	val[5]=rgfield_d->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_d->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_d->grid(i, j+1, k+1)-isoval;
    }
    int mask=0;
    int idx;
    for(idx=0;idx<8;idx++){
	if(val[idx]<0)
	    mask|=1<<idx;
    }
    if (mask==0 || mask==255) return 0;
    int firstY=(j==0);
    int firstZ=(k==0);
    Point v[8];
    v[0]=rgbase->get_point(i, j, k);
    v[1]=rgbase->get_point(i+1, j, k);
    v[2]=rgbase->get_point(i+1, j+1, k);
    v[3]=rgbase->get_point(i, j+1, k);
    v[4]=rgbase->get_point(i, j, k+1);
    v[5]=rgbase->get_point(i+1, j, k+1);
    v[6]=rgbase->get_point(i+1, j+1, k+1);
    v[7]=rgbase->get_point(i, j+1, k+1);
    TRIANGLE_CASES *tcase=triCases+mask;
    EDGE_LIST *edges=tcase->edges;
    Point p0, p1, p2;
    int vxl=0;
    for(; (*edges)>-1; edges++) {
	vxl |= 1<<*edges;
    }
    edges=tcase->edges;
    Array1<int> e_idx(12);
    int cz0=Z0;
    int cz1=Z1;
    int cz2=Z2;
    int cz3=Z3;
    int tmp;
    if (vxl & 2048) {		// edge 11
	if (last) {
	    e_idx[11]=-2-BdryLast->pop();
	} else {
	    tmp=pts.size();
	    e_idx[11]=tmp;
	    if (!lastY) {
		RY->push(tmp);	    
	    }
	    RX->push(tmp);
	    pts.add(INTERP(2, 6));
	}
    }
    if (vxl & 32) {		// edge 5
	if (last) {
	    e_idx[5]=-2-BdryLast->pop();
	} else {
	    tmp=pts.size();
	    e_idx[5]=tmp;
	    Z1=tmp;
	    RX->push(tmp);
	    pts.add(INTERP(5, 6));
	}
    }
    if (vxl & 64) {		// edge 6
	tmp=pts.size();
	e_idx[6]=tmp;
	Z2=pts.size();
	if (!lastY) {
	    RY->push(pts.size());
	}
	pts.add(INTERP(7,6));
    }
    if (vxl & 512) {		// edge 9
	if (last) {  // did the previous level generate this pt?
	    e_idx[9]=-2-BdryLast->pop();
	} else {
	    if (firstY) {  // did 11 put this in the RY ring?
		tmp=pts.size();
		e_idx[9]=tmp;
		RX->push(tmp);
		pts.add(INTERP(1,5));
	    } else {
		tmp=RY->pop();
		e_idx[9]=tmp;
		RX->push(tmp);
	    }
	}
    }
    if (vxl & 2) {		// edge 1
	if (last) {
	    e_idx[1]=-2-BdryLast->pop();
	} else {
	    if (firstZ) {  // if 5 didn't generate this pt
		tmp=pts.size();
		e_idx[1]=tmp;
		RX->push(tmp);
		pts.add(INTERP(1,2));
	    } else {
		e_idx[1]=cz1;
		RX->push(e_idx[1]);
	    }
	}
    }
    if (vxl & 4) {		// edge 2
	twoPushYFlag=1;
	if (firstZ) {  // if 6 didn't generate this pt
	    tmp=pts.size();
	    e_idx[2]=tmp;
	    if (!lastY) {
		RY->push(tmp);
	    }
	    pts.add(INTERP(3,2));
	} else {
	    e_idx[2]=cz2;
	    if (!lastY) {
		RY->push(e_idx[2]);
	    }
	}
    }
    if (vxl & 1024) {		// edge 10
	if (first) {  // 11 wasn't able to generate this pt
	    tmp=pts.size();
	    e_idx[10]=tmp;
	    if (!lastY) {
		if (twoPushYFlag) {
		    RY->swap(tmp);
		} else {
		    RY->push(tmp);
		}
	    }
	    if (!proc0) {
		BdryFirst->push(tmp);
	    }
	    pts.add(INTERP(3,7));
	} else {
	    e_idx[10]=RX->pop();
	}
    }
    if (vxl & 128) {		// edge 7
	if (first) {  // 5 wasn't able to generate this pt
	    tmp=pts.size();
	    e_idx[7]=tmp;
	    Z3=tmp;
	    if (!proc0) {
		BdryFirst->push(tmp);
	    }
	    pts.add(INTERP(4,7));
	} else {
	    e_idx[7]=RX->pop();
	}
    }
    if (vxl & 16) {		// edge 4
	if (firstY) {  // 6 wasn't able to generate this pt
	    tmp=pts.size();
	    e_idx[4]=tmp;
	    Z0=tmp;
	    pts.add(INTERP(4,5));
	} else {
	    e_idx[4]=RY->pop();
	}
    }
    if (vxl & 256) {		// edge 8
	if (first) {  // if 9 and 11 weren't able to push it...
	    if (firstY) {  // noone did!
		e_idx[8]=pts.size();
		pts.add(INTERP(0,4));
	    } else {	// 10 pushed it on Y
		e_idx[8]=RY->pop();
	    }
	} else {  // 9 or 11 pushed it on X
	    e_idx[8]=RX->pop();
	}
	if (first && !proc0) {  // we've gotta push it into Bdry
	    BdryFirst->push(e_idx[8]);
	}
    }
    if (vxl & 8) {		// edge 3
	if (first) {
	    if (firstZ) {
		tmp=pts.size();
		e_idx[3]=tmp;
		pts.add(INTERP(0,3));
		if (!proc0) {
		    BdryFirst->push(tmp);
		}
	    } else {
		e_idx[3]=cz3;
		if (!proc0) {
		    BdryFirst->push(e_idx[3]);
		}
	    }
	} else {
	    e_idx[3]=RX->pop();
	}
    }
    if (vxl & 1) {		// edge 0
	if (firstY) {
	    if (firstZ) {
		e_idx[0]=pts.size();
		pts.add(INTERP(0,1));
	    } else {
		e_idx[0]=cz0;
	    }
	} else {
	    e_idx[0]=RY->pop();
	}
    }
    int idx0, idx1, idx2;
    for (; edges[0]>-1; edges+=3) {
	idx0=e_idx[edges[0]];
	idx1=e_idx[edges[1]];
	idx2=e_idx[edges[2]];
	if (idx0>=0 && idx1>=0 && idx2>=0) {
	    p0=pts[idx0];
	    p1=pts[idx1];
	    p2=pts[idx2];
	    group->add(p0, p1, p2);
	    if (emit) elems.add(new TSElement(idx0, idx1, idx2));
	} else {
	    elems.add(new TSElement(idx0, idx1, idx2));
	}
    }
    return(tcase->nbrs);
}

int IsoSurfaceDW::iso_cubeRingFast(int i, int j, int k, double isoval,
				 GeomTrianglesP* group, 
				 Array1<TSElement*>& elems, Array1<Point>& pts,
				 Ring<int>* RX, Ring<int>* RY,int &Z1,int &Z2){
    double val[8];
    if (rgfield_uc) {
	val[0]=rgfield_uc->grid(i, j, k)-isoval;
	val[1]=rgfield_uc->grid(i+1, j, k)-isoval;
	val[2]=rgfield_uc->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_uc->grid(i, j+1, k)-isoval;
	val[4]=rgfield_uc->grid(i, j, k+1)-isoval;
	val[5]=rgfield_uc->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_uc->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_uc->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_c) {
	val[0]=rgfield_c->grid(i, j, k)-isoval;
	val[1]=rgfield_c->grid(i+1, j, k)-isoval;
	val[2]=rgfield_c->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_c->grid(i, j+1, k)-isoval;
	val[4]=rgfield_c->grid(i, j, k+1)-isoval;
	val[5]=rgfield_c->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_c->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_c->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_s) {
	val[0]=rgfield_s->grid(i, j, k)-isoval;
	val[1]=rgfield_s->grid(i+1, j, k)-isoval;
	val[2]=rgfield_s->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_s->grid(i, j+1, k)-isoval;
	val[4]=rgfield_s->grid(i, j, k+1)-isoval;
	val[5]=rgfield_s->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_s->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_s->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_i) {
	val[0]=rgfield_i->grid(i, j, k)-isoval;
	val[1]=rgfield_i->grid(i+1, j, k)-isoval;
	val[2]=rgfield_i->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_i->grid(i, j+1, k)-isoval;
	val[4]=rgfield_i->grid(i, j, k+1)-isoval;
	val[5]=rgfield_i->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_i->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_i->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_f) {
	val[0]=rgfield_f->grid(i, j, k)-isoval;
	val[1]=rgfield_f->grid(i+1, j, k)-isoval;
	val[2]=rgfield_f->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_f->grid(i, j+1, k)-isoval;
	val[4]=rgfield_f->grid(i, j, k+1)-isoval;
	val[5]=rgfield_f->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_f->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_f->grid(i, j+1, k+1)-isoval;
    } else if (rgfield_d) {
	val[0]=rgfield_d->grid(i, j, k)-isoval;
	val[1]=rgfield_d->grid(i+1, j, k)-isoval;
	val[2]=rgfield_d->grid(i+1, j+1, k)-isoval;
	val[3]=rgfield_d->grid(i, j+1, k)-isoval;
	val[4]=rgfield_d->grid(i, j, k+1)-isoval;
	val[5]=rgfield_d->grid(i+1, j, k+1)-isoval;
	val[6]=rgfield_d->grid(i+1, j+1, k+1)-isoval;
	val[7]=rgfield_d->grid(i, j+1, k+1)-isoval;
    }
    int mask=0;
    int idx;
    for(idx=0;idx<8;idx++){
	if(val[idx]<0)
	    mask|=1<<idx;
    }
    if (mask==0 || mask==255) return 0;
    int cz1=Z1;
    int cz2=Z2;
    Point v[8];
    v[0]=rgbase->get_point(i, j, k);
    v[1]=rgbase->get_point(i+1, j, k);
    v[2]=rgbase->get_point(i+1, j+1, k);
    v[3]=rgbase->get_point(i, j+1, k);
    v[4]=rgbase->get_point(i, j, k+1);
    v[5]=rgbase->get_point(i+1, j, k+1);
    v[6]=rgbase->get_point(i+1, j+1, k+1);
    v[7]=rgbase->get_point(i, j+1, k+1);
    TRIANGLE_CASES *tcase=triCases+mask;
    EDGE_LIST *edges=tcase->edges;
    Point p0, p1, p2;
    int vxl=0;
    for(; (*edges)>-1; edges++) {
	vxl |= 1<<*edges;
    }
    edges=tcase->edges;
    Array1<int> e_idx(12);
    int tmp;
    if (vxl & 2048) {		// edge 11
	tmp=pts.size();
	e_idx[11]=tmp;
	RY->push(tmp);	    
	RX->push(tmp);
	pts.add(INTERP(2, 6));
    }
    if (vxl & 32) {		// edge 5
	tmp=pts.size();
	e_idx[5]=tmp;
	Z1=tmp;
	RX->push(tmp);
	pts.add(INTERP(5, 6));
    }
    if (vxl & 64) {		// edge 6
	tmp=pts.size();
	e_idx[6]=tmp;
	Z2=tmp;
	RY->push(tmp);
	pts.add(INTERP(7,6));
    }
    if (vxl & 512) {		// edge 9
	tmp=RY->pop();
	e_idx[9]=tmp;
	RX->push(tmp);
    }
    if (vxl & 2) {		// edge 1
	e_idx[1]=cz1;
	RX->push(e_idx[1]);
    }
    if (vxl & 4) {		// edge 2
	e_idx[2]=cz2;
	RY->push(e_idx[2]);
    }
    if (vxl & 1024) {		// edge 10
	e_idx[10]=RX->pop();
    }
    if (vxl & 128) {		// edge 7
	e_idx[7]=RX->pop();
    }
    if (vxl & 16) {		// edge 4
	e_idx[4]=RY->pop();
    }
    if (vxl & 256) {		// edge 8
	e_idx[8]=RX->pop();
    }
    if (vxl & 8) {		// edge 3
	e_idx[3]=RX->pop();
    }
    if (vxl & 1) {		// edge 0
	e_idx[0]=RY->pop();
    }
    int idx0, idx1, idx2;
    for (; edges[0]>-1; edges+=3) {
	idx0=e_idx[edges[0]];
	idx1=e_idx[edges[1]];
	idx2=e_idx[edges[2]];
	if (group->add(pts[idx0], pts[idx1], pts[idx2]))
	    if (emit || idx0<0 || idx<1 || idx<2)
		elems.add(new TSElement(idx0, idx1, idx2));
    }
    return(tcase->nbrs);
}

void IsoSurfaceDW::parallel_reg_grid_rings(int proc)
{
    GeomTrianglesP* tris=all_tris[proc];

    int nx=rgbase->nx;
    int ny=rgbase->ny;
    int nz=rgbase->nz;
    int sx;
    int ex;

    blockSize=tclBlockSize.get();

    if (nx <= np*blockSize) { 	// too many processors!
	if ((proc+1)*blockSize < nx) {  // full block
	    sx=proc*blockSize;
	    ex=(proc+1)*blockSize;
	} else if (proc*blockSize < (nx-1)) { // last block is a partial
	    sx=proc*blockSize;	
	    ex=nx-1;
	} else {			// empty block -- unused processor!
	    cerr << "ERROR -- SHOULDN'T BE HERE!!!!\n";
	    return;
	}
    } else {
	sx=proc*(nx-1)/np;
	ex=(proc+1)*(nx-1)/np;
    }

    CPUTimer myThreadTime;
    CPUTimer myOuterExtract;
    CPUTimer myLace;
    CPUTimer myInnerExtract;

    myThreadTime.start();
    myOuterExtract.start();
    Array1<TSElement *>* elems=&all_elems[proc];
    Array1<Point>* pts=&all_pts[proc];
    Ring<int>* RX = new Ring<int>((ny*nz+1)*2);
    Ring<int>* RY = new Ring<int>((nz+1)*2);
    Ring<int>* BdryFirst=all_bdryRings[proc];
    Ring<int>* BdryLast;
    if (proc != np-1) {
	BdryLast=all_bdryRings[proc+1];    
    }
    int Z0, Z1, Z2, Z3;
    int proc0 = (proc==0);
    int procN = (proc==(np-1));

    int lastY=0;
    for(int j=0;j<ny-1;j++){
	if (j==ny-2)
	    lastY=1;
	for(int k=0; k<nz-1; k++){
	    iso_cubeRing(sx, j, k, the_isoval, tris, *elems, *pts,
		       1, 0, lastY, proc0, RX, RY, BdryFirst, 
		       BdryLast, Z0, Z1, Z2, Z3);
	}
    }
    if (!proc0) {
	all_sems[proc-1]->up();
    }
    for(int i=sx+1; i<ex-1; i++){
	for(int k=0; k<nz-1; k++){
	    iso_cubeRing(i, 0, k, the_isoval, tris, *elems, *pts,
		       0, 0, 0, proc0, RX, RY, BdryFirst, 
		       BdryLast, Z0, Z1, Z2, Z3);
	}
	for(j=1;j<ny-2;j++){
	    iso_cubeRing(i, j, 0, the_isoval, tris, *elems, *pts,
		       0, 0, 0, proc0, RX, RY, BdryFirst, 
		       BdryLast, Z0, Z1, Z2, Z3);
	    myInnerExtract.start();
	    for(k=1; k<nz-2; k++){
		iso_cubeRingFast(i, j, k, the_isoval, tris, *elems, *pts,
			       RX, RY, Z1, Z2);
	    }
	    myInnerExtract.stop();
	    iso_cubeRing(i, j, nz-2, the_isoval, tris, *elems, *pts,
		       0, 0, 0, proc0, RX, RY, BdryFirst, 
		       BdryLast, Z0, Z1, Z2, Z3);
	    if(sp && abort_flag)
		return;
	}
	for(k=0; k<nz-1; k++){
	    iso_cubeRing(i, ny-2, k, the_isoval, tris, *elems, *pts,
		       0, 0, 1, proc0, RX, RY, BdryFirst, 
		       BdryLast, Z0, Z1, Z2, Z3);
	}
    }
    lastY=0;
    if (!procN) {
	all_sems[proc]->down();
    }
    if (sx<ex-1) {
	for(j=0;j<ny-1;j++){
	    if (j==ny-2)
		lastY=1;
	    for(int k=0; k<nz-1; k++){
		iso_cubeRing(ex-1, j, k, the_isoval, tris, *elems, *pts,
			   0, !procN, lastY, proc0, RX, RY, BdryFirst, 
			   BdryLast, Z0, Z1, Z2, Z3);
	    }
	}
    }
    // now merge the seperate pts and elems arrays into a sungle surface...

    myOuterExtract.stop();
    barrier.wait(np);
    myLace.start();

    if(proc==0){
	composite_surf = new TriSurface();
	int npts=0, nelems=0;
	for(int i=0;i<np;i++){
	    start_pts[i]=npts; start_elems[i]=nelems;
	    npts+=all_pts[i].size(); nelems+=all_elems[i].size();
	}
	composite_surf->points.resize(npts);
	composite_surf->elements.resize(nelems);
    }

    barrier.wait(np);

    int start_elem=start_elems[proc];
    int start_pt=start_pts[proc];
    int next_start_pt;
    if (proc != np-1)
	next_start_pt=start_pts[proc+1];
    for(i=0;i<all_pts[proc].size();i++)
	composite_surf->points[i+start_pt]=all_pts[proc][i];
	
    barrier.wait(np);

    for(i=0;i<all_elems[proc].size();i++){
	TSElement* e=all_elems[proc][i];
	int flag=0;
	int ii1, ii2, ii3;
	ii1=e->i1; ii2=e->i2; ii3=e->i3;
	if (e->i1<0) {
	    flag=1;
	    e->i1 = -2 - e->i1 + next_start_pt;
	} else {
	    e->i1 += start_pt; 
	}
	if (e->i2 < 0) {
	    flag=1;
	    e->i2 = -2 - e->i2 + next_start_pt;
	} else {
	    e->i2 += start_pt;
	}
	if (e->i3 < 0) {
	    flag=1;
	    e->i3 = -2 - e->i3 + next_start_pt;
	} else {
	    e->i3 += start_pt;
	}
	composite_surf->elements[i+start_elem]=e;
	if (flag) {	// need to add this boundary GeomTri
	    all_tris[proc]->add(composite_surf->points[e->i1],
				composite_surf->points[e->i2],
				composite_surf->points[e->i3]);
	}	
    }
    myLace.stop();
    myThreadTime.stop();     
    lace.add(myLace.time());
    outerExtract.add(myOuterExtract.time());
    innerExtract.add(myInnerExtract.time());
    timer.add(myThreadTime.time());
}

void IsoSurfaceDW::iso_reg_grid_rings()
{
    WallClockTimer wct;
    wct.start();
    blockSize=tclBlockSize.get();
    if (sing) np=1;	
    else np=Min(Task::nprocessors(), ((rgbase->nx-2)/blockSize)+1);
//    np=Min(Task::nprocessors(), ((rgbase->nx-2)/blockSize)+1);
    cerr << "Parallel extraction with cache rings -- using "<<np<<" processors, blocksize="<<blockSize<<"  emit="<<emit<<"\n";

    // build them as separate surfaces...
    for (int i=0; i<all_elems.size(); i++) 
	all_elems[i].resize(0);
    for (i=0; i<all_pts.size(); i++)
	all_pts[i].resize(0);
    for (i=0; i<all_sems.size(); i++)
	delete all_sems[i];
    for (i=0; i<all_bdryRings.size(); i++)
	delete all_bdryRings[i];
    all_elems.resize(np);
    all_pts.resize(np);
    all_tris.resize(np);
    all_sems.resize(np);
    all_bdryRings.resize(np);
    start_pts.resize(np);
    start_elems.resize(np);
    for (i=0; i<np; i++) {
	all_tris[i] = new GeomTrianglesP();
	all_sems[i] = new Semaphore(0);
	all_bdryRings[i] = new Ring<int>(rgbase->ny*rgbase->nz);
    }
    
    outerExtract.clear();
    innerExtract.clear();
    lace.clear();

    Task::multiprocess(np, do_parallel_reg_grid_rings, this);

    for (i=0; i<np; i++) {
	if (all_tris[i]->size())
	    maingroup->add(all_tris[i]);
    }

    cerr << "Total outerExtract: "<<outerExtract.time()<<"\n";
    cerr << "Total innerExtract: "<<innerExtract.time()<<"\n";
    cerr << "Total lace: "<<lace.time()<<"\n";
    cerr << "TOTAL: "<<outerExtract.time()+lace.time()<<"\n";

    if (emit) {
	int total_pts=composite_surf->points.size();
	int total_tris=composite_surf->elements.size();
	if ((total_pts-2)*2 != total_tris) {
	    cerr << "NOT A SINGLE SURFACE --  # pts: "<<total_pts<<"   # tris: "<<total_tris<<"\n";
	} else {
	    cerr << "SINGLE SURFACE FOUND\n";
	}
    }

    wct.stop();
    cerr << "TOTAL WALL CLOCK TIME: "<<wct.time()<<"\n\n\n";
}

void IsoSurfaceDW::iso_reg_grid(const Point& p,
			      GeomTrianglesP* group)
{
    int nx=rgbase->nx;
    int ny=rgbase->ny;
    int nz=rgbase->nz;
    double iv;
    if(!rgbase->interpolate(p, iv)){
	error("Seed point not in rgfield boundary");
	return;
    }
    isoval.set(iv);
//    cerr << "Isoval = " << iv << "\n";
    HashTable<int, int> visitedPts;
    Queue<int> surfQ;
    int px, py, pz;
    rgbase->locate(p, px, py, pz);
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
	int nbrs=iso_cube(px, py, pz, iv, group);
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

int IsoSurfaceDW::iso_tetra(Element* element, Mesh* mesh,
			  ScalarFieldUG* field, double isoval,
			  GeomTrianglesP* group)
{
    double v1=field->data[element->n[0]]-isoval;
    double v2=field->data[element->n[1]]-isoval;
    double v3=field->data[element->n[2]]-isoval;
    double v4=field->data[element->n[3]]-isoval;
    Node* n1=mesh->nodes[element->n[0]].get_rep();
    Node* n2=mesh->nodes[element->n[1]].get_rep();
    Node* n3=mesh->nodes[element->n[2]].get_rep();
    Node* n4=mesh->nodes[element->n[3]].get_rep();
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

void IsoSurfaceDW::iso_tetrahedra(ScalarFieldUG* field, double isoval,
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


void IsoSurfaceDW::iso_tetrahedra_strip(ScalarFieldUG* field, double isoval,
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
#if NORMALIZE_NORMALS
    if (n.length2() > 0){
	n.normalize();
    }
#endif
    if (!(i&1)) // this implies a different sign convention
	n *=-1.0;
    group->add(p1,n);
}

// an "entrance" contains 6 bits - 3 bits for 1st edge, 3 for second
// you encode an "exit" with the required "entrance" bits and the following
// other high order bits:  4 bits of what faces must be checked for pushing
// and 4 bits for the face to recures (only 1 can be set at most)

int IsoSurfaceDW::iso_strip_enter(int inc, 
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

void IsoSurfaceDW::remap_element(int& rval, Element *src, Element *dst)
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
    Vector n2(Cross(pB-pC,pD-pC));
#if NORMALIZE_NORMALS
    if (n.length2() > 0)
	n.normalize();
    if (n2.length2() > 0)
	n2.normalize();
#endif    
    group->add(pA);
    group->add(pB);
    group->add(pC,n);
    group->add(pD,n2);    
}

int IsoSurfaceDW::iso_tetra_s(int nbr_status,Element *element, Mesh* mesh, 
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
#if NORMALIZE_NORMALS
	    if (n.length2() > 0){
		n.normalize();
	    }
#endif
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
#if NORMALIZE_NORMALS
	    if (n.length2() > 0){
		n.normalize();
	    }
#endif
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
#if NORMALIZE_NORMALS
	    if (n.length2() > 0){
		n.normalize();
	    }
#endif
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
#if NORMALIZE_NORMALS
	    if (n.length2() > 0){
		n.normalize();
	    }
#endif
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

void IsoSurfaceDW::iso_tetra_strip(int ix, Mesh* mesh, 
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

void IsoSurfaceDW::iso_tetrahedra_strip(ScalarFieldUG* field, const Point& p,
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

void IsoSurfaceDW::iso_tetrahedra(ScalarFieldUG* field, const Point& p,
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

void IsoSurfaceDW::find_seed_from_value(const ScalarFieldHandle& /*field*/)
{
    NOT_FINISHED("IsoSurfaceDW::find_seed_from_value");
#if 0
    int nx=field->get_nx();
    int ny=field->get_ny();
    int nz=field->get_nz();
    GeomGroup group;
    for (int i=0; i<nx-1;i++) {
	for (int j=0; j<ny-1; j++) {
	    for (int k=0; k<nz-1; k++) {
		if(iso_cube(i,j,k,isoval,&group,field)) {
		    seed_point=Point(i,j,k);
		    cerr << "New seed=" << seed_point.string() << endl;
		    return;
		}
	    }
	}
    }
#endif
}

void IsoSurfaceDW::widget_moved(int last)
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
