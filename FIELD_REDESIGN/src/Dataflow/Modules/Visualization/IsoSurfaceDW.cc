//static char *id="@(#) $Id$";

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

#include <SCICore/Containers/BitArray1.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/Queue.h>
#include <SCICore/Containers/Ring.h>
#include <SCICore/Containers/Stack.h>
#include <SCICore/Util/Timer.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Geom/GeomTriStrip.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Plane.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Barrier.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/Thread.h>

#include <map.h>
#include <iostream>
using std::cerr;
#include <sstream>
using std::ostringstream;

#define INTERP(i1, i2) (Interpolate(v[i1],v[i2],val[i1]*1./(val[i1]-val[i2])))

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Containers;
using namespace SCICore::Math;
using namespace SCICore::Thread;

class IsoSurfaceDW : public Module {

private:
    ScalarFieldIPort* infield;
    ScalarFieldIPort* incolorfield;
    ColorMapIPort* inColorMap;

    GeometryOPort* ogeom;
    SurfaceOPort* osurf;

    TCLdouble isoval;
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

    TriSurface *composite_surf;
    Array1<int> start_pts;
    Array1<int> start_elems;
    Array1<Array1<TSElement*> > all_elems;
    Array1<Array1<Point> > all_pts;
    Array1<GeomTrianglesP*> all_tris;
    Array1<Semaphore*> all_sems;
    Array1<Ring<int>* > all_bdryRings;
  
public:
    typedef map<int, int> MapIntInt;

private:
    Array1<MapIntInt*> all_bdryHashes;
  
    double old_min;
    double old_max;
    int sp;
    TCLint show_progress;
    Mutex io;

    CPUTimer outerExtract;
    CPUTimer innerExtract;
    CPUTimer lace;
    CPUTimer barrierLace;
    MapIntInt hash;
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
		     MapIntInt*, MapIntInt*, int, int);

    int iso_cubeTS(int, int, int, double, GeomTrianglesP*, 
		     Array1<TSElement*>&, Array1<Point>&);

    int iso_tetra(Element*, Mesh*, ScalarFieldUG*, double, GeomTrianglesP*);
    void iso_tetrahedra(ScalarFieldUG*, double, GeomTrianglesP*);

    ScalarFieldHandle field;
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
    TCLint logTCL;
    int np;
    void iso_reg_grid();
    void iso_reg_grid_hash();
    void iso_reg_grid_rings();
public:
    void parallel_reg_grid(int proc);
    void parallel_reg_grid_hash(int proc);
    void parallel_reg_grid_rings(int proc);
    IsoSurfaceDW(const clString& id);
    virtual ~IsoSurfaceDW();
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
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

extern "C" Module* make_IsoSurfaceDW(const clString& id) {
  return new IsoSurfaceDW(id);
}

static clString module_name("IsoSurfaceDW");
static clString surface_name("IsoSurfaceDW");

IsoSurfaceDW::IsoSurfaceDW(const clString& id)
: Module("IsoSurfaceDW", id, Filter), isoval("isoval", id, this),
  emit_surface("emit_surface", id, this),
  single("single", id, this), show_progress("show_progress", id, this), 
  tclBlockSize("tclBlockSize", id, this),
  method("method", id, this), clr_r("clr-r", id, this), 
  clr_g("clr-g", id, this), clr_b("clr-b", id, this),
  io("IsoSurfaceDW i/o lock"), hashing("IsoSurfaceDW hashing lock"),
  grouplock("IsoSurfaceDW grouplock lock"), barrier("IsoSurfaceDW barrier"),
  logTCL("logTCL", id, this)
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
    IsoSurfaceDW_id=0;

    old_min=old_max=0;
}

IsoSurfaceDW::~IsoSurfaceDW()
{
}

void IsoSurfaceDW::execute()
{
    update_state(NeedData);
    if(IsoSurfaceDW_id){
	ogeom->delObj(IsoSurfaceDW_id);
    }
    if(!infield->get(field))
	return;
    ScalarFieldHandle colorfield;
    int have_colorfield=incolorfield->get(colorfield);
    ColorMapHandle cmap;
    int have_ColorMap=inColorMap->get(cmap);
    update_state(JustStarted);

    double min, max;
    field->get_minmax(min, max);
    if(min != old_min || max != old_max){
	ostringstream str;
	double mmin=min;
	double mmax=max;
	if (logTCL.get()) {
	    if (min<1e-15) min=1e-15;
	    if (max<1e-14) max=1e-14;
	    mmin = log(min);
	    mmax = log(max);
	}
	str << id << " set_minmax " << mmin << " " << mmax;
	TCL::execute(str.str().c_str());
	old_min=min;
	old_max=max;
    }
    if (method.get() == clString("None")) return;
    sp=show_progress.get();
    double iv=isoval.get();
    cerr << "isoval="<<iv<<"\n";
    if (logTCL.get()) {
	iv=exp(iv);
	cerr << "   using e^x = "<<iv<<"\n";
    }
    Point sp;
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
    } else if(unstructured_grid){
	if (emit) {
	    surf=scinew TriSurface;
	    int pts_per_side=(int) Cbrt(unstructured_grid->mesh->nodes.size());
	    spacing/=pts_per_side;
	    surf->construct_grid(pts_per_side+2,pts_per_side+2,pts_per_side+2, 
				 minPt+(Vector(1.001,1.029,0.917)*(-.001329)),
				 spacing);
	}	
	iso_tetrahedra(unstructured_grid, iv, group);
	tgroup->add(group);
    } else {
	error("I can't IsoSurfaceDW this type of field...");
    }

    if(tgroup->size() == 0){
	delete tgroup;
#if 0
	// had to comment this out - for some reason destroying this crashes
	//   everything... go figure
	if (emit) {
	    delete composite_surf;
	}
#endif
	IsoSurfaceDW_id=0;
    } else {
	IsoSurfaceDW_id=ogeom->addObj(scinew GeomMaterial(topobj, mpick), surface_name);
	if (emit) {
	    osurf->send(SurfaceHandle(composite_surf));
	    cerr << "sent surface" << endl;
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
			     Array1<TSElement*>& elems, 
			     Array1<Point>& pts) {

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
	    idx[i]=pts.size()+i;
	}
	if (group->add(p[0], p[1], p[2]) && emit) {
	    pts.add(p[0]);
	    pts.add(p[1]);
	    pts.add(p[2]);
	    elems.add(new TSElement(idx[0], idx[1], idx[2]));
	}
    }
    return(tcase->nbrs);
}

#if 0
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
#endif

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
    Array1<TSElement *>* elems=0;
    Array1<Point>* pts=0;
    if (emit) {
	elems=&all_elems[proc];
	pts=&all_pts[proc];
    }

    for(int i=sx;i<ex;i++){
	for(int j=0;j<ny-1;j++){
	    for(int k=0;k<nz-1;k++){
		iso_cubeTS(i, j, k, the_isoval, tris, *elems, *pts);
	    }
	    if(sp && abort_flag)
		return;
	}	
    }
    myOuterExtract.stop();

    if (emit) {
	barrier.wait(np);

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

	int i;
	for(i=0;i<all_pts[proc].size();i++)
	    composite_surf->points[i+start_pt]=all_pts[proc][i];
	
	for(i=0;i<all_elems[proc].size();i++){
	    TSElement* e=all_elems[proc][i];
	    e->i1 += start_pt; 
	    e->i2 += start_pt;
	    e->i3 += start_pt;
	    composite_surf->elements[i+start_elem]=e;
	}
    }

    myThreadTime.stop();     
    outerExtract.add(myOuterExtract.time());
    timer.add(myThreadTime.time());
}

int IsoSurfaceDW::iso_cubeHash(int i, int j, int k, double isoval, 
    GeomTrianglesP* group, Array1<TSElement*>& elems, 
    Array1<Point>& pts, MapIntInt* hash, MapIntInt* Bdry,
    int first, int last)
{
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
    MapIntInt::iterator iter;
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
		    pidx[ii] = (*Bdry)[e];
		    if (pidx[ii] % 4 != 2) {
			(*Bdry)[e] = pidx[ii] + 1;
		    } else
		        Bdry->erase(e);
		    pidx[ii] = -2 - (pidx[ii]>>2);
		} else if (first && (v1==3 || v1==7 || v1==8 || v1==10)) {
		    iter = Bdry->find(e);
		    pidx[ii] = (*iter).second;
		    if (iter != Bdry->end()) {
			(*Bdry)[e] = pidx[ii] + 1;
			pidx[ii]=pidx[ii]>>2;
		    } else {
			pidx[ii]=pts.size();
			(*Bdry)[e] = pidx[ii] << 2;
			int p0=edge_table[v1][0];
			int p1=edge_table[v1][1];
			pts.add(INTERP(p0,p1));
		    }
		} else if ((iter = hash->find(e)) != hash->end()) {
		    pidx[ii] = (*iter).second;
		    if (pidx[ii]%4 != 2) {
			(*hash)[e] = pidx[ii]+1;
		    } else
			hash->erase(e);
		    pidx[ii]=pidx[ii]>>2;
		} else {
		    pidx[ii]=pts.size();
		    (*hash)[e] = pidx[ii] << 2;
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
    
    MapIntInt* hash = new MapIntInt;
    MapIntInt* BdryFirst = all_bdryHashes[proc];
    MapIntInt* BdryLast;
    
    if (proc != np-1) {
	BdryLast=all_bdryHashes[proc+1];    
    }

    int proc0=0;	
    if (proc==0) proc0=1;
    int procN=0;
    if (proc==(np-1)) procN=1;

    int j;
    for(j=0;j<ny-1;j++){
	for(int k=0; k<nz-1; k++){
	    iso_cubeHash(sx, j, k, the_isoval, tris, *elems, *pts, hash,
			 BdryFirst, !proc0, 0);
	}
    }
    if (!proc0) {
	all_sems[proc-1]->up();
    }
    int i;
    for(i=sx+1;i<ex-1;i++){
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
	//int ii1, ii2, ii3;
	//ii1=e->i1; ii2=e->i2; ii3=e->i3;
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
    else np=Min(Thread::numProcessors(), ((rgbase->nx-2)/blockSize)+1);

    cerr << "Parallel extraction with Hashing -- using "<<np<<" processors, blocksize="<<blockSize<<"  emit="<<emit<<"\n";

    // build them as separate surfaces...
    int i;
    for (i=0; i<all_elems.size(); i++) 
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
	all_sems[i] = new Semaphore("IsoSurfaceDW reg_grid_hash semaphore", 0);
	all_bdryHashes[i] = new MapIntInt;
    }
    
    outerExtract.clear();
    innerExtract.clear();
    lace.clear();

    Thread::parallel(Parallel<IsoSurfaceDW>(this, &IsoSurfaceDW::parallel_reg_grid_hash),
		     np, true);

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
	cerr << "Warning - using marching cubes, vertices won't be shared!\n";
    }
    WallClockTimer wct;
    wct.start();
    blockSize=tclBlockSize.get();
    if (sing) np=1;	
    else np=Min(Thread::numProcessors(), ((rgbase->nx-2)/blockSize)+1);
    cerr << "Parallel extraction -- using " << np
	 << " processors, blocksize=" << blockSize
	 << "  emit=" << emit << "\n";

    int i;
    if (emit) {
	for (i=0; i<all_elems.size(); i++) 
	    all_elems[i].resize(0);
	for (i=0; i<all_pts.size(); i++)
	    all_pts[i].resize(0);
	all_elems.resize(np);
	all_pts.resize(np);
	start_pts.resize(np);
	start_elems.resize(np);
    }

    all_tris.resize(np);
    for (i=0; i<np; i++) {
	all_tris[i] = new GeomTrianglesP();
    }
    
    outerExtract.clear();

    Thread::parallel(Parallel<IsoSurfaceDW>(this,
      &IsoSurfaceDW::parallel_reg_grid), np, true);

    for (i=0; i<np; i++) {
	if (all_tris[i]->size())
	    maingroup->add(all_tris[i]);
    }
    
    wct.stop();
    cerr << "Total outerExtract: "<<outerExtract.time()<<"\n";
    cerr << "TOTAL: "<<outerExtract.time()<<"\n";

    if (emit) {
	int total_pts=composite_surf->points.size();
	int total_tris=composite_surf->elements.size();
	if ((total_pts-2)*2 != total_tris) {
	    cerr << "NOT A SINGLE SURFACE --  # pts: "<<total_pts<<"   # tris: "<<total_tris<<"\n";
	    } else {
		cerr << "SINGLE SURFACE FOUND\n";
	    }
    }

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
    int j;
    for(j=0;j<ny-1;j++){
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
    int i;
    for(i=sx+1; i<ex-1; i++){
        int k;
	for(k=0; k<nz-1; k++){
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
	//int ii1, ii2, ii3;
	//ii1=e->i1; ii2=e->i2; ii3=e->i3;
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
    else np=Min(Thread::numProcessors(), ((rgbase->nx-2)/blockSize)+1);
//    np=Min(Thread::numProcessors(), ((rgbase->nx-2)/blockSize)+1);
    cerr << "Parallel extraction with cache rings -- using "<<np<<" processors, blocksize="<<blockSize<<"  emit="<<emit<<"\n";

    // build them as separate surfaces...
    int i;
    for (i=0; i<all_elems.size(); i++) 
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
	all_sems[i] = new Semaphore("IsoSurfaceDW iso_reg_grid_rings semaphore", 0);
	all_bdryRings[i] = new Ring<int>(rgbase->ny*rgbase->nz);
    }
    
    outerExtract.clear();
    innerExtract.clear();
    lace.clear();

    Thread::parallel(Parallel<IsoSurfaceDW>(this, &IsoSurfaceDW::parallel_reg_grid_rings),
		     np, true);

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

void IsoSurfaceDW::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "log") {
	if (field.get_rep()) {
	    reset_vars();
	    double iso = isoval.get();
	    double min, max;
	    field->get_minmax(min, max);
	    ostringstream str;
	    if (logTCL.get()) {
		if (min<1e-15) min=1e-15;
		if (max<1e-14) max=1e-14;
		min = log(min);
		max = log(max);
		iso = log(iso);
	    } else
		iso = exp(iso);
	    str << id << " set_minmax " << min << " " << max;
	    TCL::execute(str.str().c_str());
	    isoval.set(iso);
	}
    } else {
        Module::tcl_command(args, userdata);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.11  2000/03/17 18:47:05  dahart
// Included STL map header files where I forgot them, and removed less<>
// parameter from map declarations
//
// Revision 1.10  2000/03/17 09:27:33  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.9  2000/03/13 09:15:11  dmw
// fixed a problem with surfaceless MC
//
// Revision 1.8  2000/03/11 00:39:55  dahart
// Replaced all instances of HashTable<class X, class Y> with the
// Standard Template Library's std::map<class X, class Y, less<class X>>
//
// Revision 1.7  2000/03/10 09:09:34  dmw
// fixed SurfToGeom to create vertex normals (smooth surfaces), and IsoSurfaceDW to: autoupdate, generate surfaces for MC, and support log isovals
//
// Revision 1.6  1999/10/07 02:07:07  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/29 00:46:47  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.4  1999/08/25 03:48:09  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/18 20:20:09  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:51  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:15  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:59  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//




