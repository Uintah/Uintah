
/*
 *  SurfToGeom.cc:  Convert a surface into geoemtry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/ColorMap.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/Surface.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/TriSurface.h>
#include <Datatypes/ScalarTriSurface.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geom/Group.h>
#include <Geom/Sphere.h>
#include <Geom/Tri.h>
#include <Geom/Triangles.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class SurfToGeom : public Module {
    SurfaceIPort* isurface;
    ScalarFieldIPort* ifield;
    ColorMapIPort* icmap;
    GeometryOPort* ogeom;

    TCLdouble range_min;
    TCLdouble range_max;
    TCLint best;
    TCLint invert;
    int have_sf, have_cm;

    void surf_to_geom(const SurfaceHandle&, GeomGroup*);
public:
    SurfToGeom(const clString& id);
    SurfToGeom(const SurfToGeom&, int deep);
    virtual ~SurfToGeom();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SurfToGeom(const clString& id)
{
    return scinew SurfToGeom(id);
}
};

SurfToGeom::SurfToGeom(const clString& id)
: Module("SurfToGeom", id, Filter), range_min("range_min", id, this),
  range_max("range_max", id, this), best("best", id, this),
  invert("invert", id, this)
{
    // Create the input port
    isurface=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    ifield=scinew ScalarFieldIPort(this, "ScalarField", ScalarFieldIPort::Atomic);
    add_iport(ifield);
    icmap = scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(icmap);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

SurfToGeom::SurfToGeom(const SurfToGeom&copy, int deep)
: Module(copy, deep), range_min("range_min", id, this),
  range_max("range_max", id, this), best("best", id, this),
  invert("invert", id, this)
{
    NOT_FINISHED("SurfToGeom::SurfToGeom");
}

SurfToGeom::~SurfToGeom()
{
}

Module* SurfToGeom::clone(int deep)
{
    return scinew SurfToGeom(*this, deep);
}

MaterialHandle outmatl(new Material(Color(0,0,0), Color(0,0,0),
				    Color(0,0,0), 0));

void SurfToGeom::execute()
{
    SurfaceHandle surf;
    if (!isurface->get(surf)){
	ogeom->delAll();
	return;
    }

    ColorMapHandle cmap;
    int have_cm=icmap->get(cmap);
    ScalarFieldHandle sfield;

    int have_sf=ifield->get(sfield);
//    GeomTriangles* group = scinew GeomTriangles;
    GeomGroup* spheres = scinew GeomGroup;
    GeomTrianglesPC* PCgroup = scinew GeomTrianglesPC;
    GeomTrianglesP* Pgroup = scinew GeomTrianglesP;

    ScalarTriSurface* ss=surf->getScalarTriSurface();
    TriSurface* ts=surf->getTriSurface();
    PointsSurface* ps=surf->getPointsSurface();
    if (ss) {
	cerr << "Got it!\n";
//	if (have_cm && cmap->non_diffuse_constant) {
	if (have_cm) {
	    cerr << "hiya\n";
	    MaterialHandle mat1,mat2,mat3;
	    double min, max;
	    if (best.get()) {
		min=max=ss->data[0];
		for (int i=1; i<ss->data.size(); i++) {
		    double a=ss->data[i];
		    if (a<min) min=a;
		    else if (a>max) max=a;
		}
	    } else {
		min=range_min.get();
		max=range_max.get();
	    }
	    min--;max++;
	    if (invert.get()) {
		cmap->min=max;
		cmap->max=min;
	    } else {
		cmap->min=min;
		cmap->max=max;
	    }
	    cerr << "min="<<min<<"  max="<<max<<"\n";
	    double v1, v2, v3;
	    for (int i=0; i< ss->elements.size(); i++) {
		v1=ss->data[ss->elements[i]->i1];
		v2=ss->data[ss->elements[i]->i2];
		v3=ss->data[ss->elements[i]->i3];
		mat1=cmap->lookup(v1);
		mat2=cmap->lookup(v2);
		mat3=cmap->lookup(v3);
		PCgroup->add(ss->points[ss->elements[i]->i1],mat1->diffuse,
			     ss->points[ss->elements[i]->i2],mat2->diffuse,
			     ss->points[ss->elements[i]->i3],mat3->diffuse);
		
	    }
	} else {
	    cerr << "uhoh\n";
	    for (int i=0; i< ss->elements.size(); i++) {
		Pgroup->add(ss->points[ss->elements[i]->i1], 
			    ss->points[ss->elements[i]->i2],
			    ss->points[ss->elements[i]->i3]);
		if (PCgroup) {
		    delete PCgroup;
		    PCgroup = 0;
		}
	    }
	}
    } else if (ts) {
	int ix=0;;
	for (int i=0; i< ts->elements.size(); i++) {
	    if (have_cm && have_sf) {
		double interp;
		MaterialHandle mat1,mat2,mat3;
		int ok=1;
		if (sfield->interpolate(ts->points[ts->elements[i]->i1], 
					interp, ix, 1.e-4, 1.e-4)){
		    mat1=cmap->lookup(interp);
		} else {
		    ix=0;
		    if (sfield->interpolate(ts->points[ts->elements[i]->i1], 
					    interp, ix, 1.e-4, 30.)) {
			mat1=cmap->lookup(interp);
		    } else {
			mat1=outmatl; //ok=0;
		    }
		}
		if (sfield->interpolate(ts->points[ts->elements[i]->i2], 
				       interp, ix, 1.e-4, 1.e-4)){
		    mat2=cmap->lookup(interp);
		} else {
		    ix=0;
		    if (sfield->interpolate(ts->points[ts->elements[i]->i2], 
				       interp, ix, 1.e-4, 30.)) {
			mat2=cmap->lookup(interp);
		    } else {
			mat2=outmatl; //ok=0;
		    }
		}
		if (sfield->interpolate(ts->points[ts->elements[i]->i3], 
				       interp, ix, 1.e-4, 1.e-4)){
		    mat3=cmap->lookup(interp);
		} else {
		    ix=0;
	  	    if (sfield->interpolate(ts->points[ts->elements[i]->i3], 
				       interp, ix, 1.e-4, 30.)) {
			mat3=cmap->lookup(interp);
		    } else {
			mat3=outmatl; //ok=0;
		    }
		}
		if (ok) {
		  if (cmap->non_diffuse_constant) {
		    PCgroup->add(ts->points[ts->elements[i]->i1], mat1->diffuse,
			       ts->points[ts->elements[i]->i2],mat2->diffuse,
			       ts->points[ts->elements[i]->i3],mat3->diffuse);
		    
		  }
		  else	
		Pgroup->add(ts->points[ts->elements[i]->i1], 
			    ts->points[ts->elements[i]->i2],
			    ts->points[ts->elements[i]->i3]);
//		    Pgroup->add(ts->points[ts->elements[i]->i1], mat1,
//			       ts->points[ts->elements[i]->i2],mat2,
//			       ts->points[ts->elements[i]->i3],mat3);
		} else {
		    cerr << "One of the points was out of the field.\n";
		}
	    } else {
		Pgroup->add(ts->points[ts->elements[i]->i1], 
			    ts->points[ts->elements[i]->i2],
			    ts->points[ts->elements[i]->i3]);
		if (PCgroup) {
		    delete PCgroup;
		    PCgroup = 0;
		}
	    }
	}
    } else if(ps) {
	Array1<NodeHandle> nodes;
	ps->get_surfnodes(nodes);
	for (int ii=0; ii<nodes.size(); ii++) {
	    spheres->add(scinew GeomSphere(nodes[ii]->p, 5));
	}
    } else {
	error("Unknown representation for Surface in SurfToGeom");
    }
//    GeomObj* topobj=group;
    GeomGroup* ngroup = scinew GeomGroup;

    if (PCgroup && PCgroup->size())
	ngroup->add(PCgroup);
    if (Pgroup && Pgroup->size())
	ngroup->add(Pgroup);
    if (spheres && spheres->size())
	ngroup->add(spheres);
#if 0
    // what is this for????
    if (surf->name == "sagital.scalp") {
	topobj=scinew GeomMaterial(group, scinew Material(Color(0,0,0),
						    Color(0,.6,0), 
						    Color(.5,.5,.5), 20));
    }
#endif
    ogeom->delAll();
//    ogeom->addObj(topobj, surf->name);
    ogeom->addObj(ngroup,surf->name);
}
