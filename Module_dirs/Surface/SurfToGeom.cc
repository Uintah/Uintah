
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
#include <Dataflow/ModuleList.h>
#include <Datatypes/Colormap.h>
#include <Datatypes/ColormapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/Surface.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/TriSurface.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geom/Group.h>
#include <Geom/Tri.h>
#include <Geom/VCTri.h>
#include <TCL/TCLvar.h>


class SurfToGeom : public Module {
    SurfaceIPort* isurface;
    ScalarFieldIPort* ifield;
    ColormapIPort* icmap;
    GeometryOPort* ogeom;

    TCLdouble range_min;
    TCLdouble range_max;
    double old_smin, old_smax;
    int have_sf, have_cm;

    void surf_to_geom(const SurfaceHandle&, GeomGroup*);
public:
    SurfToGeom(const clString& id);
    SurfToGeom(const SurfToGeom&, int deep);
    virtual ~SurfToGeom();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_SurfToGeom(const clString& id)
{
    return new SurfToGeom(id);
}

static RegisterModule db1("Surfaces", "SurfToGeom", make_SurfToGeom);
static RegisterModule db2("Visualization", "SurfToGeom",
			  make_SurfToGeom);
static RegisterModule db3("Dave", "SurfToGeom", make_SurfToGeom);

SurfToGeom::SurfToGeom(const clString& id)
: Module("SurfToGeom", id, Filter), range_min("range_min", id, this),
  range_max("range_max", id, this)
{
    // Create the input port
    isurface=new SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    ifield=new ScalarFieldIPort(this, "ScalarField", ScalarFieldIPort::Atomic);
    add_iport(ifield);
    icmap = new ColormapIPort(this, "ColorMap", ColormapIPort::Atomic);
    add_iport(icmap);
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    old_smin=old_smax=0;
}

SurfToGeom::SurfToGeom(const SurfToGeom&copy, int deep)
: Module(copy, deep), range_min("range_min", id, this),
  range_max("range_max", id, this)
{
    NOT_FINISHED("SurfToGeom::SurfToGeom");
}

SurfToGeom::~SurfToGeom()
{
}

Module* SurfToGeom::clone(int deep)
{
    return new SurfToGeom(*this, deep);
}

void SurfToGeom::execute()
{
    SurfaceHandle surf;
    if (!isurface->get(surf))
	return;

    ColormapHandle cmap;
    int have_cm=icmap->get(cmap);
    ScalarFieldHandle sfield;

    int have_sf=ifield->get(sfield);
    double smin, smax;
    sfield->get_minmax(smin, smax);
    if(old_smin != smin || old_smax != smax){
	range_min.set(smin);
	range_max.set(smax);
	old_smin=smin;
	old_smax=smax;
    }
    smin=range_min.get();
    smax=range_max.get();

    GeomGroup* group = new GeomGroup;
    TriSurface* ts=surf->getTriSurface();

    if(ts){
	for (int i=0; i<ts->elements.size(); i++) {
	    if (have_cm && have_sf) {
		double interp;
		MaterialHandle mat1,mat2,mat3;
		int ok=1;
		if (sfield->interpolate(ts->points[ts->elements[i]->i1], 
				       interp))
		    mat1=cmap->lookup(interp, smin, smax);
		else ok=0;
		if (sfield->interpolate(ts->points[ts->elements[i]->i2], 
				       interp))
		    mat2=cmap->lookup(interp, smin, smax);
		else ok=0;
		if (sfield->interpolate(ts->points[ts->elements[i]->i3], 
				       interp))
		    mat3=cmap->lookup(interp, smin, smax);
		else ok=0;
		if (ok) {
		    group->add(new GeomVCTri(ts->points[ts->elements[i]->i1], 
					     ts->points[ts->elements[i]->i2],
					     ts->points[ts->elements[i]->i3],
					     mat1, mat2, mat3));
		} else {
			cerr << "One of the points was out of the field.\n";
		    }
	    } else {
		group->add(new GeomTri(ts->points[ts->elements[i]->i1], 
				       ts->points[ts->elements[i]->i2],
				       ts->points[ts->elements[i]->i3]));
	    }
	}
    } else {
	error("Unknown representation for Surface in SurfToGeom");
    }
    GeomObj* topobj=group;
    if (surf->name == "sagital.scalp") {
	topobj=new GeomMaterial(group, new Material(Color(0,0,0),
						    Color(0,.6,0), 
						    Color(.5,.5,.5), 20));
    }
    ogeom->delAll();
    ogeom->addObj(topobj, surf->name);
}
