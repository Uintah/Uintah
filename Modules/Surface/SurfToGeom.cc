
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
#include <Datatypes/SurfTree.h>
#include <Datatypes/Surface.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/TriSurface.h>
#include <Datatypes/ScalarTriSurface.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geom/Group.h>
#include <Geom/Pt.h>
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

    Array1<MaterialHandle> c;

    TCLdouble range_min;
    TCLdouble range_max;
    TCLdouble rad;
    TCLint best;
    TCLint named;
    TCLint invert;
    TCLint nodes;
    TCLint resol;
    TCLstring ntype;
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
}

SurfToGeom::SurfToGeom(const clString& id)
: Module("SurfToGeom", id, Filter), range_min("range_min", id, this),
  range_max("range_max", id, this), best("best", id, this),
  invert("invert", id, this), nodes("nodes", id, this),
  ntype("ntype", id, this), rad("rad", id, this), resol("resol", id, this),
  named("named", id, this)
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
    c.resize(7);
    c[0]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.1),Color(.5,.5,.5),20);
    c[1]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.1),Color(.5,.5,.5),20);
    c[2]=scinew Material(Color(.2,.2,.2),Color(.1,.1,.7),Color(.5,.5,.5),20);
    c[3]=scinew Material(Color(.2,.2,.2),Color(.7,.7,.1),Color(.5,.5,.5),20);
    c[4]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.7),Color(.5,.5,.5),20);
    c[5]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.7),Color(.5,.5,.5),20);
    c[6]=scinew Material(Color(.2,.2,.2),Color(.6,.6,.6),Color(.5,.5,.5),20);
}

SurfToGeom::SurfToGeom(const SurfToGeom&copy, int deep)
: Module(copy, deep), range_min("range_min", id, this),
  range_max("range_max", id, this), best("best", id, this),
  invert("invert", id, this), nodes("nodes", id, this),
  ntype("ntype", id, this), rad("rad", id, this), resol("resol", id, this),
  named("named", id, this)
{
    NOT_FINISHED("SurfToGeom::SurfToGeom");
}

SurfToGeom::~SurfToGeom()
{
	ogeom->delAll();
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
    update_state(NeedData);
    reset_vars();
    if (!isurface->get(surf)){
	ogeom->delAll();
	return;
    }
    update_state(JustStarted);
    reset_vars();
    ColorMapHandle cmap;
    int have_cm=icmap->get(cmap);
    ScalarFieldHandle sfield;

    int have_sf=ifield->get(sfield);
//    GeomTriangles* group = scinew GeomTriangles;
    GeomGroup* spheres = scinew GeomGroup;
    GeomTrianglesPC* PCgroup = scinew GeomTrianglesPC;
    GeomTrianglesP* Pgroup = scinew GeomTrianglesP;
    Array1<GeomTrianglesP* > PMgroup;
    GeomPts* ptsGroup=scinew GeomPts(0);

    ScalarTriSurface* ss=surf->getScalarTriSurface();
    TriSurface* ts=surf->getTriSurface();
    PointsSurface* ps=surf->getPointsSurface();
    SurfTree* st=surf->getSurfTree();

    reset_vars();
    int sph;
    if (ntype.get() == "spheres") sph=1; else sph=0;
    double radius=rad.get();
    int res=resol.get();

//    cerr << "sph="<<sph<<"   radius="<<radius<<"   resol="<<res<<" \n";

    if (ss) {
//	if (have_cm && cmap->non_diffuse_constant) {
	if (have_cm) {
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
	    if (nodes.get()) {
		if (!sph) { 
		    ptsGroup=new GeomPts(3*ss->points.size());
		    ptsGroup->pts.resize(3*ss->points.size());
		}		
		for (int i=0; i<ss->points.size(); i++) {
		    if (sph) {
			spheres->add(scinew GeomMaterial(scinew GeomSphere(ss->points[i], radius, res, res), cmap->lookup(ss->data[i])));
		    } else {
			Point newP=ss->points[i];
			ptsGroup->pts[i*3]=newP.x();
			ptsGroup->pts[i*3+1]=newP.y();
			ptsGroup->pts[i*3+2]=newP.z();
		    }
		}
	    } else {
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
	    }
	} else {
	    cerr << "no colormap\n";
	    if (nodes.get()) {
		if (!sph) { 
		    ptsGroup=new GeomPts(3*ss->points.size());
		    ptsGroup->pts.resize(3*ss->points.size());
		}		
		for (int i=0; i<ss->points.size(); i++) {
		    if (sph) {
			spheres->add(scinew GeomSphere(ss->points[i], radius, res, res));
		    } else {
			Point newP=ss->points[i];
			ptsGroup->pts[i*3]=newP.x();
			ptsGroup->pts[i*3+1]=newP.y();
			ptsGroup->pts[i*3+2]=newP.z();
		    }
		}
	    } else {
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
	}
    } else if (ts) {
	if (nodes.get()) {
	    surf->monitor.read_lock();
	    if (!sph) {
		ptsGroup=new GeomPts(3*ts->points.size());
		ptsGroup->pts.resize(3*ts->points.size());
	    }		
	    for (int i=0; i<ts->points.size(); i++) {
		if (sph) {
		    spheres->add(scinew GeomSphere(ts->points[i], radius, res, res));
		} else {		      
		    Point newP=ts->points[i];
		    ptsGroup->pts[i*3]=newP.x();
		    ptsGroup->pts[i*3+1]=newP.y();
		    ptsGroup->pts[i*3+2]=newP.z();
		}
	    }
	    surf->monitor.read_unlock();
	} else {
	    int ix=0;
//	    cerr << "ts->elements.size()="<<ts->elements.size();
//	    cerr << "   ts->points.size()="<<ts->points.size();
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
	}
    } else if(ps) {
	Array1<NodeHandle> nh;
	ps->get_surfnodes(nh);

//	cerr << "got "<<nh.size() <<" surfnodes.\n";
	if (!sph) { 
	    ptsGroup=new GeomPts(3*nh.size());
	    ptsGroup->pts.resize(3*nh.size());
	}		
	for (int i=0; i<nh.size(); i++) {
	    if (sph) {
		spheres->add(scinew GeomSphere(nh[i]->p, radius, res, res));
	    } else {	     
		Point newP=nh[i]->p;
		ptsGroup->pts[i*3]=newP.x();
		ptsGroup->pts[i*3+1]=newP.y();
		ptsGroup->pts[i*3+2]=newP.z();
	    }
	}
    } else if (st) {
	if (st->surfNames.size() != st->surfEls.size()) 
	    st->surfNames.resize(st->surfEls.size());
	if (nodes.get()) {
	    if (!sph) { 
		ptsGroup=new GeomPts(3*st->points.size());
		ptsGroup->pts.resize(3*st->points.size());
	    }		
	    for (int i=0; i<st->points.size(); i++) {
		if (sph) {
		    spheres->add(scinew GeomSphere(st->points[i], radius, res, res));
		} else {	     
		    Point newP=st->points[i];
		    ptsGroup->pts[i*3]=newP.x();
		    ptsGroup->pts[i*3+1]=newP.y();
		    ptsGroup->pts[i*3+2]=newP.z();
		}
	    }
	} else {
	    PMgroup.resize(st->surfEls.size());
	    for (int i=0; i<PMgroup.size(); i++)
		PMgroup[i] = scinew GeomTrianglesP;
	    for (i=0; i< st->surfEls.size(); i++) {
		if (!st->surfEls[i].size()) continue;
		for (int j=0; j < st->surfEls[i].size(); j++) {
		    int elIdx=st->surfEls[i][j];
		    PMgroup[i]->add(st->points[st->elements[elIdx]->i1],
				    st->points[st->elements[elIdx]->i2],
				    st->points[st->elements[elIdx]->i3]);
		}	
	    }	
	}
    } else {
	error("Unknown representation for Surface in SurfToGeom");
    }
    GeomGroup* ngroup = scinew GeomGroup;

    if (PCgroup && PCgroup->size())
	ngroup->add(PCgroup);
    if (Pgroup && Pgroup->size())
	ngroup->add(Pgroup);
    if (spheres && spheres->size())
	ngroup->add(scinew GeomMaterial(spheres, c[2]));
    if (ptsGroup && ptsGroup->pts.size())
	ngroup->add(ptsGroup);
    ogeom->delAll();
    if (ngroup->size()) 
	ogeom->addObj(scinew GeomMaterial(ngroup, c[0]), surf->name);

    int nmd = named.get();
    for (int i=0; i<PMgroup.size(); i++)
	if (PMgroup[i]->size()) {
	    if (st && st->surfNames[i] != clString("")) {
		ogeom->addObj(scinew GeomMaterial(PMgroup[i], c[(st->matl[i])%7]), st->surfNames[i]);
	    } else {
		if (st && !nmd)
		    ogeom->addObj(scinew GeomMaterial(PMgroup[i], c[(st->matl[i])%7]), clString("Surface ")+to_string(i));
	    }
	}
}
