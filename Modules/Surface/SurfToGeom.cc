// a better way to do this is to move most of the functionality into the
//   various surface classes.
//
// each one will have a:
//	virtual void genGeom(int nodes, int spheres, ColorMap*, ScalarField*,
//			     Array1<GeomObj*>&);
// the base Surface class will print an error -- it shouldn't be called.
//
// all this module will do is to grab the TCL vars, set the ColorMap bounds
//   if necessary, call this virtual genGeom method on the incoming surface
//   and one at a time, send the GeomObj* 's from the Array out the GeomOPort

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
#include <Geom/Arrows.h>
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
    ColorMapOPort* ocmap;
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
    TCLdouble clr_r;
    TCLdouble clr_g;
    TCLdouble clr_b;
    TCLint normals;
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
  named("named", id, this), clr_r("clr-r", id, this), clr_g("clr-g", id, this),
  clr_b("clr-b", id, this), normals("normals", id, this)
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
    ocmap=scinew ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_oport(ocmap);
    c.resize(7);
    c[0]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.1),Color(.5,.5,.5),20);
    c[1]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.1),Color(.5,.5,.5),20);
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
  named("named", id, this), clr_r("clr-r", id, this), clr_g("clr-g", id, this),
  clr_b("clr-b", id, this), normals("normals", id, this)
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

static MaterialHandle outmatl(new Material(Color(0,0,0), Color(0,0,0),
				    Color(0,0,0), 0));
static MaterialHandle black(new Material(Color(.2,.2,.2), Color(.2,.2,.2), 
				  Color(.5,.5,.5), 30));
static Color blackClr(.2,.2,.2);

void SurfToGeom::execute()
{
    SurfaceHandle surf;
    update_state(NeedData);
    reset_vars();
    if (!isurface->get(surf)){
	ogeom->delAll();
	return;
    }
    if (!surf.get_rep()) return;
    update_state(JustStarted);
    reset_vars();
    ColorMapHandle cmap;
    int have_cm=icmap->get(cmap);
    if (have_cm) cmap.detach();
    ScalarFieldHandle sfield;

    int have_sf=ifield->get(sfield);
//    GeomTriangles* group = scinew GeomTriangles;
    GeomGroup* spheres = scinew GeomGroup;
    GeomTrianglesPC* PCgroup = scinew GeomTrianglesPC;
    GeomTrianglesP* Pgroup = scinew GeomTrianglesP;
    Array1<GeomTrianglesP* > PMgroup;

    GeomTrianglesVPC* VPCgroup = scinew GeomTrianglesVPC;
    GeomTrianglesVP* VPgroup = scinew GeomTrianglesVP;
    Array1<GeomTrianglesVP* > VPMgroup;

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

    MaterialHandle mpick(new Material(Color(.2,.2,.2), 
	            Color(clr_r.get(), clr_g.get(), clr_b.get()),
		    Color(.2,.2,.2), 20));

//	HACK for 3ICIPE
//    ogeom->delAll();

    if (ss) {
	cerr << "We don't use ScalarTriSurfaces any more -- use a TriSurface\n";
	return;
    } else if (ts) {
	if (nodes.get()) {
	    surf->monitor.read_lock();
	    if (!sph) {
		ptsGroup=new GeomPts(3*ts->points.size());
		ptsGroup->pts.resize(3*ts->points.size());
	    }		
	    if (sph) {
		if (have_cm && (have_sf || ts->bcVal.size())) {
		    if (have_sf) {
			double min, max;
			if (best.get()) {
			    sfield->get_minmax(min,max);
			} else {
			    min=range_min.get();
			    max=range_max.get();
			}
			//				min--;max++;
			if (invert.get()) {
			    cmap->min=max;
			    cmap->max=min;
			} else {
			    cmap->min=min;
			    cmap->max=max;
			}

			cerr << "USING SF VALS...\n";
			for (int i=0; i< ts->points.size(); i++) {
			    double interp;
			    MaterialHandle mat1;
//			    int ok=1;
			    int ix=0;
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
			    spheres->add(scinew GeomMaterial(scinew GeomSphere(ts->points[i], radius, res, res), mat1));
			    if (ts->normType==TriSurface::PointType && 
				ts->normals.size()>i) {
				GeomArrows* ga=scinew 
				    GeomArrows(.1);
//				cerr << "radius = "<<radius<<"  ts->points[i]="<<ts->points[i]<<"  ts->normals[i]="<<ts->normals[i]<<"\n";
				ga->add(ts->points[i]-(ts->normals[i]*2), 
					ts->normals[i]*4);
				ga->set_matl(mat1, mat1, mat1);
				spheres->add(ga);
			    }
			}
		    } else {	// !have_sf
			double min, max;
			if (best.get()) {
			    min=max=ts->bcVal[0];
			    for (int i=1; i<ts->bcVal.size(); i++) {
				double a=ts->bcVal[i];
				if (a<min) min=a;
				else if (a>max) max=a;
			    }
			} else {
			    min=range_min.get();
			    max=range_max.get();
			}
			//				min--;max++;
			if (invert.get()) {
			    cmap->min=max;
			    cmap->max=min;
			} else {
			    cmap->min=min;
			    cmap->max=max;
			}
//					cerr << "min="<<min<<"  max="<<max<<"\n";

			Array1<int> used(ts->points.size());
			Array1<MaterialHandle> clr(ts->points.size());
			used.initialize(-1);
			for (int q=0; q<ts->bcIdx.size(); q++) {
			    used[ts->bcIdx[q]]=q;
			    clr[ts->bcIdx[q]]=cmap->lookup(ts->bcVal[q]);
			}
			for (int i=0; i< ts->points.size(); i++) {
//			    double interp;
			    MaterialHandle mat1;
			    int i1=i;
			    if (used[i1]!=-1) {
				mat1=clr[i1];
			    } else {
				mat1=black;
			    }
//			    HACKS FOR 3ICIPE
//			    ogeom->addObj(scinew GeomMaterial(scinew GeomSphere(ts->points[i], radius, res, res), mat1), clString("Node"+to_string(i)));
			    spheres->add(scinew GeomMaterial(scinew GeomSphere(ts->points[i], radius, res, res), mat1));
			    if (ts->normType==TriSurface::PointType && 
				ts->normals.size()>i) {
				GeomArrows* ga=scinew 
				    GeomArrows(.1);
				ts->normals[i].normalize();
//				cerr << "radius = "<<radius<<"  ts->points[i]="<<ts->points[i]<<"  ts->normals[i]="<<ts->normals[i]<<"\n";
				ga->add(ts->points[i]-(ts->normals[i]*2*radius), 
					ts->normals[i]*6*radius);
				ga->set_matl(mat1, mat1, mat1);
				spheres->add(ga);
			    }
			}
		    }
		} else {	// spheres -- no cmap
		    for (int i=0; i< ts->points.size(); i++) {
			spheres->add(scinew GeomSphere(ts->points[i], radius, res, res));
		    }
		}
	    } else {  // render nodes, but not as spheres
		for (int i=0; i<ts->points.size(); i++) {
		    Point newP=ts->points[i];
		    ptsGroup->pts[i*3]=newP.x();
		    ptsGroup->pts[i*3+1]=newP.y();
		    ptsGroup->pts[i*3+2]=newP.z();
		}
	    }
	    surf->monitor.read_unlock();
	} else {	// draw triangles! (not nodes)
	    int nrm=normals.get();
	    if (nrm) ts->bldNormals(TriSurface::VertexType);

//	    int i;
//	    for (i=0; i<ts->nodeNormals.size(); i+=100) {
//		cerr << i<<": "<<ts->nodeNormals[i]<<"\n";
//	    }
	    int ix=0;
//	    cerr << "ts->elements.size()="<<ts->elements.size();
//	    cerr << "   ts->points.size()="<<ts->points.size();
	    if (have_cm && (have_sf || ts->bcVal.size())) {
		if (have_sf) {
		    double min, max;
		    if (best.get()) {
			sfield->get_minmax(min,max);
			cerr << "SurfToGeom - min="<<min<<"\n";
			cerr << "SurfToGeom - max="<<max<<"\n";
			range_min.set(min);
			range_max.set(max);
		    } else {
			min=range_min.get();
			max=range_max.get();
		    }
		    //				min--;max++;
		    if (invert.get()) {
			cmap->min=max;
			    cmap->max=min;
		    } else {
			cmap->min=min;
			cmap->max=max;
		    }
		    cerr << "LOOKING GOOD!\n";
		    for (int i=0; i< ts->elements.size(); i++) {
			double interp;
			MaterialHandle mat1,mat2,mat3;
			int ok=1;
			ix=0;
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
			ix=0;
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
			ix=0;
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
//			    if (cmap->non_diffuse_constant) {
			    int i1=ts->elements[i]->i1;
			    int i2=ts->elements[i]->i2;
			    int i3=ts->elements[i]->i3;
			    if (nrm)
				VPCgroup->add(ts->points[i1], 
					      ts->normals[i*3], 
					      mat1->diffuse,
					      ts->points[i2], 
					      ts->normals[i*3+1], 
					      mat2->diffuse,
					      ts->points[i3], 
					      ts->normals[i*3+2], 
					      mat3->diffuse);
			    else
				PCgroup->add(ts->points[i1], mat1->diffuse,
					     ts->points[i2], mat2->diffuse,
					     ts->points[i3], mat3->diffuse);
			} else {
			    cerr << "One of the points was out of the field.\n";
			}
		    }
		} else {
//			cerr << "Using SurfToGeom w/ TriSurf, BCs and cmap!\n";
		    double min, max;
		    if (best.get()) {
			min=max=ts->bcVal[0];
			for (int i=1; i<ts->bcVal.size(); i++) {
			    double a=ts->bcVal[i];
			    if (a<min) min=a;
			    else if (a>max) max=a;
			}
			cerr << "SurfToGeom - min="<<min<<"\n";
			cerr << "SurfToGeom - max="<<max<<"\n";
			range_min.set(min);
			range_max.set(max);
		    } else {
			min=range_min.get();
			max=range_max.get();
		    }
//				min--;max++;
		    if (invert.get()) {
			cmap->min=max;
			cmap->max=min;
		    } else {
			cmap->min=min;
			cmap->max=max;
		    }
//				cerr << "min="<<min<<"  max="<<max<<"\n";

		    Array1<int> used(ts->points.size());
		    Array1<Color> clr(ts->points.size());
		    used.initialize(-1);
		    for (int q=0; q<ts->bcIdx.size(); q++) {
			used[ts->bcIdx[q]]=q;
			clr[ts->bcIdx[q]]=(cmap->lookup(ts->bcVal[q]))->diffuse;
		    }
		    for (int i=0; i< ts->elements.size(); i++) {
			if (!(i%500)) update_progress(i,ts->elements.size());
//			double interp;
			Color mat1,mat2,mat3;
			int i1=ts->elements[i]->i1; 
			int i2=ts->elements[i]->i2;
			int i3=ts->elements[i]->i3;
			if (used[i1]!=-1) {
			    mat1=clr[i1];
			} else {
			    mat1=blackClr;
			}
			if (used[i2]!=-1) {
			    mat2=clr[i2];
			} else {
			    mat2=blackClr;
			}
			if (used[i3]!=-1) {
			    mat3=clr[i3];
			} else {
			    mat3=blackClr;
			}
			if (nrm)
			    VPCgroup->add(ts->points[i1], 
					  ts->normals[i*3], 
					  mat1,
					  ts->points[i2], 
					  ts->normals[i*3+1], 
					  mat2,
					  ts->points[i3], 
					  ts->normals[i*3+2], 
					  mat3);
			else
			    PCgroup->add(ts->points[i1], mat1,
					 ts->points[i2], mat2,
					 ts->points[i3], mat3);
		    }
		}

	    } else {
		for (int i=0; i< ts->elements.size(); i++) {
		    int i1=ts->elements[i]->i1;
		    int i2=ts->elements[i]->i2;
		    int i3=ts->elements[i]->i3;
		    if (nrm)
			VPgroup->add(ts->points[i1], ts->normals[i*3],
				     ts->points[i2], ts->normals[i*3+1],
				     ts->points[i3], ts->normals[i*3+2]);
		    else
			Pgroup->add(ts->points[i1],
				    ts->points[i2],
				    ts->points[i3]);
		    if (PCgroup) {
			delete PCgroup;
			PCgroup = 0;
		    }
		    if (VPCgroup) {
			delete VPCgroup;
			VPCgroup = 0;
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
	if (nodes.get()) {
	    if (!sph) { 
		ptsGroup=new GeomPts(3*st->nodes.size());
		ptsGroup->pts.resize(3*st->nodes.size());
	    }		
	    for (int i=0; i<st->nodes.size(); i++) {
		if (sph) {
		    spheres->add(scinew GeomSphere(st->nodes[i], radius, res, res));
		} else {	     
		    Point newP=st->nodes[i];
		    ptsGroup->pts[i*3]=newP.x();
		    ptsGroup->pts[i*3+1]=newP.y();
		    ptsGroup->pts[i*3+2]=newP.z();
		}
	    }
	} else {
	    int nrm=normals.get();
	    if (nrm) {
		st->bldNormals();
		VPMgroup.resize(st->surfI.size());
		for (int i=0; i<VPMgroup.size(); i++)
		    VPMgroup[i] = scinew GeomTrianglesVP;
		for (i=0; i< st->surfI.size(); i++) {
		    if (!st->surfI[i].faces.size()) continue;
		    for (int j=0; j < st->surfI[i].faces.size(); j++) {
			int elIdx=st->surfI[i].faces[j];
			int i1=st->faces[elIdx]->i1;
			int i2=st->faces[elIdx]->i2;
			int i3=st->faces[elIdx]->i3;
			if (st->surfI.size()>i && 
			    st->surfI[i].faceOrient.size()>j && 
			    st->surfI[i].faceOrient[j])
			    VPMgroup[i]->add(st->nodes[i1], 
					     st->surfI[i].nodeNormals[i1],
					     st->nodes[i2],
					     st->surfI[i].nodeNormals[i2],
					     st->nodes[i3],
					     st->surfI[i].nodeNormals[i3]);
			else	
			    VPMgroup[i]->add(st->nodes[i1], 
					     st->surfI[i].nodeNormals[i1],
					     st->nodes[i3],
					     st->surfI[i].nodeNormals[i3],
					     st->nodes[i2],
					     st->surfI[i].nodeNormals[i2]);
		    }	
		}	
	    } else {
		PMgroup.resize(st->surfI.size());
		for (int i=0; i<PMgroup.size(); i++)
		    PMgroup[i] = scinew GeomTrianglesP;
		for (i=0; i< st->surfI.size(); i++) {
		    if (!st->surfI[i].faces.size()) continue;
		    for (int j=0; j < st->surfI[i].faces.size(); j++) {
			int elIdx=st->surfI[i].faces[j];
			int i1=st->faces[elIdx]->i1;
			int i2=st->faces[elIdx]->i2;
			int i3=st->faces[elIdx]->i3;
			if (st->surfI.size()>i && 
			    st->surfI[i].faceOrient.size()>j && 
			    st->surfI[i].faceOrient[j])
			    PMgroup[i]->add(st->nodes[i1], 
					    st->nodes[i2],
					    st->nodes[i3]);
			else	
			    PMgroup[i]->add(st->nodes[i1], 
					    st->nodes[i3],
					    st->nodes[i2]);
		    }	
		}	
	    }
	}
    } else {
	error("Unknown representation for Surface in SurfToGeom");
    }

    if (have_cm) ocmap->send(cmap);

    GeomGroup* ngroup = scinew GeomGroup;

    if (PCgroup && PCgroup->size())
	ngroup->add(PCgroup);
    if (Pgroup && Pgroup->size())
	ngroup->add(Pgroup);
    if (VPCgroup && VPCgroup->size())
	ngroup->add(VPCgroup);
    if (VPgroup && VPgroup->size())
	ngroup->add(VPgroup);
    if (spheres && spheres->size())
	ngroup->add(spheres);
    if (ptsGroup && ptsGroup->pts.size())
	ngroup->add(ptsGroup);
//    HACK FOR 3ICIPE
    ogeom->delAll();
    if (ngroup->size()) 
	ogeom->addObj(scinew GeomMaterial(ngroup, mpick), surf->name);

    int nmd = named.get();
    for (int i=0; i<PMgroup.size(); i++)
	if (PMgroup[i]->size()) {
	    if (st && st->surfI[i].name != clString("")) {
		ogeom->addObj(scinew GeomMaterial(PMgroup[i], c[(st->surfI[i].matl)%7]), st->surfI[i].name);
	    } else {
		if (st && !nmd)
		    ogeom->addObj(scinew GeomMaterial(PMgroup[i], c[(st->surfI[i].matl)%7]), clString("Surface ")+to_string(i));
	    }
	}
    for (i=0; i<VPMgroup.size(); i++)
	if (VPMgroup[i]->size()) {
	    if (st && st->surfI[i].name != clString("")) {
		ogeom->addObj(scinew GeomMaterial(VPMgroup[i], c[(st->surfI[i].matl)%7]), st->surfI[i].name);
	    } else {
		if (st && !nmd)
		    ogeom->addObj(scinew GeomMaterial(VPMgroup[i], c[(st->surfI[i].matl)%7]), clString("Surface ")+to_string(i));
	    }
	}
}
