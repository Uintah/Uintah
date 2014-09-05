//static char *id="@(#) $Id$";

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

#include <SCICore/Datatypes/BasicSurfaces.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/SurfTree.h>
#include <SCICore/Datatypes/Surface.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/TriSurface.h>
//#include <PSECommon/Datatypes/ScalarTriSurface.h>
#include <SCICore/Geom/GeomArrows.h>
#include <SCICore/Geom/GeomCylinder.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Datatypes;
using namespace PSECore::Dataflow;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;

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
  TCLstring range;
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
  virtual ~SurfToGeom();
  virtual void execute();
};

extern "C" Module* make_SurfToGeom(const clString& id) {
  return new SurfToGeom(id);
}

SurfToGeom::SurfToGeom(const clString& id)
  : Module("SurfToGeom", id, Filter), range_min("range_min", id, this),
    range_max("range_max", id, this), range("range", id, this),
    invert("invert", id, this), nodes("nodes", id, this),
    ntype("ntype", id, this), rad("rad", id, this), resol("resol", id, this),
    named("named", id, this), clr_r("clr-r", id, this), 
    clr_g("clr-g", id, this), clr_b("clr-b", id, this), 
    normals("normals", id, this)
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

SurfToGeom::~SurfToGeom()
{
  ogeom->delAll();
}

static MaterialHandle outmatl(new Material(Color(0,0,0), Color(0,0,0),
					   Color(0,0,0), 0));
static MaterialHandle black(new Material(Color(.2,.2,.2), Color(.2,.2,.2), 
					 Color(.5,.5,.5), 30));
static Color blackClr(.2,.2,.2);

//----------------------------------------------------------------------
void SurfToGeom::execute()
{
  SurfaceHandle surf;
  update_state(NeedData);
  reset_vars();
  if (!isurface->get(surf)){
    ogeom->delAll();
    cerr << "SurfToGeom: No Surface" << endl;
    return;
  }
  if (!surf.get_rep()) {
    cerr << "SurfToGeom: No Rep" << endl;
    return;
  }
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

  //    ScalarTriSurface* ss=surf->getScalarTriSurface();
  TriSurface* ts=surf->getTriSurface();
  PointsSurface* ps=surf->getPointsSurface();
  SurfTree* st=surf->getSurfTree();

  reset_vars();
  int sph=0;
  int dsk=0;
  if (ntype.get() == "spheres" || ntype.get() == "disks") sph=1;
  if (ntype.get() == "disks") dsk=1;
  double radius=rad.get();
  if (radius < 0.0000001) radius=0.0000001;
  int res=resol.get();

  //    cerr << "sph="<<sph<<"   radius="<<radius<<"   resol="<<res<<" \n";

  MaterialHandle mpick(new Material(Color(.2,.2,.2), 
				    Color(clr_r.get(), clr_g.get(), 
					  clr_b.get()),
				    Color(.2,.2,.2), 20));


  if (ts) {
    if (ts->normals.size() != ts->bcIdx.size()) dsk=0;
    if (nodes.get()) {
      surf->monitor.readLock();
      if (!sph) {
	ptsGroup=new GeomPts(3*ts->points.size());
	ptsGroup->pts.resize(3*ts->points.size());
      }		
      if (sph) {
	if (have_cm && (have_sf || ts->bcVal.size())) {
	  if (have_sf) {
	    double min, max;
	    if (range.get() == "best") {
	      sfield->get_minmax(min,max);
	    } else if (range.get() == "manual") {
	      min=range_min.get();
	      max=range_max.get();
	    } else { // range.get() == "cmap"
	      min=cmap->min;
	      max=cmap->max;
	    }
	    //				min--;max++;
	    if (invert.get()) {
	      cmap->min=max;
	      cmap->max=min;
	    } else {
	      cmap->min=min;
	      cmap->max=max;
	    }
	    MaterialHandle omatl = cmap->lookup(max+min/2);
	    cerr << "USING SF VALS...\n";
	    for (int i=0; i< ts->points.size(); i++) {
	      double interp;
	      MaterialHandle mat1;
	      //			    int ok=1;
	      int ix=0;
	      if (sfield->interpolate(ts->points[i],
				      interp, ix, 1.e-4, 1.e-4)){
		mat1=cmap->lookup(interp);
	      } else {
		ix=0;
		if (sfield->interpolate(ts->points[i],
					interp, ix, 1.e-4, 30.)) {
		  mat1=cmap->lookup(interp);
		} else {
		  mat1=omatl; //ok=0;
		}
	      }
	      spheres->add(scinew GeomMaterial(scinew GeomSphere(ts->points[i], radius, res, res), mat1));
	      if (ts->normType==TriSurface::PointType && 
		  ts->normals.size()>i) {
		GeomArrows* ga=scinew 
		  GeomArrows(.1);
		// cerr << "radius = "<<radius<<"  ts->points[i]=";
		// cerr << ts->points[i]<<"  ts->normals[i]=";
		// cerr << ts->normals[i]<<"\n";
		ga->add(ts->points[i]-(ts->normals[i]*2), 
			ts->normals[i]*4);
		ga->set_matl(mat1, mat1, mat1);
		spheres->add(ga);
	      }
	    }
	  } else {	// !have_sf
	    double min, max;
	    if (range.get() == "best") {
	      min=max=ts->bcVal[0];
	      for (int i=1; i<ts->bcVal.size(); i++) {
		double a=ts->bcVal[i];
		if (a<min) min=a;
		else if (a>max) max=a;
	      }
	    } else if (range.get() == "manual") {
	      min=range_min.get();
	      max=range_max.get();
	    } else { // range.get() == "cmap"
	      min=cmap->min;
	      max=cmap->max;
	    }
	    //				min--;max++;
	    if (invert.get()) {
	      cmap->min=max;
	      cmap->max=min;
	    } else {
	      cmap->min=min;
	      cmap->max=max;
	    }
	    //	cerr << "min="<<min<<"  max="<<max<<"\n";

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
	      // spheres->add(scinew 
	      //    GeomMaterial(scinew GeomSphere(ts->points[i], radius, 
	      //                                   res, res), mat1));
	      if (dsk && (used[i1]!=-1) && ts->normType==TriSurface::PointType && ts->normals.size()>i) {
		Vector v(ts->normals[i]);
		v.normalize();
		v*=radius/6;
		spheres->add(scinew GeomMaterial(scinew GeomCappedCylinder(ts->points[i]+v, ts->points[i]-v, radius, res, 1, 1), mat1));
	      } else
		spheres->add(scinew GeomMaterial(scinew GeomSphere(ts->points[i], radius, res, res), mat1));
#if 0
	      if (ts->normType==TriSurface::PointType && 
		  ts->normals.size()>i) {
		GeomArrows* ga=scinew 
		  GeomArrows(.1);
		ts->normals[i].normalize();
		// cerr << "radius = "<<radius<<"  ts->points[i]=";
		// cerr << ts->points[i]<<"  ts->normals[i]=";
		// cerr << ts->normals[i]<<"\n";
		ga->add(ts->points[i]-(ts->normals[i]*2*radius), 
			ts->normals[i]*6*radius);
		ga->set_matl(mat1, mat1, mat1);
		spheres->add(ga);
	      }
#endif
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
      surf->monitor.readUnlock();
    } else {	// draw triangles! (not nodes)
      int nrm=normals.get();
      if (nrm) ts->bldNormals(TriSurface::PointType);

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
	  if (range.get() == "best") {
	    sfield->get_minmax(min,max);
	    cerr << "SurfToGeom - min="<<min<<"\n";
	    cerr << "SurfToGeom - max="<<max<<"\n";
	    range_min.set(min);
	    range_max.set(max);
	  } else if (range.get() == "manual") {
	    min=range_min.get();
	    max=range_max.get();
	  } else { // range.get() == "cmap"
	    min=cmap->min;
	    max=cmap->max;
	  }
	  //				min--;max++;
	  if (invert.get()) {
	    cmap->min=max;
	    cmap->max=min;
	  } else {
	    cmap->min=min;
	    cmap->max=max;
	  }
	  MaterialHandle omatl = cmap->lookup(max+min/2);
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
		mat1=omatl; //ok=0;
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
		mat2=omatl; //ok=0;
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
		mat3=omatl; //ok=0;
	      }
	    }
	    if (ok) {
	      //    if (cmap->non_diffuse_constant) {
	      int i1=ts->elements[i]->i1;
	      int i2=ts->elements[i]->i2;
	      int i3=ts->elements[i]->i3;
	      if (nrm)
		VPCgroup->add(ts->points[i1], 
				//	ts->normals[i*3], 
			      ts->normals[i1], 
			      mat1->diffuse,
			      ts->points[i2], 
				//	ts->normals[i*3+1], 
			      ts->normals[i2], 
			      mat2->diffuse,
			      ts->points[i3], 
				//	ts->normals[i*3+2], 
			      ts->normals[i3], 
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
	  //	cerr << "Using SurfToGeom w/ TriSurf, BCs and cmap!\n";
	  double min, max;
	  if (range.get() == "best") {
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
	  } else if (range.get() == "manual") {
	    min=range_min.get();
	    max=range_max.get();
	  } else { // range.get() == "cmap"
	    min=cmap->min;
	    max=cmap->max;
	  }
	  //	min--;max++;
	  if (invert.get()) {
	    cmap->min=max;
	    cmap->max=min;
	  } else {
	    cmap->min=min;
	    cmap->max=max;
	  }
	  //	cerr << "min="<<min<<"  max="<<max<<"\n";

	  Array1<int> used;
	  Array1<Color> clr;
	  if (ts->valType == TriSurface::NodeType) {
	    used.resize(ts->points.size());
	    clr.resize(ts->points.size());
	  } else {
	    used.resize(ts->elements.size());
	    clr.resize(ts->elements.size());
	  }
	  used.initialize(-1);
	  for (int q=0; q<ts->bcIdx.size(); q++) {
	    used[ts->bcIdx[q]]=q;
	    clr[ts->bcIdx[q]]=(cmap->lookup(ts->bcVal[q]))->diffuse;
	  }
	  for (int i=0; i< ts->elements.size(); i++) {
	    if (!(i%500)) update_progress(i,ts->elements.size());
	    //			double interp;
	    Color mat1, mat2, mat3;
	    int i1=ts->elements[i]->i1; 
	    int i2=ts->elements[i]->i2;
	    int i3=ts->elements[i]->i3;
	    if (ts->valType == TriSurface::NodeType) {
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
	    } else {
	      if (used[i]!=-1) {
		mat1=clr[i];
	      } else {
		mat1=blackClr;
	      }
	      mat3=mat2=mat1;
	    }
	    if (nrm)
	      VPCgroup->add(ts->points[i1], 
			    //	 ts->normals[i*3], 
			    ts->normals[i1], 
			    mat1,
			    ts->points[i2], 
			    //	 ts->normals[i*3+1], 
			    ts->normals[i2], 
			    mat2,
			    ts->points[i3], 
			    //	 ts->normals[i*3+2], 
			    ts->normals[i3], 
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
	    //VPgroup->add(ts->points[i1], ts->normals[i*3],
	    //	     ts->points[i2], ts->normals[i*3+1],
	    //	     ts->points[i3], ts->normals[i*3+2]);
	    VPgroup->add(ts->points[i1], ts->normals[i1],
			 ts->points[i2], ts->normals[i2],
			 ts->points[i3], ts->normals[i3]);
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
	int i;
	for (i=0; i<VPMgroup.size(); i++)
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
	int i;
	for (i=0; i<PMgroup.size(); i++)
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
  ogeom->delAll();
  if (ngroup->size()) 
    ogeom->addObj(scinew GeomMaterial(ngroup, mpick), surf->name);

  int nmd = named.get();
  int i;
  for (i=0; i<PMgroup.size(); i++)
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.12.2.4  2000/11/01 23:02:58  mcole
// Fix for previous merge from trunk
//
// Revision 1.12.2.2  2000/10/26 10:03:43  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.12.2.1  2000/09/28 03:16:09  mcole
// merge trunk into FIELD_REDESIGN branch
//
// Revision 1.14  2000/10/29 04:34:56  dmw
// BuildFEMatrix -- ground an arbitrary node
// SolveMatrix -- when preconditioning, be careful with 0's on diagonal
// MeshReader -- build the grid when reading
// SurfToGeom -- support node normals
// IsoSurface -- fixed tet mesh bug
// MatrixWriter -- support split file (header + raw data)
//
// LookupSplitSurface -- split a surface across a place and lookup values
// LookupSurface -- find surface nodes in a sfug and copy values
// Current -- compute the current of a potential field (- grad sigma phi)
// LocalMinMax -- look find local min max points in a scalar field
//
// Revision 1.13  2000/08/04 19:19:44  dmw
// adding TransformSurface.cc to makefile
//
// Revision 1.12  2000/03/17 09:27:22  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.11  2000/03/11 00:39:54  dahart
// Replaced all instances of HashTable<class X, class Y> with the
// Standard Template Library's std::map<class X, class Y, less<class X>>
//
// Revision 1.10  2000/03/10 09:09:33  dmw
// fixed SurfToGeom to create vertex normals (smooth surfaces), and IsoSurfaceDW to: autoupdate, generate surfaces for MC, and support log isovals
//
// Revision 1.9  2000/03/04 00:20:17  dmw
// new bc in buildfematrix, fixed normals in surftogeom
//
// Revision 1.8  2000/02/02 05:51:56  dmw
// added new module to index.cc and fixed bugs
//
// Revision 1.7  1999/10/07 02:07:01  sparker
// use standard iostreams and complex type
//
// Revision 1.6  1999/09/16 00:38:12  dmw
// fixed TCL files for SurfToGeom and SolveMatrix and added SurfToGeom to the Makefile
//
// Revision 1.5  1999/09/05 05:32:27  dmw
// updated and added Modules from old tree to new
//
// Revision 1.4  1999/08/25 03:48:01  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/18 20:19:59  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:44  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:59  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:29  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
