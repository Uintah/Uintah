/*
 *  SurfToVectGeom.cc:  Convert a surface into geometry of vectors
 *
 *  Written by:
 *   Robert Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Datatypes/BasicSurfaces.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/SurfTree.h>
#include <Core/Datatypes/Surface.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/TriSurface.h>
#include <Core/Geom/GeomArrows.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/Color.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Pt.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>

#include <Packages/RobV/Core/Datatypes/MEG/VectorFieldMI.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <math.h>

namespace RobV {
using namespace SCIRun;

class SurfToVectGeom : public Module {
    SurfaceIPort* isurface;
    VectorFieldIPort* ifield;
    ColorMapIPort* icmap;
    GeometryOPort* ogeom;
  
    VectorFieldHandle vfield;  
    ColorMapHandle cmap;
    int have_cmap;
    int grid_id;
    GeomArrows* arrows;
    Mutex mutex;
    double lenscale;
    GuiDouble length_scale;
    GuiDouble width_scale;
    GuiDouble head_length;
    GuiInt exhaustive_flag;
    GuiInt max_vect;
    GuiInt drawcylinders;
    GuiDouble shaft_rad;
    int numSurfPts;
    int np;
    int exhaustive;
    TriSurface* ts;
    Array1<Vector> vectArr;
    double maxLength, minLength;
    Array1<Point> ptArr;

public:
    SurfToVectGeom(const clString& id);
    virtual ~SurfToVectGeom();
    virtual void execute();
    void parallel(int proc);
};


extern "C" Module* make_SurfToVectGeom(const clString& id)
{
    return scinew SurfToVectGeom(id);
}

static clString module_name("SurfToVectGeom");

SurfToVectGeom::SurfToVectGeom(const clString& id)
: Module("SurfToVectGeom", id, Filter),
  length_scale("length_scale", id, this),
  width_scale("width_scale", id, this),
  head_length("head_length", id, this),
  drawcylinders("drawcylinders", id, this),
  shaft_rad("shaft_rad", id, this),
  exhaustive_flag("exhaustive_flag", id, this),
  max_vect("max_vect", id, this), 
  mutex("SurfToVectGeom vector adding mutex")
{
    // Create the input port
    isurface=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    ifield=scinew VectorFieldIPort(this, "VectorField", VectorFieldIPort::Atomic);
    add_iport(ifield);
    icmap = scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(icmap);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    grid_id=0;
    drawcylinders.set(0);
    
    maxLength = -(HUGE_VAL);
    minLength = HUGE_VAL;

}

SurfToVectGeom::~SurfToVectGeom()
{
	ogeom->delAll();
}

void SurfToVectGeom::parallel(int proc)
{
  
  int su=proc*numSurfPts/np;
  int eu=(proc+1)*numSurfPts/np;

  for (int i = su; i < eu; i++) {

    Point p = ts->points[i];
  
     // Query the vector field...
     Vector vv;
     int ii;
     if (vfield->interpolate( p, vv, ii, exhaustive)){
        mutex.lock();
        vectArr.add(vv);
	ptArr.add(p);
	double l = vv.length();
	if (l > maxLength) maxLength = l;
	if (l < minLength) minLength = l;
	mutex.unlock();
     }
  }
}

void SurfToVectGeom::execute()
{
  int old_grid_id = grid_id;

    SurfaceHandle surf;
     if (!isurface->get(surf)){
	ogeom->delAll();
	return;
    }
    if (!surf.get_rep()) return;

    if (!ifield->get( vfield )) return;

    ts=surf->getTriSurface();
    numSurfPts = ts->points.size();

    have_cmap=icmap->get( cmap );

    lenscale = length_scale.get();
    double widscale = width_scale.get(),
    headlen = head_length.get();
    exhaustive = exhaustive_flag.get();
    arrows = new GeomArrows(widscale, 1.0-headlen, drawcylinders.get(), shaft_rad.get() );
    

    VectorFieldMI* mi = dynamic_cast<VectorFieldMI*> (vfield.get_rep());
  
    //parallel
    if (mi != NULL) {   //parallel for only vfmi
      np=Thread::numProcessors();
      if (np>4) np/=2;	// using half the processors.
    } else np = 1;

    Thread::parallel(Parallel<SurfToVectGeom>(this, &SurfToVectGeom::parallel), np, true);

    if (have_cmap) {
      cmap->Scale(minLength,maxLength);
    }

    for (int i=0; i<vectArr.size(); i++) {

      Point p = ptArr[i];
      Vector vv = vectArr[i];
      if(vv.length2()*lenscale > 1.e-3) {
	 if (have_cmap && vv.length() <= max_vect.get()) {
	   MaterialHandle matl = cmap->lookup(vv.length()*lenscale);
	   arrows->add(p, vv*lenscale, matl, matl, matl);
	 } else if (vv.length() <= max_vect.get()) {
	   arrows->add(p, vv*lenscale);
	 }
      }// else cerr << vv.length()<<"\n";
    }

    // delete the old grid/cutting plane
    if (old_grid_id != 0)
	ogeom->delObj( old_grid_id );

    grid_id = ogeom->addObj(arrows, module_name);
} // End namespace RobV
}

