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

#include <DaveW/Datatypes/General/VectorFieldMI.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>
#include <math.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using namespace SCICore::TclInterface;
using DaveW::Datatypes::VectorFieldMI;
using SCICore::Thread::Mutex;
using SCICore::Thread::Parallel;
using SCICore::Thread::Thread;

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
    TCLdouble length_scale;
    TCLdouble width_scale;
    TCLdouble head_length;
    TCLint exhaustive_flag;
    TCLint max_vect;
    TCLint drawcylinders;
    TCLdouble shaft_rad;
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


Module* make_SurfToVectGeom(const clString& id)
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
}
} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.2  1999/09/08 02:26:29  sparker
// Various #include cleanups
//
// Revision 1.1  1999/09/02 04:27:09  dmw
// Rob V's modules
//
//
