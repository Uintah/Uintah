/*
 *  MagneticScalarField.cc:  Unfinished modules
 *
 *  Written by:
 *   Robert Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   August 1999
 *
 *  Copyright (C) 1999 SCI Group
 */


#include <DaveW/Datatypes/General/VectorFieldMI.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>
#include <iostream>
using std::cerr;

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using DaveW::Datatypes::VectorFieldMI;
using SCICore::Thread::Mutex;
using SCICore::Thread::Parallel;
using SCICore::Thread::Thread;

class MagneticScalarField : public Module {
  VectorFieldHandle vfh;
  VectorFieldIPort* magneticFieldP;
  SurfaceIPort* isurfaceP;
  SurfaceOPort* magnitudeFieldP;
public:   
  MagneticScalarField(const clString& id);
  virtual ~MagneticScalarField();
  virtual void execute();
  void parallel(int proc);

  int np;
  int nelems;
  TriSurface* magField;
  Mutex mutex;
  MeshHandle mesh;
  VectorFieldHandle mField;
};


extern "C" Module* make_MagneticScalarField(const clString& id)
{
    return new MagneticScalarField(id);
}

MagneticScalarField::MagneticScalarField(const clString& id): Module("MagneticScalarField", id, Filter), mutex("MagneticScalarField lock for adding bc's")
{
  magneticFieldP=new VectorFieldIPort(this, "MagneticField", VectorFieldIPort::Atomic);
  add_iport(magneticFieldP);

  isurfaceP = new SurfaceIPort(this, "ISurface", SurfaceIPort::Atomic);
  add_iport(isurfaceP);

  // Create the output port
  magnitudeFieldP=new SurfaceOPort(this, "magnitudeField", SurfaceIPort::Atomic);
  add_oport(magnitudeFieldP);
}

MagneticScalarField::~MagneticScalarField()
{
}

void MagneticScalarField::parallel(int proc)
{

  int su=proc*nelems/np;
  int eu=(proc+1)*nelems/np;

  for (int i=su; i<eu; i++) {
    
    Element* e = mesh->elems[i];

    Vector magV;
    Point centroid = e->centroid();
    int ii = i;
    ((VectorFieldMI*)(mField.get_rep()))->interpolate(centroid, magV,ii);  

    double l = magV.length();

    mutex.lock();
    magField->bcIdx.add(i);  //correct use?
    magField->bcVal.add(l);
    mutex.unlock();
  }
}

void MagneticScalarField::execute() {

  if (!magneticFieldP->get(mField)) return;

  SurfaceHandle isurf;
  if(!isurfaceP->get(isurf))
    return;
  TriSurface *ts=isurf->getTriSurface();

  ////  //copy over input surface  //correct?

  magField = new TriSurface();

  //  Array1<Vector> normals;
  //isurf->get_surfnodes(nodes);
  //Array1<Point> pts(nodes.size());

  *magField=*ts;

  //  for (i=0; ts && i<isurf->normals.size(); i++) 
  //  normals.add((transformPt(m, ts->normals[i].point(), Vector(0,0,0), 1,1,1)).vector());
//if (normals.size()) magField->normals=normals;
//magField->points=pts;
  //////

  mesh = ((VectorFieldMI*)mField.get_rep())->getMesh();

  if (mesh.get_rep() == NULL) {
    cerr << "Error getting mesh\n";
    exit(0);
  }

  nelems = mesh->elems.size(); 

  VectorFieldMI* mi = dynamic_cast<VectorFieldMI*> (mField.get_rep());
  
  //parallel
  if (mi != NULL) {   //parallel for only vfmi
    np=Thread::numProcessors();
    if (np>4) np/=2;	// using half the processors.
  } else np = 1;
 
  Thread::parallel(Parallel<MagneticScalarField>(this, &MagneticScalarField::parallel), np, true);

//  Task::multiprocess(np, do_parallel, this);
  
  magnitudeFieldP->send(magField);

}
} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.4  2000/03/17 09:25:50  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  1999/10/07 02:06:38  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/09/08 02:26:29  sparker
// Various #include cleanups
//
// Revision 1.1  1999/09/02 04:27:08  dmw
// Rob V's modules
//
//
