/*
 *  MakeCurrentDensityField.cc:  Unfinished modules
 *
 *  Written by:
 *   Robert Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <DaveW/Datatypes/General/VectorFieldMI.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using DaveW::Datatypes::VectorFieldMI;

class MakeCurrentDensityField : public Module {
  TCLint interpolate;
  VectorFieldHandle vfh;
  VectorFieldIPort* electricFieldP;
  MatrixIPort* sourceLocationsP;
  VectorFieldOPort* currentDensityFieldP;
public:   
  MakeCurrentDensityField(const clString& id);
  virtual ~MakeCurrentDensityField();
  virtual void execute();

private:
  Vector mult(Array1<double> matrix, Vector ElemField);
};


Module* make_MakeCurrentDensityField(const clString& id)
{
    return new MakeCurrentDensityField(id);
}

MakeCurrentDensityField::MakeCurrentDensityField(const clString& id): Module("MakeCurrentDensityField", id, Filter), interpolate("interpolate", id, this)
{
  electricFieldP=new VectorFieldIPort(this, "ElectricField", VectorFieldIPort::Atomic);
  add_iport(electricFieldP);

  sourceLocationsP=new MatrixIPort(this, "SourceLocations", MatrixIPort::Atomic);
  add_iport(sourceLocationsP);

  // Create the output port
  currentDensityFieldP=new VectorFieldOPort(this, "currentDensityField", VectorFieldIPort::Atomic);
  add_oport(currentDensityFieldP);
}

MakeCurrentDensityField::~MakeCurrentDensityField()
{
}

void MakeCurrentDensityField::execute() {

  VectorFieldHandle eField;
  if (!electricFieldP->get(eField)) return;
  MeshHandle mesh = ((VectorFieldUG*)(eField.get_rep()))->mesh;

  //MatrixHandle sourceLocsM;
  //if (!sourceLocationsP->get(sourceLocsM)) return;

  //DenseMatrix* sourceLocations = dynamic_cast<DenseMatrix*>(sourceLocsM.get_rep());

  VectorFieldUG* currentDensityField = new VectorFieldUG(mesh,VectorFieldUG::ElementValues);

  int nelems = mesh->elems.size();

  for (int i=0; i<nelems; i++) {
  
    Element* e = mesh->elems[i];

    Array1<double> matrix = mesh->cond_tensors[(e->cond)];
 

    Vector elemField;
    Point centroid = e->centroid();
    int ii = i;
    ((VectorFieldUG*)(eField.get_rep()))->interpolate(centroid, elemField,ii);

    Vector condElect = mult(matrix,elemField);  //matrix-vextor mult (must write)

    //    int nSources = sourceLocations->ncols();

//comment out source magnture current density until figure out scale
    
    /*    for(int j=0; j<nSources;j++) {

      double x = (*sourceLocations)[0][j];
      double y = (*sourceLocations)[1][j];
      double z = (*sourceLocations)[2][j];
      double theta = (*sourceLocations)[3][j];
      double phi = (*sourceLocations)[4][j];
      double magnitude = (*sourceLocations)[5][j];
     
      Point  pt (x,y,z);

      Vector dir(Sin(theta)*Cos(phi), Sin(theta)*Sin(phi), Cos(theta));

      int ix=0;
      
      if (mesh->locate2(pt,ix) == i) {
	  condElect += (dir*magnitude);
      }
     }*/

    currentDensityField->data[i] = condElect;
  }
  VectorFieldMI* currentDensityFieldMI = new VectorFieldMI(currentDensityField); 

  vfh=currentDensityFieldMI;

  currentDensityFieldP->send(vfh);
}

Vector MakeCurrentDensityField::mult(Array1<double> matrix, Vector elemField) {

  return(Vector(matrix[0]*elemField.x()+matrix[1]*elemField.y()+matrix[2]*elemField.z(),matrix[1]*elemField.x()+matrix[3]*elemField.y()+matrix[4]*elemField.z(),matrix[2]*elemField.x()+matrix[4]*elemField.y()+matrix[5]*elemField.z()));

}
} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.1  1999/09/02 04:27:09  dmw
// Rob V's modules
//
//
