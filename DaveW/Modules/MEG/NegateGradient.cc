/*
 *  NegateGradient.cc:  Negate Gradient Field to be used as Electric Field
 *
 *  Written by:
 *   Robert Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   Sept 1999
 *
 *  Copyright (C) 1999 SCI Group
 */


#include <DaveW/Datatypes/General/VectorFieldMI.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <SCICore/Datatypes/Matrix.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
//#include <iostream.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

class NegateGradient : public Module {
  VectorFieldIPort* gFieldP;
  VectorFieldOPort* eFieldP;
public:   
  NegateGradient(const clString& id);
  virtual ~NegateGradient();
  virtual void execute();
};


Module* make_NegateGradient(const clString& id)
{
    return new NegateGradient(id);
}

NegateGradient::NegateGradient(const clString& id): Module("NegateGradient", id, Filter)
{
  gFieldP=new VectorFieldIPort(this, "GradientField", VectorFieldIPort::Atomic);
  add_iport(gFieldP);

  // Create the output port
  eFieldP=new VectorFieldOPort(this, "ElectricField", VectorFieldIPort::Atomic);
  add_oport(eFieldP);
}

NegateGradient::~NegateGradient()
{
}

void NegateGradient::execute() {

  VectorFieldHandle gField;
  if (!gFieldP->get(gField)) return;

  MeshHandle mesh = ((VectorFieldUG*)(gField.get_rep()))->mesh;

  int ndata = ((VectorFieldUG*)(gField.get_rep()))->data.size();

  VectorFieldUG* eField = new VectorFieldUG(mesh,VectorFieldUG::NodalValues);

  for (int i=0; i<ndata; i++) {
    eField->data[i] = ((VectorFieldUG*)(gField.get_rep()))->data[i] * -1;
  }

  eFieldP->send(eField);
}
} // End namespace Modules
} // End namespace DaveW
