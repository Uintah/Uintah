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


#include <Packages/RobV/Core/Datatypes/MEG/VectorFieldMI.h>
#include <Core/Math/Trig.h>
#include <Core/Geometry/Point.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Datatypes/VectorFieldUG.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Ports/ColumnMatrixPort.h>
//#include <iostream.h>

namespace RobV {
using namespace SCIRun;

class NegateGradient : public Module {
  VectorFieldIPort* gFieldP;
  VectorFieldOPort* eFieldP;
public:   
  NegateGradient(const clString& id);
  virtual ~NegateGradient();
  virtual void execute();
};


extern "C" Module* make_NegateGradient(const clString& id)
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
} // End namespace RobV
}
