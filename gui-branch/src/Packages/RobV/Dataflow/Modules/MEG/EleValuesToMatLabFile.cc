/*
 *  EleValuesToMatLabFile.cc
 *
 *  Written by:
 *   Robert Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   October 1999
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

class EleValuesToMatLabFile : public Module {
  ScalarFieldIPort* scalarFieldP;
public:   
  EleValuesToMatLabFile(const clString& id);
  virtual ~EleValuesToMatLabFile();
  virtual void execute();

};

extern "C" Module* make_EleValuesToMatLabFile(const clString& id)
{
    return new EleValuesToMatLabFile(id);
}

EleValuesToMatLabFile::EleValuesToMatLabFile(const clString& id): Module("MagneticFieldAtPoints", id, Filter)
{

  scalarFieldP=new ScalarFieldIPort(this, "ScalarField", ScalarFieldIPort::Atomic);
  add_iport(scalarFieldP);
}

EleValuesToMatLabFile::~EleValuesToMatLabFile()
{
}

void EleValuesToMatLabFile::execute() {

  ScalarFieldHandle sfieldH;
  if (!scalarFieldP->get(sfieldH)) return;

  MeshHandle meshH = ((ScalarFieldUG*)(sfieldH.get_rep()))->mesh;

  Mesh* mesh = meshH.get_rep();

  int nnodes = mesh->nodes.size();

  FILE* out = fopen("error.dat", "w");

  int count=0;

  for (int i=0; i<((ScalarFieldUG*)(sfieldH.get_rep()))->data.size(); i++) {
    // NodeHandle n = mesh->nodes[i];
    // double x = n->p.x();
    //double y = n->p.y();
    //double z = n->p.z();
    
    //  if (z >= -0.1 && z <= 0.1) {
    //      count++;
      double value =  ((ScalarFieldUG*)(sfieldH.get_rep()))->data[i];
      if (value == 0.0) cerr << "ZERO VALUE\n";
      //      fprintf (out, "%g %g %g\n", x, y, value);
      // }
      fprintf (out, "%d %g \n", i, value);
  }
  //  cerr << "Num Points: " << count << "\n";
  fclose(out);
}
} // End namespace RobV


