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

class EleValuesToMatLabFile : public Module {
  ScalarFieldIPort* scalarFieldP;
public:   
  EleValuesToMatLabFile(const clString& id);
  virtual ~EleValuesToMatLabFile();
  virtual void execute();

};

Module* make_EleValuesToMatLabFile(const clString& id)
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

} // End namespace Modules
} // End namespace DaveW

