/*
 *  OldMeshtoNewTetVol.cc: Converter
 *
 *  Written by:
 *   Martin Cole
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <FieldConverters/Core/Datatypes/Mesh.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "MeshToTet.h"

using std::cerr;
using std::cout;
using std::ifstream;
using std::endl;

using namespace SCIRun;
using namespace FieldConverters;


int
main(int argc, char **argv) {
  
  if (argc !=3) {
    cerr << "Usage: " << argv[0] << " Old Mesh to New TetVol"<< endl;
    cerr << "       " << "argv[1] Input File (Old Mesh)" << endl;
    cerr << "       " << "argv[2] Output File (TetVol)" << endl;
    exit(0);
  }

  MeshHandle handle;

  Piostream* instream = auto_istream(argv[1]);
  if (!instream) {
    cerr << "Error: couldn't open file " << argv[1] 
	 << ".  Exiting..." << endl;
    exit(0);
  }
  Pio(*instream, handle);
  if (!handle.get_rep()) {
    cerr << "Error: reading Mesh from file " << argv[1] 
	 << ".  Exiting..." << endl;
    exit(0);
  }
  
  Mesh *mesh = dynamic_cast<Mesh*>(handle.get_rep());
  if (!mesh) {
    cerr << "Error: input Field wasn't a Mesh."
	 << ".  Exiting..." << endl;
    exit(0);
  }
  // Set up neighbors.
  mesh->compute_neighbors();
  mesh->compute_face_neighbors();


  // A Mesh is Geometry only, so attach no data to the new TetVol.
  TetVol<double> *field = new TetVol<double>(Field::NODE);
  FieldHandle fH(field); 

  TetVolMeshHandle tvm = field->get_typed_mesh();
  
  load_mesh(mesh, tvm);

  TextPiostream outstream(argv[2], Piostream::Write);
  Pio(outstream, fH);

  return 0;  
}    
