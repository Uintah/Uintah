/*
 *  OldMeshtoNewTetVol.cc: Converter
 *
 *  Written by:
 *   Martin Cole
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <FieldConverters/Core/Datatypes/Mesh.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;
using namespace FieldConverters;
int
main(int argc, char **argv) {
  
  if (argc !=3) {
    cerr << "Usage: " << argv[0] << " OldMesh to NewTetVol"<< endl;
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
  
  Mesh *base = dynamic_cast<Mesh*>(handle.get_rep());
  if (!base) {
    cerr << "Error: input Field wasn't a Mesh."
	 << ".  Exiting..." << endl;
    exit(0);
  }

  // A Mesh is Geometry only, so attach no data to the new TetVol.
  TetVol<double> *field = new TetVol<double>(Field::NODE);
  FieldHandle fH(field); 

  TetVolMeshHandle tvm = field->get_typed_mesh();

  // Assume that the old Mesh and the new arrange nodes the same way.
  for (int i = 0; i < handle->nodesize(); i++) {
    cout << "node " << i << ": " << handle->node(i).p << endl;
  }
  
  // TO_DO:
  // make a new Field, set it to fH, and give it a mesh and data like base's

  BinaryPiostream outstream(argv[2], Piostream::Write);
  Pio(outstream, fH);

  return 0;  
}    
