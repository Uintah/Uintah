/*
 *  OldSFUGtoNewTetVol.cc: Converter
 *
 *  Written by:
 *   Martin Cole
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <FieldConverters/Core/Datatypes/ScalarFieldUG.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#include "MeshToTet.h"

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;
using namespace FieldConverters;

int
main(int argc, char **argv) {
  
  if (argc !=3) {
    cerr << "Usage: " << argv[0] << " Old ScalarFieldUG to New TetVol"<< endl;
    cerr << "       " << "argv[1] Input File (Old calarFieldUG)" << endl;
    cerr << "       " << "argv[2] Output File (TetVol)" << endl;
    exit(0);
  }
  typedef LockingHandle<ScalarFieldUG> ScalarFieldUGHandle;
  ScalarFieldUGHandle handle;
  
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
  
  ScalarFieldUG *sf = dynamic_cast<ScalarFieldUG*>(handle.get_rep());
  if (!sf) {
    cerr << "Error: input Field wasn't a ScalarFieldUG."
	 << ".  Exiting..." << endl;
    exit(0);
  }

  MeshHandle mesh = sf->mesh;

  // Set up neighbors.
  mesh->compute_neighbors();
  mesh->compute_face_neighbors();
  
  TetVol<double> *field = 0;
  // A Mesh is Geometry only, so attach no data to the new TetVol.
  if (sf->typ == ScalarFieldUG::NodalValues) {
    field = new TetVol<double>(Field::NODE);
  } else {
    field = new TetVol<double>(Field::CELL);
  }
  FieldHandle fH(field); 
  TetVolMeshHandle tvm = field->get_typed_mesh();
  
  // load the mesh.
  load_mesh(mesh.get_rep(), tvm);

  // load the data
  // create iterators for Array1 data
  double *begin = &sf->data[0];
  double *end = begin + sf->data.size();
  TetVol<double>::fdata_type &fd = field->fdata();
  fd.insert(fd.end(), begin, end);

  TextPiostream outstream(argv[2], Piostream::Write);
  Pio(outstream, fH);

  return 0;  
}    
