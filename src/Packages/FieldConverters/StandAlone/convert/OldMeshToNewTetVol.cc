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

using std::cerr;
using std::cout;
using std::ifstream;
using std::endl;

using namespace SCIRun;
using namespace FieldConverters;


// Functor to extract a Point from a NodeHandle.
struct NodePointFtor{
  Point operator()(NodeHandle nh) {
    static int i = 0;
    cout << "NodePointFtor: " << i++ << endl;
    return nh->p;
  }
};

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
  
  Mesh *mesh = dynamic_cast<Mesh*>(handle.get_rep());
  if (!mesh) {
    cerr << "Error: input Field wasn't a Mesh."
	 << ".  Exiting..." << endl;
    exit(0);
  }

  // A Mesh is Geometry only, so attach no data to the new TetVol.
  TetVol<double> *field = new TetVol<double>(Field::NODE);
  FieldHandle fH(field); 

  TetVolMeshHandle tvm = field->get_typed_mesh();

  NodeHandle &s = mesh->nodes[0];
  NodeHandle *begin = &s;
  NodeHandle *end = begin + mesh->nodes.size();
  tvm->fill_points(begin, end, NodePointFtor());

  // FIX_ME fill the cells and neihbor info up.
  TextPiostream outstream(argv[2], Piostream::Write);
  Pio(outstream, fH);

  return 0;  
}    
