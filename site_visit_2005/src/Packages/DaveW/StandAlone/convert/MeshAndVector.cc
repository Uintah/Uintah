
/*
 *  MatrixInfo: Read in a Matrix, output info about it
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Containers/String.h>
#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/Matrix.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

main(int argc, char **argv) {
  if (argc !=4) {
    cerr <<"Usage: "<<argv[0]<<" mesh vector sfug\n";
    exit(0);
  }
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    printf("Couldn't open file %s.  Exiting...\n", argv[1]);
    exit(0);
  }
  MeshHandle mesh;
  Pio(*stream, mesh);
  if (!mesh.get_rep()) {
    printf("Error reading Matrix from file %s.  Exiting...\n", argv[1]);
    exit(0);
  }
  stream=auto_istream(argv[2]);
  if (!stream) {
    printf("Couldn't open file %s.  Exiting...\n", argv[2]);
    exit(0);
  }
  MatrixHandle matrix;
  Pio(*stream, matrix);
  if (!matrix.get_rep()) {
    printf("Error reading Matrix from file %s.  Exiting...\n", argv[2]);
    exit(0);
  }
  if (mesh->elems.size()*3 != matrix->ncols()) {
    cerr << "Error - this only works for cell-centered vector data.\n";
    exit(0);
  }
  ScalarFieldUG *sfug = new ScalarFieldUG(mesh, 

  MeshHandle mesh;
  Pio(*stream, mesh);
  if (!handle.get_rep()) {
    printf("Error reading Matrix from file %s.  Exiting...\n", argv[1]);
    exit(0);
  }
  cerr << "Matrix typeinfo = "<<handle->getType()<<"\n";
  cerr << "nrows = "<<handle->nrows()<<"  ncols = "<<handle->ncols()<<"\n";

  return 0;
}    
