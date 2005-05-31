
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

#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/BBox.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

main(int argc, char **argv) {

  FieldHandle handle;

  if (argc !=2) {
    printf("Need the file name!\n");
    exit(0);
  }
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    printf("Couldn't open file %s.  Exiting...\n", argv[1]);
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    printf("Error reading Matrix from file %s.  Exiting...\n", argv[1]);
    exit(0);
  }
  cerr << "Type = "<<handle->get_type_name(-1)<<"\n";
  cerr << "Data at = ";
  if (handle->data_at() == Field::NODE) cerr << "NODE";
  else if (handle->data_at() == Field::EDGE) cerr << "EDGE";
  else if (handle->data_at() == Field::FACE) cerr << "FACE";
  else if (handle->data_at() == Field::CELL) cerr << "CELL";
  else if (handle->data_at() == Field::NONE) cerr << "NONE";
  cerr << "\n";
  if (handle->get_type_name(0) == "TetVolField") {
    MeshHandle mbH = handle->mesh();
    TetVolMesh *tvm = dynamic_cast<TetVolMesh *>(mbH.get_rep());
    vector<pair<string, Tensor> > cond_table;
    if (handle->get("conductivity_table", cond_table))
    {
      cerr << "Conductivities: \n";
      for (int i=0; i<cond_table.size(); i++) {
	cerr << "  cond value " << i << "(" << cond_table[i].first << ") : " <<
	  cond_table[i].second.mat_[0][0] << "\n";
      }
    }
    TetVolMesh::Node::size_type nsize;
    TetVolMesh::Edge::size_type esize;
    TetVolMesh::Face::size_type fsize;
    TetVolMesh::Cell::size_type csize;
    tvm->size(nsize);
    tvm->size(esize);
    tvm->size(fsize);
    tvm->size(csize);
    cerr << "Number of nodes: "<<nsize<<"\n";
    cerr << "Number of edges: "<<esize<<"\n";
    cerr << "Number of faces: "<<fsize<<"\n";
    cerr << "Number of cells: "<<csize<<"\n";
    BBox b;
    for (int i=0; i<nsize; i++) b.extend(tvm->point(i));
    Point min, max;
    cerr << "Bounding box: "<<b.min()<<" -- "<<b.max()<<"\n";
  }
  cerr << "\n";
  return 0;
}
    
