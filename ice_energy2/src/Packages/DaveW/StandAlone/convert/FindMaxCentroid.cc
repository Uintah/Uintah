/*
 *  ExtractCC.cc: Split the connected components of a TriSurfField into
 *                separate files
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/TetVolField.h>
#include <Core/Persistent/Pstreams.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {

  // check for correct usage and valid input
  if (argc != 2) {
    cerr << "Usage: "<<argv[0]<<" CellCenteredTetVolFieldOfVectors\n";
    return 0;
  }
  FieldHandle handle;
  Piostream *stream=auto_istream(argv[1]);
  if (!stream) {
    cerr << "Error - couldn't open input file "<<argv[1]<<"\n";
    return 0;
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error - input file "<<argv[1]<<" didn't contant a field.\n";
    return 0;
  }
  TetVolField<Vector> *tv = dynamic_cast<TetVolField<Vector> *>(handle.get_rep());
  if (!tv) {
    cerr << "Error - input field wasn't a TetVolField<Vector>, it was a "<<handle->get_type_name();
    return 0;
  }
  if (tv->data_at() != Field::CELL) {
    cerr << "Error - data was supposed to be at the cells.\n";
    return 0;
  }
  TetVolMeshHandle tvm = tv->get_typed_mesh();
  TetVolMesh::Cell::iterator ci, cie;
  tvm->begin(ci);
  tvm->end(cie);
  double maxVal = tv->fdata()[*ci].length();
  TetVolMesh::Cell::index_type maxIdx = *ci;
  ++ci;
  for (; ci != cie; ++ci) {
    double newVal = tv->fdata()[*ci].length();
    if (newVal > maxVal) {
      maxVal = newVal;
      maxIdx = *ci;
    }
  }
  Point c;
  tvm->get_center(c, maxIdx);
  cerr << "Max value was: "<<maxVal<<"  Max index was: "<<maxIdx<<"  Max centroid was "<<c<<"\n";

  // input looks good -- split the surface into connected components
  return 0;
}
