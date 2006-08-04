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

#include <Core/Datatypes/TriSurfField.h>
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
  if (argc < 3) {
    cerr << "Usage: "<<argv[0]<<" input outputbasename\n";
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
  TriSurfField<double> *ts = dynamic_cast<TriSurfField<double> *>(handle.get_rep());
  if (!ts) {
    cerr << "Error - input field wasn't a TriSurfField<double>, it was a "<<handle->get_type_name();
    return 0;
  }
  TriSurfMeshHandle tsm = ts->get_typed_mesh();
  TriSurfMesh::Face::size_type fsize;
  tsm->size(fsize);
  if (!(unsigned int)fsize) {
    cerr << "Error -- input surface has no faces.\n";
    return 0;
  }

  TriSurfMesh::Node::size_type nsize;
  tsm->size(nsize);
  cerr << "Input surface has "<<((unsigned int)nsize)<<" points and "
       << ((unsigned int)fsize) <<" faces.\n";

  // input looks good -- split the surface into connected components
  return 0;
}
