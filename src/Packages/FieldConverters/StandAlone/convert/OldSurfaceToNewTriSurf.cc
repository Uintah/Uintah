/*
 *  OldSurfaceToNewTriSurf.cc
 *
 *  Written by:
 *   David Weinstein and Michae Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Packages/FieldConverters/Core/Datatypes/TriSurface.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;
using namespace FieldConverters;

int
main(int argc, char **argv) {
  SurfaceHandle handle;
  
  // Read in a TriSurface
  if (argc !=3) {
    cerr << "Usage: "<<argv[0]<<" OldSurf NewTriSurf\n";
    exit(0);
  }
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    cerr << "Error - couldn't open file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading Surface from file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  TriSurface *ts = dynamic_cast<TriSurface*>(handle.get_rep());
  if (!ts) {
    cerr << "Error - surface wasn't a TriSurface.\n";
  }

  
  TriSurfMesh *tsm = new TriSurfMesh();


  int i;
  for (i=0; i<ts->points.size(); i++)
  {
    tsm->add_point(ts->points[i]);
  }

  for (i=0; i<ts->elements.size(); i++)
  {
    const TSElement *ele = ts->elements[i];
    tsm->add_triangle(ele->i1, ele->i2, ele->i3);
  }

  tsm->connect();
  
  
  //bcidx
  //bcval
  //valtype

  //normtype
  //normals
  //havenormals

  // Write out the new field.
  BinaryPiostream out_stream(argv[2], Piostream::Write);
  TriSurfMeshHandle tsmh(tsm);
  Pio(out_stream, tsmh);

  return 0;  
}    
