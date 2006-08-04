/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  OldSurfaceToNewTriSurfField.cc
 *
 *  Written by:
 *   David Weinstein and Michae Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Packages/FieldConverters/Core/Datatypes/TriSurfFieldace.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <Core/Geometry/Vector.h>

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
  
  // Read in a TriSurfFieldace
  if (argc !=3) {
    cerr << "Usage: "<<argv[0]<<" OldSurf NewTriSurfField\n";
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
  TriSurfFieldace *ts = dynamic_cast<TriSurfFieldace*>(handle.get_rep());
  if (!ts) {
    cerr << "Error - surface wasn't a TriSurfFieldace.\n";
  }

  
  TriSurfMesh *tsm = new TriSurfMesh;
  TriSurfMeshHandle tsmh(tsm);

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

  TriSurfField<double> *tsd = scinew TriSurfField<double>(tsmh, Field::NODE);
  FieldHandle tsdh(tsd);

  for (i=0; i<ts->points.size(); i++) {
    tsd->fdata()[i] = 0;
  }
  for (i=0; i<ts->bcIdx.size(); i++) {
    tsd->fdata()[ts->bcIdx[i]] = ts->bcVal[i];
  }

  BinaryPiostream out_stream(argv[2], Piostream::Write);

  // Write out the new field.
  
  Pio(out_stream, tsdh);
  return 0;  
}    
