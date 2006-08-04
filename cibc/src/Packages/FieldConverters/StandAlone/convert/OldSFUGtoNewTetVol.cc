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
 *  OldSFUGtoNewTetVolField.cc: Converter
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
#include <Core/Datatypes/TetVolField.h>
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
    cerr << "Usage: " << argv[0] << " Old ScalarFieldUG to New TetVolField"<< endl;
    cerr << "       " << "argv[1] Input File (Old calarFieldUG)" << endl;
    cerr << "       " << "argv[2] Output File (TetVolField)" << endl;
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
  
  TetVolField<double> *field = 0;
  // A Mesh is Geometry only, so attach no data to the new TetVolField.
  if (sf->typ == ScalarFieldUG::NodalValues) {
    field = new TetVolField<double>(Field::NODE);
  } else {
    field = new TetVolField<double>(Field::CELL);
  }
  FieldHandle fH(field); 
  TetVolMeshHandle tvm = field->get_typed_mesh();
  
  // load the mesh.
  load_mesh(mesh.get_rep(), tvm);

  // load the data
  // create iterators for Array1 data
  double *begin = &sf->data[0];
  double *end = begin + sf->data.size();
  TetVolField<double>::fdata_type &fd = field->fdata();
  fd.insert(fd.end(), begin, end);

  TextPiostream outstream(argv[2], Piostream::Write);
  Pio(outstream, fH);

  return 0;  
}    
