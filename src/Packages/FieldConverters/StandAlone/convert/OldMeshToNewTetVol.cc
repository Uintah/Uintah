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
 *  OldMeshtoNewTetVolField.cc: Converter
 *
 *  Written by:
 *   Martin Cole
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <FieldConverters/Core/Datatypes/Mesh.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "MeshToTet.h"

using std::cerr;
using std::cout;
using std::ifstream;
using std::endl;

using namespace SCIRun;
using namespace FieldConverters;


int
main(int argc, char **argv) {
  
  if (argc !=3) {
    cerr << "Usage: " << argv[0] << " Old Mesh to New TetVolField"<< endl;
    cerr << "       " << "argv[1] Input File (Old Mesh)" << endl;
    cerr << "       " << "argv[2] Output File (TetVolField)" << endl;
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
  // Set up neighbors.
  mesh->compute_neighbors();
  mesh->compute_face_neighbors();


  // A Mesh is Geometry only, so attach no data to the new TetVolField.
  TetVolField<double> *field = new TetVolField<double>(Field::NODE);
  FieldHandle fH(field); 

  TetVolMeshHandle tvm = field->get_typed_mesh();
  
  load_mesh(mesh, tvm);

  TextPiostream outstream(argv[2], Piostream::Write);
  Pio(outstream, fH);

  return 0;  
}    
