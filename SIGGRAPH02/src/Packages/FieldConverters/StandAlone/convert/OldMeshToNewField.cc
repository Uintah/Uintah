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
 *  OldMeshToNewField.cc: Converter for old meshes with conductivities
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <FieldConverters/Core/Datatypes/Mesh.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Geometry/Tensor.h>
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

int main(int argc, char **argv) {
  
  if (argc !=3) {
    cerr << "Usage: " << argv[0] << " Old Mesh to New Field"<< endl;
    cerr << "       " << "argv[1] Input File (Old Mesh)" << endl;
    cerr << "       " << "argv[2] Output File (Field)" << endl;
    exit(0);
  }
  typedef LockingHandle<Mesh> MeshHandle;
  MeshHandle handle;
  
  Piostream* instream = auto_istream(argv[1]);
  if (!instream) {
    cerr << "Error: couldn't open file " << argv[1] 
	 << ".  Exiting..." << endl;
    exit(0);
  }
  Pio(*instream, handle);
  Mesh *mesh;
  if (! (mesh = handle.get_rep())) {
    cerr << "Error: reading Mesh from file " << argv[1] 
	 << ".  Exiting..." << endl;
    exit(0);
  }
  
  // Set up neighbors.
  mesh->compute_neighbors();
  mesh->compute_face_neighbors();

  if (!mesh->cond_tensors.size()) {
    cerr << "Error - mesh didn't have any conductivity information or Dirichlet nodes... use OldMeshToNewTetVolField instead.\n";
    return 0;
  }

  // Build the new mesh and field
  TetVolField<int> *field;
  TetVolMeshHandle tvm = scinew TetVolMesh;
  load_mesh(mesh, tvm);
  field = new TetVolField<int>(tvm, Field::CELL);

  // Set the conductivities as indexed data
  vector<pair<string, Tensor> > conds;
  for (int i=0; i<mesh->cond_tensors.size(); i++)
  {
    conds.push_back(pair<string, Tensor>(to_string(i),
					 Tensor(mesh->cond_tensors[i])));
  }
  field->store("data_storage", string("table"));
  field->store("name", string("conductivity"));
  field->store("conductivity_table", conds);
  int cell_counter=0;
  TetVolMesh::Cell::iterator ci;
  TetVolMesh::Cell::iterator cie;
  tvm->begin(ci); tvm->end(cie);
  for (; ci != cie; ++ci, cell_counter++)
    field->fdata()[*ci] = mesh->elems[cell_counter]->cond;

  // Add any Dirichlet conditions to as a Property of the mesh
  vector<pair<int,double> > Dirichlet;
  int i;
  for (i=0; i<mesh->nodes.size(); i++) 
    if (mesh->nodes[i]->bc)
      Dirichlet.push_back(pair<int,double>(i,mesh->nodes[i]->bc->value));
  if (Dirichlet.size())
    tvm->store("Dirichlet", Dirichlet);

  // Write it to disk
  BinaryPiostream outstream(argv[2], Piostream::Write);
  FieldHandle fH(field);
  Pio(outstream, fH);

  return 0;  
}    
