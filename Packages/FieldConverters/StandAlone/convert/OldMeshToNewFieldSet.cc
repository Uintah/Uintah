/*
 *  OldSFUGtoNewTetVol.cc: Converter
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
#include <Core/Datatypes/FieldSet.h>
#include <Core/Datatypes/MaskedTetVol.h>
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

main(int argc, char **argv) {
  
  if (argc !=3) {
    cerr << "Usage: " << argv[0] << " Old Mesh to New FieldSet"<< endl;
    cerr << "       " << "argv[1] Input File (Old Mesh)" << endl;
    cerr << "       " << "argv[2] Output File (FieldSet)" << endl;
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

  FieldSet* fs = scinew FieldSet;
  TetVolMeshHandle tvm=0;
  if (mesh->cond_tensors.size()) {
    TetVol<Tensor> *field = new TetVol<Tensor>(Field::CELL);
    field->set_string("name", "conductivity");
    tvm = field->get_typed_mesh();
    load_mesh(mesh, tvm);
    field->resize_fdata();
    int cell_counter=0;
    TetVolMesh::cell_iterator ci;
    for (ci = tvm->cell_begin(); ci != tvm->cell_end(); ++ci, cell_counter++) {
      Tensor t(mesh->cond_tensors[mesh->elems[cell_counter]->cond]);
      field->fdata()[*ci] = t;
    }
    FieldHandle fH(field);
    fs->add(fH);
    cerr << "Added `conductivity' field (tensors at cells) to fieldset\n";
  }
  
  Array1<int> dirichlet;
  int i;
  for (i=0; i<mesh->nodes.size(); i++) 
    if (mesh->nodes[i]->bc) dirichlet.add(i);

  if (dirichlet.size()) {
    MaskedTetVol<double> *field;
    if (!tvm.get_rep()) {
      field = new MaskedTetVol<double>(Field::NODE);
      tvm = field->get_typed_mesh();
      load_mesh(mesh, tvm);
      field->resize_fdata();
      field->initialize_mask(0);
    } else {
      field = new MaskedTetVol<double>(tvm, Field::NODE);
      field->initialize_mask(0);
    }

    field->set_string("name", "dirichlet");
    int node_counter=0;
    TetVolMesh::node_iterator ni;
    for (ni = tvm->node_begin(); ni != tvm->node_end(); ++ni, node_counter++) {
      if (mesh->nodes[node_counter]->bc) {
	field->fdata()[*ni]=mesh->nodes[node_counter]->bc->value;
	field->mask()[*ni]=1;
      }
    }

    FieldHandle fH(field);
    fs->add(fH);
    cerr << "Added `dirichlet' maskedfield (doubles at nodes) to fieldset\n";
  }

  if (fs->field_begin() == fs->field_end()) {
    cerr << "Error - mesh didn't have any conductivity information or Dirichlet nodes... use OldMeshToNewTetVol instead.\n";
    return 0;
  }

  TextPiostream outstream(argv[2], Piostream::Write);
  FieldSetHandle fsH(fs);
  Pio(outstream, fsH);

  return 0;  
}    
