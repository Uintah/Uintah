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
 *  FDMtoFEM: Translate a FDM model into an equivalent FEM model
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2001
 *
 *  Copyright (C) 1994, 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Dataflow/Ports/FieldPort.h>

#include <iostream>
#include <stdio.h>
#include <math.h>

namespace BioPSE {
using namespace SCIRun;

class FDMtoFEM : public Module {
  
  //! Private Data

  //! input ports
  FieldIPort*  ifdm_;

  //! output port
  FieldOPort*  ofem_;

  //! cached data
  int ifdmGen_;
  FieldHandle ofemH_;
public:
  
  FDMtoFEM(GuiContext *context);
  virtual ~FDMtoFEM();
  virtual void execute();
};


DECLARE_MAKER(FDMtoFEM)


FDMtoFEM::FDMtoFEM(GuiContext *context)
  : Module("FDMtoFEM", context, Filter, "Forward", "BioPSE"),
    ifdmGen_(-1), ofemH_(0)
{
}

FDMtoFEM::~FDMtoFEM()
{
}

void FDMtoFEM::execute() {
  
  update_state(NeedData);
  ifdm_ = (FieldIPort *)get_iport("FDM (LVT, cell sigmas)");
  ofem_ = (FieldOPort *)get_oport("FEM (TVT, cell sigmas)");
  FieldHandle ifdmH;

  if (!ifdm_) {
    error("Unable to initialize iport 'FDM (LVT, cell sigmas)'.");
    return;
  }
  if (!ofem_) {
    error("Unable to initialize oport 'FEM (TVT, cell sigmas)'.");
    return;
  }
  
  if (!ifdm_->get(ifdmH)) {
    error("Can't get input finite difference field.");
    return;
  }
  
  if (!ifdmH.get_rep()) {
    error("Empty input field.");
    return;
  }
 
  if (ifdmH->mesh()->get_type_description()->get_name() !=
      get_type_description((LatVolMesh *)0)->get_name())
  {
    error("Need a lattice volume field  as input.");
    return;
  }

  if (ifdmH->data_at() != Field::CELL) {
    error("Need cell centered data as input.");
    return;
  }
    
  vector<pair<string, Tensor> > conds;
  if (ifdmH->get_type_name(1) == "Tensor" ||
      (ifdmH->get_type_name(1) == "int" && 
       ifdmH->get_property("conductivity_table", conds))) {

    // first check to see if it's the same field as last time, though...
    //   if so, resend old results
    if (ofemH_.get_rep() && ifdmH->generation == ifdmGen_) {
      remark("Sending cached output field.");
      ofem_->send(ofemH_);
      return;
    }

    // build the new TetVolMesh based on nodes and cells from the LatVolMesh
    LatVolMesh *lvm = dynamic_cast<LatVolMesh*>(ifdmH->mesh().get_rep());
    TetVolMesh *tvm = scinew TetVolMesh;
    Array3<TetVolMesh::Node::index_type> node_lookup(lvm->get_nz(), 
						     lvm->get_ny(),
						     lvm->get_nx());

    // store a table of which nodes are valid (a node is valid if any of
    //   its surrounding cells are valid)
    Array3<char> valid_nodes(lvm->get_nz(), lvm->get_ny(), lvm->get_nx());

    // store a table of which cells are valid
    FData3d<char> mask;

    // if the input Field was a Masked Lattice Vol, copy the mask
    if (ifdmH->get_type_name(0) == "MaskedLatVolField") {
      valid_nodes.initialize(0);
      if (ifdmH->get_type_name(1) == "int") {
	MaskedLatVolField<int> *mlv =
	  dynamic_cast<MaskedLatVolField<int> *>(ifdmH.get_rep());
	mask.copy(mlv->mask());
      } else { // Tensor
	MaskedLatVolField<Tensor> *mlv =
	  dynamic_cast<MaskedLatVolField<Tensor> *>(ifdmH.get_rep());
	mask.copy(mlv->mask());
      }
      LatVolMesh::Node::array_type lat_nodes(8);
      LatVolMesh::Cell::iterator ci, cie;
      lvm->begin(ci); lvm->end(cie);
      while (ci != cie) {
	if (mask[*ci]) {
	  lvm->get_nodes(lat_nodes, *ci);
	  for (int i=0; i<8; i++) 
	    valid_nodes(lat_nodes[i].k_, lat_nodes[i].j_, lat_nodes[i].i_) = 1;
	}
	++ci;
      }
    // otherwise, just initialize the mask to 1's
    } else {
      mask.newsize(lvm->get_nz()-1, lvm->get_ny()-1, lvm->get_nx()-1);
      mask.initialize(1);
      valid_nodes.initialize(1);
    }

    // add all of the nodes from the FDM mesh into the FEM mesh
    // store a node_lookup table to translate from FDM node_indices to
    //   FEM node_indices
    LatVolMesh::Node::iterator ni; lvm->begin(ni);
    LatVolMesh::Node::iterator nie; lvm->end(nie);
    while (ni != nie) {
      if (valid_nodes(ni.k_, ni.j_, ni.i_)) {
	Point p;
	lvm->get_center(p, *ni);
	node_lookup(ni.k_, ni.j_, ni.i_) = tvm->add_point(p);
      }
      ++ni;
    }

    // for each FDM cell, if it's valid bust it into 5 FEM tets
    LatVolMesh::Node::array_type lat_nodes(8);
    TetVolMesh::Node::array_type tet_nodes(8);
    LatVolMesh::Cell::iterator ci, cie;
    lvm->begin(ci); lvm->end(cie);
    while (ci != cie) {
      if (!mask[*ci]) { ++ci; continue; }
      lvm->get_nodes(lat_nodes, *ci);
      for (int i=0; i<8; i++)
	tet_nodes[i] = 
	  node_lookup(lat_nodes[i].k_, lat_nodes[i].j_, lat_nodes[i].i_);
      bool parity = (ci.k_ + ci.j_ + ci.i_) % 2;
      if (parity) {
	tvm->add_tet(tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[5]);
	tvm->add_tet(tet_nodes[0], tet_nodes[2], tet_nodes[3], tet_nodes[7]);
	tvm->add_tet(tet_nodes[0], tet_nodes[2], tet_nodes[5], tet_nodes[7]);
	tvm->add_tet(tet_nodes[0], tet_nodes[4], tet_nodes[5], tet_nodes[7]);
	tvm->add_tet(tet_nodes[2], tet_nodes[5], tet_nodes[6], tet_nodes[7]);
      } else {
	tvm->add_tet(tet_nodes[1], tet_nodes[0], tet_nodes[3], tet_nodes[4]);
	tvm->add_tet(tet_nodes[1], tet_nodes[3], tet_nodes[2], tet_nodes[6]);
	tvm->add_tet(tet_nodes[1], tet_nodes[3], tet_nodes[4], tet_nodes[6]);
	tvm->add_tet(tet_nodes[1], tet_nodes[5], tet_nodes[4], tet_nodes[6]);
	tvm->add_tet(tet_nodes[3], tet_nodes[4], tet_nodes[7], tet_nodes[6]);
      }
      ++ci;
    }

    // now that the mesh has been built, we have to set conductivities (fdata)
    TetVolField<int> *tv;
    if (ifdmH->get_type_name(1) == "Tensor") {
      // we have tensors, no need to get sigmas out of the property manager.
      // allocate the new finite element mesh so the node indices remain
      //   consistent with the finite difference nodes
      LatVolField<Tensor> *lv = 
	dynamic_cast<LatVolField<Tensor> *>(ifdmH.get_rep());
      tv = scinew TetVolField<int>(tvm, Field::CELL);
      LatVolMesh::Cell::iterator ci;
      lvm->begin(ci);
      TetVolMesh::Cell::iterator ci_tet; tvm->begin(ci_tet);
      TetVolMesh::Cell::iterator ci_tet_end; tvm->end(ci_tet_end);
      int count=0;
      while (ci_tet != ci_tet_end) {
	conds.push_back(pair<string, Tensor>(to_string((int)*ci_tet),
					     lv->fdata()[*ci]));
	for (int i=0; i<5; i++, ++ci_tet) {
	  tv->fdata()[*ci_tet]=count;
	}
	count++;
	++ci;
      }
    } else {
      // we have int's -- the "conds" array has the tensors
      LatVolField<int> *lv = 
	dynamic_cast<LatVolField<int> *>(ifdmH.get_rep());
      tv = scinew TetVolField<int>(tvm, Field::CELL);
      LatVolMesh::Cell::iterator ci;
      lvm->begin(ci);
      TetVolMesh::Cell::iterator ci_tet; tvm->begin(ci_tet);
      TetVolMesh::Cell::iterator ci_tet_end; tvm->end(ci_tet_end);
      while (ci_tet != ci_tet_end) {
	for (int i=0; i<5; i++, ++ci_tet) {
	  tv->fdata()[*ci_tet]=lv->fdata()[*ci];
	}
	++ci;
      }      
    }
    msgStream_ << "Conds->size() == "<<conds.size()<<"\n";
    tv->set_property("data_storage", string("table"), false);
    tv->set_property("name", string("conductivity"), false);
    tv->set_property("conductivity_table", conds, false);
    ofemH_ = tv;
    ofem_->send(ofemH_);
  } else {
    error("Could not get tensors from the lattice vol.");
    return;
  }
}

} // End namespace BioPSE
