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
 *  ApplyFEMCurrentSource.cc:  Builds the RHS of the FE matrix for current sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TetVol.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace BioPSE {

using namespace SCIRun;

class ApplyFEMCurrentSource : public Module {

  FieldIPort*   iportField_;
  FieldIPort*   iportSource_;
  MatrixIPort*  iportRhs_;

  MatrixOPort*  oportRhs_;

  int gen;
  TetVolMesh::cell_index loc;

public:
  ApplyFEMCurrentSource(const clString& id);
  virtual ~ApplyFEMCurrentSource();
  virtual void execute();
};

extern "C" Module* make_ApplyFEMCurrentSource(const clString& id)
{
  return scinew ApplyFEMCurrentSource(id);
}

ApplyFEMCurrentSource::ApplyFEMCurrentSource(const clString& id)
  : Module("ApplyFEMCurrentSource", id, Filter) 
{
  // Create the input port
  iportField_ = scinew FieldIPort(this, "Mesh", FieldIPort::Atomic);
  add_iport(iportField_);

  iportSource_=scinew FieldIPort(this, "Source", MatrixIPort::Atomic);
  add_iport(iportSource_);

  iportRhs_=scinew MatrixIPort(this, "Input RHS", MatrixIPort::Atomic);
  add_iport(iportRhs_);

  // Create the output ports
  oportRhs_=scinew MatrixOPort(this,"Output RHS", MatrixIPort::Atomic);
  add_oport(oportRhs_);
}

ApplyFEMCurrentSource::~ApplyFEMCurrentSource()
{
}

void ApplyFEMCurrentSource::execute()
{
  FieldHandle hField;
  
  if (!iportField_->get(hField) || !hField.get_rep()) {
    msgStream_ << "Cann't get handle to mesh. Returning..." << endl;
    return;
  }

  TetVolMeshHandle hMesh;
  LockingHandle<TetVol<int> > hCondField;

  if (hField->get_type_name(0)!="TetVol" && hField->get_type_name(1)!="int"){
    msgStream_ << "Supplied field is not of type TetVol<int>. Returning..." << endl;
    return;
  }
  else {
    hCondField = dynamic_cast<TetVol<int>*> (hField.get_rep());
    hMesh = hCondField->get_typed_mesh();
  }
  
  FieldHandle hSource;
  if (!iportSource_->get(hSource) || !hSource.get_rep()) {
    msgStream_ << "Cann't get handle to source field. Returning..." << endl;
    return;
  }
  
  LockingHandle<TetVol<Vector> > hDipField;
  
  if (hSource->get_type_name(0)!="TetVol" || hSource->get_type_name(1)!="Vector"){
    msgStream_ << "Supplied field is not of type TetVol<Vector>. Returning..." << endl;
    return;
  }
  else {
    hDipField = dynamic_cast<TetVol<Vector>*> (hSource.get_rep());
  }
  
  MatrixHandle  hRhsIn;
  ColumnMatrix* rhsIn;
  
  ColumnMatrix* rhs = scinew ColumnMatrix(hMesh->nodes_size());
 
  // -- if the user passed in a vector the right size, copy it into ours 
  if (iportRhs_->get(hRhsIn) && 
      (rhsIn=dynamic_cast<ColumnMatrix*>(hRhsIn.get_rep())) && 
      (rhsIn->nrows()==hMesh->nodes_size())){
    for (int i=0; i<hMesh->nodes_size(); i++) 
      (*rhs)[i]=(*rhsIn)[i];
  }
  else{
    msgStream_ << "The supplied RHS doesn't correspond to the mesh in size. Creating own one..." << endl;
    rhs->zero();
  }
  
  TetVolMesh::node_iterator ii;
  
  for (ii=hDipField->get_typed_mesh()->node_begin(); ii!=hDipField->get_typed_mesh()->node_end(); ++ii) {
    
    Vector dir = hDipField->value(*ii);
    Point p;
    hDipField->get_typed_mesh()->get_point(p, *ii);
   
    if (hMesh->locate(loc, p)) {
      double s1, s2, s3, s4;
      Vector g1, g2, g3, g4;
      hMesh->get_gradient_basis(loc, g1, g2, g3, g4);
  
      s1=Dot(g1,dir);
      s2=Dot(g2,dir);
      s3=Dot(g3,dir);
      s4=Dot(g4,dir);
      TetVolMesh::node_array cell_nodes;
      
      hMesh->get_nodes(cell_nodes, loc);

      (*rhs)[cell_nodes[0]]+=s1;
      (*rhs)[cell_nodes[1]]+=s2;
      (*rhs)[cell_nodes[2]]+=s3;
      (*rhs)[cell_nodes[3]]+=s4;
     
#if 0
      msgStream_ << "ApplyFEMCurrentSource :: Here's the RHS vector: ";
      for (int jj=0; jj<hMesh->nodes_size(); jj++) {
	if ((*rhs)[jj]!=0){
	  msgStream_ << "i = " << jj << ", val=" << (*rhs)[jj]<< endl;
	}
      }
      msgStream_ << "ApplyFEMCurrentSource :: Here's the dipole: ";
      msgStream_ << "Pos: " << p << ", Moment=" << dir  << endl;
#endif

    } else {
      loc=0;
      dir=Vector(0,0,0);
      msgStream_ << "Dipole: "<< p <<" not located within mesh!\n";
    }
    
    gen=hSource->generation;
  }

  oportRhs_->send(MatrixHandle(rhs)); 
}
} // End namespace BioPSE




