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
 *  ApplyFEMVoltageSource.cc:  Builds the RHS of the FE matrix for current sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *  Modified by:
 *   Alexei Samsonov
 *   March 2001
 *  Copyright (C) 1999, 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace BioPSE {

using namespace SCIRun;

class ApplyFEMVoltageSource : public Module {
public:
  GuiString bcFlag_; // "none", "GroundZero", or "DirSub"

  //! Constructor/Destructor
  ApplyFEMVoltageSource(GuiContext *context);
  virtual ~ApplyFEMVoltageSource();
  
  //! Public methods
  virtual void execute();
};

DECLARE_MAKER(ApplyFEMVoltageSource)


ApplyFEMVoltageSource::ApplyFEMVoltageSource(GuiContext *context)
  : Module("ApplyFEMVoltageSource", context, Filter, "Forward", "BioPSE"),
    bcFlag_(context->subVar("bcFlag"))
{
}

ApplyFEMVoltageSource::~ApplyFEMVoltageSource()
{
}

void ApplyFEMVoltageSource::execute()
{
  FieldIPort *iportField_ = (FieldIPort *)get_iport("Mesh");
  MatrixIPort *iportMatrix_ = (MatrixIPort *)get_iport("Stiffness Matrix");
  MatrixIPort *iportRhs_ = (MatrixIPort *)get_iport("RHS");
  MatrixOPort *oportMatrix_ = (MatrixOPort *)get_oport("Forward Matrix");
  MatrixOPort *oportRhs_ = (MatrixOPort *)get_oport("RHS");

  if (!iportField_) {
    error("Unable to initialize iport 'Mesh'.");
    return;
  }
  if (!iportMatrix_) {
    error("Unable to initialize iport 'Stiffness Matrix'.");
    return;
  }
  if (!iportRhs_) {
    error("Unable to initialize iport 'RHS'.");
    return;
  }
  if (!oportMatrix_) {
    error("Unable to initialize oport 'Forward Matrix'.");
    return;
  }
  if (!oportRhs_) {
    error("Unable to initialize oport 'RHS'.");
    return;
  }
  
  //! Obtaining handles to computation objects
  FieldHandle hField;
  
  if (!iportField_->get(hField) || !hField.get_rep()) {
    msgStream_ << "Can't get handle to mesh. Returning..." << endl;
    return;
  }

  TetVolMeshHandle mesh;
  TetVolField<int> *field;

  if (hField->get_type_name(0)!="TetVolField" && hField->get_type_name(1)!="int"){
    msgStream_ << "Supplied field is not of type TetVolField<int>. Returning..." << endl;
    return;
  }
  else {
    field = dynamic_cast<TetVolField<int>*> (hField.get_rep());
    mesh = field->get_typed_mesh();
  }
  
  //-- polling Field for Dirichlet BC
  vector<pair<int, double> > dirBC;
  string bcFlag = bcFlag_.get();
  if (bcFlag != "none") {
    if (bcFlag=="GroundZero"){
//      dirBC.erase(dirBC.begin(), dirBC.end());
      dirBC.push_back(pair<int, double>(0,0.0));
    } else { // bcFlag == DirSub
      if (!hField->get_property("dirichlet", dirBC)){
	msgStream_ << "The input field doesn't contain Dirichlet boundary conditions" << endl;
//	return;
      }
    }
  }

  MatrixHandle  hRhsIn;
  ColumnMatrix* rhsIn;
  
  TetVolMesh::Node::size_type nsize; mesh->size(nsize);
  ColumnMatrix* rhs = scinew ColumnMatrix(nsize);
 
  // -- if the user passed in a vector the right size, copy it into ours 
  if (iportRhs_->get(hRhsIn) && 
      (rhsIn=dynamic_cast<ColumnMatrix*>(hRhsIn.get_rep())) && 
      (rhsIn->nrows() == nsize))
  {
    string units;
    if (rhsIn->get_property("units", units))
      rhs->set_property("units", units, false);

    for (int i=0; i < nsize; i++) 
      (*rhs)[i]=(*rhsIn)[i];
  }
  else{
    rhs->set_property("units", string("volts"), false);
 //   msgStream_ << "The supplied RHS doesn't correspond to the mesh in size. Creating own one..." << endl;
    rhs->zero();
  }
  

  MatrixHandle  hMatIn;
  if (!iportMatrix_->get(hMatIn) || !hMatIn.get_rep()) {
    cerr << "Error - need a stiffness matrix.\n";
    return;
  }
  SparseRowMatrix *matIn;
  if (!(matIn = dynamic_cast<SparseRowMatrix*>(hMatIn.get_rep()))) {
    cerr << "Error - input stiffness matrix wasn't sparse.\n";
    return;
  }
  if (matIn->nrows() != matIn->ncols()) {
    cerr << "Error - input stiffness matrix wasn't square.\n";
    return;
  }
  if (nsize != matIn->nrows()) {
    cerr << "Error - input stiffness matrix was "<<nsize<<" nodes, matrix has "<<matIn->nrows()<<" rows.\n";
    return;
  }

  SparseRowMatrix *mat = matIn->clone();

  //! adjusting matrix for Dirichlet BC
  Array1<int> idcNz;
  Array1<double> valNz;

  TetVolMesh::Node::array_type nind;
  vector<double> dbc;
  unsigned int idx;
  for(idx = 0; idx<dirBC.size(); ++idx){
    int ni = dirBC[idx].first;
    double val = dirBC[idx].second;
    
    // -- getting column indices of non-zero elements for the current row
    mat->getRowNonzeros(ni, idcNz, valNz);
    
    // -- updating rhs
    for (int i=0; i<idcNz.size(); ++i){
      int j = idcNz[i];
      (*rhs)[j] +=-val*valNz[i]; 
    }
  }
  
  //! zeroing matrix row and column corresponding to the dirichlet nodes
  for(idx = 0; idx<dirBC.size(); ++idx){
    int ni = dirBC[idx].first;
    double val = dirBC[idx].second;
    
    mat->getRowNonzeros(ni, idcNz, valNz);
      
    for (int i=0; i<idcNz.size(); ++i){
      int j = idcNz[i];
      mat->put(ni, j, 0);
      mat->put(j, ni, 0); 
    }
      
    //! updating dirichlet node and corresponding entry in rhs
    mat->put(ni, ni, 1);
    (*rhs)[ni] = val;
  }

  //! Sending result
  oportRhs_->send(MatrixHandle(rhs)); 
  oportMatrix_->send(MatrixHandle(mat)); 
}
} // End namespace BioPSE
