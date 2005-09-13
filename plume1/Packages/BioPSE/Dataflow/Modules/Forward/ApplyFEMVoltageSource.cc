/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Containers/StringUtil.h>
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

  //! Obtaining handles to computation objects
  FieldHandle hField;
  
  if (!iportField_->get(hField) || !hField.get_rep()) {
    error("Can't get handle to mesh.");
    return;
  }

  TetVolMeshHandle mesh;
  TetVolField<int> *field;

  if (hField->get_type_name(0)!="TetVolField" && hField->get_type_name(1)!="int"){
    error("Supplied field is not of type TetVolField<int>.");
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
	warning("The input field doesn't contain Dirichlet boundary conditions.");
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
      ((unsigned int)(rhsIn->nrows()) == nsize))
  {
    string units;
    if (rhsIn->get_property("units", units))
      rhs->set_property("units", units, false);

    for (unsigned int i=0; i < nsize; i++) 
      (*rhs)[i]=(*rhsIn)[i];
  }
  else{
    rhs->set_property("units", string("volts"), false);
 //   msgStream_ << "The supplied RHS doesn't correspond to the mesh in size. Creating own one..." << endl;
    rhs->zero();
  }
  

  MatrixHandle  hMatIn;
  if (!iportMatrix_->get(hMatIn) || !hMatIn.get_rep()) {
    error("Need a stiffness matrix.");
    return;
  }
  SparseRowMatrix *matIn;
  if (!(matIn = dynamic_cast<SparseRowMatrix*>(hMatIn.get_rep()))) {
    error("Input stiffness matrix wasn't sparse.");
    return;
  }
  if (matIn->nrows() != matIn->ncols()) {
    error("Input stiffness matrix wasn't square.");
    return;
  }
  if (nsize != (unsigned int)(matIn->nrows())) {
    error("Input stiffness matrix was " + to_string(nsize)  +
	  " nodes, matrix has " + to_string(matIn->nrows()) + " rows.");
    return;
  }

  SparseRowMatrix *mat = matIn->clone();

  //! adjusting matrix for Dirichlet BC
  int *idcNz;
  double *valNz;
  int idcNzsize;
  int idcNzstride;

  TetVolMesh::Node::array_type nind;
  vector<double> dbc;
  unsigned int idx;
  for(idx = 0; idx<dirBC.size(); ++idx){
    int ni = dirBC[idx].first;
    double val = dirBC[idx].second;
    
    // -- getting column indices of non-zero elements for the current row
    mat->getRowNonzerosNoCopy(ni, idcNzsize, idcNzstride, idcNz, valNz);
    
    // -- updating rhs
    for (int i=0; i<idcNzsize; ++i){
      int j = idcNz?idcNz[i*idcNzstride]:i;
      (*rhs)[j] += - val * valNz[i*idcNzstride]; 
    }
  }
  
  //! zeroing matrix row and column corresponding to the dirichlet nodes
  for(idx = 0; idx<dirBC.size(); ++idx){
    int ni = dirBC[idx].first;
    double val = dirBC[idx].second;
    
    mat->getRowNonzerosNoCopy(ni, idcNzsize, idcNzstride, idcNz, valNz);
      
    for (int i=0; i<idcNzsize; ++i){
      int j = idcNz?idcNz[i*idcNzstride]:i;
      mat->put(ni, j, 0.0);
      mat->put(j, ni, 0.0); 
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
