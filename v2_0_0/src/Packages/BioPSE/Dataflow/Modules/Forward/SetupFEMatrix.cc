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
 *  SetupFEMatrix.cc:  Setups the global finite element matrix
 *
 *  Written by:
 *   Ruth Nicholson Klepfer
 *   Department of Bioengineering
 *   University of Utah
 *   October 1994
 *
 *  Modified:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   March 2001    
 *  Copyright (C) 1994, 2001 SCI Group
 *
 * Modified:
 *  Sascha Moehrs
 *  January 2003
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildFEMatrix.h>
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildHexFEMatrix.h>
#include <Core/Datatypes/HexVolField.h>

#include <iostream>

using std::endl;

namespace BioPSE {

using namespace SCIRun;
typedef LockingHandle<TetVolField<int> >    CondMeshHandleTet;
typedef LockingHandle<HexVolField<int> >    LHHexVolField;

class SetupFEMatrix : public Module {
  
  //! Private data
  FieldIPort*        iportField_;
  MatrixOPort*       oportMtrx_;
  
  GuiInt             uiUseCond_;
  int                lastUseCond_;
  
  GuiInt             uiUseBasis_;
  int                lastUseBasis_;
  
  MatrixHandle       hGblMtrx_;
  int                gen_;

  //! For the per-conductivity bases
  int                meshGen_;
  Array1<Array1<double> > dataBasis_;
  MatrixHandle       AmatH_;  // shape information

  GuiString          nprocessors_;

  void build_basis_matrices(FieldHandle fldH, unsigned int nconds, 
			    double unitsScale, int num_procs);
  MatrixHandle build_composite_matrix(const vector<pair<string,Tensor> >&tens);

  bool tet;

public:
  
  //! Constructor/Destructor
  SetupFEMatrix(GuiContext *context);
  virtual ~SetupFEMatrix();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(SetupFEMatrix);


SetupFEMatrix::SetupFEMatrix(GuiContext *context) : 
  Module("SetupFEMatrix", context, Filter, "Forward", "BioPSE"), 
  uiUseCond_(context->subVar("UseCondTCL")),
  lastUseCond_(-1),
  uiUseBasis_(context->subVar("UseBasisTCL")),
  lastUseBasis_(-1),
  gen_(-1),
  meshGen_(-1),
  nprocessors_(context->subVar("nprocs"))
{}


SetupFEMatrix::~SetupFEMatrix()
{
}


void
SetupFEMatrix::build_basis_matrices(FieldHandle fldH, 
				    unsigned int nconds,
				    double unitsScale,
				    int num_procs)
{
  TetVolFieldIntHandle tvH;
  tvH = dynamic_cast<TetVolField<int> *>(fldH.get_rep());
  Tensor zero(0);
  Tensor identity(1);

  MatrixHandle aH;
  vector<pair<string, Tensor> > tens(nconds, pair<string, Tensor>("", zero));
  BuildFEMatrix::build_FEMatrix(tvH, tens, aH, unitsScale, num_procs);
  AmatH_ = aH;
  AmatH_.detach(); //! Store our matrix shape

  dataBasis_.resize(nconds);
  for (unsigned int i=0; i<nconds; i++) {
    tens[i].first=to_string(i);
    tens[i].second=identity;
    BuildFEMatrix::build_FEMatrix(tvH, tens, aH, unitsScale, num_procs);
    SparseRowMatrix *m = dynamic_cast<SparseRowMatrix*>(aH.get_rep());
    dataBasis_[i].resize(m->nnz);
    for (int j=0; j<m->nnz; j++)
      dataBasis_[i][j] = m->a[j];
    tens[i].second=zero;
  }
}


//! Scale the basis matrix data by the conductivities and sum
MatrixHandle
SetupFEMatrix::build_composite_matrix(const vector<pair<string, Tensor> > &tens)
{
  MatrixHandle fem_mat = AmatH_;
  fem_mat.detach();
  SparseRowMatrix *m = dynamic_cast<SparseRowMatrix*>(fem_mat.get_rep());
  double *sum = m->a;
  for (unsigned int i=0; i<tens.size(); i++)
  {
    double weight = tens[i].second.mat_[0][0];
    for (int j=0; j<dataBasis_[i].size(); j++)
    {
      sum[j] += weight*dataBasis_[i][j];
    }
  }
  return fem_mat;
}


void
SetupFEMatrix::execute()
{
  
  iportField_ = (FieldIPort *)get_iport("Mesh");
  oportMtrx_ = (MatrixOPort *)get_oport("Stiffness Matrix");

  if (!iportField_) {
    error("Unable to initialize iport 'Mesh'.");
    return;
  }
  if (!oportMtrx_) {
    error("Unable to initialize oport 'Stiffness Matrix'.");
    return;
  }

  //! Validate input
  FieldHandle hField;
  if(!iportField_->get(hField) || !hField.get_rep()){
    error("Can not get input field.");
    return;
  }

  if (hField->get_type_name(0) == "TetVolField" &&
      (hField->get_type_name(1) == "int")) {
    tet = true;
    remark("Input is a 'TetVolField'");
  }
  else {
    if((hField->get_type_name(0) == "HexVolField") &&
       (hField->get_type_name(1) == "int")) {
      tet = false;
      remark("Input is a 'HexVolField'");
    }
    else {
      error("input field is not 'TetVolField' nor 'HexVolField'.");
      return;
    }
  }

  CondMeshHandleTet hCondMeshTet;
  LHHexVolField hLHHexVolField;

  if(tet)
    hCondMeshTet = dynamic_cast<TetVolField<int>* >(hField.get_rep());
  else
    hLHHexVolField = dynamic_cast<HexVolField<int>* >(hField.get_rep());

  if (hField->generation == gen_ 
      && hGblMtrx_.get_rep() 
      && lastUseCond_==uiUseCond_.get()
      && lastUseBasis_==uiUseBasis_.get()) {
    oportMtrx_->send(hGblMtrx_);
    return;
  }

  //! Either use supplied tensors, or make an array of identity tensors
  vector<pair<string, Tensor> > tens;
 
  if(tet) {
    if (uiUseCond_.get()==1 &&
	hCondMeshTet->get_property("conductivity_table", tens)){
      remark("Using supplied conductivity tensors.");
    } else {
      remark("Using identity conductivity tensors.");
      pair<int,int> minmax;
      minmax.second=1;
      field_minmax(*(hCondMeshTet.get_rep()), minmax);
      tens.resize(minmax.second+1);
      vector<double> t(6);
      t[0] = t[3] = t[5] = 1;
      t[1] = t[2] = t[4] = 0;
      Tensor ten(t);
      for (unsigned int i = 0; i < tens.size(); i++) {
	tens[i] = pair<string, Tensor>(to_string((int)i), ten);
      }
    }
  }
  else {
    if((uiUseCond_.get()==1) && (hLHHexVolField->get_property("conductivity_table", tens))) {
      remark("Using supplied conductivity tensors.");
    } 
    else {
      remark("Using identity conductivity tensors.");
      pair<int,int> minmax;
      minmax.second=1;
      field_minmax(*(hLHHexVolField.get_rep()), minmax);
      tens.resize(minmax.second+1);
      vector<double> t(6);
      t[0] = t[3] = t[5] = 1;
      t[1] = t[2] = t[4] = 0;
      Tensor ten(t);
      for (unsigned int i = 0; i < tens.size(); i++) {
	tens[i] = pair<string, Tensor>(to_string((int)i), ten);
      }
    }
  }

  //! Cache data values for comparison next time
  gen_ = hField->generation;
  lastUseCond_ = uiUseCond_.get();
  lastUseBasis_ = uiUseBasis_.get();


  //! Compute the scale of this geometry based on its "units" property
  double unitsScale = 1.;
  string units;
  if (uiUseCond_.get()==1 /*&& hField->mesh()->get_property("units", units)*/) {
    if(tet) {
      if(hCondMeshTet->mesh()->get_property("units", units)) {
	msgStream_  << "units = "<< units <<"\n";
	if (units == "mm") unitsScale = 1./1000;
	else if (units == "cm") unitsScale = 1./100;
	else if (units == "dm") unitsScale = 1./10;
	else if (units == "m") unitsScale = 1./1;
	else {
	  warning("Did not recognize units of mesh '" + units + "'.");
	}
	msgStream_ << "unitsScale = "<< unitsScale <<"\n";
      }
    }
    else {
      if(hLHHexVolField->mesh()->get_property("units", units)) {
	msgStream_  << "units = "<< units <<"\n";
	if (units == "mm") unitsScale = 1./1000;
	else if (units == "cm") unitsScale = 1./100;
	else if (units == "dm") unitsScale = 1./10;
	else if (units == "m") unitsScale = 1./1;
	else {
	  warning("Did not recognize units of mesh '" + units + "'.");
	}
	msgStream_ << "unitsScale = "<< unitsScale <<"\n";
      }
    }
  }

  int nprocs = atoi(nprocessors_.get().c_str());
  if (nprocs > Thread::numProcessors() * 4)
  {
    nprocs = Thread::numProcessors() * 4;
  }
  
  if(!tet) {
    BuildHexFEMatrix *hexmat = scinew BuildHexFEMatrix(hLHHexVolField, tens, unitsScale);
    hGblMtrx_ = hexmat->buildMatrix();
  }
  else {
    //! If the user wants to use basis matrices, 
    //!    first check to see if we need to recompute them
    if (lastUseBasis_) {
      if (hField->mesh()->generation != meshGen_ || 
	  tens.size() != (unsigned int)(dataBasis_.size())) {
	meshGen_ = hField->mesh()->generation;
	//! Need to build basis matrices
	build_basis_matrices(hField, tens.size(), unitsScale, nprocs);
      }
      //! Have basis matrices, compute combined matrix
      hGblMtrx_ = build_composite_matrix(tens);
    } else {
      BuildFEMatrix::build_FEMatrix(hCondMeshTet, tens, hGblMtrx_,
				    unitsScale, nprocs);
    }
  }

  oportMtrx_->send(hGblMtrx_);
}


} // End namespace BioPSE
