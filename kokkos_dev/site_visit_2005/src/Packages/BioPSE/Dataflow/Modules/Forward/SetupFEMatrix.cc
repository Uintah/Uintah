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
 *
 *  Lorena Kreda, Northeastern University, November 2003
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/TriSurfField.h>
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
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildTriFEMatrix.h>

#include <iostream>

using std::endl;

namespace BioPSE {

using namespace SCIRun;
typedef LockingHandle<TetVolField<int> >    TetVolFieldIntHandle;
typedef LockingHandle<TetVolField<Tensor> > TetVolFieldTensorHandle;

typedef LockingHandle<HexVolField<int> >    HexVolFieldIntHandle;
typedef LockingHandle<HexVolField<Tensor> > HexVolFieldTensorHandle;

typedef LockingHandle<TriSurfField<int> >   TriSurfFieldIntHandle;
typedef LockingHandle<TriSurfField<Tensor> > TriSurfFieldTensorHandle;


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

  void build_basis_matrices(TetVolFieldIntHandle tviH, unsigned int nconds, 
			    double unitsScale, int num_procs);
  void build_TriBasis_matrices(TriSurfFieldIntHandle, unsigned int nconds, 
			       double unitsScale, int num_procs);
  MatrixHandle build_composite_matrix(const vector<pair<string,Tensor> >&tens);

  bool tet;
  bool hex;
  bool tri;

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
SetupFEMatrix::build_basis_matrices(TetVolFieldIntHandle tviH,
				    unsigned int nconds,
				    double unitsScale,
				    int num_procs)
{
  TetVolFieldTensorHandle tvtH;
  Tensor zero(0);
  Tensor identity(1);

  MatrixHandle aH;
  vector<pair<string, Tensor> > tens(nconds, pair<string, Tensor>("", zero));
  BuildFEMatrix::build_FEMatrix(tviH, tvtH, true, tens, 
				aH, unitsScale, num_procs);
  AmatH_ = aH;
  AmatH_.detach(); //! Store our matrix shape

  dataBasis_.resize(nconds);
  for (unsigned int i=0; i<nconds; i++) {
    tens[i].first=to_string(i);
    tens[i].second=identity;
    BuildFEMatrix::build_FEMatrix(tviH, tvtH, true, tens, aH, 
				  unitsScale, num_procs);
    SparseRowMatrix *m = dynamic_cast<SparseRowMatrix*>(aH.get_rep());
    dataBasis_[i].resize(m->nnz);
    for (int j=0; j<m->nnz; j++)
      dataBasis_[i][j] = m->a[j];
    tens[i].second=zero;
  }
}

void SetupFEMatrix::build_TriBasis_matrices(TriSurfFieldIntHandle tsiH, 
					    unsigned int nconds,
					    double unitsScale,
					    int num_procs) 
{
  TriSurfFieldTensorHandle tstH;
  Tensor zero(0);
  Tensor identity(1);

  MatrixHandle aH;
  vector<pair<string, Tensor> > tens(nconds, pair<string, Tensor>("", zero));
  BuildTriFEMatrix::build_FEMatrix(tsiH, tstH, true, tens, 
				   aH, unitsScale, num_procs);
  AmatH_ = aH;
  AmatH_.detach(); //! Store our matrix shape
  dataBasis_.resize(nconds);
  for (unsigned int i=0; i<nconds; i++) {
    tens[i].first=to_string(i);
    tens[i].second=identity;
    BuildTriFEMatrix::build_FEMatrix(tsiH, tstH, true, tens, aH, 
				     unitsScale, num_procs);
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

  //! Validate input
  FieldHandle hField;
  if(!iportField_->get(hField) || !hField.get_rep()){
    error("Can not get input field.");
    return;
  }
  
  tet = false;
  hex = false;
  tri = false;

  bool index_based = true;
  if (hField->get_type_name(0) == "TetVolField") 
  {
    remark("Input is a 'TetVolField'");
    if (hField->get_type_name(1) == "int") {
      tet = true;
    } else if (hField->get_type_name(1) == "Tensor") {
      tet = true;
      index_based = false;
    } else {
      error("Input TetVolField is not of type 'int' or 'Tensor'.");
      return;
    }
    if (hField->basis_order() != 0)
    {
      error("Input TetVolField field must contain cell centered data.");
      return;
    }
  } 
  else if (hField->get_type_name(0) == "HexVolField") 
  {
    remark("Input is a 'HexVolField'");
    if (hField->get_type_name(1) == "int") {
      hex = true;
    } else if (hField->get_type_name(1) == "Tensor") {
      hex = true;
      index_based = false;
    } else {
      error("Input HexVolField is not of type 'int' or 'Tensor'.");
      return;
    }
    if (hField->basis_order() != 0)
    {
      error("Input HexVolField field must contain cell centered data.");
      return;
    }
  }
  else if (hField->get_type_name(0) == "TriSurfField") 
  {
    remark("Input is a 'TriSurfField'");
    if (hField->get_type_name(1) == "int") {
      tri = true;
    }  else if (hField->get_type_name(1) == "Tensor") {
       tri = true;
       index_based = false;
    } else {
      error("Input TriSurfField is not of type 'int' or 'Tensor'.");
      return;
    }
    if (hField->basis_order() != 0)
    {
      error("Input TriSurfField field must have constant basis.");
      return;
    }
  }
  else 
  {
    error("Input field is not 'TetVolField' or 'HexVolField' or 'TriSurfField'.");
    return;
  }

  TetVolFieldIntHandle tvfiH;
  TetVolFieldTensorHandle tvftH;

  HexVolFieldIntHandle hvfiH;
  HexVolFieldTensorHandle hvftH;

  TriSurfFieldIntHandle tsfiH;
  TriSurfFieldTensorHandle tsftH;

  if (tet) 
  {
    if (index_based)
      tvfiH = dynamic_cast<TetVolField<int>* >(hField.get_rep());
    else
      tvftH = dynamic_cast<TetVolField<Tensor>* >(hField.get_rep());
  } 
  else if (hex)
  {
    if (index_based)
      hvfiH = dynamic_cast<HexVolField<int>* >(hField.get_rep());
    else
      hvftH = dynamic_cast<HexVolField<Tensor>* >(hField.get_rep());
  }
  else if (tri)
  {
    if (index_based)
      tsfiH = dynamic_cast<TriSurfField<int>* >(hField.get_rep());
    else
      tsftH = dynamic_cast<TriSurfField<Tensor>* >(hField.get_rep());
  } 

  if (hField->generation == gen_ 
      && hGblMtrx_.get_rep() 
      && lastUseCond_==uiUseCond_.get()
      && lastUseBasis_==uiUseBasis_.get()) {
    oportMtrx_->send(hGblMtrx_);
    return;
  }

  //! Either use supplied tensors, or make an array of identity tensors
  vector<pair<string, Tensor> > tens;
 
  if (tet && index_based) 
  {
    if (uiUseCond_.get()==1 &&
	tvfiH->get_property("conductivity_table", tens)) {
      remark("Using supplied conductivity tensors.");
    } else {
      remark("Using identity conductivity tensors.");
      pair<int,int> minmax;
      minmax.second=1;
      field_minmax(*(tvfiH.get_rep()), minmax);
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
  else if (hex && index_based) 
  {
    if((uiUseCond_.get()==1) && 
       hvfiH->get_property("conductivity_table", tens)) {
      remark("Using supplied conductivity tensors.");
    } else {
      remark("Using identity conductivity tensors.");
      pair<int,int> minmax;
      minmax.second=1;
      field_minmax(*(hvfiH.get_rep()), minmax);
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
  else if (tri && index_based)
  {
    if (uiUseCond_.get()==1 &&
	tsfiH->get_property("conductivity_table", tens)) {
      remark("Using supplied conductivity tensors.");
    } else {
      remark("Using identity conductivity tensors.");
      pair<int,int> minmax;
      minmax.second=1;
      field_minmax(*(tsfiH.get_rep()), minmax);
      tens.resize(minmax.second+1);
      vector<double> t(6);
      t[0] = t[3] = t[5] = 1;
      t[1] = t[2] = t[4] = 0;
      Tensor ten(t);
      for (unsigned int i = 0; i < tens.size(); i++)
      {
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
      if(hField->get_property("units", units)) {
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
    else if (hex) {
      if(hField->mesh()->get_property("units", units)) {
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
    else if (tri) {
      if(hField->mesh()->get_property("units", units)) {
          msgStream_ << "units = "<<units<<"\n";
          if (units == "mm") unitsScale = 1./1000;
          else if (units == "cm") unitsScale = 1./100;
          else if (units == "dm") unitsScale = 1./10;
          else if (units == "m") unitsScale = 1./1;
          else {
            warning("Did not recognize units of mesh '" + units + "'.");
          }
          msgStream_ << "unitsScale = "<<unitsScale<<"\n";
       }
    }
  }

  int nprocs = atoi(nprocessors_.get().c_str());
  if (nprocs > Thread::numProcessors() * 4)
  {
    nprocs = Thread::numProcessors() * 4;
  }
 
  if (hex) {      // HEXES
    BuildHexFEMatrix *hexmat = 
      scinew BuildHexFEMatrix(hvfiH, hvftH, 
			      index_based, tens, unitsScale);
    hGblMtrx_ = hexmat->buildMatrix();
  } else if (tet && index_based && lastUseBasis_) { // TETS -- indexed & bases
    //! If the user wants to use basis matrices, 
    //!    first check to see if we need to recompute them
    if (hField->mesh()->generation != meshGen_ || 
	tens.size() != (unsigned int)(dataBasis_.size())) {
      meshGen_ = hField->mesh()->generation;
      //! Need to build basis matrices
      build_basis_matrices(tvfiH, tens.size(), unitsScale, nprocs);
    }
    //! Have basis matrices, compute combined matrix
    hGblMtrx_ = build_composite_matrix(tens);
  } else if (tet) {          // TETS -- (non-indexed or bases)
    BuildFEMatrix::build_FEMatrix(tvfiH, tvftH,
				  index_based, tens, hGblMtrx_, 
				  unitsScale, nprocs);
  } else if (tri) {  // TRIS -- indexed 
    //! If the user wants to use basis matrices, 
    //!    first check to see if we need to recompute them
    if (lastUseBasis_) {
      if (hField->mesh()->generation != meshGen_ || 
	  tens.size() != (unsigned int)(dataBasis_.size())) {
        meshGen_ = hField->mesh()->generation;
        //! Need to build basis matrices
        build_TriBasis_matrices(tsfiH, (int)tens.size(), unitsScale, nprocs);
      }
      //! Have basis matrices, compute combined matrix
      hGblMtrx_ = build_composite_matrix(tens);
    } else if (tri) {
      BuildTriFEMatrix::build_FEMatrix(tsfiH, tsftH, index_based,
				       tens, hGblMtrx_, unitsScale);
    }  
  }
  oportMtrx_->send(hGblMtrx_);
}


} // End namespace BioPSE



