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
 *  BuildFDMatrix.cc:
 *
 *  Written by:
 *   Alexei Samsonov, Dave Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 17, 2000
 *   
 *   Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>

#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/LatticeGeom.h>
#include <Core/Datatypes/IndexAttrib.h>
#include <Core/Datatypes/AccelAttrib.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/SymSparseRowMatrix.h>
#include <Packages/BioPSE/Core/Datatypes/NeumannBC.h>
#include <Packages/BioPSE/Core/Datatypes/TypeName.h>
#include <Core/Containers/Array1.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

#include <map>
#include <iostream>
#include <string>

namespace BioPSE {

using namespace SCIRun;

typedef Array1<double>                  TensorArray;
typedef DiscreteAttrib<TensorArray >    TensorAttrib;
typedef LockingHandle<TensorAttrib >    TensorAttribHandle;

typedef DiscreteAttrib<double>          SourceAttrib;
typedef LockingHandle<SourceAttrib >    SourceAttribHandle;

typedef DiscreteAttrib<NeumannBC>       NmnBCAttrib;
typedef LockingHandle<NmnBCAttrib >     NmnBCAttribHandle;

typedef DiscreteAttrib<double>          DrhBCAttrib;
typedef LockingHandle<DrhBCAttrib >     DrhBCAttribHandle;

#define NUMZERO 10e-13

// -------------------------------------------------------------------------------
class BuildFDMatrix : public Module {
  
  // GROUP: Private Data 
  ///////////////////////////
  //

  //////////
  // Input port pointer
  FieldIPort*        iportField_;

  //////////
  // Output ports pointers
  ColumnMatrixOPort* oportRhs_;
  MatrixOPort*       oportMatrix_;
  FieldOPort*        oportMapField_;
  
  MatrixHandle       hMtrx_;                     // handle to matrix copy
  ColumnMatrixHandle hRhs_;                      // handle to Rhs copy
  
  // step sizes in every dimension
  double dx_;
  double dy_;
  double dz_;
  
public:
  
  // GROUP: Constructors
  ///////////////////////////
  //
  // Constructor
  
  BuildFDMatrix(const clString& id);
  
  // GROUP: Destructors
  ///////////////////////////
  //
  // Destructor  
  virtual ~BuildFDMatrix();

  // GROUP: interface functions
  //////////
  //
  virtual void execute();

};

//////////
// Module maker
extern "C" Module* make_BuildFDMatrix(const clString& id) {
  return new BuildFDMatrix(id);
}

// -------------------------------------------------------------------------------

//////////
// Constructor/Destructor

BuildFDMatrix::BuildFDMatrix(const clString& id)
  : Module("BuildFDMatrix", id, Source),
    dx_(1.0),
    dy_(1.0),
    dz_(1.0)
{
  // Create the input ports
  iportField_ = scinew FieldIPort(this, "Conductivity Field", FieldIPort::Atomic);
  add_iport(iportField_);

  // Create the output ports
  oportMatrix_ = scinew MatrixOPort(this, "FDM Matrix", MatrixIPort::Atomic);
  add_oport(oportMatrix_);
  
  oportRhs_ = scinew ColumnMatrixOPort(this,"RHS", ColumnMatrixIPort::Atomic);
  add_oport(oportRhs_);

  oportMapField_ = scinew FieldOPort(this, "Map Field", FieldIPort::Atomic);
  add_oport(oportMapField_);
  
  hMtrx_ = new SparseRowMatrix();
  hRhs_  = new ColumnMatrix(1);
  
}

BuildFDMatrix::~BuildFDMatrix(){
}

// -------------------------------------------------------------------------------

//////////
// Module execution
void BuildFDMatrix::execute()
{
  FieldHandle hField;
 
  if(!iportField_->get(hField)) { 
    return;
  }

  GeomHandle        hTmpGeom = hField->getGeom();

  if (!hTmpGeom.get_rep()){
    cerr << "No geometry supplied in the field" << endl;
    return;
  }

  LatticeGeomHandle hGeom = hTmpGeom->downcast((LatticeGeom*)0);
  
  if (!hGeom.get_rep()){
    cerr << "The supplied geometry is not of LatticeGeom type\n" << endl;
    return;
  }

  int nx=hGeom->getSizeX();
  int ny=hGeom->getSizeY();
  int nz=hGeom->getSizeZ();

  Vector diag;
  hGeom->getDiagonal(diag);
  
  dx_=diag.x()/(nx-1);
  dy_=diag.y()/(ny-1);
  dz_=diag.z()/(nz-1);
  
  double m=Max(dx_,dy_,dz_);
  
  dx_/=m;
  dy_/=m;
  dz_/=m;
  
  AttribHandle hTmpAttrib; 
  // -- getting sources
  hTmpAttrib = hField->getAttrib("Sources");
  if (!hTmpAttrib.get_rep()){
    cerr << "No Sources Attrib  supplied in the field" << endl;
    return;
  }

  SourceAttribHandle hSources = hTmpAttrib->downcast((SourceAttrib*) 0);
  if (!hSources.get_rep()){
    cerr << "Module BuildFDMatrix hasn't found Source Attributes." << endl;
    return;                      // no tensor attributes are found
  }
  
  // -- getting boundary conditions
  hTmpAttrib = hField->getAttrib("NeumannBC");
  NmnBCAttribHandle  hNmnBC(0);
  AccelAttrib<NeumannBC>* dummyAttr = dynamic_cast<AccelAttrib<NeumannBC>*>(hTmpAttrib.get_rep());
  if (!hTmpAttrib.get_rep()){
    cerr << "No NeumannBC Attrib  supplied in the field" << endl;    
  }
  else {
    hNmnBC  =  hTmpAttrib->downcast((NmnBCAttrib*) 0);
  }

  if (hNmnBC.get_rep()==0){
    cout << "No Newman cast done" << endl;
    // -- create dummy attribute with no-valid nodes
    hNmnBC=new NmnBCAttrib(nx, ny, nz);
    
    // -- invalidate all the nodes in the attribute
    hNmnBC->setValidBit(IntVector(0, 0, 0), IntVector(nx-1, ny-1, nz-1), false);
  }
  
  hTmpAttrib = hField->getAttrib("DirichletBC");

  DrhBCAttribHandle  hDrhBC(0);

  if (!hTmpAttrib.get_rep()){
    cerr << "No DirichletBC Attrib  supplied in the field" << endl;
  }
  else {
    hDrhBC  =  hTmpAttrib->downcast((DrhBCAttrib*) 0);
  }

  if (hDrhBC.get_rep()==0){
    // -- create dummy attribute with no-valid nodes
    hDrhBC=new DrhBCAttrib(nx, ny, nz);
    
    // -- invalidate all the nodes in the attribute
    hDrhBC->setValidBit(IntVector(0, 0, 0), IntVector(nx-1, ny-1, nz-1), false);
  }
  
  //////////
  // -- getting the attributes out of the field
  
  // -- getting tensor attributes
  hTmpAttrib = hField->getAttrib("Tensors");
  
  if (!hTmpAttrib.get_rep()){
    cerr << "No Tensors supplied in the field" << endl;
    return;
  }
  
  typedef IndexAttrib<Array1<double>, int> iAttr;
  typedef LockingHandle<iAttr > hIAttr;
  const hIAttr hIndex = hTmpAttrib->downcast((iAttr*)0);
  
  TensorAttribHandle hTensors = hTmpAttrib->downcast((TensorAttrib*)0);
 
  if (hTensors.get_rep()){
    cout << "Cast to Tensors is done!" << endl;
  }
  else {
    cout << "No succeseful cast to Tensors is done " << endl;
    return;
  }


  if (!hTensors.get_rep()){
    cerr << "Module BuildFDMatrix hasn't found Tensor Attributes." << endl;
    return;                      // no tensor attributes are found
  }
  else {
    cerr << "Tensor Attributes are found!!!"<< endl;
  }
  
  // -- finding nodes participating in the discretization
  // and creating local matrices for them

  int i=0, j=0, k=0;
  Array1<IntVector> activeNodes;
 
  for (i = 0; i<nx; i++) {
    for (j = 0; j<ny; j++) {
      for (k = 0; k<nz; k++) {
	if ( hTensors->isValid(i, j, k)){
	  // -- the node will be in the matrix
	  activeNodes.add(IntVector(i, j, k));
	}
      }
    }
  }
  
  hRhs_->resize(activeNodes.size());
  hRhs_->zero();

  double* rhs = hRhs_.get_rep()->get_rhs();
  int mSize = activeNodes.size();

  Array1<Array1<double> > lclMtrs(mSize);
  Array1<double> tmp(7);
  
  tmp.initialize(0.0);
  lclMtrs.initialize(tmp);
  
  int ii=0, jj=0, kk=0;
  
  Array1<int> rows(mSize);
  Array1<int> cols;
  Array1<unsigned char> nearNeibs(mSize);

  // -- walking over all the participating nodes
  for (i=0; i<mSize; i++){
    rows[i] = cols.size();

    ii = activeNodes[i].x();
    jj = activeNodes[i].y();
    kk = activeNodes[i].z();

    rhs[i] = hSources->get3(ii, jj, kk);
    Array1<double>& currLcl = lclMtrs[i];
    
    if (hDrhBC->isValid(ii, jj, kk)){
      rhs[i] = hDrhBC->get3(ii, jj, kk);
      currLcl[3] = 1;
      cols.add(i);
    }
    else {
      TensorArray& ownSigma = hTensors->get3(ii, jj, kk);
      
      double tmpSigma = 0;
      NeumannBC nmnBC;    
      Vector dir(0, 0, 0);
      double dval, val;


      // -- determining valid neighborhood
      unsigned char empty=0;
      // -----------------------------------------------------------------------
      // --  if it has neighboor with Neumann and Dirichlet BC at the same time,
      // just set it to the combined value of the BC's
      if ( kk!=0 && hTensors->isValid(ii, jj, kk-1) 
	   && hDrhBC->getValid3(ii, jj, kk-1, dval) 
	   && hNmnBC->getValid3(ii, jj, kk-1, nmnBC) 
	   && nmnBC.dir.z()){
	
	val = dval+dz_*nmnBC.val;
	val = (nmnBC.dir.z()>0)?nmnBC.val:-nmnBC.val;
	rhs[i] = val;
	currLcl[3] = 1;
	cols.add(i);
	goto lNextNode;
      }

      if ( kk!=nz-1 && hTensors->isValid(ii, jj, kk+1) 
	   && hDrhBC->getValid3(ii, jj, kk+1, dval) 
	   && hNmnBC->getValid3(ii, jj, kk+1, nmnBC) 
	   && nmnBC.dir.z()){
	
	val = dval-dz_*nmnBC.val;
	val = (nmnBC.dir.z()>0)?nmnBC.val:-nmnBC.val;
	rhs[i] = val;
	currLcl[3] = 1;
	cols.add(i);
	goto lNextNode;
      }
      
      if ( jj!=0 && hTensors->isValid(ii, jj-1, kk) 
	   && hDrhBC->getValid3(ii, jj-1, kk, dval) 
	   && hNmnBC->getValid3(ii, jj-1, kk, nmnBC) 
	   && nmnBC.dir.y()){
	
	val = dval+dz_*nmnBC.val;
	val = (nmnBC.dir.y()>0)?nmnBC.val:-nmnBC.val;
	rhs[i] = val;
	currLcl[3] = 1;
	cols.add(i);
	goto lNextNode;
      }
      
      if ( jj!=ny-1 && hTensors->isValid(ii, jj+1, kk) 
	   && hDrhBC->getValid3(ii, jj+1, kk, dval) 
	   && hNmnBC->getValid3(ii, jj+1, kk, nmnBC) 
	   && nmnBC.dir.y()){
	
	val = dval-dz_*nmnBC.val;
	val = (nmnBC.dir.y()>0)?nmnBC.val:-nmnBC.val;
	rhs[i] = val;
	currLcl[3] = 1;
	cols.add(i);
	goto lNextNode;
      }

      if ( ii!=0 && hTensors->isValid(ii-1, jj, kk) 
	   && hDrhBC->getValid3(ii-1, jj, kk, dval) 
	   && hNmnBC->getValid3(ii-1, jj, kk, nmnBC) 
	   && nmnBC.dir.x()){
	
	val = dval+dz_*nmnBC.val;
	val = (nmnBC.dir.x()>0)?nmnBC.val:-nmnBC.val;
	rhs[i] = val;
	currLcl[3] = 1;
	cols.add(i);
	goto lNextNode;
      }
      
      if ( ii!=nx-1 && hTensors->isValid(ii+1, jj, kk) 
	   && hDrhBC->getValid3(ii+1, jj, kk, dval) 
	   && hNmnBC->getValid3(ii+1, jj, kk, nmnBC) 
	   && nmnBC.dir.x()){
	
	val = dval-dz_*nmnBC.val;
	val = (nmnBC.dir.x()>0)?nmnBC.val:-nmnBC.val;
	rhs[i] = val;
	currLcl[3] = 1;
	cols.add(i);
	goto lNextNode;
      }
      
      // --------------------------------------------------------------
      if ( kk==0 || !hTensors->isValid(ii, jj, kk-1)){                  // ----1-
	empty |= 2;
      }
      else {
	tmpSigma=(hTensors->get3(ii, jj, kk-1))[5];
	
	if (fabs(tmpSigma-ownSigma[5])>NUMZERO){
	  currLcl[0] = 2*tmpSigma*ownSigma[5]/(tmpSigma+ownSigma[5])/(dz_*dz_);
	}
	else {
	  currLcl[0] = ownSigma[5]/(dz_*dz_);
	}
	cols.add(i-ny*nz);
      }
      
      // --------------------------------------------------------------
      if ( jj==0 || !hTensors->isValid(ii, jj-1, kk)){                  // --1---
	empty |= 8;
      }
      else {
	tmpSigma=(hTensors->get3(ii, jj-1, kk))[3];
	if (fabs(tmpSigma-ownSigma[3])>NUMZERO){
	  currLcl[1] = 2*tmpSigma*ownSigma[3]/(tmpSigma+ownSigma[3])/(dy_*dy_);
	}
	else {
	  currLcl[1] = ownSigma[3]/(dy_*dy_);
	}
	cols.add(i-ny);
      }
      
      // --------------------------------------------------------------
      if (ii==0 || !hTensors->isValid(ii-1, jj, kk)){                  // 1-----  
	empty |= 32;
      }
      else {
	tmpSigma=(hTensors->get3(ii-1, jj, kk))[0];
	if (fabs(tmpSigma-ownSigma[0])>NUMZERO){
	  currLcl[2] = 2*tmpSigma*ownSigma[0]/(tmpSigma+ownSigma[0])/(dx_*dx_);
	}
	else {
	  currLcl[2] = ownSigma[0]/(dx_*dx_);
	}
	cols.add(i-1);
      }
      
      // --------------------------------------------------------------
      cols.add(i);
      
      // --------------------------------------------------------------
      if (ii==nx-1 || !hTensors->isValid(ii+1, jj, kk)){                  // -1----
	empty |= 16;
      }
      else {
	tmpSigma=(hTensors->get3(ii+1, jj, kk))[0];
	
	if (fabs(tmpSigma-ownSigma[0])>NUMZERO){
	  currLcl[4] = 2*tmpSigma*ownSigma[0]/(tmpSigma+ownSigma[0])/(dx_*dx_);
	}
	else {
	  currLcl[4] = ownSigma[0]/(dx_*dx_);
	}
	cols.add(i+1);
      }
      
      // --------------------------------------------------------------
      if ( jj==ny-1 || !hTensors->isValid(ii, jj+1, kk)){                  // ---1--
	empty |= 4;
      }
      else {
	tmpSigma=(hTensors->get3(ii, jj+1, kk))[3];

	if (fabs(tmpSigma-ownSigma[3])>NUMZERO){
	  currLcl[5] = 2*tmpSigma*ownSigma[3]/(tmpSigma+ownSigma[3])/(dy_*dy_);
	}
	else {
	  currLcl[5] = ownSigma[3]/(dy_*dy_);
	}
	cols.add(i+ny);
      }

      // --------------------------------------------------------------
      if ( kk==nz-1 || !hTensors->isValid(ii, jj, kk+1)){                  // -----1
	empty |= 1;
      }
      else {
	tmpSigma=(hTensors->get3(ii, jj, kk+1))[5];

	if (fabs(tmpSigma-ownSigma[5])>NUMZERO){
	  currLcl[6] = 2*tmpSigma*ownSigma[5]/(tmpSigma+ownSigma[5])/(dz_*dz_);
	}
	else {
	  currLcl[6] = ownSigma[5]/(dz_*dz_);
	}
	cols.add(i+ny*nz);
      }
      
      nearNeibs.add(empty);
      
      // -- making adjustments for the empty neighbors
      double val1;

      if (hNmnBC->getValid3(ii, jj, kk, nmnBC)){
	// --  adjustment to rhs due to the pure Neumann node
	dir=nmnBC.dir;
	val=nmnBC.val;
      }
       
      // 1-----
      if (empty & 32) {
	currLcl[1]*=2;
	if (dir.x()!=0){                 // Neumann node
	  val1 = (dir.x()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[1]/dx_;
	  dir.x(0.0);                    // Neumann BC is handled
	}
      }

      // -1----
      if (empty & 16) {
	currLcl[0]*=2;
	if (dir.x()!=0){                 // Neumann node
	  val1 = (dir.x()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[0]/dx_;
	  dir.x(0.0);                    // Neumann BC is handled
	}
      }
      
      // --1---
      if (empty & 8)  {
	currLcl[3]*=2;
	if (dir.y()!=0){                 // Neumann node
	  val1 = (dir.y()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[3]/dy_;
	   dir.y(0.0);                    // Neumann BC is handled
	}
      }

      // ---1--
      if (empty & 4)  {
	currLcl[2]*=2;
	if (dir.y()!=0){                 // Neumann node
	  val1 = (dir.y()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[2]/dy_;
	  dir.y(0.0);                    // Neumann BC is handled 
	}
      } 	     
      
      // ----1-
      if (empty & 2)  {
	currLcl[5]*=2;
	if (dir.z()!=0){                 // Neumann node
	  val1 = (dir.z()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[5]/dz_;
	  dir.z(0.0);                    // Neumann BC is handled 
	}
      }
      
      // -----1
      if (empty & 1)  {
	currLcl[4]*=2;
	if (dir.z()!=0){                 // Neumann node
	  val1 = (dir.z()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[4]/dz_;
	  dir.z(0.0);                    // Neumann BC is handled 
	}
      }	
      
      // ----------------------------------------------------------------------------
      // handling pure Neumann BC left inside the volume
      if (dir.x()){
	val1 = (dir.x()>0)?val:-val;
	rhs[i]-=2*val1*currLcl[1]/dx_;
	currLcl[0] *=2;
	currLcl[1] = 0;
      }
      else if(dir.y()){
	val1 = (dir.y()>0)?val:-val;
	rhs[i]-=2*val1*currLcl[3]/dy_;
	currLcl[2] *=2;
	currLcl[3] = 0;
      }
      else if (dir.z()){
	val1 = (dir.z()>0)?val:-val;
	rhs[i]-=2*val1*currLcl[5]/dz_;
	currLcl[4] *=2;
	currLcl[5] = 0;
      }
      
      currLcl[3]=-(currLcl[0]+currLcl[1]+currLcl[2]+currLcl[4]+currLcl[5]+currLcl[6]);
    
    lNextNode:
      // -- the node is found to have Robson (mixed) type BC neighboor
      int dummy = 0;        // to make compiler stop complaining
    }
  }

  // -- creating field mapping from the original field to rhs-vector
  FieldHandle mapField = scinew Field ();
  mapField->setGeometry(GeomHandle(scinew LatticeGeom(nx, ny, nz)));
  
  AccelAttrib<int>* pMapAttrib = scinew AccelAttrib<int>(nx, ny, nz);
  pMapAttrib->initialize(-1);
  pMapAttrib->setName("MapField");
  
  int nalloc;
  const unsigned int* pBitVector = hTensors->getValidBits(nalloc);
  pMapAttrib->copyValidBits(pBitVector, nalloc);
  
  // -- making adjustments for Neumann nodes with Dirichlet conditions
  for (i=0; i<mSize; i++){
    ii = activeNodes[i].x();
    jj = activeNodes[i].y();
    kk = activeNodes[i].z();
    
    pMapAttrib->set3(ii, jj, kk, i);
  }
  
  ////////
  // TODO: implement Sparse matrix to be able not to predefine its structure
  //       
  
  // -- populating sparse matrix
  double* pElems = scinew double[cols.size()];
  int elPos = 0;
  for (i=0; i<mSize; i++){
    Array1<double>& lcl = lclMtrs[i];
    cout << "Node #" << i << " has local matrix " ;
    for (j=0; j<7; j++){
      cout << lcl[j] << ", ";
      if (lcl[j]!=0)
	pElems[++elPos]=lcl[j];
    }
    cout << endl;
  }
  
  SparseRowMatrix* sm = scinew SparseRowMatrix(mSize, mSize, rows, cols);
  sm->a = pElems;
  hMtrx_ = sm;
  
  mapField->addAttribute(AttribHandle(pMapAttrib));
  
  // -- sending handles
  oportMatrix_->send(MatrixHandle(hMtrx_->clone()));
  oportRhs_->send(ColumnMatrixHandle(hRhs_->clone()));
  oportMapField_->send(mapField);
}

} // End namespace BioPSE

