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

typedef DiscreteAttrib<NewmannBC>       NmnBCAttrib;
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
  FieldIPort*        d_iportField;

  //////////
  // Output ports pointers
  ColumnMatrixOPort* d_oportRhs;
  MatrixOPort*       d_oportMatrix;
  FieldOPort*        d_oportMapField;
  
  MatrixHandle       d_hMtrx;                     // handle to matrix copy
  ColumnMatrixHandle d_hRhs;                      // handle to Rhs copy
  
  // step sizes in every dimension
  double d_dx;
  double d_dy;
  double d_dz;
  
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
    d_dx(1.0),
    d_dy(1.0),
    d_dz(1.0)
{
  // Create the input ports
  d_iportField = scinew FieldIPort(this, "Conductivity Field", FieldIPort::Atomic);
  add_iport(d_iportField);

  // Create the output ports
  d_oportMatrix = scinew MatrixOPort(this, "FDM Matrix", MatrixIPort::Atomic);
  add_oport(d_oportMatrix);
  
  d_oportRhs = scinew ColumnMatrixOPort(this,"RHS", ColumnMatrixIPort::Atomic);
  add_oport(d_oportRhs);

  d_oportMapField = scinew FieldOPort(this, "Map Field", FieldIPort::Atomic);
  add_oport(d_oportMapField);
  
  d_hMtrx = new SparseRowMatrix();
  d_hRhs  = new ColumnMatrix(1);
  
}

BuildFDMatrix::~BuildFDMatrix(){
}

// -------------------------------------------------------------------------------

//////////
// Module execution
void BuildFDMatrix::execute()
{
  FieldHandle hField;
 
  if(!d_iportField->get(hField)) { 
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
  
  d_dx=diag.x()/(nx-1);
  d_dy=diag.y()/(ny-1);
  d_dz=diag.z()/(nz-1);
  
  double m=Max(d_dx,d_dy,d_dz);
  
  d_dx/=m;
  d_dy/=m;
  d_dz/=m;
  
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
  hTmpAttrib = hField->getAttrib("NewmannBC");
  NmnBCAttribHandle  hNmnBC(0);
  AccelAttrib<NewmannBC>* dummyAttr = dynamic_cast<AccelAttrib<NewmannBC>*>(hTmpAttrib.get_rep());
  if (!hTmpAttrib.get_rep()){
    cerr << "No NewmannBC Attrib  supplied in the field" << endl;    
  }
  else {
    hNmnBC  =  hTmpAttrib->downcast((NmnBCAttrib*) 0);
  }

  if (hNmnBC.get_rep()==0){
    cout << "No Newman cast done" << endl;
    // -- create dummy attribute with no-valid nodes
    hNmnBC=new NmnBCAttrib();
    hNmnBC->resize(nx, ny, nz);
    
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
    hDrhBC=new DrhBCAttrib();
    hDrhBC->resize(nx, ny, nz);
    
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
  
  d_hRhs->resize(activeNodes.size());
  d_hRhs->zero();

  double* rhs = d_hRhs.get_rep()->get_rhs();
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
      NewmannBC nmnBC;    
      Vector dir(0, 0, 0);
      double dval, val;

      // -----------------------------------------------------------------------
      // --  if it has neighboor with Newmann and Dirichlet BC at the same time,
      // just set it to the combined value of the BC's
      if ( kk!=0 && hTensors->isValid(ii, jj, kk-1) 
	   && hDrhBC->getValid3(ii, jj, kk-1, dval) 
	   && hNmnBC->getValid3(ii, jj, kk-1, nmnBC) 
	   && nmnBC.dir.z()){
	
	val = dval+d_dz*nmnBC.val;
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
	
	val = dval-d_dz*nmnBC.val;
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
	
	val = dval+d_dz*nmnBC.val;
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
	
	val = dval-d_dz*nmnBC.val;
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
	
	val = dval+d_dz*nmnBC.val;
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
	
	val = dval-d_dz*nmnBC.val;
	val = (nmnBC.dir.x()>0)?nmnBC.val:-nmnBC.val;
	rhs[i] = val;
	currLcl[3] = 1;
	cols.add(i);
	goto lNextNode;
      }

      // -- determining valid neighborhood
      unsigned char empty=0;
      
      // --------------------------------------------------------------
      if ( kk==0 || !hTensors->isValid(ii, jj, kk-1)){                  // ----1-
	empty |= 2;
      }
      else {
	tmpSigma=(hTensors->get3(ii, jj, kk-1))[5];
	
	if (abs(tmpSigma-ownSigma[5])>NUMZERO){
	  currLcl[0] = 2*tmpSigma*ownSigma[5]/(tmpSigma+ownSigma[5])/(d_dz*d_dz);
	}
	else {
	  currLcl[0] = ownSigma[5]/(d_dz*d_dz);
	}
	cols.add(i-ny*nz);
      }
      
      // --------------------------------------------------------------
      if ( jj==0 || !hTensors->isValid(ii, jj-1, kk)){                  // --1---
	empty |= 8;
      }
      else {
	tmpSigma=(hTensors->get3(ii, jj-1, kk))[3];
	if (abs(tmpSigma-ownSigma[3])>NUMZERO){
	  currLcl[1] = 2*tmpSigma*ownSigma[3]/(tmpSigma+ownSigma[3])/(d_dy*d_dy);
	}
	else {
	  currLcl[1] = ownSigma[3]/(d_dy*d_dy);
	}
	cols.add(i-ny);
      }
      
      // --------------------------------------------------------------
      if (ii==0 || !hTensors->isValid(ii-1, jj, kk)){                  // 1-----  
	empty |= 32;
      }
      else {
	tmpSigma=(hTensors->get3(ii-1, jj, kk))[0];
	if (abs(tmpSigma-ownSigma[0])>NUMZERO){
	  currLcl[2] = 2*tmpSigma*ownSigma[0]/(tmpSigma+ownSigma[0])/(d_dx*d_dx);
	}
	else {
	  currLcl[2] = ownSigma[0]/(d_dx*d_dx);
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
	
	if (abs(tmpSigma-ownSigma[0])>NUMZERO){
	  currLcl[4] = 2*tmpSigma*ownSigma[0]/(tmpSigma+ownSigma[0])/(d_dx*d_dx);
	}
	else {
	  currLcl[4] = ownSigma[0]/(d_dx*d_dx);
	}
	cols.add(i+1);
      }
      
      // --------------------------------------------------------------
      if ( jj==ny-1 || !hTensors->isValid(ii, jj+1, kk)){                  // ---1--
	empty |= 4;
      }
      else {
	tmpSigma=(hTensors->get3(ii, jj+1, kk))[3];

	if (abs(tmpSigma-ownSigma[3])>NUMZERO){
	  currLcl[5] = 2*tmpSigma*ownSigma[3]/(tmpSigma+ownSigma[3])/(d_dy*d_dy);
	}
	else {
	  currLcl[5] = ownSigma[3]/(d_dy*d_dy);
	}
	cols.add(i+ny);
      }

      // --------------------------------------------------------------
      if ( kk==nz-1 || !hTensors->isValid(ii, jj, kk+1)){                  // -----1
	empty |= 1;
      }
      else {
	tmpSigma=(hTensors->get3(ii, jj, kk+1))[5];

	if (abs(tmpSigma-ownSigma[5])>NUMZERO){
	  currLcl[6] = 2*tmpSigma*ownSigma[5]/(tmpSigma+ownSigma[5])/(d_dz*d_dz);
	}
	else {
	  currLcl[6] = ownSigma[5]/(d_dz*d_dz);
	}
	cols.add(i+ny*nz);
      }
      
      nearNeibs.add(empty);
      
      // -- making adjustments for the empty neighbors
      double val1;

      if (hNmnBC->getValid3(ii, jj, kk, nmnBC)){
	// --  adjustment to rhs due to the pure Newmann node
	dir=nmnBC.dir;
	val=nmnBC.val;
      }
       
      // 1-----
      if (empty & 32) {
	currLcl[1]*=2;
	if (dir.x()!=0){                 // Newmann node
	  val1 = (dir.x()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[1]/d_dx;
	  dir.x(0.0);                    // Newmann BC is handled
	}
      }

      // -1----
      if (empty & 16) {
	currLcl[0]*=2;
	if (dir.x()!=0){                 // Newmann node
	  val1 = (dir.x()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[0]/d_dx;
	  dir.x(0.0);                    // Newmann BC is handled
	}
      }
      
      // --1---
      if (empty & 8)  {
	currLcl[3]*=2;
	if (dir.y()!=0){                 // Newmann node
	  val1 = (dir.y()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[3]/d_dy;
	   dir.y(0.0);                    // Newmann BC is handled
	}
      }

      // ---1--
      if (empty & 4)  {
	currLcl[2]*=2;
	if (dir.y()!=0){                 // Newmann node
	  val1 = (dir.y()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[2]/d_dy;
	  dir.y(0.0);                    // Newmann BC is handled 
	}
      } 	     
      
      // ----1-
      if (empty & 2)  {
	currLcl[5]*=2;
	if (dir.z()!=0){                 // Newmann node
	  val1 = (dir.z()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[5]/d_dz;
	  dir.z(0.0);                    // Newmann BC is handled 
	}
      }
      
      // -----1
      if (empty & 1)  {
	currLcl[4]*=2;
	if (dir.z()!=0){                 // Newmann node
	  val1 = (dir.z()>0)?val:-val;
	  rhs[i]-=2*val1*currLcl[4]/d_dz;
	  dir.z(0.0);                    // Newmann BC is handled 
	}
      }	
      
      // ----------------------------------------------------------------------------
      // handling pure Newmann BC left inside the volume
      if (dir.x()){
	val1 = (dir.x()>0)?val:-val;
	rhs[i]-=2*val1*currLcl[1]/d_dx;
	currLcl[0] *=2;
	currLcl[1] = 0;
      }
      else if(dir.y()){
	val1 = (dir.y()>0)?val:-val;
	rhs[i]-=2*val1*currLcl[3]/d_dy;
	currLcl[2] *=2;
	currLcl[3] = 0;
      }
      else if (dir.z()){
	val1 = (dir.z()>0)?val:-val;
	rhs[i]-=2*val1*currLcl[5]/d_dz;
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
  
  // -- making adjustments for Newmann nodes with Dirichlet conditions
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
  d_hMtrx = sm;
  
  mapField->addAttribute(AttribHandle(pMapAttrib));
  
  // -- sending handles
  d_oportMatrix->send(MatrixHandle(d_hMtrx->clone()));
  d_oportRhs->send(ColumnMatrixHandle(d_hRhs->clone()));
  d_oportMapField->send(mapField);
}

} // End namespace BioPSE

