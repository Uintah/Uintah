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
 *  BuildBEMatrix.cc: constructs matrix Zbh to relate potentials on surfaces in 
 *                    boundary value problems
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   December, 2000
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
#include <Core/Datatypes/UnstructuredGeom.h>
#include <Core/Datatypes/DiscreteAttrib.h>
#include <Core/Datatypes/SurfaceGeom.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <math.h>

#include <map>
#include <iostream>
#include <string>

namespace BioPSE {

using namespace SCIRun;

typedef DiscreteAttrib<double>      Potentials;
typedef LockingHandle<Potentials >  PotentialsHandle;
typedef LockingHandle<DenseMatrix>  DenseMatrixHandle;

#define NUMZERO 10e-13
#define PI 3.1415

// -------------------------------------------------------------------------------
class BuildBEMatrix : public Module {
  
  // GROUP: Private Data 
  ///////////////////////////
  //

  //////////
  // Input port pointer
  FieldIPort*        iportSurfOut_;
  FieldIPort*        iportSurfIn_;

  //////////
  // Output ports pointers
  MatrixOPort*       oportMatrix_;
  ColumnMatrixOPort* oportPhiHeart_;
 
  DenseMatrixHandle  hZbh_;
  ColumnMatrixHandle hPhiH_;

  //////////
  // matrices used in calculation of integrals
  Array1<DenseMatrix> baseMtrx_;
  DenseMatrix         coef16_;
  DenseMatrix         coef64_;
  DenseMatrix         coef256_;

  // GROUP: Private Methods
  ///////////////////////////
  //
  // Methods to fill intermediate matrices for Zbh calculation
  bool makePbb(DenseMatrix&, SurfaceGeomHandle);
  bool makePbh(DenseMatrix&, SurfaceGeomHandle,  SurfaceGeomHandle);
  bool makePhb(DenseMatrix&, SurfaceGeomHandle,  SurfaceGeomHandle);
  bool makePhh(DenseMatrix&, SurfaceGeomHandle);
  bool makeGbh(DenseMatrix&, SurfaceGeomHandle,  SurfaceGeomHandle);
  bool makeGhh(DenseMatrix&, SurfaceGeomHandle);
  
  //////////
  // Initializers of coefficient matrices for triangle subdivision
  // Called in the constructor
  void initBase();
  void init16();
  void init64();
  void init128();
  
  //////////
  // Estimators of integral values in BEM
  inline double getOmega(const Point&, const Point&, const Point&, const Point&);
  inline double getIntegral(const Point&, const Point&, const Point&, const Point&, int);
  
  void releaseHandles();
public:
  
  // GROUP: Constructors
  ///////////////////////////
  //
  // Constructor
  
  BuildBEMatrix(const clString& id);
  
  // GROUP: Destructors
  ///////////////////////////
  //
  // Destructor  
  virtual ~BuildBEMatrix();

  // GROUP: interface functions
  //////////
  //
  virtual void execute();

};

//////////
// Module maker
extern "C" Module* make_BuildBEMatrix(const clString& id) {
  return new BuildBEMatrix(id);
}

// -------------------------------------------------------------------------------
//////////
// Constructor/Destructor

BuildBEMatrix::BuildBEMatrix(const clString& id): 
  Module("BuildBEMatrix", id, Source),
  baseMtrx_(),
  coef16_(16, 3),
  coef64_(64, 3),
  coef256_(256, 3)  
{
  // Create the input ports
  iportSurfOut_ = scinew FieldIPort(this, "Outer Surface", FieldIPort::Atomic);
  add_iport(iportSurfOut_);

  iportSurfIn_ = scinew FieldIPort(this, "Inner Surface", FieldIPort::Atomic);
  add_iport(iportSurfIn_);

  // Create the output ports
  oportMatrix_ = scinew MatrixOPort(this, "Zbh Matrix", MatrixIPort::Atomic);
  add_oport(oportMatrix_);
  
  oportPhiHeart_ = scinew ColumnMatrixOPort(this, "Inner Surf Potentials", ColumnMatrixIPort::Atomic);
  add_oport(oportPhiHeart_);
  
  hPhiH_  = new ColumnMatrix(1);
  DenseMatrix tmpMtrx(4, 4);
  tmpMtrx.zero();

  baseMtrx_.add(tmpMtrx);
  baseMtrx_.add(tmpMtrx);
  baseMtrx_.add(tmpMtrx);
  baseMtrx_.add(tmpMtrx);

  initBase();
  init16();
  init64();
  init128();
 
}

BuildBEMatrix::~BuildBEMatrix(){
}

// -------------------------------------------------------------------------------
//////////
// Module execution
void BuildBEMatrix::execute()
{
  FieldHandle hFieldOut;
  FieldHandle hFieldIn;
  
  if(!iportSurfOut_->get(hFieldOut)) { 
    msgStream_ << "BuildBEMatrix -- couldn't get outer surface. Returning." << endl;
    return;
  }
  if(!iportSurfIn_->get(hFieldIn)) { 
    msgStream_ << "BuildBEMatrix -- couldn't get inner surface. Returning." << endl;
    return;
  }

  // -- processing supplied field handles
  GeomHandle hGeomOuter = hFieldOut->getGeom();
  GeomHandle hGeomInner = hFieldIn->getGeom();

  AttribHandle hAttribInner = hFieldIn->getAttrib();
  
  // -- casting attributes and geometries to the expected types
  SurfaceGeomHandle hSurfOut = hGeomOuter->downcast((SurfaceGeom*)0);
  SurfaceGeomHandle hSurfIn  = hGeomInner->downcast((SurfaceGeom*)0);
  
  PotentialsHandle  hPhiIn   = hAttribInner->downcast((Potentials*)0);
  
  if (hSurfOut.get_rep()){
    msgStream_ << "BuildBEMatrix -- couldn't cast Geom to SurfaceGeom for outer surface. Returning." << endl;
    releaseHandles();
    return;
  }
  
  if (hSurfIn.get_rep()){
    msgStream_ << "BuildBEMatrix -- couldn't cast Geom to SurfaceGeom for inner surface. Returning." << endl;
    releaseHandles();
    return;
  }

  if (hPhiIn.get_rep()){
    msgStream_ << "BuildBEMatrix -- couldn't cast Attrib to DiscreateAttrib<double> for inner surface potentials. Returning." << endl;
    releaseHandles();
    return;
  }
  
  // -- allocating matrices
  int nnIn = hSurfIn->face_.size();
  int nnOut= hSurfOut->face_.size();

  hZbh_ = scinew DenseMatrix(nnOut, nnIn);
  
  // -- STARTING CALCULATIONS
  // -- Zbh<-Gbh
  if(!makeGbh(*hZbh_.get_rep(), hSurfOut, hSurfIn)){
    msgStream_ << "BuildBEMatrix: Cann't construct Gbh. Returning." << endl;
    return;
  }

  // -- calculating Ghh
  DenseMatrixHandle hGhh = scinew DenseMatrix(nnIn, nnIn);
  if(!makeGhh(*hGhh.get_rep(), hSurfIn)){
    msgStream_ << "BuildBEMatrix: Cann't construct Ghh. Returning." << endl;
    return;
  }

  // -- Ghh<-(Ghh^-1)
  hGhh->invert();
  
  // -- tmpBH<-Zbh*(Ghh^-1)
  DenseMatrixHandle hTmpBH = scinew DenseMatrix(nnOut, nnIn);
  Mult(*hTmpBH.get_rep(), *hZbh_.get_rep(), *hGhh.get_rep());
  
  // -- Ybb<-tmpBH*Phb
  DenseMatrixHandle hPhb = scinew DenseMatrix(nnIn, nnOut);
  DenseMatrixHandle hYbb = scinew DenseMatrix(nnOut, nnOut);
  
  if (!makePhb(*hPhb.get_rep(), hSurfIn, hSurfOut)){
    msgStream_ << "BuildBEMatrix: Cann't construct Phb. Returning." << endl;
    return;
  }

  Mult(*hYbb.get_rep(), *hTmpBH.get_rep(), *hPhb.get_rep());
  
  // -- Ybb <- Pbb-Ybb
  DenseMatrixHandle hPbb = scinew DenseMatrix(nnOut, nnOut);

  if(!makePbb(*hPbb.get_rep(), hSurfOut)){
    msgStream_ << "BuildBEMatrix: Cann't construct Pbb. Returning." << endl;
    return;
  }

  Add(1.0, *hYbb.get_rep(), -1.0, *hPbb.get_rep());
  
  // -- Ybb <- (Ybb^-1)
  hYbb->invert();

  // -- Zbh <- tmpBH*Phh
  DenseMatrixHandle hPhh = scinew DenseMatrix(nnIn, nnIn);
  
  if(!makePhh(*hPhh.get_rep(), hSurfIn)){
    msgStream_ << "BuildBEMatrix: Cann't construct Phh. Returning." << endl;
    return;
  }

  Mult(*hZbh_.get_rep(), *hTmpBH.get_rep(), *hPhh.get_rep());
  
  // -- Zbh <- Zbh-Pbh
  DenseMatrixHandle hPbh = scinew DenseMatrix(nnOut, nnIn);
  if (!makePbh(*hPbh.get_rep(), hSurfOut, hSurfIn)){
    msgStream_ << "BuildBEMatrix: Cann't construct Pbh. Returning." << endl;
    return;
  }

  Add(1.0, *hZbh_.get_rep(), -1.0, *hPbh.get_rep());

  // -- tmpBH <- Ybb*Zbh
  Mult(*hTmpBH.get_rep(), *hYbb.get_rep(), *hZbh_.get_rep());

  // -- Zbh <- tmpBH
  hZbh_ = hTmpBH;

  // TODO: PhiHeart vector, mapping from original geometry to the vector

  // -- sending handles to cloned objects
  oportMatrix_->send(MatrixHandle(hZbh_->clone()));
  oportPhiHeart_->send(ColumnMatrixHandle(hPhiH_->clone()));
  releaseHandles();
}

//////////
// Methods to fill intermediate matrices for Zbh calculation
bool BuildBEMatrix::makePbb(DenseMatrix& mP, SurfaceGeomHandle hSurf){
  
  mP.zero();
  
  // -- getting raw pointer for fast access
  double** pmP = mP.getData2D();
  int nr = mP.ncols();
  int nc = mP.nrows();

  int nNodes = hSurf->node_.size();
  int nFaces = hSurf->face_.size();
  
  ASSERTEQ(nr, nc);
  ASSERTEQ(nr, nNodes);

  vector<NodeSimp>& nodes = hSurf->node_;
  
  int n1, n2, n3;
  double tmp = 0;
  double autoAngle = 0;  // accumulator for autosolid angle value

  for (int i=0; i<nNodes; i++){               // -- for every node on the first surface
    Point& rNode = nodes[i].p;
    autoAngle = 0;
    for (int j=0; j<nFaces; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurf->face_[j].nodes[0];
      n2 = hSurf->face_[j].nodes[1];
      n3 = hSurf->face_[j].nodes[2];
      
      Point& rn1 = nodes[n1].p;
      Point& rn2 = nodes[n2].p;
      Point& rn3 = nodes[n3].p;

      if (rn1!=rNode && rn2!=rNode && rn3!=rNode){
	tmp = getOmega(rn1, rn2, rn3, rNode)/(4*PI*3);
	pmP[i][n1]+=tmp;
	pmP[i][n2]+=tmp;
	pmP[i][n3]+=tmp;
	autoAngle += 3*tmp;
      }     
    }
    
    // -- find Rc-coeffiecient for the current node
    pmP[i][i] = -autoAngle;
  }
  
  return true;
}

bool BuildBEMatrix::makePhh(DenseMatrix& mP, SurfaceGeomHandle hSurf){
  
  mP.zero();
  
  // -- getting raw pointer for fast access
  double** pmP = mP.getData2D();
  int nr = mP.ncols();
  int nc = mP.nrows();

  int nNodes = hSurf->node_.size();
  int nFaces = hSurf->face_.size();
  
  ASSERTEQ(nr, nc);
  ASSERTEQ(nr, nNodes);

  vector<NodeSimp>& nodes = hSurf->node_;
  
  int n1, n2, n3;
  double tmp = 0;
  double autoAngle = 0;  // accumulator for autosolid angle value

  for (int i=0; i<nNodes; i++){               // -- for every node on the first surface
    Point& rNode = nodes[i].p;
    autoAngle = 0;
    for (int j=0; j<nFaces; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurf->face_[j].nodes[0];
      n2 = hSurf->face_[j].nodes[1];
      n3 = hSurf->face_[j].nodes[2];
      
      Point& rn1 = nodes[n1].p;
      Point& rn2 = nodes[n2].p;
      Point& rn3 = nodes[n3].p;
      
      if (rn1!=rNode && rn2!=rNode && rn3!=rNode){
	tmp = getOmega(rn1, rn2, rn3, rNode)/(-4*PI*3);
	pmP[i][n1]+=tmp;
	pmP[i][n2]+=tmp;
	pmP[i][n3]+=tmp;
	autoAngle += 3*tmp;
      }     
    }
    
    // -- find Rc-coeffiecient for the current node
    pmP[i][i] = -1-autoAngle;
  }
  
  return true;
}

bool BuildBEMatrix::makePbh(DenseMatrix& mP, SurfaceGeomHandle hSurfB, SurfaceGeomHandle hSurfH){
  
  mP.zero();
  
  // -- getting raw pointer for fast access
  double** pmP = mP.getData2D();
  int nr = mP.ncols();
  int nc = mP.nrows();

  int nNodesB = hSurfB->node_.size();
  int nNodesH = hSurfH->node_.size();
  int nFacesH = hSurfH->face_.size();
  
  ASSERTEQ(nr, nNodesB);
  ASSERTEQ(nc, nNodesH);

  vector<NodeSimp>& nodesB = hSurfB->node_;
  vector<NodeSimp>& nodesH = hSurfH->node_;
  
  int n1, n2, n3;
  double tmp = 0;

  for (int i=0; i<nNodesB; i++){               // -- for every node on the first surface
    Point& rNode = nodesB[i].p;
    for (int j=0; j<nFacesH; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurfH->face_[j].nodes[0];
      n2 = hSurfH->face_[j].nodes[1];
      n3 = hSurfH->face_[j].nodes[2];
      
      Point& rn1 = nodesH[n1].p;
      Point& rn2 = nodesH[n2].p;
      Point& rn3 = nodesH[n3].p;

      tmp = getOmega(rn1, rn2, rn3, rNode)/(-4*PI*3);
      pmP[i][n1]+=tmp;
      pmP[i][n2]+=tmp;
      pmP[i][n3]+=tmp;	
    }
  }
  
  return true;
}
bool BuildBEMatrix::makePhb(DenseMatrix& mP, SurfaceGeomHandle hSurfH, SurfaceGeomHandle hSurfB){
  
  mP.zero();
  
  // -- getting raw pointer for fast access
  double** pmP = mP.getData2D();
  int nr = mP.ncols();
  int nc = mP.nrows();

  int nNodesB = hSurfB->node_.size();
  int nNodesH = hSurfH->node_.size();
  int nFacesB = hSurfB->face_.size();
  
  ASSERTEQ(nc, nNodesB);
  ASSERTEQ(nr, nNodesH);

  vector<NodeSimp>& nodesB = hSurfB->node_;
  vector<NodeSimp>& nodesH = hSurfH->node_;
  
  int n1, n2, n3;
  double tmp = 0;

  for (int i=0; i<nNodesH; i++){               // -- for every node on the first surface
    Point& rNode = nodesH[i].p;
    for (int j=0; j<nFacesB; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurfB->face_[j].nodes[0];
      n2 = hSurfB->face_[j].nodes[1];
      n3 = hSurfB->face_[j].nodes[2];
      
      Point& rn1 = nodesB[n1].p;
      Point& rn2 = nodesB[n2].p;
      Point& rn3 = nodesB[n3].p;

      tmp = getOmega(rn1, rn2, rn3, rNode)/(4*PI*3);
      pmP[i][n1]+=tmp;
      pmP[i][n2]+=tmp;
      pmP[i][n3]+=tmp;	
    }
  }
  
  return true;
}

bool BuildBEMatrix::makeGbh(DenseMatrix& mG, SurfaceGeomHandle hSurfB, SurfaceGeomHandle hSurfH){
  
  mG.zero();
  
  // -- getting raw pointer for fast access
  double** pmG = mG.getData2D();
  int nr = mG.ncols();
  int nc = mG.nrows();

  int nNodesB = hSurfB->node_.size();
  int nNodesH = hSurfH->node_.size();
  int nFacesH = hSurfB->face_.size();
  
  ASSERTEQ(nc, nNodesB);
  ASSERTEQ(nr, nNodesH);

  vector<NodeSimp>& nodesB = hSurfB->node_;
  vector<NodeSimp>& nodesH = hSurfH->node_;
  
  int n1, n2, n3;
  double tmp = 0;

  for (int i=0; i<nNodesB; i++){               // -- for every node on the first surface
    Point& rNode = nodesB[i].p;
    for (int j=0; j<nFacesH; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurfH->face_[j].nodes[0];
      n2 = hSurfH->face_[j].nodes[1];
      n3 = hSurfH->face_[j].nodes[2];
      
      Point& rn1 = nodesH[n1].p;
      Point& rn2 = nodesH[n2].p;
      Point& rn3 = nodesH[n3].p;

      tmp = getIntegral(rn1, rn2, rn3, rNode, 64)/(4*PI*3);
      pmG[i][n1]+=tmp;
      pmG[i][n2]+=tmp;
      pmG[i][n3]+=tmp;	
    }
  }
  
  return true;
}

bool BuildBEMatrix::makeGhh(DenseMatrix& mG, SurfaceGeomHandle hSurfH){
  
  mG.zero();
  
  // -- getting raw pointer for fast access
  double** pmG = mG.getData2D();
  int nr = mG.ncols();
  int nc = mG.nrows();

  int nNodesH = hSurfH->node_.size();
  int nFacesH = hSurfH->face_.size();
  
  ASSERTEQ(nc, nNodesH);
  ASSERTEQ(nr, nNodesH);

  vector<NodeSimp>& nodesH = hSurfH->node_;
  
  int n1, n2, n3;
  double tmp = 0;

  for (int i=0; i<nNodesH; i++){               // -- for every node on the first surface
    Point& rNode = nodesH[i].p;
    for (int j=0; j<nFacesH; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurfH->face_[j].nodes[0];
      n2 = hSurfH->face_[j].nodes[1];
      n3 = hSurfH->face_[j].nodes[2];
      
      Point& rn1 = nodesH[n1].p;
      Point& rn2 = nodesH[n2].p;
      Point& rn3 = nodesH[n3].p;
      
      // TODO: find out if no singular integral is in the part
      tmp = getIntegral(rn1, rn2, rn3, rNode, 64)/(-4*PI*3);
      pmG[i][n1]+=tmp;
      pmG[i][n2]+=tmp;
      pmG[i][n3]+=tmp;	
    }
  }
  
  return true;
}

void BuildBEMatrix::releaseHandles(){
  hZbh_ = 0;
  hPhiH_ = 0;
}

//////////
// Getting omega by standard method
double BuildBEMatrix::getOmega(const Point& p1, const Point& p2, const Point& p3, const Point& pp){
  Vector r3 = p3 - pp;
  Vector r2 = p2 - pp;
  Vector r1 = p1 - pp;
  
  Vector v1 = Cross(r3, r1);
  Vector v2 = Cross(r1, r2);
  
  double alpha = atan(Cross(v1, v2).length()/(-Dot(v1, v2)));
  
  v1 = v2;
  v2 = Cross(r2, r3);
  
  double beta = atan(Cross(v1, v2).length()/(-Dot(v1, v2)));
  
  v1 = v2;
  v2 = Cross(r3, r1);
  
  double gamma = atan(Cross(v1, v2).length()/(-Dot(v1, v2)));
  
  Vector c = (p1.vector()+p2.vector()+p3.vector())/3 - pp.vector();                     // triangle centroid
  Vector n = Cross((p2-p1), (p3-p1)).normal();  // outward normal

  double dot = Dot(c, n);
  if (abs(dot)>NUMZERO)
    return (alpha+beta+gamma-PI)*dot/abs(dot);
  else
    return 0;
}

double BuildBEMatrix::getIntegral(const Point& p1, const Point& p2, const Point& p3, const Point& pp, int nt){

  Vector areaV = Cross(Vector(p2-p1), Vector(p3-p1))*0.5;
  Vector v1(p1), v2(p2), v3(p3), v(pp);

  //  center vector
  Vector cV;  
  double resOm = 0;
  DenseMatrix* pM = 0;
  
  switch (nt){
  case 16:
    pM = &coef16_;
    break;
  case 64:
    pM = &coef64_;
    break;
  case 128:
    pM = &coef256_;
    break;
  }
  DenseMatrix& coeffs = *pM;
  
  double sArea = areaV.length()/nt;
  for (int i=0; i<nt; i++){
    cV = v1*coeffs[i][0]*v1 + v2*coeffs[i][1] + v3*coeffs[i][2];
    resOm+=sArea/(cV-v).length();
  }
  return resOm;
}

//////////
// Initialize matrix used in calculations of subdivision coefficients
void BuildBEMatrix::initBase(){
  // -- initializing base matrix
  baseMtrx_[0][0][0] = 0.5; baseMtrx_[0][0][1] = 0.5;   baseMtrx_[0][0][2] = 0;
  baseMtrx_[0][1][0] = 0;   baseMtrx_[0][1][1] = 0.5;   baseMtrx_[0][1][2] = 0.5;
  baseMtrx_[0][2][0] = 0.5; baseMtrx_[0][2][1] = 0.0;   baseMtrx_[0][2][2] = 0.5;

  baseMtrx_[1][0][0] = 1.0; baseMtrx_[1][0][1] = 0.0;   baseMtrx_[1][0][2] = 0.0;
  baseMtrx_[1][1][0] = 0.5; baseMtrx_[1][1][1] = 0.5;   baseMtrx_[1][1][2] = 0.0;
  baseMtrx_[1][2][0] = 0.5; baseMtrx_[1][2][1] = 0.0;   baseMtrx_[1][2][2] = 0.5;

  baseMtrx_[2][0][0] = 0.5; baseMtrx_[2][0][1] = 0.5;   baseMtrx_[2][0][2] = 0.0;
  baseMtrx_[2][1][0] = 0.0; baseMtrx_[2][1][1] = 1.0;   baseMtrx_[2][1][2] = 0.0;
  baseMtrx_[2][2][0] = 0.0; baseMtrx_[2][2][1] = 0.5;   baseMtrx_[2][2][2] = 0.5;

  baseMtrx_[3][0][0] = 0.5; baseMtrx_[3][0][1] = 0.0;   baseMtrx_[3][0][2] = 0.5;
  baseMtrx_[3][1][0] = 0.0; baseMtrx_[3][1][1] = 0.5;   baseMtrx_[3][1][2] = 0.5;
  baseMtrx_[3][2][0] = 0.0; baseMtrx_[3][2][1] = 0.0;   baseMtrx_[3][2][2] = 1.0;
}

//////////
// Initialize matrices of subdivision of a triangle for integrals calculations
void BuildBEMatrix::init16(){
  // -- initializing subdivision matrix for n=16
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++){
      DenseMatrix tmpMtrx(4, 4);
      Mult(tmpMtrx, baseMtrx_[i], baseMtrx_[j]);
      for (int k=0; k<3; k++)
	coef16_[j+i*4][k] = tmpMtrx.sumOfCol(k)/3.0;
    }
}

void BuildBEMatrix::init64(){
  // -- initializing subdivision matrix for n=64
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      for (int j1=0; j1<4; j1++){
	DenseMatrix tmpMtrx(4, 4);
	Mult(tmpMtrx, baseMtrx_[i], baseMtrx_[j]);
	DenseMatrix tmpMtrx2(4, 4);
	Mult(tmpMtrx2, tmpMtrx, baseMtrx_[j1]);
	for (int k=0; k<3; k++)
	  coef64_[j1+j*4+i*4*4][k] = tmpMtrx2.sumOfCol(k)/3.0;
      }
}

void BuildBEMatrix::init128(){
  // -- initializing subdivision matrix for n=128
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      for (int j1=0; j1<4; j1++)
	for (int j2=0; j2<4; j2++){
	  DenseMatrix tmpMtrx1(4, 4);
	  Mult(tmpMtrx1, baseMtrx_[i], baseMtrx_[j]);
	  DenseMatrix tmpMtrx2(4, 4);
	  Mult(tmpMtrx2, tmpMtrx1, baseMtrx_[j1]);
	  Mult(tmpMtrx1, tmpMtrx2, baseMtrx_[j2]);
	  for (int k=0; k<3; k++)
	    coef256_[j2+4*(j1+4*(j+i*4))][k] = tmpMtrx1.sumOfCol(k)/3.0;
      }
}

} // end namespace BioPSE
