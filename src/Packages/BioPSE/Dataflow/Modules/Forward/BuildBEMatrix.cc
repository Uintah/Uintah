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
  FieldIPort*        d_iportSurfOut;
  FieldIPort*        d_iportSurfIn;

  //////////
  // Output ports pointers
  MatrixOPort*       d_oportMatrix;
  ColumnMatrixOPort* d_oportPhiHeart;
 
  DenseMatrixHandle  d_hZbh;
  ColumnMatrixHandle d_hPhiH;

  //////////
  // matrices used in calculation of integrals
  Array1<DenseMatrix> d_baseMtrx;
  DenseMatrix         d_coef16;
  DenseMatrix         d_coef64;
  DenseMatrix         d_coef256;

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
  d_baseMtrx(),
  d_coef16(16, 3),
  d_coef64(64, 3),
  d_coef256(256, 3)  
{
  // Create the input ports
  d_iportSurfOut = scinew FieldIPort(this, "Outer Surface", FieldIPort::Atomic);
  add_iport(d_iportSurfOut);

  d_iportSurfIn = scinew FieldIPort(this, "Inner Surface", FieldIPort::Atomic);
  add_iport(d_iportSurfIn);

  // Create the output ports
  d_oportMatrix = scinew MatrixOPort(this, "Zbh Matrix", MatrixIPort::Atomic);
  add_oport(d_oportMatrix);
  
  d_oportPhiHeart = scinew ColumnMatrixOPort(this, "Inner Surf Potentials", ColumnMatrixIPort::Atomic);
  add_oport(d_oportPhiHeart);
  
  d_hPhiH  = new ColumnMatrix(1);
  DenseMatrix tmpMtrx(4, 4);
  tmpMtrx.zero();

  d_baseMtrx.add(tmpMtrx);
  d_baseMtrx.add(tmpMtrx);
  d_baseMtrx.add(tmpMtrx);
  d_baseMtrx.add(tmpMtrx);

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
  
  if(!d_iportSurfOut->get(hFieldOut)) { 
    d_msgStream << "BuildBEMatrix -- couldn't get outer surface. Returning." << endl;
    return;
  }
  if(!d_iportSurfIn->get(hFieldIn)) { 
    d_msgStream << "BuildBEMatrix -- couldn't get inner surface. Returning." << endl;
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
    d_msgStream << "BuildBEMatrix -- couldn't cast Geom to SurfaceGeom for outer surface. Returning." << endl;
    releaseHandles();
    return;
  }
  
  if (hSurfIn.get_rep()){
    d_msgStream << "BuildBEMatrix -- couldn't cast Geom to SurfaceGeom for inner surface. Returning." << endl;
    releaseHandles();
    return;
  }

  if (hPhiIn.get_rep()){
    d_msgStream << "BuildBEMatrix -- couldn't cast Attrib to DiscreateAttrib<double> for inner surface potentials. Returning." << endl;
    releaseHandles();
    return;
  }
  
  // -- allocating matrices
  int nnIn = hSurfIn->d_face.size();
  int nnOut= hSurfOut->d_face.size();

  d_hZbh = scinew DenseMatrix(nnOut, nnIn);
  
  // -- STARTING CALCULATIONS
  // -- Zbh<-Gbh
  if(!makeGbh(*d_hZbh.get_rep(), hSurfOut, hSurfIn)){
    d_msgStream << "BuildBEMatrix: Cann't construct Gbh. Returning." << endl;
    return;
  }

  // -- calculating Ghh
  DenseMatrixHandle hGhh = scinew DenseMatrix(nnIn, nnIn);
  if(!makeGhh(*hGhh.get_rep(), hSurfIn)){
    d_msgStream << "BuildBEMatrix: Cann't construct Ghh. Returning." << endl;
    return;
  }

  // -- Ghh<-(Ghh^-1)
  hGhh->invert();
  
  // -- tmpBH<-Zbh*(Ghh^-1)
  DenseMatrixHandle hTmpBH = scinew DenseMatrix(nnOut, nnIn);
  Mult(*hTmpBH.get_rep(), *d_hZbh.get_rep(), *hGhh.get_rep());
  
  // -- Ybb<-tmpBH*Phb
  DenseMatrixHandle hPhb = scinew DenseMatrix(nnIn, nnOut);
  DenseMatrixHandle hYbb = scinew DenseMatrix(nnOut, nnOut);
  
  if (!makePhb(*hPhb.get_rep(), hSurfIn, hSurfOut)){
    d_msgStream << "BuildBEMatrix: Cann't construct Phb. Returning." << endl;
    return;
  }

  Mult(*hYbb.get_rep(), *hTmpBH.get_rep(), *hPhb.get_rep());
  
  // -- Ybb <- Pbb-Ybb
  DenseMatrixHandle hPbb = scinew DenseMatrix(nnOut, nnOut);

  if(!makePbb(*hPbb.get_rep(), hSurfOut)){
    d_msgStream << "BuildBEMatrix: Cann't construct Pbb. Returning." << endl;
    return;
  }

  Add(1.0, *hYbb.get_rep(), -1.0, *hPbb.get_rep());
  
  // -- Ybb <- (Ybb^-1)
  hYbb->invert();

  // -- Zbh <- tmpBH*Phh
  DenseMatrixHandle hPhh = scinew DenseMatrix(nnIn, nnIn);
  
  if(!makePhh(*hPhh.get_rep(), hSurfIn)){
    d_msgStream << "BuildBEMatrix: Cann't construct Phh. Returning." << endl;
    return;
  }

  Mult(*d_hZbh.get_rep(), *hTmpBH.get_rep(), *hPhh.get_rep());
  
  // -- Zbh <- Zbh-Pbh
  DenseMatrixHandle hPbh = scinew DenseMatrix(nnOut, nnIn);
  if (!makePbh(*hPbh.get_rep(), hSurfOut, hSurfIn)){
    d_msgStream << "BuildBEMatrix: Cann't construct Pbh. Returning." << endl;
    return;
  }

  Add(1.0, *d_hZbh.get_rep(), -1.0, *hPbh.get_rep());

  // -- tmpBH <- Ybb*Zbh
  Mult(*hTmpBH.get_rep(), *hYbb.get_rep(), *d_hZbh.get_rep());

  // -- Zbh <- tmpBH
  d_hZbh = hTmpBH;

  // TODO: PhiHeart vector, mapping from original geometry to the vector

  // -- sending handles to cloned objects
  d_oportMatrix->send(MatrixHandle(d_hZbh->clone()));
  d_oportPhiHeart->send(ColumnMatrixHandle(d_hPhiH->clone()));
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

  int nNodes = hSurf->d_node.size();
  int nFaces = hSurf->d_face.size();
  
  ASSERTEQ(nr, nc);
  ASSERTEQ(nr, nNodes);

  vector<NodeSimp>& nodes = hSurf->d_node;
  
  int n1, n2, n3;
  double tmp = 0;
  double autoAngle = 0;  // accumulator for autosolid angle value

  for (int i=0; i<nNodes; i++){               // -- for every node on the first surface
    Point& rNode = nodes[i].p;
    autoAngle = 0;
    for (int j=0; j<nFaces; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurf->d_face[j].nodes[0];
      n2 = hSurf->d_face[j].nodes[1];
      n3 = hSurf->d_face[j].nodes[2];
      
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

  int nNodes = hSurf->d_node.size();
  int nFaces = hSurf->d_face.size();
  
  ASSERTEQ(nr, nc);
  ASSERTEQ(nr, nNodes);

  vector<NodeSimp>& nodes = hSurf->d_node;
  
  int n1, n2, n3;
  double tmp = 0;
  double autoAngle = 0;  // accumulator for autosolid angle value

  for (int i=0; i<nNodes; i++){               // -- for every node on the first surface
    Point& rNode = nodes[i].p;
    autoAngle = 0;
    for (int j=0; j<nFaces; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurf->d_face[j].nodes[0];
      n2 = hSurf->d_face[j].nodes[1];
      n3 = hSurf->d_face[j].nodes[2];
      
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

  int nNodesB = hSurfB->d_node.size();
  int nNodesH = hSurfH->d_node.size();
  int nFacesH = hSurfH->d_face.size();
  
  ASSERTEQ(nr, nNodesB);
  ASSERTEQ(nc, nNodesH);

  vector<NodeSimp>& nodesB = hSurfB->d_node;
  vector<NodeSimp>& nodesH = hSurfH->d_node;
  
  int n1, n2, n3;
  double tmp = 0;

  for (int i=0; i<nNodesB; i++){               // -- for every node on the first surface
    Point& rNode = nodesB[i].p;
    for (int j=0; j<nFacesH; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurfH->d_face[j].nodes[0];
      n2 = hSurfH->d_face[j].nodes[1];
      n3 = hSurfH->d_face[j].nodes[2];
      
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

  int nNodesB = hSurfB->d_node.size();
  int nNodesH = hSurfH->d_node.size();
  int nFacesB = hSurfB->d_face.size();
  
  ASSERTEQ(nc, nNodesB);
  ASSERTEQ(nr, nNodesH);

  vector<NodeSimp>& nodesB = hSurfB->d_node;
  vector<NodeSimp>& nodesH = hSurfH->d_node;
  
  int n1, n2, n3;
  double tmp = 0;

  for (int i=0; i<nNodesH; i++){               // -- for every node on the first surface
    Point& rNode = nodesH[i].p;
    for (int j=0; j<nFacesB; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurfB->d_face[j].nodes[0];
      n2 = hSurfB->d_face[j].nodes[1];
      n3 = hSurfB->d_face[j].nodes[2];
      
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

  int nNodesB = hSurfB->d_node.size();
  int nNodesH = hSurfH->d_node.size();
  int nFacesH = hSurfB->d_face.size();
  
  ASSERTEQ(nc, nNodesB);
  ASSERTEQ(nr, nNodesH);

  vector<NodeSimp>& nodesB = hSurfB->d_node;
  vector<NodeSimp>& nodesH = hSurfH->d_node;
  
  int n1, n2, n3;
  double tmp = 0;

  for (int i=0; i<nNodesB; i++){               // -- for every node on the first surface
    Point& rNode = nodesB[i].p;
    for (int j=0; j<nFacesH; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurfH->d_face[j].nodes[0];
      n2 = hSurfH->d_face[j].nodes[1];
      n3 = hSurfH->d_face[j].nodes[2];
      
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

  int nNodesH = hSurfH->d_node.size();
  int nFacesH = hSurfH->d_face.size();
  
  ASSERTEQ(nc, nNodesH);
  ASSERTEQ(nr, nNodesH);

  vector<NodeSimp>& nodesH = hSurfH->d_node;
  
  int n1, n2, n3;
  double tmp = 0;

  for (int i=0; i<nNodesH; i++){               // -- for every node on the first surface
    Point& rNode = nodesH[i].p;
    for (int j=0; j<nFacesH; j++){             // -- find Rt-coefficient for each triangle
      n1 = hSurfH->d_face[j].nodes[0];
      n2 = hSurfH->d_face[j].nodes[1];
      n3 = hSurfH->d_face[j].nodes[2];
      
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
  d_hZbh = 0;
  d_hPhiH = 0;
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
    pM = &d_coef16;
    break;
  case 64:
    pM = &d_coef64;
    break;
  case 128:
    pM = &d_coef256;
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
  d_baseMtrx[0][0][0] = 0.5; d_baseMtrx[0][0][1] = 0.5;   d_baseMtrx[0][0][2] = 0;
  d_baseMtrx[0][1][0] = 0;   d_baseMtrx[0][1][1] = 0.5;   d_baseMtrx[0][1][2] = 0.5;
  d_baseMtrx[0][2][0] = 0.5; d_baseMtrx[0][2][1] = 0.0;   d_baseMtrx[0][2][2] = 0.5;

  d_baseMtrx[1][0][0] = 1.0; d_baseMtrx[1][0][1] = 0.0;   d_baseMtrx[1][0][2] = 0.0;
  d_baseMtrx[1][1][0] = 0.5; d_baseMtrx[1][1][1] = 0.5;   d_baseMtrx[1][1][2] = 0.0;
  d_baseMtrx[1][2][0] = 0.5; d_baseMtrx[1][2][1] = 0.0;   d_baseMtrx[1][2][2] = 0.5;

  d_baseMtrx[2][0][0] = 0.5; d_baseMtrx[2][0][1] = 0.5;   d_baseMtrx[2][0][2] = 0.0;
  d_baseMtrx[2][1][0] = 0.0; d_baseMtrx[2][1][1] = 1.0;   d_baseMtrx[2][1][2] = 0.0;
  d_baseMtrx[2][2][0] = 0.0; d_baseMtrx[2][2][1] = 0.5;   d_baseMtrx[2][2][2] = 0.5;

  d_baseMtrx[3][0][0] = 0.5; d_baseMtrx[3][0][1] = 0.0;   d_baseMtrx[3][0][2] = 0.5;
  d_baseMtrx[3][1][0] = 0.0; d_baseMtrx[3][1][1] = 0.5;   d_baseMtrx[3][1][2] = 0.5;
  d_baseMtrx[3][2][0] = 0.0; d_baseMtrx[3][2][1] = 0.0;   d_baseMtrx[3][2][2] = 1.0;
}

//////////
// Initialize matrices of subdivision of a triangle for integrals calculations
void BuildBEMatrix::init16(){
  // -- initializing subdivision matrix for n=16
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++){
      DenseMatrix tmpMtrx(4, 4);
      Mult(tmpMtrx, d_baseMtrx[i], d_baseMtrx[j]);
      for (int k=0; k<3; k++)
	d_coef16[j+i*4][k] = tmpMtrx.sumOfCol(k)/3.0;
    }
}

void BuildBEMatrix::init64(){
  // -- initializing subdivision matrix for n=64
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      for (int j1=0; j1<4; j1++){
	DenseMatrix tmpMtrx(4, 4);
	Mult(tmpMtrx, d_baseMtrx[i], d_baseMtrx[j]);
	DenseMatrix tmpMtrx2(4, 4);
	Mult(tmpMtrx2, tmpMtrx, d_baseMtrx[j1]);
	for (int k=0; k<3; k++)
	  d_coef64[j1+j*4+i*4*4][k] = tmpMtrx2.sumOfCol(k)/3.0;
      }
}

void BuildBEMatrix::init128(){
  // -- initializing subdivision matrix for n=128
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      for (int j1=0; j1<4; j1++)
	for (int j2=0; j2<4; j2++){
	  DenseMatrix tmpMtrx1(4, 4);
	  Mult(tmpMtrx1, d_baseMtrx[i], d_baseMtrx[j]);
	  DenseMatrix tmpMtrx2(4, 4);
	  Mult(tmpMtrx2, tmpMtrx1, d_baseMtrx[j1]);
	  Mult(tmpMtrx1, tmpMtrx2, d_baseMtrx[j2]);
	  for (int k=0; k<3; k++)
	    d_coef256[j2+4*(j1+4*(j+i*4))][k] = tmpMtrx1.sumOfCol(k)/3.0;
      }
}

} // end namespace BioPSE
