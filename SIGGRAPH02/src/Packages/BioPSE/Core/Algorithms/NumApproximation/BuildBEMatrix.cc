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
 *  BuildBEMatrix.cc:  Build finite element matrix
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   March 2001   
 *  Copyright (C) 2001 SCI Group
 */

#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildBEMatrix.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <Core/Util/Timer.h>

namespace BioPSE {

#define NUMZERO 10e-13
#define PI 3.141592653589738

using namespace SCIRun;

//! static field initialization
Array1<DenseMatrix> BuildBEMatrix::base_matrix_;
DenseMatrix BuildBEMatrix::c16_(16, 3);
DenseMatrix BuildBEMatrix::c64_(64, 3);
DenseMatrix BuildBEMatrix::c256_(256, 3);

//! Constructor
// -- it's private, no occasional object creation
BuildBEMatrix::BuildBEMatrix(TriSurfMeshHandle hInn, 
			     TriSurfMeshHandle hOut, 
			     MatrixHandle& hA,
			     const DenseMatrix& cf,
			     int np)
:
  // ---------------------------------------------
  hInnerSurf_(hInn),
  hOuterSurf_(hOut),
  avInn_(),
  avOut_(),
  cf_(cf),
  hA_(hA),
  lock_Pbb_("Pbb mutex"),
  lock_Phh_("Phh mutex"),
  lock_Pbh_("Pbh mutex"),
  lock_Phb_("Phb mutex"),
  lock_Gbh_("Gbh mutex"),
  lock_Ghh_("Ghh mutex"),
  lock_avInn_("avInn mutex"),
  lock_avOut_("avOut mutex"),
  lock_print_("print lock"),
  np_(np),
  barrier_("BuildBEMatrix barrier")
{
  nsubs_ = cf_.nrows();
}

BuildBEMatrix::~BuildBEMatrix(){}

bool BuildBEMatrix::build_BEMatrix(TriSurfMeshHandle hInn,    // handle to inner surface
				     TriSurfMeshHandle hOut,    // handle to outer surface
				     MatrixHandle& hA,          // handle to result matrix
				     int prec)                  // level of precision in solid angle calculation
//------------------------------------------------
{
  static Mutex lock_init("BuildBEMatrix::build_BEMatrix lock");
  static bool isInit = false;

  //! initializing subdivision coefficients
  lock_init.lock();  
  if (!isInit){
    isInit = true;
    init_base();
    init_16();
    init_64();
    init_256();
  }
  lock_init.unlock();
  
  //! choosing which coefficients to use
  DenseMatrix* pCoeff;
  switch (prec){
  case 1:
    pCoeff = &c16_;
    break;
  case 2:
    pCoeff = &c64_;
    break;
  case 3:
    pCoeff = &c256_;
    break;
  default:
    pCoeff = &c16_;
  }

  int np=Thread::numProcessors();
  
  if (np>10) {
    np=6;
  }

  //! by now using one processor
  np = 1;
  hA = 0;

  BuildBEMatrixHandle hMaker = new BuildBEMatrix(hInn, hOut, hA, *pCoeff, np);

  Thread::parallel(Parallel<BuildBEMatrix>(hMaker.get_rep(), &BuildBEMatrix::parallel),
		   np, true);

  if (hA.get_rep())
    return true;
  else
    return false;
}

// -- callback routine to execute in parallel
void BuildBEMatrix::parallel(int proc)
{
  //! precalculating triangle areas
  // -- for inner surface
  if (lock_avInn_.tryLock()) {
    if (avInn_.size()==0){
      calc_tri_area(hInnerSurf_, avInn_);
    }
    lock_avInn_.unlock();
  }
  
  // -- for outer surface
  if (lock_avOut_.tryLock() ){ 
    if(avOut_.size()==0){
      calc_tri_area(hOuterSurf_, avOut_);
    }
    lock_avOut_.unlock();
  }
  
  //! area vectors are used by every processor, wait
  barrier_.wait(np_);

  //! distributing matrix calculation among processors
  //! Pbb
  if (lock_Pbb_.tryLock() ){ 
    if(!hPbb_.get_rep()){
      makePbb();
    }
    lock_Pbb_.unlock();
  }

  //! Phh
  if (lock_Phh_.tryLock() ){ 
    if(!hPhh_.get_rep()){
      makePhh();
    }
    lock_Phh_.unlock();
  }

  //! Pbh
  if (lock_Pbh_.tryLock() ){ 
    if(!hPbh_.get_rep()){
      makePbh();
    }
    lock_Pbh_.unlock();
  }
  
  //! Phb
  if (lock_Phb_.tryLock() ){ 
    if(!hPhb_.get_rep()){
      makePhb();
    }
    lock_Phb_.unlock();
  }

  //! Ghh
  if (lock_Ghh_.tryLock() ){ 
    if(!hGhh_.get_rep()){
      makeGhh();
      //! we only need inverted Ghh
      hGhh_->invert();
    }
    lock_Ghh_.unlock();
  }

  //! Gbh and partial Zbh construction
  if (lock_Gbh_.tryLock()){ 
    if(!hGbh_.get_rep()){
      makeGbh();
  
      //! when locking, Ghh already done
      lock_Ghh_.lock();
      hZbh_ = new DenseMatrix(hGbh_->nrows(), hGbh_->ncols());
      //! Zbh <- Gbh*Ghh^-1
      Mult(*(hZbh_.get_rep()), *(hGbh_.get_rep()), *(hGhh_.get_rep()));
      lock_Ghh_.unlock();
    }
    lock_Gbh_.unlock();
  }
  
  barrier_.wait(np_);

  // two processors work
  //! Zbh
  if (proc==0 || proc==1){
    
    if (proc == 0)
      if (lock_Pbb_.tryLock()){
	DenseMatrix tmpBB(hPbb_->nrows(), hPbb_->ncols());
	Mult(tmpBB, *hZbh_.get_rep(), *hPhb_.get_rep());
	Add(1, *(hPbb_.get_rep()), -1, tmpBB);
	hPbb_->invert();
	lock_Pbb_.unlock();
      }
    
    if (proc == 1 || proc ==0 && np_==1)
      if (lock_Pbh_.tryLock()){
	DenseMatrix tmpBH(hPbh_->nrows(), hPbh_->ncols());
	Mult(tmpBH, *hZbh_.get_rep(), *hPhh_.get_rep());
	Add(-1, *(hPbh_.get_rep()), 1, tmpBH);
	lock_Pbh_.unlock();
      }
    
    barrier_.wait((np_==1)?1:2);
    
    if (proc==0){
      Mult(*(hZbh_.get_rep()), *(hPbb_.get_rep()), *(hPbh_.get_rep()));
      hA_ = hZbh_.get_rep();
    }
  }
}

void BuildBEMatrix::io(Piostream&){
}

//! static functions for initializations
void BuildBEMatrix::init_base(){

  DenseMatrix tmp(3, 3);
  tmp[0][0] = 0.5; tmp[0][1] = 0.5;   tmp[0][2] = 0;
  tmp[1][0] = 0;   tmp[1][1] = 0.5;   tmp[1][2] = 0.5;
  tmp[2][0] = 0.5; tmp[2][1] = 0.0;   tmp[2][2] = 0.5;

  base_matrix_.add(tmp);

  tmp[0][0] = 1.0; tmp[0][1] = 0.0;   tmp[0][2] = 0.0;
  tmp[1][0] = 0.5; tmp[1][1] = 0.5;   tmp[1][2] = 0.0;
  tmp[2][0] = 0.5; tmp[2][1] = 0.0;   tmp[2][2] = 0.5;
  base_matrix_.add(tmp);

  tmp[0][0] = 0.5; tmp[0][1] = 0.5;   tmp[0][2] = 0.0;
  tmp[1][0] = 0.0; tmp[1][1] = 1.0;   tmp[1][2] = 0.0;
  tmp[2][0] = 0.0; tmp[2][1] = 0.5;   tmp[2][2] = 0.5;
  base_matrix_.add(tmp);

  tmp[0][0] = 0.5; tmp[0][1] = 0.0;   tmp[0][2] = 0.5;
  tmp[1][0] = 0.0; tmp[1][1] = 0.5;   tmp[1][2] = 0.5;
  tmp[2][0] = 0.0; tmp[2][1] = 0.0;   tmp[2][2] = 1.0;
  base_matrix_.add(tmp);
}

//! initialization of coefficients for triangle subdivisions
void BuildBEMatrix::init_16(){
  DenseMatrix tmpMtrx(3, 3);
  int i, j, k;
  int ind = 0;
  for(i=0; i<4; ++i){
    for (j=0; j<4; ++j){
      Mult(tmpMtrx, base_matrix_[i], base_matrix_[j]);
      
      for (k=0; k<3; ++k){
	c16_[ind][k] = tmpMtrx.sumOfCol(k)/3.0;
      }
      ind++;
    }
  }
}

void BuildBEMatrix::init_64(){
  
  DenseMatrix tmpMtrx1(3, 3);
  DenseMatrix tmpMtrx2(3, 3);
  int i, j, k, l;
  int ind = 0;
  
  for(i=0; i<4; ++i){
    for (j=0; j<4; ++j){
      Mult(tmpMtrx1, base_matrix_[i], base_matrix_[j]);
      
      for (l=0; l<4; ++l){	
	Mult(tmpMtrx2, tmpMtrx1, base_matrix_[l]);
	
	for (k=0; k<3; ++k){
	  c64_[ind][k] = tmpMtrx2.sumOfCol(k)/3.0;
	  
	}
	ind++;
      }
    }
  }
}

void BuildBEMatrix::init_256(){
  DenseMatrix tmpMtrx1(3, 3);
  DenseMatrix tmpMtrx2(3, 3);
  DenseMatrix tmpMtrx3(3, 3);
  int i, j, k, l, m;
  int ind = 0;
  
  for(i=0; i<4; ++i){
    for (j=0; j<4; ++j){
      Mult(tmpMtrx1, base_matrix_[i], base_matrix_[j]);

      for (l=0; l<4; ++l){
	Mult(tmpMtrx2, tmpMtrx1, base_matrix_[l]);

	for (m=0; m<4; ++m){
	  Mult(tmpMtrx3, tmpMtrx2, base_matrix_[m]);

	  for (k=0; k<3; ++k){
	    c256_[ind][k] = tmpMtrx3.sumOfCol(k)/3.0; 
	  }

	  ind++;
	}
      }
    }
  }
}

void BuildBEMatrix::makePbb(){
 
  TriSurfMeshHandle hsurf = hOuterSurf_->clone();
 
  TriSurfMesh::Node::size_type nsize; hsurf->size(nsize);
  unsigned int nnodes = nsize;
  DenseMatrix* tmp = new DenseMatrix(nnodes, nnodes);
  hPbb_ = tmp;
  DenseMatrix& Pbb = *tmp;
  Pbb.zero();
  
  DenseMatrix omega(nsubs_, 1);
  DenseMatrix  pts(3, 3);

  const double mult = 1/(4*PI);

  TriSurfMesh::Node::array_type nodes;
  DenseMatrix coef(1, 3);
  DenseMatrix cVector(nsubs_, 3);

  TriSurfMesh::Node::iterator ni, nie;
  TriSurfMesh::Face::iterator fi, fie;

  double tt = 0;

  unsigned int i;

  hsurf->begin(ni); hsurf->end(nie);
  for (; ni != nie; ++ni){ //! for every node
    TriSurfMesh::Node::index_type ppi = *ni;
    Point pp = hsurf->point(ppi);

    hsurf->begin(fi); hsurf->end(fie);
    for (; fi != fie; ++fi) { //! find contributions from every triangle

      hsurf->get_nodes(nodes, *fi);     
      if (ppi!=nodes[0] && ppi!=nodes[1] && ppi!=nodes[2]){
	 Vector v1 = hsurf->point(nodes[0]) - pp;
	 Vector v2 = hsurf->point(nodes[1]) - pp;
	 Vector v3 = hsurf->point(nodes[2]) - pp;
	 pts[0][0] = v1.x(); pts[0][1] = v1.y();  pts[0][2] = v1.z();
	 pts[1][0] = v2.x(); pts[1][1] = v2.y();  pts[1][2] = v2.z();
	 pts[2][0] = v3.x(); pts[2][1] = v3.y();  pts[2][2] = v3.z();
	 
	 WallClockTimer timer;
	 timer.start();
	 getOmega(pts, avOut_[*fi], cVector, omega, coef);
	 tt +=timer.time();
	 timer.stop();
	 for (i=0; i<3; ++i)
	   Pbb[ppi][nodes[i]]+=coef[0][i]*mult;
      }
    }
  }
  
  //! accounting for autosolid angle
  for (i=0; i<nnodes; ++i){
    Pbb[i][i] = -Pbb.sumOfRow(i);
  }
}

void BuildBEMatrix::makePhh(){
  
  TriSurfMeshHandle hsurf = hInnerSurf_->clone();

  TriSurfMesh::Node::size_type nsize; hsurf->size(nsize);
  unsigned int nnodes = nsize;
  DenseMatrix* tmp = new DenseMatrix(nnodes, nnodes);
  hPhh_ = tmp;
  DenseMatrix& Phh = *tmp;
  Phh.zero();

  DenseMatrix omega(nsubs_, 1);
  DenseMatrix  pts(3, 3);
  const double mult = 1/(-4*PI);

  TriSurfMesh::Node::array_type nodes;
  DenseMatrix coef(1, 3);
  DenseMatrix cVector(nsubs_, 3);

  TriSurfMesh::Node::iterator  ni, nie;
  TriSurfMesh::Face::iterator  fi, fie;

  unsigned int i;

  hsurf->begin(ni); hsurf->end(nie);
  for (; ni != nie; ++ni){ //! for every node
    TriSurfMesh::Node::index_type ppi = *ni;
    Point pp = hsurf->point(ppi);

    hsurf->begin(fi); hsurf->end(fie);
    for (; fi != fie; ++fi){ //! find contributions from every triangle

      hsurf->get_nodes(nodes, *fi);     
      if (ppi!=nodes[0] && ppi!=nodes[1] && ppi!=nodes[2]){
	 Vector v1 = hsurf->point(nodes[0]) - pp;
	 Vector v2 = hsurf->point(nodes[1]) - pp;
	 Vector v3 = hsurf->point(nodes[2]) - pp;
	 pts[0][0] = v1.x(); pts[0][1] = v1.y();  pts[0][2] = v1.z();
	 pts[1][0] = v2.x(); pts[1][1] = v2.y();  pts[1][2] = v2.z();
	 pts[2][0] = v3.x(); pts[2][1] = v3.y();  pts[2][2] = v3.z();
	 getOmega(pts, avInn_[*fi], cVector, omega, coef);
	 
	 for (i=0; i<3; ++i)
	   Phh[ppi][nodes[i]]+=coef[0][i]*mult;
      }
    }
  }
  
  //! accounting for autosolid angle
  for (i=0; i<nnodes; ++i)
  {
    Phh[i][i] = -1-Phh.sumOfRow(i);
  }
}

void BuildBEMatrix::makePbh(){
 
  TriSurfMeshHandle hsurf1 = hOuterSurf_->clone();
  TriSurfMeshHandle hsurf2 = hInnerSurf_->clone();

  TriSurfMesh::Node::size_type nsize1; hsurf1->size(nsize1);
  TriSurfMesh::Node::size_type nsize2; hsurf2->size(nsize2);
  DenseMatrix* tmp = new DenseMatrix(nsize1, nsize2);
  hPbh_ = tmp;
  DenseMatrix& Pbh = *tmp;
  Pbh.zero();
  
  DenseMatrix omega(nsubs_, 1);
  DenseMatrix  pts(3, 3);
  const double mult = 1/(-4*PI);

  TriSurfMesh::Node::array_type nodes;
  DenseMatrix coef(1, 3);
  DenseMatrix cVector(nsubs_, 3);
  int i;

  TriSurfMesh::Node::iterator  ni, nie;
  TriSurfMesh::Face::iterator  fi, fie;
  
  hsurf1->begin(ni); hsurf1->end(nie);
  for (; ni != nie; ++ni){ //! for every node
    TriSurfMesh::Node::index_type ppi = *ni;
    Point pp = hsurf1->point(ppi);

    hsurf2->begin(fi); hsurf2->end(fie);
    for (; fi != fie; ++fi){ //! find contributions from every triangle
      
      hsurf2->get_nodes(nodes, *fi);
      Vector v1 = hsurf2->point(nodes[0]) - pp;
      Vector v2 = hsurf2->point(nodes[1]) - pp;
      Vector v3 = hsurf2->point(nodes[2]) - pp;
      pts[0][0] = v1.x(); pts[0][1] = v1.y();  pts[0][2] = v1.z();
      pts[1][0] = v2.x(); pts[1][1] = v2.y();  pts[1][2] = v2.z();
      pts[2][0] = v3.x(); pts[2][1] = v3.y();  pts[2][2] = v3.z();
      getOmega(pts, avInn_[*fi], cVector, omega, coef);
      
      for (i=0; i<3; ++i)
	Pbh[ppi][nodes[i]]+=coef[0][i]*mult;
    }
  }
}

void BuildBEMatrix::makePhb(){
  
  TriSurfMeshHandle hsurf1 = hInnerSurf_->clone();
  TriSurfMeshHandle hsurf2 = hOuterSurf_->clone();

  TriSurfMesh::Node::size_type nsize1; hsurf1->size(nsize1);
  TriSurfMesh::Node::size_type nsize2; hsurf2->size(nsize2);
  DenseMatrix* tmp = new DenseMatrix(nsize1, nsize2);
  hPhb_ = tmp;
  DenseMatrix& Phb = *tmp;
  Phb.zero();
  
  DenseMatrix omega(nsubs_, 1);
  DenseMatrix  pts(3, 3);
  const double mult = 1/(4*PI);

  TriSurfMesh::Node::array_type nodes;
  DenseMatrix coef(1, 3);
  DenseMatrix cVector(nsubs_, 3);
  int i;

  TriSurfMesh::Node::iterator  ni, nie;
  TriSurfMesh::Face::iterator  fi, fie;
  
  hsurf1->begin(ni); hsurf1->end(nie);
  for (; ni != nie; ++ni){ //! for every node
    TriSurfMesh::Node::index_type ppi = *ni;
    Point pp = hsurf1->point(ppi);

    hsurf2->begin(fi); hsurf2->end(fie);
    for (; fi != fie; ++fi){ //! find contributions from every triangle
      
      hsurf2->get_nodes(nodes, *fi);
      Vector v1 = hsurf2->point(nodes[0]) - pp;
      Vector v2 = hsurf2->point(nodes[1]) - pp;
      Vector v3 = hsurf2->point(nodes[2]) - pp;
      pts[0][0] = v1.x(); pts[0][1] = v1.y();  pts[0][2] = v1.z();
      pts[1][0] = v2.x(); pts[1][1] = v2.y();  pts[1][2] = v2.z();
      pts[2][0] = v3.x(); pts[2][1] = v3.y();  pts[2][2] = v3.z();
      getOmega(pts, avOut_[*fi], cVector, omega, coef);
      
      for (i=0; i<3; ++i)
	Phb[ppi][nodes[i]]+=coef[0][i]*mult;
    }
  }
}

void BuildBEMatrix::makeGhh(){
  
  TriSurfMeshHandle hsurf = hInnerSurf_->clone();

  TriSurfMesh::Node::size_type nsize; hsurf->size(nsize);
  unsigned int nnodes = nsize;
  DenseMatrix* tmp = new DenseMatrix(nnodes, nnodes);
  hGhh_ = tmp;
  DenseMatrix& Ghh = *tmp;
  Ghh.zero();

  DenseMatrix omega(nsubs_, 1);
  DenseMatrix  pts(3, 3);
  const double mult = 1/(-4*PI);

  TriSurfMesh::Node::array_type nodes;
  DenseMatrix coef(1, 3);
  DenseMatrix cVector(nsubs_, 3);
  unsigned int i;

  TriSurfMesh::Node::iterator ni, nie;
  TriSurfMesh::Face::iterator fi, fie;

  hsurf->begin(ni);
  hsurf->end(nie);
  for (; ni != nie; ++ni){ //! for every node
    TriSurfMesh::Node::index_type ppi = *ni;
    Point pp = hsurf->point(ppi);

    hsurf->begin(fi); hsurf->end(fie);
    for (; fi != fie; ++fi){ //! find contributions from every triangle

      hsurf->get_nodes(nodes, *fi);     
      
      Vector v1 = hsurf->point(nodes[0]) - pp;
      Vector v2 = hsurf->point(nodes[1]) - pp;
      Vector v3 = hsurf->point(nodes[2]) - pp;
      pts[0][0] = v1.x(); pts[0][1] = v1.y();  pts[0][2] = v1.z();
      pts[1][0] = v2.x(); pts[1][1] = v2.y();  pts[1][2] = v2.z();
      pts[2][0] = v3.x(); pts[2][1] = v3.y();  pts[2][2] = v3.z();
      getIntegral(pts, avInn_[*fi], cVector, omega, coef);
      
      for (i=0; i<3; ++i)
	Ghh[ppi][nodes[i]]+=coef[0][i]*mult; 
    }
  }
}

void BuildBEMatrix::makeGbh(){
 
  TriSurfMeshHandle hsurf1 = hOuterSurf_->clone();
  TriSurfMeshHandle hsurf2 = hInnerSurf_->clone();

  TriSurfMesh::Node::size_type nsize1; hsurf1->size(nsize1);
  TriSurfMesh::Node::size_type nsize2; hsurf2->size(nsize2);
  DenseMatrix* tmp = new DenseMatrix(nsize1, nsize2);
  hGbh_ = tmp;
  DenseMatrix& Gbh = *tmp;
  Gbh.zero();
  
  DenseMatrix omega(nsubs_, 1);
  DenseMatrix  pts(3, 3);
  const double mult = 1/(-4*PI);

  TriSurfMesh::Node::array_type nodes;
  DenseMatrix coef(1, 3);
  DenseMatrix cVector(nsubs_, 3);
  int i;

  TriSurfMesh::Node::iterator  ni, nie;
  TriSurfMesh::Face::iterator  fi, fie;
  
  hsurf1->begin(ni); hsurf1->end(nie);
  for (; ni != nie; ++ni){ //! for every node
    TriSurfMesh::Node::index_type ppi = *ni;
    Point pp = hsurf1->point(ppi);

    hsurf2->begin(fi); hsurf2->end(fie);
    for (; fi != fie; ++fi){ //! find contributions from every triangle
      
      hsurf2->get_nodes(nodes, *fi);
      Vector v1 = hsurf2->point(nodes[0]) - pp;
      Vector v2 = hsurf2->point(nodes[1]) - pp;
      Vector v3 = hsurf2->point(nodes[2]) - pp;
      pts[0][0] = v1.x(); pts[0][1] = v1.y();  pts[0][2] = v1.z();
      pts[1][0] = v2.x(); pts[1][1] = v2.y();  pts[1][2] = v2.z();
      pts[2][0] = v3.x(); pts[2][1] = v3.y();  pts[2][2] = v3.z();
      getIntegral(pts, avInn_[*fi], cVector, omega, coef);
      
      for (i=0; i<3; ++i)
	Gbh[ppi][nodes[i]]+=coef[0][i]*mult;
    }
  }
}

//! precalculate triangles area
void BuildBEMatrix::calc_tri_area(TriSurfMeshHandle& hsurf, vector<Vector>& areaV){
 
  TriSurfMesh::Face::iterator  fi, fie;
  TriSurfMesh::Node::array_type     nodes;
  
  hsurf->begin(fi); hsurf->end(fie);
  for (; fi != fie; ++fi) {
    hsurf->get_nodes(nodes, *fi);
    Vector v1 = hsurf->point(nodes[1]) - hsurf->point(nodes[0]);
    Vector v2 = hsurf->point(nodes[2]) - hsurf->point(nodes[1]);
    areaV.push_back(Cross(v1, v2)*0.5);
  }
}
 
} // end namespace BioPSE
