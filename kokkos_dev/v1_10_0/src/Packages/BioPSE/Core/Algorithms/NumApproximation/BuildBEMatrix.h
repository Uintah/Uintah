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
 *  BuildBEMatrix.h:  class to build Boundary Elements matrix
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   March 2001   
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <iostream>
#include <math.h>

namespace BioPSE {

using namespace SCIRun;

class BuildBEMatrix;

typedef LockingHandle<BuildBEMatrix>   BuildBEMatrixHandle;
typedef LockingHandle<DenseMatrix>     DenseMatrixHandle;

class BuildBEMatrix: public Datatype {

  //! Private data members
  TriSurfMeshHandle        hInnerSurf_;
  TriSurfMeshHandle        hOuterSurf_;
  vector<Vector>           avInn_;
  vector<Vector>           avOut_;
  const DenseMatrix&       cf_;

  DenseMatrixHandle  hPbb_;
  DenseMatrixHandle  hPhh_;
  DenseMatrixHandle  hPbh_;
  DenseMatrixHandle  hPhb_;
  DenseMatrixHandle  hGbh_;
  DenseMatrixHandle  hGhh_;
  DenseMatrixHandle  hZbh_;
  MatrixHandle&      hA_;

  Mutex             lock_Pbb_;
  Mutex             lock_Phh_;
  Mutex             lock_Pbh_;
  Mutex             lock_Phb_;
  Mutex             lock_Gbh_;
  Mutex             lock_Ghh_;
  Mutex             lock_avInn_;
  Mutex             lock_avOut_;
  Mutex             lock_print_;
  
  int               np_;
  Barrier           barrier_;
 
  //! number of subdivisions used in the approximating surface integrals
  int               nsubs_;

  //! static private data
  static Array1<DenseMatrix> base_matrix_;
  static DenseMatrix c16_;
  static DenseMatrix c64_;
  static DenseMatrix c256_;

  //! Private methods
  void parallel(int);
  
  void makePbb();
  void makePbh();
  void makePhb();
  void makePhh();
  void makeGbh();
  void makeGhh();
  
  //! precalculation of triangle areas
  void calc_tri_area(TriSurfMeshHandle&, vector<Vector>&);
  
  //!
  inline void getOmega(const DenseMatrix&,
		       const Vector&,
		       DenseMatrix&,
		       DenseMatrix&,
		       DenseMatrix&);
  
  //!
  inline void getIntegral(const DenseMatrix&,
			  const Vector&,
			  DenseMatrix&,
			  DenseMatrix&,
			  DenseMatrix&);
  
  //! static private methods
  static void init_base();
  static void init_16();
  static void init_64();
  static void init_256();

  //! Constructor
  BuildBEMatrix(TriSurfMeshHandle, TriSurfMeshHandle, MatrixHandle&, const DenseMatrix&, int);
  
public:
 
  static bool build_BEMatrix(TriSurfMeshHandle,     // handle to inner surface
			     TriSurfMeshHandle,     // handle to outer surface
			     MatrixHandle&,         // handle to result matrix
			     int );                 // level of precision in solid angle calculation(1-3)
  //! Destuctor
  virtual ~BuildBEMatrix();
  virtual void io(Piostream&);
};

inline void  BuildBEMatrix::getOmega(const DenseMatrix& pp,
				     const Vector& areaV,
				     DenseMatrix& cVector,
				     DenseMatrix& omega,
				     DenseMatrix& coef)
{
  //! obtaining vectors to centers of subdivision triangles
  Mult(cVector, cf_, pp);
  
  DenseMatrix av(3, 1);
  av[0][0] = areaV.x()/nsubs_;
  av[1][0] = areaV.y()/nsubs_;
  av[2][0] = areaV.z()/nsubs_;

  //! finding coeffs for each subdivision triangle
  Mult(omega, cVector, av);
  
  int i, ii=0;
  double tmp;
  double* raw = cVector.getData();
  
  for (i=0; i<nsubs_; ++i){
    tmp = sqrt(raw[ii]*raw[ii] + raw[ii+1]*raw[ii+1] + raw[ii+2]*raw[ii+2]);
    tmp = tmp*tmp*tmp;
    ii+=3;
    omega[i][0]/=tmp;
  }
  
  //! summing contributions from each triangle
  Mult_trans_X(coef, omega, cf_);
}

inline void  BuildBEMatrix::getIntegral(const DenseMatrix& pp,
					const Vector& areaV,
					DenseMatrix& cVector,
					DenseMatrix& omega,
					DenseMatrix& coef)
{
  //! obtaining vectors to centers of subdivision triangles
  Mult(cVector, cf_, pp);
  
  double areaVal = areaV.length()/nsubs_;
  int i, ii=0;
  double tmp;
  double* raw = cVector.getData();
  
  for (i=0; i<nsubs_; ++i){
    tmp = sqrt(raw[ii]*raw[ii] + raw[ii+1]*raw[ii+1] + raw[ii+2]*raw[ii+2]);
    ii+=3;
    omega[i][0] = areaVal/tmp;
  }
  
  //! summing contriibutions from each triangle
  Mult_trans_X(coef, omega, cf_);
}

} // end namespace BioPSE
