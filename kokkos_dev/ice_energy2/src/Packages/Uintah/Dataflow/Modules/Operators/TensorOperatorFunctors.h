/********************************************************
 * TensorOperatorFunctors.h
 *
 * Author: Wayne Witzel
 * Scientific Computing and Imaging,
 * University of Utah
 * Copyright 2000
 */

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <iostream>

/* Functors used by TensorFieldOperator and TensorParticlesOperator */

namespace Uintah {

  /************************************
   * Template

     struct MyNewTensorOp
     {
       MyNewTensorOp(optional_parameters):
         class_member_initialization(optional_parameters)
       { // do initialization stuff if you want }

       // This is the function which does the operation you want.
       inline double operator()(Matrix3 M)
       { return myoperation_on_M(M); }

       // These are entirely optional
       class_member_initialization;
     };
     
  */

  
  struct TensorElementExtractionOp
  {
    TensorElementExtractionOp(int row, int column)
      : m_row(row), m_column(column) { }

    inline double operator()(Matrix3 M)
    { return M(m_row-1, m_column-1); }

    int m_row;
    int m_column;
  };

#define Eigen2DOp(plane) \
struct Eigen2D##plane##Op \
{ \
  Eigen2D##plane##Op() {} \
  inline double operator()(Matrix3 M) \
  { \
    double e1, e2; \
    int num_eigen_values = M.get##plane##EigenValues(e1, e2); \
    return (num_eigen_values == 2) ? e1 - e2 : 0; \
  } \
}
  Eigen2DOp(XY); // declares Eigen2DXYOp struct from above macro
  Eigen2DOp(XZ); // declares Eigen2DXZOp struct from above macro
  Eigen2DOp(YZ); // declares Eigen2DYZOp struct from above macro

#define Eigen2DCosOp(plane) \
struct Eigen2D##plane##CosOp \
{ \
  Eigen2D##plane##CosOp(double delta) \
  : m_delta(delta) {} \
  inline double operator()(Matrix3 M) \
  { \
    double e1, e2; \
    int num_eigen_values = M.get##plane##EigenValues(e1, e2); \
    return (num_eigen_values == 2) ? cos((e1 - e2) / m_delta) : 1; \
  } \
  double m_delta; \
}
Eigen2DCosOp(XY); // declares Eigen2DXYCosOp struct from above macro
Eigen2DCosOp(XZ); // declares Eigen2DXZCosOp struct from above macro
Eigen2DCosOp(YZ); // declares Eigen2DYZCosOp struct from above macro

struct PressureOp
{
  PressureOp() {}
  inline double operator()(Matrix3 M)
  { return -M.Trace()/3.; }
};

struct EquivalentStressOp
{
  EquivalentStressOp() {}
  inline double operator()(Matrix3 M)
  { 
    // Calculate deviatoric part
    Matrix3 one; one.Identity();
    Matrix3 Mdev = M - one*(M.Trace()/3.0);
    return sqrt(Mdev.NormSquared()*1.5); 
  }
};

struct OctShearStressOp
{
  OctShearStressOp() {}
  inline double operator()(Matrix3 M)
  { return sqrt( (M(0,0)-M(1,1))*(M(0,0)-M(1,1))+
		 (M(1,1)-M(2,2))*(M(1,1)-M(2,2))+
		 (M(2,2)-M(0,0))*(M(2,2)-M(0,0))+
		 6*(M(0,1)*M(0,1)+M(1,2)*M(1,2)+M(0,2)*M(0,2))
		 )/3.0; }
};

struct NDotSigmaDotTOp
{
  NDotSigmaDotTOp(double nx, double ny, double nz, 
                  double tx, double ty, double tz)
  { 
    n = Vector(nx,ny,nz);
    if(n.length() > 0.0) n/=n.length();
    t = Vector(tx,ty,tz);
    if(t.length() > 0.0) t/=t.length();
    //    std::cout << " n = " << n << " t = " << t << std::endl;
  }

  // This is the function which does the operation you want.
  inline double operator()(Matrix3 M)
  { 
    double nsigmat = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        nsigmat += n[i]*M(i,j)*t[j];
      }
    }
    return nsigmat;
  }

  Vector n;
  Vector t;
};

/*
  struct EigenOp
  {
  EigenOp(int chosen)
  : m_chosen(chosen) { }
  inline pair<double, Vector> operator()(Matrix3 M);
  int m_chosen;
  };

  inline pair<double, Vector> EigenOp::operator()(Matrix3 M)
  {
  double e[3];
  int num_eigen_values = M.getEigenValues(e[0], e[1], e[2]);
  if (num_eigen_values <= m_chosen)
  return pair<double, Vector>(0, Vector(0, 0, 0));
  else {
  std::vector<Vector> eigenVectors;
  double eigenValue = e[m_chosen];
  eigenVectors = M.getEigenVectors(eigenValue, e[0]);
  if (eigenVectors.size() != 1)
  return pair<double, Vector>(eigenValue, Vector(0, 0, 0));
  else
  return pair<double, Vector>(eigenValue, eigenVectors[0].normal());
  }
  }
*/

struct GreenLagrangeStrainTensorOp
{
  GreenLagrangeStrainTensorOp() {}
  inline Matrix3 operator()(const Matrix3& F)
  { 
    Matrix3 one; one.Identity();
    Matrix3 E = (F.Transpose()*F - one)*0.5;
    return E;
  }
};

struct CauchyGreenDeformationTensorOp
{
  CauchyGreenDeformationTensorOp() {}
  inline Matrix3 operator()(const Matrix3& F)
  { 
    Matrix3 C = F.Transpose()*F;
    return C;
  }
};

struct FingerDeformationTensorOp
{
  FingerDeformationTensorOp() {}
  inline Matrix3 operator()(const Matrix3& F)
  { 
    Matrix3 B = F*F.Transpose();
    return B;
  }
};

struct NullTensorOp
{
  NullTensorOp() {}
  inline Matrix3 operator()(const Matrix3& F)
  { 
    Matrix3 T = F;
    return T;
  }
};

}

