/********************************************************
 * TensorOperatorFunctors.h
 *
 * Author: Wayne Witzel
 * Scientific Computing and Imaging,
 * University of Utah
 * Copyright 2000
 */

#include <Uintah/Components/MPM/Util/Matrix3.h>

/* Functors used by TensorFieldOperator and TensorParticlesOperator */

struct TensorElementExtractionOp
{
  TensorElementExtractionOp(int row, int column)
    : m_row(row), m_column(column) { }

  inline double operator()(Matrix3 M)
  { return M(m_row, m_column); }

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

#define Eigen2DSinOp(plane) \
struct Eigen2D##plane##SinOp \
{ \
  Eigen2D##plane##SinOp(double delta) \
  : m_delta(delta) {} \
  inline double operator()(Matrix3 M) \
  { \
    double e1, e2; \
    int num_eigen_values = M.get##plane##EigenValues(e1, e2); \
    return (num_eigen_values == 2) ? sin((e1 - e2) / m_delta) : 0; \
  } \
  double m_delta; \
}
Eigen2DSinOp(XY); // declares Eigen2DXYSinOp struct from above macro
Eigen2DSinOp(XZ); // declares Eigen2DXZSinOp struct from above macro
Eigen2DSinOp(YZ); // declares Eigen2DYZSinOp struct from above macro

struct PressureOp
{
  PressureOp() {}
  inline double operator()(Matrix3 M)
  { return -(M(1,1) + M(2,2) + M(3,3)) / 3; }
};

struct EquivalentStressOp
{
  EquivalentStressOp() {}
  inline double operator()(Matrix3 M)
  { return M.NormSquared()/2; }
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



