/*
 *  Tensor.h:  Symetric, positive definite tensors (diffusion, conductivity)
 *
 *  Written by:
 *   Author: David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef Geometry_Tensor_h
#define Geometry_Tensor_h 1

#include <Core/share/share.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Vector.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {

class SCICORESHARE Tensor {
private:
  Vector e1_, e2_, e3_;
  double l1_, l2_, l3_;
  int valid_eigens_;
public:
  Tensor();
  Tensor(const Tensor&);
  Tensor(int);
  Tensor(const Array1<double> &);
  Tensor(const double *);
  Tensor(const double **);
  Tensor(const Vector&, const Vector&, const Vector&);
  Tensor& operator=(const Tensor&);
  ~Tensor();
  
  Tensor operator+(const Tensor&) const;
  Tensor& operator+=(const Tensor&);
  Tensor operator*(const double) const;

  static string type_name();
  
  double mat_[3][3];
  void build_eigens();
  void get_eigenvectors(Vector &e1, Vector &e2, Vector &e3);
  void get_eigenvalues(double &l1, double &l2, double &l3);
  double aniso_index();
  
  friend void SCICORESHARE Pio(Piostream&, Tensor&);
};

} // End namespace SCIRun

#endif // Geometry_Tensor_h
