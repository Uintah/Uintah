/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  Tensor.h:  Symetric, positive definite tensors (diffusion, conductivity)
 *
 *  Written by:
 *   Author: David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2001
 *
 */

#ifndef Geometry_Tensor_h
#define Geometry_Tensor_h 1

#include <Core/Geometry/Vector.h>

#include <iosfwd>
#include <vector>

namespace Uintah {

template<class T> class Array1;

class Tensor {
private:
  Vector e1_, e2_, e3_;  // these are already scaled by the eigenvalues
  double l1_, l2_, l3_;
  int have_eigens_;
public:

  Tensor();
  Tensor(const Tensor&);
  Tensor(int);
  Tensor(double);
  Tensor(const Array1<double> &); // 6 values
  Tensor(const std::vector<double> &); // 6 values
  Tensor(const double *); // 6 values
  Tensor(const double **);
  Tensor(const Vector&, const Vector&, const Vector&);
  Tensor& operator=(const Tensor&);
  Tensor& operator=(const double&);
  virtual ~Tensor();
  
  // checks if one tensor is exactly the same as another
  int operator==(const Tensor&) const;
  int operator!=(const Tensor&) const;

  Tensor operator+(const Tensor&) const;
  Tensor& operator+=(const Tensor&);
  Tensor operator-(const Tensor&) const;
  Tensor& operator-=(const Tensor&);
  Tensor operator*(const double) const;
  Vector operator*(const Vector) const;

  static std::string type_name(int i = -1);
  
  double mat_[3][3];
  void build_mat_from_eigens();
  void build_eigens_from_mat();

  double norm();

  bool have_eigens() { return have_eigens_; }
  void get_eigenvectors(Vector &e1, Vector &e2, Vector &e3);
  const Vector &get_eigenvector1() { ASSERT(have_eigens_); return e1_; }
  const Vector &get_eigenvector2() { ASSERT(have_eigens_); return e2_; }
  const Vector &get_eigenvector3() { ASSERT(have_eigens_); return e3_; }
  void get_eigenvalues(double &l1, double &l2, double &l3);
  void set_eigens(const Vector &e1, const Vector &e2, const Vector &e3);

  // This directly sets the eigenvectors and values in the tensor.  It
  // is meant to be used in conjunction with custom eigenvector/value
  // computation, such as that found in the package.
  void set_outside_eigens(const Vector &e1, const Vector &e2,
			  const Vector &e3,
			  double v1, double v2, double v3);

  //! support dynamic compilation
  static const std::string& get_h_file_path();

};

inline bool operator<(Tensor t1, Tensor t2)
{
  return(t1.norm()<t2.norm());
}

inline bool operator<=(Tensor t1, Tensor t2)
{
  return(t1.norm()<=t2.norm());
}

inline bool operator>(Tensor t1, Tensor t2)
{
  return(t1.norm()>t2.norm());
}

inline bool operator>=(Tensor t1, Tensor t2)
{
  return(t1.norm()>=t2.norm());
}

inline 
Tensor operator*(double d, const Tensor &t) {
  return t*d;
}
const TypeDescription* get_type_description(Tensor*);

std::ostream& operator<<(std::ostream& os, const Tensor& t);
std::istream& operator>>(std::istream& os, Tensor& t);

} // End namespace Uintah

#endif // Geometry_Tensor_h
