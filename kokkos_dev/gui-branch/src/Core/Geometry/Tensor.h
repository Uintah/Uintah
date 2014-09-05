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
  Tensor(double);
  Tensor(const Array1<double> &);
  Tensor(const double *);
  Tensor(const double **);
  Tensor(const Vector&, const Vector&, const Vector&);
  Tensor& operator=(const Tensor&);
  virtual ~Tensor();
  
  Tensor operator+(const Tensor&) const;
  Tensor& operator+=(const Tensor&);
  Tensor operator*(const double) const;

  static string type_name(int i = -1);
  
  double mat_[3][3];
  void build_eigens();
  void get_eigenvectors(Vector &e1, Vector &e2, Vector &e3);
  void get_eigenvalues(double &l1, double &l2, double &l3);
  double aniso_index();
  
  //! support dynamic compilation
  static const string& get_h_file_path();

  friend void SCICORESHARE Pio(Piostream&, Tensor&);
};

const TypeDescription* get_type_description(Tensor*);

SCICORESHARE std::ostream& operator<<(std::ostream& os, const Tensor& t);
SCICORESHARE std::istream& operator>>(std::istream& os, Tensor& t);

} // End namespace SCIRun

#endif // Geometry_Tensor_h
