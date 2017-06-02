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
 *  Tensor.cc: Symetric, positive definite tensors (diffusion, conductivity)
 *
 *  Written by:
 *   Author: David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2001
 *
 */


#include <Core/Geometry/Tensor.h>
#include <Core/Containers/Array1.h>
#include <Core/Util/Assert.h>
#include <Core/Math/MiscMath.h>
#include <iostream>

using namespace std;

#include <cstdio>
#include <Core/Exceptions/InternalError.h>


namespace Uintah {

Tensor::Tensor() : have_eigens_(0)
{
  mat_[0][0] = 0.0;
  mat_[0][1] = 0.0;
  mat_[0][2] = 0.0;
  mat_[1][0] = 0.0;
  mat_[1][1] = 0.0;
  mat_[1][2] = 0.0;
  mat_[2][0] = 0.0;
  mat_[2][1] = 0.0;
  mat_[2][2] = 0.0;
}

Tensor::Tensor(const Tensor& copy)
{
  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
      mat_[i][j]=copy.mat_[i][j];
  have_eigens_=copy.have_eigens_;
  if (have_eigens_) {
    e1_=copy.e1_; e2_=copy.e2_; e3_=copy.e3_;
    l1_=copy.l1_; l2_=copy.l2_; l3_=copy.l3_;
  }
}

Tensor::Tensor(const Array1<double> &t) {
  mat_[0][0]=t[0];
  mat_[0][1]=mat_[1][0]=t[1];
  mat_[0][2]=mat_[2][0]=t[2];
  mat_[1][1]=t[3];
  mat_[1][2]=mat_[2][1]=t[4];
  mat_[2][2]=t[5];

  have_eigens_=0;
}

Tensor::Tensor(const vector<double> &t)
{
  ASSERT(t.size() > 5);

  mat_[0][0]=t[0];
  mat_[0][1]=mat_[1][0]=t[1];
  mat_[0][2]=mat_[2][0]=t[2];
  mat_[1][1]=t[3];
  mat_[1][2]=mat_[2][1]=t[4];
  mat_[2][2]=t[5];

  have_eigens_=0;
}

Tensor::Tensor(const double *t)
{
  mat_[0][0]=t[0];
  mat_[0][1]=mat_[1][0]=t[1];
  mat_[0][2]=mat_[2][0]=t[2];
  mat_[1][1]=t[3];
  mat_[1][2]=mat_[2][1]=t[4];
  mat_[2][2]=t[5];

  have_eigens_=0;
}

//! Initialize the diagonal to this value
Tensor::Tensor(double v) {
  have_eigens_=0;
  for (int i=0; i<3; i++) 
    for (int j=0; j<3; j++)
      if (i==j) mat_[i][j]=v;
      else mat_[i][j]=0;
}

//! Initialize the diagonal to this value
Tensor::Tensor(int v) {
  have_eigens_=0;
  for (int i=0; i<3; i++) 
    for (int j=0; j<3; j++)
      if (i==j) mat_[i][j]=v;
      else mat_[i][j]=0;
}

Tensor::Tensor(const Vector &e1, const Vector &e2, const Vector &e3) :
  e1_(e1), e2_(e2), e3_(e3), 
  l1_(e1.length()), l2_(e2.length()), l3_(e3.length())
{
  build_mat_from_eigens();
  have_eigens_=1;
}

Tensor::Tensor(const double **cmat) {
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      mat_[i][j]=cmat[i][j];
  have_eigens_=0;
}

void Tensor::build_mat_from_eigens() {
  if (!have_eigens_) return;
  double E[3][3];
  double S[3][3];
  double SE[3][3];
  Vector e1n(e1_);
  Vector e2n(e2_);
  Vector e3n(e3_);
  if (l1_ != 0) e1n.normalize();
  if (l2_ != 0) e2n.normalize();
  if (l3_ != 0) e3n.normalize();
  
  E[0][0] = e1n.x(); E[0][1] = e1n.y(); E[0][2] = e1n.z();
  E[1][0] = e2n.x(); E[1][1] = e2n.y(); E[1][2] = e2n.z();
  E[2][0] = e3n.x(); E[2][1] = e3n.y(); E[2][2] = e3n.z();
  S[0][0] = l1_; S[0][1] = 0;   S[0][2] = 0;
  S[1][0] = 0;   S[1][1] = l2_; S[1][2] = 0;
  S[2][0] = 0;   S[2][1] = 0;   S[2][2] = l3_;
  int i,j,k;
  for (i=0; i<3; i++)
    for (j=0; j<3; j++) {
      SE[i][j]=0;
      for (k=0; k<3; k++)
	SE[i][j] += S[i][k] * E[j][k];  // S x E-transpose
    }
  for (i=0; i<3; i++)
    for (j=0; j<3; j++) {
      mat_[i][j]=0;
      for (k=0; k<3; k++)
	mat_[i][j] += E[i][k] * SE[k][j];
    }
}

int Tensor::operator==(const Tensor& t) const
{
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      if( mat_[i][j]!=t.mat_[i][j])
	return false;

  return true;
}

int Tensor::operator!=(const Tensor& t) const
{
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      if( mat_[i][j]!=t.mat_[i][j])
	return true;

  return false;
}

Tensor& Tensor::operator=(const Tensor& copy)
{
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      mat_[i][j]=copy.mat_[i][j];
  have_eigens_=copy.have_eigens_;
  if (have_eigens_) {
    e1_=copy.e1_; e2_=copy.e2_; e3_=copy.e3_;
    l1_=copy.l1_; l2_=copy.l2_; l3_=copy.l3_;
  }
  return *this;
}

Tensor& Tensor::operator=(const double& d)
{
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      mat_[i][j]=d;
  have_eigens_=false;
  return *this;
}

/* matrix max norm */
double Tensor::norm()
{
  double a = 0.0;
  double sum;
  for (int i=0;i<3;i++)
  {
    sum = 0.0;
    for (int j=0;j<3;j++) sum += Abs(mat_[i][j]);
    if (sum > a) a = sum;
  }
  return (a);
}


Tensor::~Tensor()
{
}

string Tensor::type_name(int) {
  static const string str("Tensor");
  return str;
}

Tensor Tensor::operator-(const Tensor& t) const
{
  Tensor t1(*this);
  t1.have_eigens_=0;
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      t1.mat_[i][j]-=t.mat_[i][j];
  return t1;
}

Tensor& Tensor::operator-=(const Tensor& t)
{
  have_eigens_=0;
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      mat_[i][j]-=t.mat_[i][j];
  return *this;
}

Tensor Tensor::operator+(const Tensor& t) const
{
  Tensor t1(*this);
  t1.have_eigens_=0;
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      t1.mat_[i][j]+=t.mat_[i][j];
  return t1;
}

Tensor& Tensor::operator+=(const Tensor& t)
{
  have_eigens_=0;
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      mat_[i][j]+=t.mat_[i][j];
  return *this;
}

Tensor Tensor::operator*(const double s) const
{
  Tensor t1(*this);
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      t1.mat_[i][j]*=s;
  if (t1.have_eigens_) {
    t1.e1_*=s; t1.e2_*=s; t1.e3_*=s;
    t1.l1_*=s; t1.l2_*=s; t1.l3_*=s;
  }
  return t1;
}

Vector Tensor::operator*(const Vector v) const
{
  return Vector(v.x()*mat_[0][0]+v.y()*mat_[0][1]+v.z()*mat_[0][2],
		v.x()*mat_[1][0]+v.y()*mat_[1][1]+v.z()*mat_[1][2],
		v.x()*mat_[2][0]+v.y()*mat_[2][1]+v.z()*mat_[2][2]);
}

void Tensor::build_eigens_from_mat()
{
  if (have_eigens_) return;
  float eval[3];
  float evec[9];
  throw InternalError("Trying to eigensolve without Teem", __FILE__, __LINE__);
  e1_ = Vector(evec[0], evec[1], evec[2]);
  e2_ = Vector(evec[3], evec[4], evec[5]);
  e3_ = Vector(evec[6], evec[7], evec[8]);
  l1_ = eval[0];
  l2_ = eval[1];
  l3_ = eval[2];
  have_eigens_ = true;
}

void Tensor::get_eigenvectors(Vector &e1, Vector &e2, Vector &e3)
{
  if (!have_eigens_) build_eigens_from_mat();
  e1=e1_; e2=e2_; e3=e3_;
}

void Tensor::get_eigenvalues(double &l1, double &l2, double &l3) 
{
  if (!have_eigens_) build_eigens_from_mat();
  l1=l1_; l2=l2_; l3=l3_;
}

void Tensor::set_eigens(const Vector &e1, const Vector &e2, const Vector &e3) {
  e1_ = e1; e2_ = e2; e3_ = e3;
  l1_ = e1.length(); l2_ = e2.length(); l3_ = e3.length();
  build_mat_from_eigens();
  have_eigens_ = 1;
}

void Tensor::set_outside_eigens(const Vector &e1, const Vector &e2,
				const Vector &e3,
				double v1, double v2, double v3)
{
  e1_ = e1; e2_ = e2; e3_ = e3;
  l1_ = v1; l2_ = v2; l3_ = v3;
  have_eigens_ = 1;
}





ostream& operator<<( ostream& os, const Tensor& t )
{
  os << "/" << t.mat_[0][0] << " " << t.mat_[0][1] << " " << t.mat_[0][2] << "\\\n"
     << "|         "               << t.mat_[1][1] << " " << t.mat_[1][2] << "|\n"
     << "\\         " << "         "               << " " << t.mat_[2][2] << "/";

  return os;
}

istream& operator>>( istream& is, Tensor& t)
{
  t = Tensor();
  is >> t.mat_[0][0] >> t.mat_[0][1] >> t.mat_[0][2]
     >> t.mat_[1][0] >> t.mat_[1][1] >> t.mat_[1][2]
     >> t.mat_[2][0] >> t.mat_[2][1] >> t.mat_[2][2];
     
  return is;
}


} // End namespace Uintah

