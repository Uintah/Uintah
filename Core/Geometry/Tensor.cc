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
 *  Tensor.cc: Symetric, positive definite tensors (diffusion, conductivity)
 *
 *  Written by:
 *   Author: David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2001
 *
 *  Copyright (C) 200  SCI Group
 */

#include <Core/Geometry/Tensor.h>
#include <Core/Containers/Array1.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Math/MiscMath.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
using std::istream;
using std::ostream;
#include <stdio.h>

namespace SCIRun {

Tensor::Tensor() : have_eigens_(0)
{
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

// compute the tensor from 7 diffusion channels
Tensor::Tensor(const double * /* channels */) {

  // TODO: compute mat

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
	SE[i][j] += S[i][k] * E[k][j];
    }
  for (i=0; i<3; i++)
    for (j=0; j<3; j++) {
      mat_[i][j]=0;
      for (k=0; k<3; k++)
	mat_[i][j] += E[i][k] * SE[k][j];
    }
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

Tensor::~Tensor()
{
}

string Tensor::type_name(int) {
  static const string str("Tensor");
  return str;
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
  ASSERTFAIL("not yet implemented");
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

void SCICORESHARE Pio(Piostream& stream, Tensor& t){
  
  stream.begin_cheap_delim();
 
  Pio(stream, t.mat_[0][0]);
  Pio(stream, t.mat_[0][1]);
  Pio(stream, t.mat_[0][2]);
  Pio(stream, t.mat_[1][1]);
  Pio(stream, t.mat_[1][2]);
  Pio(stream, t.mat_[2][2]);

  t.mat_[1][0]=t.mat_[0][1];
  t.mat_[2][0]=t.mat_[0][2];
  t.mat_[2][1]=t.mat_[1][2];

  Pio(stream, t.have_eigens_);
  if (t.have_eigens_) {
    Pio(stream, t.e1_);
    Pio(stream, t.e2_);
    Pio(stream, t.e3_);
    Pio(stream, t.l1_);
    Pio(stream, t.l2_);
    Pio(stream, t.l3_);
  }

  stream.end_cheap_delim();
}

const string& 
Tensor::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

const TypeDescription* get_type_description(Tensor*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("Tensor", Tensor::get_h_file_path(), 
				"SCIRun");
  }
  return td;
}


ostream& operator<<( ostream& os, const Tensor& t )
{
  os << '[' << t.mat_[0][0] << ' ' << t.mat_[0][1] << ' ' << t.mat_[0][2]
     << ' ' << t.mat_[1][0] << ' ' << t.mat_[1][1] << ' ' << t.mat_[1][2]
     << ' ' << t.mat_[2][0] << ' ' << t.mat_[2][1] << ' ' << t.mat_[2][2]
     << ']';

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


} // End namespace SCIRun
