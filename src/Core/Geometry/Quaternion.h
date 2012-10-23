/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

/*----------------------------------------------------------------------
CLASS
    Quaternion

    Unit quaternion representation

GENERAL INFORMATION
    
    Alexei Samsonov
    Department of Computer Science
    University of Utah
    July 2000
    

KEYWORDS
    None

DESCRIPTION

   Quaternion is a "brother" of rotational matrix. This representation provides
   almost complete set of operations for quaternion manipulations and rotations. 
   Default constructor creates quaternion corresponding to zero rotation of frame,
   i.e. (1, 0, 0, 0).

PATTERNS
    None

WARNING   
    exp(q) function for unit quaternion is defined in order to implement pow()
    function (q^n=exp(n*log(q)) and only for q=(0, theta*v), where ||v||=1. 
    Then exp(q)=(cos(theta), sin(theta)*v).

POSSIBLE REVISIONS
    implement persistent representation
    add tests
----------------------------------------------------------------------*/

#ifndef Geometry_Quaternion_h
#define Geometry_Quaternion_h 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Point.h>
#include <cmath>

#include <Core/Geometry/share.h>

namespace SCIRun {

#define NUM_ZERO 10e-9

class Point;
class Transform;

class Quaternion {
  double a;
  Vector v;
  inline Quaternion log();
  inline Quaternion exp();          // exp-function should be used only after log()
				    // doesn't make sense otherwise in most cases	 
  void from_matrix(const double matr[4][4]);
  friend Quaternion Slerp(const Quaternion&, const Quaternion&, double);
  friend Quaternion Pow(const Quaternion&, double);
public:

  inline Quaternion(): a(1), v(Vector(0, 0, 0)){};
  inline Quaternion(double angle, const Vector& vect): a(angle), v(vect){
    this->normalize();
  };
  SCISHARE Quaternion(Vector, Vector);
  explicit Quaternion(const Transform&);
  explicit Quaternion(const Vector&);

  inline void normalize();
  inline double norm() const;
  
  void to_matrix(Transform&);
  inline void from_matrix(const Transform&);
  
  //rotate the vector using current quaternion V'=qVinv(q)
  inline Vector rotate(const Vector&) const;
  
  bool operator==(const Quaternion&);
  inline Quaternion& operator+=(const Quaternion&);
  inline Quaternion& operator*=(const Quaternion&);
  
  inline Quaternion get_inv() const;
  inline void inverse();

  // to retrieve lookdirection frame axis
  void get_frame(Vector&, Vector&, Vector&);
  
  friend inline Quaternion operator+(const Quaternion&, const Quaternion&);
  friend inline Quaternion operator*(const Quaternion&, const Quaternion&);   

  friend inline double Dot(const Quaternion&, const Quaternion&);
  friend std::ostream& operator<<(std::ostream&, const Quaternion&);
  friend void Pio(Piostream&, Quaternion&);
};

inline void Quaternion::normalize(){
  double n=norm();
  if (n > NUM_ZERO){
    a/=n;	
    v*=1/n;
  }
  else {
    a=1;
    v=Vector(0, 0, 0);
  }
}

inline double Quaternion::norm() const{
  double tmp=a*a+v.length()*v.length();
  if (tmp>NUM_ZERO)
    return sqrt(tmp);
  else 
    return 0;
}

inline void Quaternion::from_matrix(const Transform& m){
  const double (&matr)[4][4]=m.mat;
  from_matrix(matr);
}

inline Quaternion& Quaternion::operator+=(const Quaternion& q){
  a+=q.a;
  v+=q.v;
  this->normalize();
  return *this;
}

inline Quaternion& Quaternion::operator*=(const Quaternion& q){
  a=(a * q.a) - Dot(v, q.v);
  v=Cross(v, q.v) + (v * q.a) + (q.v * a);
  this->normalize();
  return *this;
}

inline Quaternion operator*(const Quaternion& fq, const Quaternion& sq){
  return Quaternion((fq.a * sq.a) - Dot(fq.v, sq.v), Cross(fq.v, sq.v) + (fq.v * sq.a) + (sq.v * fq.a));
}

//**********************************************************************************
// exp(q) is defined only in case q=(0, theta*v), where ||v||=1
// and exp(q)=(cos(theta), sin(theta)*v)

inline Quaternion Quaternion::exp(){
  double arg=v.length();
  if (arg>NUM_ZERO)
      return Quaternion(cos(arg), v.normal()*sin(arg));
  else
      return Quaternion();
}

inline Quaternion Quaternion::log(){
  double arg=v.length();
  if (arg>NUM_ZERO)
     return Quaternion(0, v.normal()*acos(a));
  else
     return Quaternion();
}

//**********************************************************************************

inline Vector Quaternion::rotate(const Vector& V) const{
  double value=V.length();
  Quaternion Res(0, V);
  return  (((*this)*Res*this->get_inv()).v).normal()*value;
}

inline Quaternion Quaternion::get_inv() const{
  Quaternion tmp(a, -v);
  tmp.normalize();
  return tmp;
}

inline void Quaternion::inverse(){
  v=-v;
}

inline double Dot(const Quaternion& lq, const Quaternion& rq){
  return lq.a*rq.a+Dot(lq.v, rq.v);
}

inline Quaternion operator+(const Quaternion& first, const Quaternion& second){
  return Quaternion(first.a+second.a, first.v+second.v);
}

} // End namespace SCIRun

#endif  //Geometry_Quaternion_h







