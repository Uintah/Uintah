/*
 * \file Numeric3Vec.h
 *
 *  \date Aug 16, 2013
 *      \author John Hutchins
 * 
 * Copyright (c) 2014 The University of Utah
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

#ifndef Numeric3Vec_H_
#define Numeric3Vec_H_

#include <ostream>
#include <vector>
#include <cassert>
#include <iosfwd>


namespace SpatialOps{

/**
 * \class Numeric3Vec
 * \author John Hutchins
 * \brief provides a template lightweight class for storing 3d vectors of numbers.
 */
template<typename T>
class Numeric3Vec
{
  T ijk[3];

 public:

  Numeric3Vec(){ ijk[0]=0; ijk[1]=0; ijk[2]=0; }

   inline Numeric3Vec( const T i, const T j, const T k ){
     ijk[0]=i; ijk[1]=j; ijk[2]=k;
   }

   inline Numeric3Vec( const T vec[3] ){
     ijk[0]=vec[0]; ijk[1]=vec[1]; ijk[2]=vec[2];
   }

   inline Numeric3Vec( const std::vector<T>& vec ){
     assert(vec.size() == 3);
     ijk[0]=vec[0]; ijk[1]=vec[1]; ijk[2]=vec[2];
   }

   inline Numeric3Vec( const Numeric3Vec& x ){
     ijk[0]=x.ijk[0];  ijk[1]=x.ijk[1];  ijk[2]=x.ijk[2];
   }

   template<typename T1>
   inline Numeric3Vec(const Numeric3Vec<T1>& x){
     ijk[0]=(T)(x[0]);  ijk[1]=(T)(x[1]);  ijk[2]=(T)(x[2]);
   }

   inline T  operator[](const size_t i) const{
#    ifndef NDEBUG
     assert(i<3);
#    endif
     return ijk[i];
   }
   inline T& operator[](const size_t i){
#    ifndef NDEBUG
     assert(i<3);
#    endif
     return ijk[i];
   }

   template<typename T2>
   Numeric3Vec& operator=(const Numeric3Vec<T2>& x){
     ijk[0] = x[0];
     ijk[1] = x[1];
     ijk[2] = x[2];
     return *this;
   }

   inline bool operator==(const Numeric3Vec& v) const{
     return (ijk[0]==v[0]) & (ijk[1]==v[1]) & (ijk[2]==v[2]);
   }
   inline bool operator!=(const Numeric3Vec& v) const{
     return (ijk[0]!=v[0]) | (ijk[1]!=v[1]) | (ijk[2]!=v[2]);
   }

   template<typename T2>
   inline bool operator<(const Numeric3Vec<T2>& v) const{
     return (ijk[0]<v[0]) & (ijk[1]<v[1]) & (ijk[2]<v[2]);
   }
   template<typename T2>
   inline bool operator<=(const Numeric3Vec<T2>& v) const{
     return (ijk[0]<=v[0]) & (ijk[1]<=v[1]) & (ijk[2]<=v[2]);
   }
   template<typename T2>
   inline bool operator>(const Numeric3Vec<T2>& v) const{
     return (ijk[0]>v[0]) & (ijk[1]>v[1]) & (ijk[2]>v[2]);
   }
   template<typename T2>
   inline bool operator>=(const Numeric3Vec<T2>& v) const{
     return (ijk[0]>=v[0]) & (ijk[1]>=v[1]) & (ijk[2]>=v[2]);
   }


   template<typename T2>
   inline Numeric3Vec operator+( const Numeric3Vec<T2>& v ) const{
     return Numeric3Vec( ijk[0] + v[0],
                         ijk[1] + v[1],
                         ijk[2] + v[2] );
   }
   template<typename T2>
   inline Numeric3Vec operator-( const Numeric3Vec<T2>& v ) const{
     return Numeric3Vec( ijk[0] - v[0],
                         ijk[1] - v[1],
                         ijk[2] - v[2] );
   }
   template<typename T2>
   inline Numeric3Vec operator*( const Numeric3Vec<T2>& v ) const{
     return Numeric3Vec( ijk[0] * v[0],
                         ijk[1] * v[1],
                         ijk[2] * v[2] );
   }
   template<typename T2>
   inline Numeric3Vec operator/( const Numeric3Vec<T2>& v ) const{
     return Numeric3Vec( ijk[0] / v[0],
                         ijk[1] / v[1],
                         ijk[2] / v[2] );
   }
   inline Numeric3Vec operator-() const{
     return Numeric3Vec( - ijk[0],
                         - ijk[1],
                         - ijk[2] );
   }

   template<typename T1>
   inline Numeric3Vec operator+( const T1 v ) const{
     return Numeric3Vec( ijk[0] + v,
                         ijk[1] + v,
                         ijk[2] + v );
   }
   template<typename T1>
   inline Numeric3Vec operator*( const T1 v ) const{
     return Numeric3Vec( ijk[0] * v,
                         ijk[1] * v,
                         ijk[2] * v );
   }
   template<typename T1>
   inline Numeric3Vec operator/( const T1 v) const{
     return Numeric3Vec(ijk[0]/v, ijk[1]/v, ijk[2]/v);
   }

   template<typename T2>
   inline Numeric3Vec& operator+=( const Numeric3Vec<T2>& v ){
     ijk[0] += v[0];
     ijk[1] += v[1];
     ijk[2] += v[2];
     return *this;
   }
   template<typename T2>
   inline Numeric3Vec& operator-=( const Numeric3Vec<T2>& v ){
     ijk[0] -= v[0];
     ijk[1] -= v[1];
     ijk[2] -= v[2];
     return *this;
   }

   inline T sum() const{
     return ijk[0]+ijk[1]+ijk[2];
   }

 };

template<typename T>
 inline std::ostream& operator<<( std::ostream& os, const Numeric3Vec<T>& v ){
   os << "[ " << v[0] << ","  << v[1] << ","  << v[2] << " ]";
   return os;
 }

} // namespace SpatialOps

#endif /* Numeric3Vec_H_ */
