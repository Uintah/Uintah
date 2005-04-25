/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#ifndef __VECTOR_H__
#define __VECTOR_H__

#if defined __cplusplus

#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace std;

class Vector
{

public:
  
  Vector( void ) { }
  Vector( const Vector& a ) : _x( a._x ), _y( a._y ), _z( a._z ) { }
  Vector( float x, float y, float z ): _x( x ), _y( y ), _z( z ) { }
  void Normalize( void ) {
    float square = sqrt( _x*_x + _y*_y + _z*_z);
    _x = _x/square;
    _y = _y/square;
    _z = _z/square;
  }
  float Length( void ) const {
    return sqrt( _x*_x + _y*_y + _z*_z );
  }

  float x( void ) const { return _x; }
  float& x( void ) { return _x; }
  float y( void ) const { return _y; }
  float& y( void ) { return _y; }
  float z( void ) const { return _z; }
  float& z( void ) { return _z; }
  
  friend ostream& operator<< ( ostream& os, Vector& v ) {
    os << v._x << ", " << v._y << ", " << v._z;  
    return os;
  }

  friend Vector operator- ( const Vector & a, const Vector & b )
  {
    return Vector ( a._x - b._x, a._y - b._y, a._z - b._z);
  }
  
  friend Vector operator+ ( const Vector & a, const Vector & b )
  {
    return Vector ( a._x + b._x, a._y + b._y, a._z + b._z );
  }

  friend float operator* ( const Vector & a, const Vector & b )
  {
    return( a._x * b._x + a._y * b._y + a._z + b._z );
  }
  
  friend Vector operator* ( float a, const Vector & b )
  {
    return Vector ( a*b._x, a*b._y, a*b._z );
  }
  
  friend Vector operator* ( double a, const Vector & b )
  {
    return Vector( a*b._x, a*b._y, a*b._z );
  }
  
private:
  
  float _x, _y, _z;
};


#endif
#endif
