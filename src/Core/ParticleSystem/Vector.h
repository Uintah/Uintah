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
