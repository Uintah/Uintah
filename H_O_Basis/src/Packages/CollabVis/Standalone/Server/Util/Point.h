/*
 *
 * Point: Utility for 3D integer points.
 *
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */


#ifndef __POINT_H_
#define __POINT_H_

#include <values.h>
#include <math.h>
#include <Util/stringUtil.h>
#include <Util/MinMax.h>
#include <iostream>

namespace SemotusVisum {

#ifndef PI
#define PI 3.14159265358979323846
#endif

inline double DtoR(const double d)
{
    return d*PI/180.;
}

inline double RtoD(const double r)
{
   return r*180./PI; 
}


/**
 * Enapsulates the concept of an integer 3d point
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Point {
public:
  
  /**
   * Constructor
   *
   * @param x   X coordinate  
   * @param y   Y coordinate
   * @param z   Z coordinate
   */
  Point( const int x, const int y, const int z=0 ) : x(x), y(y), z(z) {}
  
  /**
   * Destructor
   *
   */
  ~Point() {}
  
  /**
   * Converts the point to a string
   *
   * @return String rep of the point
   */
  inline string toString() const {
    return "( " + mkString(x) + ", " + mkString( y ) + ", " +
      mkString(z) + ")";
  }
  
  /// Spatial Coordinate
  int x, y, z;
};

class Vector3d;

/**
 * Enapsulates the concept of a floating-point 3d point
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Point3d {
public:
  /**
   *  Constructor. Sets values to defaults
   *
   */
  Point3d() : x(-MAXFLOAT), y(-MAXFLOAT), z(-MAXFLOAT) {
    _array[0] = x; _array[1] = y; _array[2] = z;
  }
  
  /**
   * Constructor - takes all values
   *
   * @param x   X coordinate  
   * @param y   Y coordinate
   * @param z   Z coordinate
   */
  Point3d( const double x, const double y, const double z ) :
    x(x), y(y), z(z) {
    _array[0] = x; _array[1] = y; _array[2] = z;
  }

  inline explicit Point3d( const Vector3d &v );
  
  /**
   *  Constructor - takes all values
   *
   * @param _x   X coordinate      
   * @param _y   Y coordinate   
   * @param _z   Z coordinate    
   * @param w    Homogenous 4D point. 
   */
  Point3d( const double _x, const double _y, const double _z,
	   const double w ) {
    if ( w == 0 ) {
      cerr << "Degenerate Point! " << endl;
      x = y = z = 0.0;
    }
    else {
      x = _x / w;
      y = _y / w;
      z = _z / w;
    }
    _array[0] = x; _array[1] = y; _array[2] = z;
  }
  
  /**
   * Copy constructor
   *
   * @param p     Point to copy
   */
  Point3d( const Point3d &p ) : x(p.x), y(p.y), z(p.z) {
    _array[0] = x; _array[1] = y; _array[2] = z;
  }
  
  /**
   * Destructor
   *
   */
  ~Point3d() {}
  
  /**
   * Returns true if the point is set
   *
   * @return True if point values are not defaults
   */
  inline bool set() const { return (x != -MAXFLOAT ||
				    y != -MAXFLOAT ||
				    z != -MAXFLOAT ); }
  
  /**
   * Equality operator
   *
   * @param p     Point to test
   * @return  True if the point is equal to this.
   */
  inline bool equals( const Point3d &p ) const {
    return ( p.x == x && p.y == y && p.z == z );
  }
  
  /**
   * Returns the point as an array
   *
   * @return Point array
   */
  inline float * array() { return _array; }

  
  /**
   * Subtracts the point from this to produce a vector
   *
   * @param p Point to subtract
   *
   * @return Vector between points.
   */
  inline Vector3d operator-(const Point3d& p) const;

  inline Vector3d operator+(const Point3d& p) const;
  inline Vector3d operator+(const Vector3d& p) const;
  
  inline int operator==(const Point3d&p) const {
    return p.x == x && p.y == y && p.z == z;
  }
  inline Point3d operator*(const double s) const {
    return Point3d(x*s, y*s, z*s);
  }
  /**
   * Returns a string rep of the point
   *
   * @return String rep of the point
   */
  inline string toString() const {
    return "( " + mkString(x) + ", " + mkString( y ) + ", " +
      mkString(z) + ")";
  }
  
  /// Coords
  double x, y, z;
  /// Array of coords
  float _array[3];
};

/**
 * Returns the maximum values of the two points
 *
 * @param p1    Point 1
 * @param p2    Point 2
 * @return      Max values of the two points
 */
Point3d Max(const Point3d& p1, const Point3d& p2);

/**
 * Returns the minimum values of the two points
 *
 * @param p1    Point 1    
 * @param p2    Point 2   
 * @return      Min values of the two points
 */
Point3d Min(const Point3d& p1, const Point3d& p2);

/**
 * Projects the point by the matrix
 *
 * @param mat  Matrix 
 * @param p    Point 
 * @return     Matrix * Point
 */
Point3d project(float mat[4][4], const Point3d& p);


/**
 * Enapsulates the concept of a floating-point 4d point
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Point4d {
public:
  /**
   *  Constructor. Sets values to defaults
   *
   */
  Point4d() : x(-MAXFLOAT), y(-MAXFLOAT), z(-MAXFLOAT), w(0) {
    _array[0] = x; _array[1] = y; _array[2] = z; _array[3] = w;
  }
  
  /**
   *  Constructor - takes all values
   *
   * @param x   X coordinate      
   * @param y   Y coordinate   
   * @param z   Z coordinate    
   * @param w    Homogenous 4D point. 
   */
  Point4d( double x, double y, double z, double w=0 ) : x(x), y(y), z(z), w(w){
    _array[0] = x; _array[1] = y; _array[2] = z; _array[3] = w;
  }
  
  /**
   * Copy constructor
   *
   * @param p     Point to copy
   */
  Point4d( const Point4d &p ) : x(p.x), y(p.y), z(p.z), w(p.w) {
    _array[0] = x; _array[1] = y; _array[2] = z; _array[3] = w;
  }
  
  /**
   * Destructor
   *
   */
  ~Point4d() {}

  /**
   * Returns true if the point is set
   *
   * @return True if point values are not defaults
   */
  inline bool set() const { return (x != -MAXFLOAT ||
				    y != -MAXFLOAT ||
				    z != -MAXFLOAT ); }
 
  /**
   * Equality operator
   *
   * @param p     Point to test
   * @return  True if the point is equal to this.
   */
  inline bool equals( const Point4d &p ) const {
    return ( p.x == x && p.y == y && p.z == z && p.w == w );
  }

  
  /**
   * Returns the point as an array
   *
   * @return Point array
   */
  inline float * array() { return _array; }

  /// Coordinates
  double x, y, z, w;

  /// Array rep of the point
  float _array[4];
};



/**
 * Enapsulates the concept of a 3D vector
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Vector3d {
public:
  /**
   *  Constructor. Sets values to defaults
   *
   */
  Vector3d() : x(-MAXFLOAT), y(-MAXFLOAT), z(-MAXFLOAT) {}
  
  
  /**
   * Constructor - takes all values
   *
   * @param x   X component
   * @param y   Y component
   * @param z   Z component
   */
  Vector3d( double x, double y, double z ) : x(x), y(y), z(z) {}
  
  /**
   * Copy constructor
   *
   * @param p     Vector to copy
   */
  Vector3d( const Vector3d &p ) : x(p.x), y(p.y), z(p.z) {}
  
  inline explicit Vector3d( const Point3d &p );
  
  /**
   * Destructor
   *
   */
  ~Vector3d() {}
    
  /**
   * Returns true if the vector is set
   *
   * @return True if vector values are not defaults
   */
  inline bool set() const { return (x != -MAXFLOAT ||
				    y != -MAXFLOAT ||
				    z != -MAXFLOAT ); }
    
  /**
   * Equality operator
   *
   * @param p     Vector to test
   * @return  True if the vector is equal to this.
   */
  inline bool equals( const Vector3d &p ) const {
    return ( p.x == x && p.y == y && p.z == z );
  }

  inline int operator==(const Vector3d&p) const {
    return p.x == x && p.y == y && p.z == z;
  }
  inline double length() const
  {
    return sqrt(x*x+y*y+z*z);
  }

  /**
   * Converts the vector to a string
   *
   * @return String rep of the vector
   */
  inline string toString() {
    return "( " + mkString(x) + ", " + mkString( y ) + ", " +
      mkString(z) + ")";
  }
  
  /**
   * Returns the cross product of the two vectors
   *
   * @param v1    Vector 1
   * @param v2    Vector 2
   * @return      v1 X v2
   */
  static inline Vector3d cross( const Vector3d &v1,
				const Vector3d &v2 ) {
    return Vector3d(
		    v1.y*v2.z-v1.z*v2.y,
		    v1.z*v2.x-v1.x*v2.z,
		    v1.x*v2.y-v1.y*v2.x);
  }
  
  /**
   *  Normalizes the vector to unit length
   *
   */
  inline double normalize() {
    double l2=x*x+y*y+z*z;
    double l=sqrt(l2);
    x/=l;
    y/=l;
    z/=l;
    return l;
  }

  inline Vector3d& operator*=(const double d) {
    x*=d;
    y*=d;
    z*=d;
    return *this;
  }
  inline Vector3d operator+(const Vector3d& p) const
  {
    return Vector3d(x+p.x, y+p.y, z+p.z);
  }
  inline Vector3d operator-(const Point3d& p) const
  {
    return Vector3d(x-p.x, y-p.y, z-p.z);
  }
  inline Vector3d operator*(const double s) const {
    return Vector3d(x*s, y*s, z*s);
  }

  /**
   * Negates the vector
   *
   */
  inline void negate() {
    x = -x;
    y = -y;
    z = -z;
  }
  
  /**
   * Returns the length squared of the vector
   *
   * @return Square of the length of the vector
   */
  inline double length2() {
    return x*x+y*y+z*z;
  }

  /// Vector component
  double x, y, z;
};

/**
 * Dot product of the point and vector
 *
 * @param p     Point
 * @param v     Vector
 * @return      P . V
 */
inline double Dot(const Point3d& p, const Vector3d& v)
{
    return p.x*v.x+p.y*v.y+p.z*v.z;
}

/**
 * Dot product of the vector and vector
 *
 * @param p     Vector
 * @param v     Vector
 * @return      P . V
 */
inline double Dot(const Vector3d& v1, const Vector3d& v)
{
    return v1.x*v.x+v1.y*v.y+v1.z*v.z;
}


/**
 * Dot product of the point and vector
 *
 * @param p     Point
 * @param v     Vector
 * @return      P . V
 */
inline double Dot(const Vector3d& v, const Point3d& p)
{
    return p.x*v.x+p.y*v.y+p.z*v.z;
}

inline Vector3d Point3d::operator-(const Point3d& p) const
{
    return Vector3d(x-p.x, y-p.y, z-p.z);
}
inline Vector3d Point3d::operator+(const Point3d& p) const
{
    return Vector3d(x+p.x, y+p.y, z+p.z);
}
inline Vector3d Point3d::operator+(const Vector3d& p) const
{
    return Vector3d(x+p.x, y+p.y, z+p.z);
}


inline Point3d::Point3d( const Vector3d &v ) : x(v.x), y(v.y), z(v.z) {
  _array[0] = x; _array[1] = y; _array[2] = z;
}

inline Vector3d::Vector3d( const Point3d &v ) : x(v.x), y(v.y), z(v.z) {}


}
#endif
