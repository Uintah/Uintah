#ifndef POINT4D_H
#define POINT4D_H

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <stdio.h>

#define POINT4D_EPS 1E-9
#define APPROX_ZERO(a) (fabs(a) < POINT4D_EPS ? true : false)
#define INV(w) (APPROX_ZERO(w) ? 1. : 1./w)

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Dot;
using SCIRun::Cross;

class Point4D
{

public:
  double _x,_y,_z,_w;

  inline Point4D()
    {
      zero();
    }
    
    inline Point4D(const double cx, const double cy, 
		   const double cz, const double ch)
    {
	_x = cx;
	_y = cy;
	_z = cz;
	_w = ch;
    }

    inline Point4D(const Point4D &p)
    {
	_x = p._x;
	_y = p._y;
	_z = p._z;
	_w = p._w;
    }
    
    inline void zero()
      {
	_x = _y = _z = _w = 0.;
      }

    inline void set(const double& x,
		    const double& y,
		    const double& z,
		    const double& w)
      {
	_x = x;
	_y = y;
	_z = z;
	_w = w;
      }

    inline double x() const {
        return _x;
    }
    
    inline double y() const {
        return _y;
    }
    
    inline double z() const {
        return _z;
    }

    inline double w() const {
        return _w;
    }
    
    inline Point e3() const {

/*        double invw = INV(_w); */

      double invw = 1./_w;
      
      return Point(invw*_x,
		   invw*_y,
		   invw*_z);
    }

    inline void addscaled(const Point4D &p, const double &c)
      {
	_x += p._x*c;
	_y += p._y*c;
	_z += p._z*c;
	_w += p._w*c;
      }

    inline Point4D &operator=(const Point4D &p)
    {
	_x = p._x;
	_y = p._y;
	_z = p._z;
	_w = p._w;

	return *this;
    }
};


inline void blend(const Point4D &p1, const Point4D &p2,
                  const double t, Point4D &pr)
{

  double t0 = 1.0-t;
  double t1 = t;
  pr.set(p1._x*t0 + p2._x*t1,
	 p1._y*t0 + p2._y*t1,
	 p1._z*t0 + p2._z*t1,
	 p1._w*t0 + p2._w*t1);
}


} // end namespace rtrt

#endif


