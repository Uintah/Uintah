#ifndef POINT2D_H
#define POINT2D_H

#include <math.h>
#include <stdio.h>

namespace rtrt {

class Point2D 
{
public:
  double _x, _y;

  inline Point2D()
    {
      _x = 0;
      _y = 0;
    }
  
  Point2D(const double& x, const double& y) 
    {
      _x = x;
      _y = y;
    }

    inline void x(const double& c) {
        _x = c;
    }

    inline void y(const double& c) {
        _y = c;
    }

    inline double x() const {
        return _x;
    }

    inline double y() const {
        return _y;
    }

  inline double operator[](int i) 
    {
      return (i==0) ? _x : _y;
    }

  inline double normalize()
    {
      double l = sqrt(_x*_x+_y*_y);
      
      _x /= l;
      _y /= l;
      
      return l;
    }
  
  inline double length()
    {
      return sqrt(_x*_x+_y*_y);
    }
  
  inline Point2D operator+(const Point2D &p) const 
    {
      return Point2D(_x+p._x,_y+p._y);
    }
  
  inline Point2D operator-(const Point2D &p) const 
    {
      return Point2D(_x-p._x,_y-p._y);
    }
  
  inline Point2D operator*(const double c) const 
    {
      return Point2D(c*_x,c*_y);
    }

  inline Point2D &operator=(const Point2D &p)  {
    _x = p._x;
    _y = p._y;
    
    return *this;
  }
  void Print() const {
    printf("%lf %lf\n",_x,_y);
  }

  friend inline double Dot(const Point2D&, const Point2D&);

};
 
 inline double Dot(const Point2D& p1, const Point2D& p2)
   {
     return (p1._x*p2._x+p1._y*p2._y);
   }
 
} // end namespace rtrt


#endif
