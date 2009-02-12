/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef POINT2D_H
#define POINT2D_H

#include <cmath>
#include <cstdio>

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
