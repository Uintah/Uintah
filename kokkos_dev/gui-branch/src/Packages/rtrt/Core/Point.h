
#ifndef POINT_H
#define POINT_H 1

//class ostream;
#include <iostream>

namespace rtrt {
  
using namespace std;


class Vector;

//using namespace std;
//#include <ostream>
//#ifndef ostream
//typedef basic_ostream ostream;
//#endif

class Point {
  double d[3];
  friend class Vector;
public:
  inline Point(double x, double y, double z);
  inline Point();
  inline Vector operator-(const Point& p) const;
  inline Point operator+(const Vector& v) const;
  inline Point operator+(const Point&  p) const;
  inline Point operator-(const Vector& v) const;
  inline double x() const;
  inline double y() const;
  inline double z() const;
  friend ostream& operator<<(ostream& out, const Point& p);
  friend inline Point Max(const Point& p1, const Point& p2);
  friend inline Point Min(const Point& p1, const Point& p2);
  inline Point& operator+=(const Vector& v);
  inline bool operator != (const Point& p) const;
  inline double* ptr() const {return (double*)&d[0];}
  inline void blend(const Point &p1, const Point &p2,
		    const double t);
  inline double dot(const Vector &p) const;
  inline void addscaled(const Point &p, const double w);
  inline Vector asVector() const;
  inline Point operator/(const double c) const;
    inline void x(double xx) {
	d[0]=xx;
    }
    inline void y(double yy) {
	d[1]=yy;
    }
    inline void z(double zz) {
	d[2]=zz;
    }
};

inline Point::Point(double x, double y, double z) {
    d[0]=x; d[1]=y; d[2]=z;
}

inline Point::Point() {
}
} // end namespace rtrt

#include "Vector.h"

namespace rtrt {
  
inline Vector Point::operator-(const Point& p) const {
    return Vector(d[0]-p.d[0], d[1]-p.d[1], d[2]-p.d[2]);
}

inline Point Point::operator+(const Vector& v) const {
    return Point(d[0]+v.d[0], d[1]+v.d[1], d[2]+v.d[2]);
}

inline Point Point::operator+(const Point& p) const {
    return Point(d[0]+p.d[0], d[1]+p.d[1], d[2]+p.d[2]);
}

inline Point Point::operator-(const Vector& v) const {
    return Point(d[0]-v.d[0], d[1]-v.d[1], d[2]-v.d[2]);
}


inline Point Point::operator/(const double c) const {
  return Point(d[0]/c,d[1]/c,d[2]/c);
}

inline double Point::x() const {
    return d[0];
}

inline double Point::y() const {
    return d[1];
}

inline double Point::z() const {
    return d[2];
}

inline Point& Point::operator+=(const Vector& v) {
    d[0]+=v.d[0];
    d[1]+=v.d[1];
    d[2]+=v.d[2];
    return *this;
}
} // end namespace rtrt

#include "MinMax.h"

namespace rtrt {

inline Point Max(const Point& p1, const Point& p2)
{
    return Point(Max(p1.x(), p2.x()),
		 Max(p1.y(), p2.y()),
		 Max(p1.z(), p2.z()));
}

inline Point Min(const Point& p1, const Point& p2)
{
    return Point(Min(p1.x(), p2.x()),
		 Min(p1.y(), p2.y()),
		 Min(p1.z(), p2.z()));
}

inline bool Point::operator != (const Point& v) const{
    return d[0] != v.d[0] || d[1] != v.d[1] || d[2] != v.d[2];
}

inline Vector Point::asVector() const {
    return Vector(d[0], d[1], d[2]);
}

inline double Point::dot(const Vector &v) const {
  return d[0]*v.d[0]+d[1]*v.d[1]+d[2]*v.d[2];
}

inline void Point::blend(const Point &p1, const Point &p2,
			 const double t)
{
    double w1,w2;
    
    w1 = (1.-t);
    w2 = t;

    d[0] = (w1*p1.x() + w2*p2.x());
    d[1] = (w1*p1.y() + w2*p2.y());
    d[2] = (w1*p1.z() + w2*p2.z());
}

inline void Point::addscaled(const Point &p, const double scale) {
  d[0] += p.d[0]*scale;
  d[1] += p.d[1]*scale;
  d[2] += p.d[2]*scale;
}


} // end namespace rtrt

#endif
