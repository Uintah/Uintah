
#ifndef VECTOR_H
#define VECTOR_H 1

//class ostream;
//class istream;
#include <iostream>
#include <math.h>

namespace rtrt {
  using namespace std;

class Point;
class Point4D;

class Vector {
    double d[3];
    friend class Point;
  friend class Point4D;
public:
    inline Vector(double x, double y, double z);
    inline Vector(const Vector& v);
    inline Vector();
    inline double length() const;
    inline double length2() const;
    inline Vector& operator=(const Vector& v);
    inline Vector& operator=(const Point& p);
    int operator==(const Vector& v) const;
    inline Vector operator*(double s) const;
    inline Vector operator*(const Vector& v) const;
    inline Vector operator/(const Vector& v) const;
    inline Vector& operator*=(double s);
    Vector operator/(double s) const;
    inline Vector operator+(const Vector& v) const;
    inline Vector& operator+=(const Vector& v);
    inline Vector operator-() const;
    inline Vector operator-(const Vector& v) const;
    Vector& operator-=(const Vector& v);
    inline double normalize();
    Vector normal() const;
    inline Vector cross(const Vector& v) const;
    inline double dot(const Vector& v) const;
    inline double dot(const Point& p) const;
    inline void x(double xx) {
	d[0]=xx;
    }
    inline double x() const;
    inline void y(double yy) {
	d[1]=yy;
    }
    inline double y() const;
    inline void z(double zz) {
	d[2]=zz;
    }
    inline double z() const;
    inline double minComponent() const;
    Point asPoint() const;
    friend ostream& operator<<(ostream& os, const Vector& p);
    friend istream& operator>>(istream& os, Vector& p);
    inline bool operator != (const Vector& v) const;
    inline double* ptr() const {return (double*)&d[0];}

    void make_ortho(Vector&, Vector&) const;
  inline void weighted_diff(Point &p1, double w1, Point &p2, double w2);
    
};

inline Vector::Vector(double x, double y, double z) {
    d[0]=x;
    d[1]=y;
    d[2]=z;
}

inline Vector::Vector(const Vector& v) {
    d[0]=v.d[0];
    d[1]=v.d[1];
    d[2]=v.d[2];
}

inline Vector::Vector() {
}
} // end namespace rtrt

#include "Point.h"

namespace rtrt {

inline double Vector::length() const {
    return sqrt(length2());
}

inline double Vector::length2() const {
    return d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
}

inline Vector& Vector::operator=(const Vector& v) {
    d[0]=v.d[0];
    d[1]=v.d[1];
    d[2]=v.d[2];
    return *this;
}

inline Vector& Vector::operator=(const Point& p) {
  d[0] = p.d[0];
  d[1] = p.d[1];
  d[2] = p.d[2];
  return *this;
}

inline Vector Vector::operator*(double s) const {
    return Vector(d[0]*s, d[1]*s, d[2]*s);
}

inline Vector operator*(double s, const Vector& v) {
    return v*s;
}

inline Vector Vector::operator*(const Vector& v) const {
    return Vector(d[0]*v.d[0], d[1]*v.d[1], d[2]*v.d[2]);
}

inline Vector Vector::operator/(const Vector& v) const {
    return Vector(d[0]/v.d[0], d[1]/v.d[1], d[2]/v.d[2]);
}

inline Vector Vector::operator+(const Vector& v) const {
    return Vector(d[0]+v.d[0], d[1]+v.d[1], d[2]+v.d[2]);
}

inline Vector& Vector::operator+=(const Vector& v) {
    d[0]+=v.d[0];
    d[1]+=v.d[1];
    d[2]+=v.d[2];
    return *this;
}

inline Vector& Vector::operator*=(double s) {
    d[0]*=s;
    d[1]*=s;
    d[2]*=s;
    return *this;
}

inline Vector Vector::operator-() const {
    return Vector(-d[0], -d[1], -d[2]);
}

inline Vector Vector::operator-(const Vector& v) const {
    return Vector(d[0]-v.d[0], d[1]-v.d[1], d[2]-v.d[2]);
}

inline double Vector::normalize() {
    double l=length();
    d[0]/=l;
    d[1]/=l;
    d[2]/=l;
    return l;
}

inline Vector Vector::cross(const Vector& v) const {
    return Vector(d[1]*v.d[2]-d[2]*v.d[1],
    	      d[2]*v.d[0]-d[0]*v.d[2],
    	      d[0]*v.d[1]-d[1]*v.d[0]);
}

inline double Vector::dot(const Vector& v) const {
    return d[0]*v.d[0]+d[1]*v.d[1]+d[2]*v.d[2];
}

inline double Vector::dot(const Point& p) const {
    return d[0]*p.d[0]+d[1]*p.d[1]+d[2]*p.d[2];
}

inline double Vector::x() const {
    return d[0];
}

inline double Vector::y() const {
    return d[1];
}

inline double Vector::z() const {
    return d[2];
}

inline double Vector::minComponent() const {
    return (d[0]<d[1] && d[0]<d[2])?d[0]:d[1]<d[2]?d[1]:d[2];
}

inline bool Vector::operator != (const Vector& v) const {
    return d[0] != v.d[0] || d[1] != v.d[1] || d[2] != v.d[2];
}

inline void Vector::weighted_diff(Point &p1, double w1, 
				  Point &p2, double w2)
{
  d[0] = p1.d[0]*w1 - p2.d[0]*w2;
  d[1] = p1.d[1]*w1 - p2.d[1]*w2;
  d[2] = p1.d[2]*w1 - p2.d[2]*w2;
}

} // end namespace rtrt

#endif
