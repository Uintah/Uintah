#ifndef POINT4D_H
#define POINT4D_H

#include "Vector.h"
#include "stdio.h"

namespace rtrt {

class Point4D
{

public:
    
    inline Point4D(const double cx=0, const double cy=0, const double
		 cz=0, const double ch=1)
    {
	coord[0] = cx;
	coord[1] = cy;
	coord[2] = cz;
	coord[3] = ch;
    }

    inline Point4D(const Point4D &p)
    {
	coord[0] = p.coord[0];
	coord[1] = p.coord[1];
	coord[2] = p.coord[2];
	coord[3] = p.coord[3];
    }
    inline double x() const {
        return coord[0];
    }
    
    inline double y() const {
        return coord[1];
    }
    
    inline double z() const {
        return coord[2];
    }

    inline Vector operator-(const Point4D &p) const
    {
	return Vector(coord[0]-p.coord[0],
		      coord[1]-p.coord[1],
		      coord[2]-p.coord[2]
		      );
    }

    inline Vector operator-(const Vector &v) const
    {
	return Vector(coord[0]-v.x(),
		      coord[1]-v.y(),
		      coord[2]-v.z());
    }

    inline Point4D operator+(const Vector &v) const
    {
	return Point4D(coord[0]+v.x(),
		     coord[1]+v.y(),
		     coord[2]+v.z(),
		     1.);
    }

    inline Point4D operator+(const Point4D &p) const
    {
	return Point4D(coord[0]+p.coord[0],
		     coord[1]+p.coord[1],
		     coord[2]+p.coord[2],
		     1.);
    }

    inline Point4D &operator+=(const Point4D &p)
    {
	coord[0] += p.coord[0];
	coord[1] += p.coord[1];
	coord[2] += p.coord[2];
	coord[3] = 1.;
	
	return (*this);
    }

    inline Point4D &operator+=(const Vector &v)
    {
	coord[0] += v.x();
	coord[1] += v.y();
	coord[2] += v.z();
	coord[3] = 1.;

	return (*this);
    }
    
    inline Point4D &operator-=(const Point4D &p)
    {
	coord[0] -= p.coord[0];
	coord[1] -= p.coord[1];
	coord[2] -= p.coord[2];
	coord[3] = 1.;

	return *this;
    }

    inline Point4D ToBasis(const Point4D &c, const Vector &u,
			 const Vector &v, const Vector &w) const
    {
	Vector v1((*this)-c);

	return Point4D(v1.dot(u),v1.dot(v),v1.dot(w));
    }

    inline Point4D ToCanonical(const Point4D &c, const Vector &u,
			     const Vector &v, const Vector &w) const
    {
	return Point4D(coord[0]*u.x()+
                       coord[1]*v.x()+
                       coord[2]*w.x(),
                       coord[0]*u.y()+
                       coord[1]*v.y()+
                       coord[2]*w.y(),
                       coord[0]*u.z()+
                       coord[1]*v.z()+
                       coord[2]*w.z()) + c;
    }

    inline Point4D &operator=(const Vector &v)
    {
	coord[0] = v.x();
	coord[1] = v.y();
	coord[2] = v.z();
	coord[3] = 1.;
	
	return *this;
    }

    inline Point4D &operator=(const Point4D &p)
    {
	coord[0] = p.coord[0];
	coord[1] = p.coord[1];
	coord[2] = p.coord[2];
	coord[3] = p.coord[3];

	return *this;
    }

    inline Vector operator*(const Vector &v2) const
    {
	Vector v1 = Vector(coord[0],coord[1],coord[2]);
	return v1.cross(v2);
    }

    inline Point4D operator*(const double c) const
    {
	return Point4D(coord[0]*c,coord[1]*c,coord[2]*c,coord[3]);
    }

    inline Point4D operator/(double c) const
    {
	double div = 1./c;

	return Point4D(coord[0]*div,coord[1]*div,coord[2]*div);
    }
    
    inline double dot(const Vector &v) const
      {
	return
	    coord[0]*v.x() +
	    coord[1]*v.y() +
	    coord[2]*v.z();
    }

    inline double operator,(const Vector &v) const
    {
	return
	    coord[0]*v.x() +
	    coord[1]*v.y() +
	    coord[2]*v.z();
    }

  inline void addscaled(const Point4D &p, const double scale) {
    coord[0] += p.coord[0]*scale;
    coord[1] += p.coord[1]*scale;
    coord[2] += p.coord[2]*scale;
  }

  inline void weighted_set(const Point4D &p, const double w) {
    coord[0] = p.coord[0]*w;
    coord[1] = p.coord[1]*w;
    coord[2] = p.coord[2]*w;
  }

    void Print() {
	printf("%lf %lf %lf %lf\n",coord[0],coord[1],coord[2],coord[3]);
    }

    inline operator Vector() const
    {
	return (Vector(coord[0],coord[1],coord[2]));
    }
    
    double coord[4];
    
};

inline void blend(const Point4D &p1, const Point4D &p2,
		  const double t, Point4D &pr)
{
    double w1,w2;
    
    w1 = p1.coord[3]*(1.-t);
    w2 = p2.coord[3]*t;

    pr.coord[3] = w1+w2;

    w1 /= pr.coord[3];
    w2 = 1.-w1;

    pr.coord[0] = w1*p1.coord[0] + w2*p2.coord[0];
    pr.coord[1] = w1*p1.coord[1] + w2*p2.coord[1];
    pr.coord[2] = w1*p1.coord[2] + w2*p2.coord[2];
}

inline Vector &Sub(Point4D &p1, Point4D &p2, Vector &vout)
{
    vout.x(p1.coord[0]-p2.coord[0]);
    vout.y(p1.coord[1]-p2.coord[1]);
    vout.z(p1.coord[2]-p2.coord[2]);

    return vout;
}

inline Point4D &Sub(Point4D &p1, Point4D &p2, Point4D &pout)
{
    pout.coord[0] = p1.coord[0]-p2.coord[0];
    pout.coord[1] = p1.coord[1]-p2.coord[1];
    pout.coord[2] = p1.coord[2]-p2.coord[2];

    return pout;
}

inline Vector &Sub(Point4D &p, Vector &v, Vector &vout)
{
    vout.x(p.coord[0]-v.x());
    vout.y(p.coord[1]-v.y());
    vout.z(p.coord[2]-v.z());

    return vout;
}

inline double distance(const Point4D &p1, const Point4D &p2)
{
    return
	sqrt((p1.coord[0]-p2.coord[0])*(p1.coord[0]-p2.coord[0]) +
	     (p1.coord[1]-p2.coord[1])*(p1.coord[1]-p2.coord[1]) +
	     (p1.coord[2]-p2.coord[2])*(p1.coord[2]-p2.coord[2]));
}

inline Point4D &Add(Point4D &p, Vector &v, Point4D &pout)
{
    pout.coord[0] = p.coord[0]+v.x();
    pout.coord[1] = p.coord[1]+v.y();
    pout.coord[2] = p.coord[2]+v.z();
    pout.coord[3] = 1.;

    return pout;
}

inline Point4D &Add(Point4D &p1, Point4D &p2, Point4D &pout)
{
    pout.coord[0] = p1.coord[0]+p2.coord[0];
    pout.coord[1] = p1.coord[1]+p2.coord[1];
    pout.coord[2] = p1.coord[2]+p2.coord[2];
    pout.coord[3] = 1.;

    return pout;
}


inline Point4D &ToBasis(Point4D &pin, Point4D &c,
		      Vector &u, Vector &v, Vector &w,
		      Point4D &pout)
{
    Vector tempv;

    Sub(pin,c,tempv);

    pout.coord[0] = tempv.dot(u);
    pout.coord[1] = tempv.dot(v);
    pout.coord[2] = tempv.dot(w);
    
    return pout;
}

inline Point4D &ToCanonical(Point4D &pin, Point4D &c,
			  Vector &u, Vector &v, Vector &w,
			  Point4D &pout)
{
    pout.coord[0] =
	pin.coord[0]*u.x()+
	pin.coord[1]*v.x()+
	pin.coord[2]*w.x();
    pout.coord[1] = 
	pin.coord[0]*u.y()+
	pin.coord[1]*v.y()+
	pin.coord[2]*w.y();
    pout.coord[2] =
	pin.coord[0]*u.z()+
	pin.coord[1]*v.z()+
	pin.coord[2]*w.z();
    pout.coord[3] = 1.;
    
    pout += c;
    
    return pout;
}

inline Vector &Cross(Point4D &p, Vector &v, Vector &vout)
{
    vout.x(p.coord[1] * v.z() - v.y() * p.coord[2]);
    vout.y(p.coord[2] * v.x() - v.z() * p.coord[0]);
    vout.z(p.coord[0] * v.y() - v.x() * p.coord[1]);
    
    return vout;
}

inline Point4D &Mult(Point4D &p, double c, Point4D &pout)
{
    pout.coord[0] = p.coord[0]*c;
    pout.coord[1] = p.coord[1]*c;
    pout.coord[2] = p.coord[2]*c;
    pout.coord[3] = 1.;
    
    return pout;
}

inline Point4D &Mult(Vector &v, double c, Point4D &pout)
{
    pout.coord[0] = v.x()*c;
    pout.coord[1] = v.y()*c;
    pout.coord[2] = v.z()*c;
    pout.coord[3] = 1.;
    
    return pout;
}

inline Point4D &Div(Point4D &p, double c, Point4D &pout)
{
    double div = 1./c;
    
    pout.coord[0] = p.coord[0]*div;
    pout.coord[1] = p.coord[1]*div;
    pout.coord[2] = p.coord[2]*div;
    pout.coord[3] = 1.;

    return pout;
}

inline void weighted_diff(Point4D &p1, double w1,
			  Point4D &p2, double w2,
			  Vector &v) 
{
  v.x(p1.coord[0]*w1-p2.coord[0]*w2);
  v.y(p1.coord[1]*w1-p2.coord[1]*w2);
  v.z(p1.coord[2]*w1-p2.coord[2]*w2);
}

} // end namespace rtrt

#endif

