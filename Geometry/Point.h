
#ifndef Geometry_Point_h
#define Geometry_Point_h 1

#ifdef __GNUG__
#pragma interface
#endif

#include <Classlib/Assert.h>

class Vector;
class clString;
class Piostream;
class ostream;

class Point {
    double _x,_y,_z;
#if SCI_ASSERTION_LEVEL >= 4
    int uninit;
#endif
public:
    inline Point(double x, double y, double z): _x(x), _y(y), _z(z)
#if SCI_ASSERTION_LEVEL >= 4
	, uninit(0)
#endif
	    {}
    Point(double, double, double, double);
    inline Point(const Point&);
    inline Point();
    int operator==(const Point&) const;
    int operator!=(const Point&) const;
    Point& operator=(const Point&);
    Vector operator-(const Point&) const;
    Point operator+(const Vector&) const;
    Point operator-(const Vector&) const;
    Point operator*(double) const;
    Point& operator*=(const double);
    Point& operator+=(const Vector&);
    Point& operator-=(const Vector&);
    Point& operator/=(const double);
    Point operator/(const double) const;
    Point operator-() const;
    void x(const double);
    inline double x() const;
    void y(const double);
    inline double y() const;
    void z(const double);
    inline double z() const;
    Vector vector() const;
    
    clString string() const;
    
    friend class Vector;
    friend double Dot(const Point&, const Point&);
    friend double Dot(const Vector&, const Point&);
//    friend double Dot(const Point&, const Vector&);
    friend Point Min(const Point&, const Point&);
    friend Point Max(const Point&, const Point&);
    friend Point Interpolate(const Point&, const Point&, double);
    friend Point AffineCombination(const Point&, double,
				   const Point&, double,
				   const Point&, double);
    friend Point AffineCombination(const Point&, double,
				   const Point&, double);
    friend void Pio(Piostream&, Point&);
};

ostream& operator<<(ostream& os, Point& p);

#include <Geometry/Vector.h>
#include <Math/MinMax.h>

#include <Geometry/Point.icc>
#endif
