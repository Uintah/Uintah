
#ifndef Geometry_Vector_h
#define Geometry_Vector_h 1

#ifdef __GNUG__
#pragma interface
#endif

#include <Classlib/Assert.h>

class Point;
class clString;
class Piostream;

class Vector {
    double _x,_y,_z;
#if SCI_ASSERTION_LEVEL >= 4
    int uninit;
#endif
public:
    inline Vector(double x, double y, double z): _x(x), _y(y), _z(z)
#if SCI_ASSERTION_LEVEL >= 4
	, uninit(0)
#endif
	    { }
    inline Vector(const Vector&);
    inline Vector();
    double length() const;
    double length2() const;
    friend double Dot(const Vector&, const Vector&);
    friend double Dot(const Point&, const Vector&);
    friend double Dot(const Vector&, const Point&);
    Vector& operator=(const Vector&);
    Vector operator*(const double) const;
    Vector& operator*=(const double);
    Vector operator/(const double) const;
    Vector operator+(const Vector&) const;
    Vector& operator+=(const Vector&);
    Vector operator-() const;
    Vector operator-(const Vector&) const;
    Vector& operator-=(const Vector&);
    double normalize();
    Vector normal() const;
    friend Vector Cross(const Vector&, const Vector&);
    friend Vector Abs(const Vector&);
    void x(double);
    double x() const;
    void y(double);
    double y() const;
    void z(double);
    double z() const;
    void rotz90(const int);
    Point point() const;
    
    clString string() const;
    
    friend class Point;
    friend class Transform;
    
    friend Vector Interpolate(const Vector&, const Vector&, double);
    
    void find_orthogonal(Vector&, Vector&);
    
    friend void Pio(Piostream&, Vector&);
};

#include <Math/Expon.h>
#include <Math/MiscMath.h>
#include <Geometry/Point.h>

#include "Vector.icc"

#endif
