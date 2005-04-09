

#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Classlib/Assert.h>
#include <Classlib/Persistent.h>
#include <Classlib/String.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>
#include <iostream.h>

Point Interpolate(const Point& p1, const Point& p2, double w)
{
    return Point(
	Interpolate(p1._x, p2._x, w),
	Interpolate(p1._y, p2._y, w),
	Interpolate(p1._z, p2._z, w));
}

clString Point::string() const
{
    return clString("[")
	+to_string(_x)+clString(", ")
	    +to_string(_y)+clString(", ")
		+to_string(_z)+clString("]");
}

int Point::operator==(const Point& p) const
{
    return p._x == _x && p._y == _y && p._z == _z;
}

int Point::operator!=(const Point& p) const
{
    return p._x != _x || p._y != _y || p._z != _z;
}

Point::Point(double x, double y, double z, double w)
{
    if(w==0){
	cerr << "degenerate point!" << endl;
	_x=_y=_z=0;
    } else {
	_x=x/w;
	_y=y/w;
	_z=z/w;
    }
#if SCI_ASSERTION_LEVEL >= 4
    uninit=0;
#endif
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2)
{
    return Point(p1._x*w1+p2._x*w2,
		 p1._y*w1+p2._y*w2,
		 p1._z*w1+p2._z*w2);
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2,
			const Point& p3, double w3)
{
    return Point(p1._x*w1+p2._x*w2+p3._x*w3,
		 p1._y*w1+p2._y*w2+p3._y*w3,
		 p1._z*w1+p2._z*w2+p3._z*w3);
}

ostream& operator<<( ostream& os, Point& p )
{
   os << p.string();
   return os;
}

void Pio(Piostream& stream, Point& p)
{
    stream.begin_cheap_delim();
    Pio(stream, p._x);
    Pio(stream, p._y);
    Pio(stream, p._z);
    stream.end_cheap_delim();
}
