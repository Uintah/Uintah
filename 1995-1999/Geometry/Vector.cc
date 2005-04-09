#if 1
#include <Geometry/Vector.h>
#include <Geometry/Point.h>
#include <Classlib/Assert.h>
#include <Classlib/Persistent.h>
#include <Classlib/String.h>
#include <Math/Expon.h>
#include <Math/MiscMath.h>
#include <iostream.h>
#include <stdio.h>

clString Vector::string() const
{
#if 0
    return clString("[")
	+to_string(_x)+clString(", ")
	    +to_string(_y)+clString(", ")
		+to_string(_z)+clString("]");
#endif
    char buf[100];
    sprintf(buf, "[%g, %g, %g]", _x, _y, _z);
    return clString(buf);
}

void Vector::find_orthogonal(Vector& v1, Vector& v2) const
{
    ASSERTL4(!uninit);
    Vector v0(Cross(*this, Vector(1,0,0)));
    if(v0.length2() == 0){
	v0=Cross(*this, Vector(0,1,0));
    }
    v1=Cross(*this, v0);
    v1.normalize();
    v2=Cross(*this, v1);
    v2.normalize();
}

void Pio(Piostream& stream, Vector& p)
{
    stream.begin_cheap_delim();
    Pio(stream, p._x);
    Pio(stream, p._y);
    Pio(stream, p._z);
    stream.end_cheap_delim();
}

Vector Vector::normal() const
{
   Vector v(*this);
   v.normalize();
   return v;			// 
}

ostream& operator<<( ostream& os, const Vector& v )
{
   os << v.string();
   return os;
}

istream& operator>>( istream& is, Vector& v)
{
  double x, y, z;
  char st;
  is >> st >> x >> st >> y >> st >> z >> st;
  v=Vector(x,y,z);
  return is;
}

int
Vector::operator== ( const Vector& v ) const
{
    return v._x == _x && v._y == _y && v._z == _z;
}
#endif
