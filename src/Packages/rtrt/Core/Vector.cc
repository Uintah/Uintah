
#include "Vector.h"
#include <iostream>

using namespace rtrt;

namespace rtrt {
  ostream& operator<<(ostream& out, const Vector& p) {
    out << '[' << p.x() << ", " << p.y() << ", " << p.z() << ']';
    return out;
  }
} // end namespace rtrt

void Vector::make_ortho(Vector& v1, Vector& v2) const
{
    Vector v0(this->cross(Vector(1,0,0)));
    if(v0.length2() == 0){
	v0=this->cross(this->cross(Vector(0,1,0)));
    }
    v1=this->cross(v0);
    v1.normalize();
    v2=this->cross(v1);
    v2.normalize();
}

Vector Vector::normal() const {
    Vector v1(*this);
    v1.normalize();
    return v1;
}
