
/*
 *  Ray.cc:  The Ray datatype
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geometry/Ray.h>

namespace SCICore {
namespace Geometry {

  
Ray::Ray(const Point& o, const Vector& d)
: o(o), d(d)
{
}

Ray::Ray(const Ray& copy)
: o(copy.o), d(copy.d)
{
}

Ray::~Ray()
{
}

Ray& Ray::operator=(const Ray& copy)
{
    o=copy.o;
    d=copy.d;
    return *this;
}

Point Ray::origin() const
{
    return o;
}


Vector Ray::direction() const
{
    return d;
}

Point Ray::parameter(double t) const
{
  return o + d*t;
}

void Ray::normalize()
{
  d.normalize();
}

void Ray::direction(const Vector& newdir)
{
    d=newdir;
}


}
} 

