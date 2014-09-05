/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Geometry/Ray.h>
#include <Core/Persistent/Persistent.h>
namespace SCIRun {

  
Ray::Ray(const Point& o, const Vector& d)
: o_(o), d_(d)
{
}

Ray::Ray(const Ray& copy)
: o_(copy.o_), d_(copy.d_)
{
}

Ray::~Ray()
{
}

Ray& Ray::operator=(const Ray& copy)
{
    o_=copy.o_;
    d_=copy.d_;
    return *this;
}

Point Ray::origin() const
{
    return o_;
}


Vector Ray::direction() const
{
    return d_;
}

Point Ray::parameter(double t) const
{
  return o_ + d_*t;
}


bool
Ray::planeIntersectParameter(const Vector& N, const Point& P, double& t) const
{
  //! Computes the ray parameter t at which the ray R will

  //! point P

  /*  Dot(N, ((O + t0*V) - P)) = 0   solve for t0 */

  Point O(o_);
  Vector V(d_);
  double D = -(N.x()*P.x() + N.y()*P.y() + N.z()*P.z());
  double NO = (N.x()*O.x() + N.y()*O.y() + N.z()*O.z());

  double NV = Dot(N,V);

  if (NV == 0) // ray is parallel to plane
    return false;
  else {
    t =  -(D + NO)/NV;  
    return true;
  }
}

void Ray::normalize()
{
  d_.normalize();
}

void Ray::direction(const Vector& newdir)
{
    d_=newdir;
}

void Pio(Piostream& stream, Ray& ray)
{
    stream.begin_cheap_delim();
    Pio(stream, ray.o_);
    Pio(stream, ray.d_);
    stream.end_cheap_delim();
}

} // End namespace SCIRun

