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

namespace SCIRun {

  
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


} // End namespace SCIRun

