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
 *  Ray.h:  The Ray datatype
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef Geometry_Ray_h
#define Geometry_Ray_h

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {


class Ray {
    Point o;
    Vector d;
public:
    Ray(){}
    Ray(const Point&, const Vector&);
    
    Ray(const Ray&);
    ~Ray();
    Ray& operator=(const Ray&);

    Point origin() const;
    Vector direction() const;
  Point parameter(double t) const; // returns the Point at parameter t
			     //  does not pre-normalized direction
  
  void normalize(); // normalizes the direction vector
    void direction(const Vector& newdir);
};


} // End namespace SCIRun
#endif
