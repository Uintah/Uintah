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

class Piostream;


class SCICORESHARE Ray {
    Point o_;
    Vector d_;
public:
  //! Constructors
  Ray(){}
  Ray(const Point&, const Vector&);
  Ray(const Ray&);

  //! Destructor
  ~Ray();

  //! Copy Constructor
  Ray& operator=(const Ray&);
  
  //! Return data
  Point origin() const;
  Vector direction() const;

  /*!
    Returns the Point at parameter t, but does not pre-normalize d
  */
  Point parameter(double t) const;

  /*! 
    Computes the ray parameter t at which the ray will
    intersect the plane specified by the normal N and the 
    point P, such that the plane intersect point Ip: 
    Ip = o + d*t.  Returns true if there is an intersection,
    false if the vector is parallel to the plane.
  */
  bool planeIntersectParameter(const Vector& N, const Point& P, double& t) const;
  
  //! Modifiers
  void normalize(); //! normalizes the direction vector d
  void direction(const Vector& newdir); //! changes d

  friend SCICORESHARE void Pio( Piostream&, Ray&);
};


} // End namespace SCIRun
#endif
