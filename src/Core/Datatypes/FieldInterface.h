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

//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   May 2000
//
//  Copyright (C) 2000 SCI Institute
//
//
//
// To add an Interface, a new class should be created in this file,
// and the appropriate query pure virtual should be added in the Field class.
//


#ifndef Datatypes_FieldInterface_h
#define Datatypes_FieldInterface_h

#include <Core/Geometry/Point.h>

namespace SCIRun {

//! base class for interpolation objects
class InterpBase {
};  

//! generic interpolation class
template <class Data>
class GenericInterpolate : public InterpBase {
public:
  virtual bool interpolate(const Point& p, Data &value) const = 0;
};

//! type needed to support query_interpolate_to_scalar() interface
typedef GenericInterpolate<double> InterpolateToScalar;

} // end namespace SCIRun


#endif // Datatypes_FieldInterface_h


