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

#include <Core/Datatypes/Datatype.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class Point;
  class Vector;
  class Tensor;
  using std::vector;

class ScalarFieldInterface: public Datatype 
{
public:

  ScalarFieldInterface() {}
  ScalarFieldInterface(const ScalarFieldInterface&) {}
  virtual ~ScalarFieldInterface() {}

  virtual bool compute_min_max(double &minout, double &maxout,
			       bool cache = true) = 0;
  virtual bool interpolate(double &result, const Point &p) const = 0;
  virtual bool interpolate_many(vector<double> &results,
				const vector<Point> &points) const = 0;

  virtual double find_closest(double &result, const Point &p) const = 0;
};


class VectorFieldInterface: public Datatype 
{
public:
  VectorFieldInterface() {}
  VectorFieldInterface(const VectorFieldInterface&) {}
  virtual ~VectorFieldInterface() {}

  virtual bool compute_min_max(Vector &minout, Vector &maxout,
			       bool cache = true) = 0;
  virtual bool interpolate(Vector &result, const Point &p) const = 0;
  virtual bool interpolate_many(vector<Vector> &results,
				const vector<Point> &points) const = 0;
  virtual double find_closest(Vector &result, const Point &p) const = 0;
};


class TensorFieldInterface: public Datatype
{
public:
  TensorFieldInterface() {}
  TensorFieldInterface(const TensorFieldInterface&) {}
  virtual ~TensorFieldInterface() {}

  virtual bool interpolate(Tensor &result, const Point &p) const = 0;
  virtual bool interpolate_many(vector<Tensor> &results,
				const vector<Point> &points) const = 0;
  virtual double find_closest(Tensor &result, const Point &p) const = 0;
};


} // end namespace SCIRun


#endif // Datatypes_FieldInterface_h


