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
#include <Core/Datatypes/Field.h>

namespace SCIRun {


class ScalarFieldInterface {
public:

  ScalarFieldInterface() {}
  ScalarFieldInterface(const ScalarFieldInterface&) {}
  virtual ~ScalarFieldInterface() {}

  virtual bool compute_min_max(double &minout, double &maxout) const = 0;
  virtual bool interpolate(double &result, const Point &p) const = 0;
};


//! Should only be instantiated for fields with scalar data.
template <class F>
class SFInterface : public ScalarFieldInterface {
public:
  SFInterface(const F *fld) :
    fld_(fld)
  {}
  
  virtual bool compute_min_max(double &minout, double &maxout) const;
  virtual bool interpolate(double &result, const Point &p) const;

private:

  const F        *fld_;
};


template <class F>
bool
SFInterface<F>::interpolate(double &result, const Point &p) const
{
  typename F::mesh_handle_type mesh = fld_->get_typed_mesh();
  switch(fld_->data_at())
  {
  case F::NODE:
    {
      typename F::mesh_type::Node::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::EDGE:
    {
      typename F::mesh_type::Edge::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::FACE:
    {
      typename F::mesh_type::Face::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::CELL:
    {
      typename F::mesh_type::Cell::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::NONE:
    return false;
  }
  return true;
}


template <class F>
bool
SFInterface<F>::compute_min_max(double &minout, double &maxout) const
{
  bool result = false;
  minout = 1.0e6;
  maxout = -1.0e6;
  typename F::mesh_handle_type mesh = fld_->get_typed_mesh();
  switch (fld_->data_at())
  {
  case F::NODE:
    {
      typename F::mesh_type::Node::iterator bi = mesh->node_begin();
      typename F::mesh_type::Node::iterator ei = mesh->node_end();
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val < minout) minout = val;
	  if (val > maxout) maxout = val;
	  result = true;
	}
      }      
    }
    break;

  case F::EDGE:
    {
      typename F::mesh_type::Edge::iterator bi = mesh->edge_begin();
      typename F::mesh_type::Edge::iterator ei = mesh->edge_end();
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val < minout) minout = val;
	  if (val > maxout) maxout = val;
	  result = true;
	}
      }      
    }
    break;

  case F::FACE:
    {
      typename F::mesh_type::Face::iterator bi = mesh->face_begin();
      typename F::mesh_type::Face::iterator ei = mesh->face_end();
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val < minout) minout = val;
	  if (val > maxout) maxout = val;
	  result = true;
	}
      }      
    }
    break;

  case F::CELL:
    {
      typename F::mesh_type::Cell::iterator bi = mesh->cell_begin();
      typename F::mesh_type::Cell::iterator ei = mesh->cell_end();
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val < minout) minout = val;
	  if (val > maxout) maxout = val;
	  result = true;
	}
      }      
    }
    break;
    
  case F::NONE:
    break;
  }
  return result;
}


class VectorFieldInterface {
};


class TensorFieldInterface {
};

} // end namespace SCIRun


#endif // Datatypes_FieldInterface_h


