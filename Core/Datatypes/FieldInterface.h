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
    fld_(fld),
    interp_(this)
  {}
  
  virtual bool compute_min_max(double &minout, double &maxout) const;
  virtual bool interpolate(double &result, const Point &p) const;

private:
  friend class linear_interp;

  class linear_interp {
  public:
    typedef typename F::mesh_type Mesh;
    const SFInterface *par_;

    linear_interp(const SFInterface *par) :
      par_(par)
    {}
    
    //! do linear interp at data location.
    bool operator()(double &result, const Point &p) const {
      typename Mesh::Cell::index_type ci;
      const typename F::mesh_handle_type &mesh = par_->fld_->get_typed_mesh();

      switch (par_->fld_->data_at()) {
      case F::NODE :
	{
	  if (! mesh->locate(ci, p)) return false;	  

	  typename Mesh::Node::array_type nodes;
	  mesh->get_nodes(nodes, ci);

	  typename F::value_type tmp;
	  int i = 0;
	  Point center;
	  typename Mesh::Node::array_type::iterator iter = nodes.begin();
	  while (iter != nodes.end()) {
	    if (par_->fld_->value(tmp, *iter)) { 
	      mesh->get_point(center, *iter);
	      const double w = (p - center).length();
	      result += w * (double)tmp;
	    }
	    ++iter; ++i;
	  }
	}
      break;
      case F::EDGE:
	{
	  ASSERTFAIL("Edge Data not yet supported");
	}
	break;
      case F::FACE:
	{  
	 ASSERTFAIL("Face Data not yet supported");
	}
	break;
      case F::CELL:
	{
	  if (! mesh->locate(ci, p)) return false;
	  
	  typename Mesh::Cell::array_type cells;
	  mesh->get_neighbors(cells, ci);
	  
	  typename F::value_type tmp;
	  int i = 0;
	  typename Mesh::Cell::array_type::iterator iter = cells.begin();
	  Point center;
	  while (iter != cells.end()) {
	    if (par_->fld_->value(tmp, *iter)) { 
	      mesh->get_center(center, *iter);
	      const double w = (p - center).length();
	      result += w * (double)tmp;
	    } else {
	      return false;
	    }
	    ++iter; ++i;
	  }
	}
	break;
      case F::NONE:
	cerr << "Error: Field data at location NONE!!" << endl;
	return false;
      } 
      return false;
    }    
  };

  const F        *fld_;
  linear_interp   interp_;
};


template <class Fld>
bool
SFInterface<Fld>::interpolate(double &result, const Point &p) const
{
  return interp_(result, p);
}

template <class Fld>
bool
SFInterface<Fld>::compute_min_max(double &minout, double &maxout) const
{
  ASSERTFAIL("not implemented");
}

class VectorFieldInterface {
};


class TensorFieldInterface {
};

} // end namespace SCIRun


#endif // Datatypes_FieldInterface_h


