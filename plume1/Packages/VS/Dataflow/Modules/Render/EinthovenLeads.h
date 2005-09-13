//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : EinthovenLeads.h
//    Author : Martin Cole
//    Date   : Mon Mar  7 11:47:51 2005

#if !defined(EinthovenLeads_h)
#define EinthovenLeads_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>

namespace VS {

using namespace SCIRun;

class EinthovenLeadsAlgo : public DynamicAlgoBase
{
public:
  virtual bool get_values(FieldHandle field, unsigned I, unsigned II, 
			  unsigned III, vector<double> &values,
			  vector<Point> &pos) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftype);
};


template <class FIELD>
class EinthovenLeadsAlgoT : public EinthovenLeadsAlgo
{
public:
  bool get_values(FieldHandle field, unsigned I, unsigned II, unsigned III,
		  vector<double> &values, vector<Point> &pos);
};

template <class T>
bool
to_double(const T& tmp, double &val)
{
  val = (double)tmp;
  return true;
}

template <>
bool
to_double(const Vector&, double &);

template <>
bool
to_double(const Tensor&, double &);


template <class FIELD>
bool
EinthovenLeadsAlgoT<FIELD>::get_values(FieldHandle field, unsigned I, 
				       unsigned II, unsigned III, 
				       vector<double> &values,
				       vector<Point> &pos)
{
  
  FIELD *torso = dynamic_cast<FIELD*>(field.get_rep());
  typedef typename FIELD::mesh_type mesh_t;
  typedef typename FIELD::mesh_handle_type mesh_ht;
  mesh_ht mesh = torso->get_typed_mesh();

  bool rval = true;
  typename FIELD::value_type val;
  typedef typename mesh_t::Node::index_type node_idx_t;
  rval = rval && torso->value(val, (node_idx_t)I);
  rval = rval && to_double(val, values[0]); 
  mesh->get_center(pos[0], (node_idx_t)I);

  rval = rval && torso->value(val, (node_idx_t)II);
  rval = rval && to_double(val, values[1]); 
  mesh->get_center(pos[1], (node_idx_t)II);

  rval = rval && torso->value(val, (node_idx_t)III);
  rval = rval && to_double(val, values[2]); 
  mesh->get_center(pos[2], (node_idx_t)III);

  return rval;
}

} // End namespace VS

#endif
