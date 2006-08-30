/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


#ifndef CORE_ALGORITHMS_FIELDS_SCALEFIELD_H
#define CORE_ALGORITHMS_FIELDS_SCALEFIELD_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <Core/Geometry/BBox.h>
#include <sci_hash_map.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class ScaleFieldAlgo : public DynamicAlgoBase
{
public:
  virtual bool ScaleField(ProgressReporter *pr, FieldHandle input, FieldHandle& output,double datascale,double meshscale, bool scale_from_center);
};


template <class FIELD>
class ScaleFieldAlgoT : public ScaleFieldAlgo
{
public:
  virtual bool ScaleField(ProgressReporter *pr, FieldHandle input, FieldHandle& output,double datascale,double meshscale, bool scale_from_center);
};


template <class FIELD>
bool ScaleFieldAlgoT<FIELD>::ScaleField(ProgressReporter *pr, FieldHandle input, FieldHandle& output,double datascale,double meshscale, bool scale_from_center)
{
  FIELD *ifield = dynamic_cast<FIELD *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("ScaleField: Could not obtain input field");
    return (false);
  }

  output = input->clone();
  FIELD *ofield = dynamic_cast<FIELD *>(output.get_rep());
  if (ofield == 0)
  {
    pr->error("ScaleField: Could not copy input field");
    return (false);
  }
  
  BBox box = input->mesh()->get_bounding_box();
  Vector center = 0.5*(box.min()+box.max());
  
  // scale mesh, only when needed
  if (scale_from_center || (meshscale != 1.0))
  {
    ofield->mesh_detach();
    Transform tf;
    tf.load_identity();
    if (scale_from_center) tf.pre_translate(-center);
    tf.pre_scale(Vector(meshscale,meshscale,meshscale));
    if (scale_from_center) tf.pre_translate(center);
    ofield->mesh()->transform(tf);
  }
  
  typename FIELD::mesh_handle_type omesh = ofield->get_typed_mesh();

  // scale data
  if (ofield->basis_order() == 0)
  {
    typename FIELD::mesh_type::Elem::iterator it, it_end;
    typename FIELD::value_type val;
    val = 0;
    
    omesh->begin(it);
    omesh->end(it_end);
    while (it != it_end)
    {
      ifield->value(val,*it);
      val = val*datascale;
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  else if (ofield->basis_order() == 1)
  {
    typename FIELD::mesh_type::Node::iterator it, it_end;
    typename FIELD::value_type val;
    val = 0;
    
    omesh->begin(it);
    omesh->end(it_end);
    while (it != it_end)
    {
      ifield->value(val,*it);
      val = val*datascale;
      ofield->set_value(val,*(it));
      ++it;
    }  
  }
  
	output->copy_properties(input.get_rep());
  return (true);
}

} // end namespace SCIRunAlgo

#endif 

