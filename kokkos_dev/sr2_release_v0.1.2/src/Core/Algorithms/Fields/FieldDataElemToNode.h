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


#ifndef CORE_ALGORITHMS_FIELDS_FIELDDATAELEMTONODE_H
#define CORE_ALGORITHMS_FIELDS_FIELDDATAELEMTONODE_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <algorithm>
#include <sgi_stl_warnings_on.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class FieldDataElemToNodeAlgo : public DynamicAlgoBase
{
public:

  virtual bool FieldDataElemToNode(ProgressReporter *pr,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method);
};


template<class FIELD, class OFIELD>
class FieldDataElemToNodeAlgoT : public FieldDataElemToNodeAlgo
{
public:
  virtual bool FieldDataElemToNode(ProgressReporter *pr,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method);                          
};


template<class FIELD, class OFIELD>
bool FieldDataElemToNodeAlgoT<FIELD,OFIELD>::FieldDataElemToNode(ProgressReporter *pr,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method)
{     
  output = 0;
  
  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());
  if (field == 0)
  {
    pr->error("FieldDataElemToNode: Object is not valid");
    return(false);
  }

  if (field->basis_order() > 0)
  {
     pr->warning("FieldDataElemToNode: Data is already located at nodes");
     output = input;
     return(true);   
  }
  
  if (field->basis_order() != 0)
  {
     pr->error("FieldDataElemToNode: Data is not located at elements");
     return(false);   
  }

  // Create the field with the new mesh and data location.
  OFIELD *ofield = scinew OFIELD(field->get_typed_mesh());
  ofield->resize_fdata();
  
  typename FIELD::mesh_handle_type mesh = field->get_typed_mesh();
  typename FIELD::mesh_type::Elem::array_type elems;
  typename FIELD::mesh_type::Node::iterator it, eit;

  mesh->synchronize(SCIRun::Mesh::NODE_NEIGHBORS_E);
  mesh->begin(it);
  mesh->end(eit);

  if ((method == "Interpolate")||(method == "Average"))
  {
    while (it != eit)
    {
      mesh->get_elems(elems, *(it));
      int nsize = elems.size();
      typename FIELD::value_type val = 0.0;
      for (size_t p = 0; p < nsize; p++)
      {
        val = val + field->value(elems[p]);
      }
      val = static_cast<typename FIELD::value_type>(val*(1.0/static_cast<double>(nsize)));
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  
  if (method == "Max")
  {
    while (it != eit)
    {
      mesh->get_elems(elems, *(it));
      int nsize = elems.size();
      typename FIELD::value_type val = 0.0;
      typename FIELD::value_type tval = 0.0;
      if (nsize > 0)
      {
        val = field->value(elems[0]);
        for (size_t p = 1; p < nsize; p++)
        {
          tval = field->value(elems[p]);
          if (tval > val) val = tval;
        }
      }
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  
  if (method == "Min")
  {
    while (it != eit)
    {
      mesh->get_elems(elems, *it);
      int nsize = elems.size();
      typename FIELD::value_type val = 0.0;
      typename FIELD::value_type tval = 0.0;
      if (nsize > 0)
      {
        val = field->value(elems[0]);
        for (size_t p = 1; p < nsize; p++)
        {
          tval = field->value(elems[p]);
          if (tval < val) val = tval;
        }
      }
      ofield->set_value(val,*(it));
      ++it;
    }    
  }

  if (method == "Sum")
  {
    while (it != eit)
    {
      mesh->get_elems(elems, *(it));
      int nsize = elems.size();
      typename FIELD::value_type val = 0.0;
      for (size_t p = 0; p < nsize; p++)
      {
        val += field->value(elems[p]);
      }
      ofield->set_value(val,*(it));
      ++it;
    }
  }

  if (method == "Median")
  {
    while (it != eit)
    {
      mesh->get_elems(elems, *(it));
      int nsize = elems.size();
      std::vector<typename FIELD::value_type> valarray(nsize);
      for (size_t p = 0; p < nsize; p++)
      {
        valarray[p] = field->value(elems[p]);
      }
      sort(valarray.begin(),valarray.end());
      int idx = static_cast<int>((valarray.size()/2));
      ofield->set_value(valarray[idx],*(it));
      ++it;
    }
  }

  output = dynamic_cast<SCIRun::Field *>(ofield);
  return(true);
}


} // namespace SCIRunAlgo

#endif
