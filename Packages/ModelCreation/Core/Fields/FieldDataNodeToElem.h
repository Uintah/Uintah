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


#ifndef MODELCREATION_CORE_FIELDS_FIELDDATANODETOELEM_H
#define MODELCREATION_CORE_FIELDS_FIELDDATANODETOELEM_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <algorithm>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class FieldDataNodeToElemAlgo : public DynamicAlgoBase
{
public:

  virtual bool FieldDataNodeToElem(ProgressReporter *pr,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method);
};


template<class FIELD, class OFIELD>
class FieldDataNodeToElemAlgoT : public FieldDataNodeToElemAlgo
{
public:
  virtual bool FieldDataNodeToElem(ProgressReporter *pr,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method);                              
};


template<class FIELD, class OFIELD>
bool FieldDataNodeToElemAlgoT<FIELD,OFIELD>::FieldDataNodeToElem(ProgressReporter *pr,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method)
{ 
  output = 0;
                             
  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());
  if (field == 0)
  {
    pr->error("FieldDataNodeToElem: Object is not valid");
    return(false);
  }

  if (field->basis_order() < 1)
  {
     pr->error("FieldDataNodeToElem: Data is not located at nodes");
     return(false);   
  }

  // Create the field with the new mesh and data location.
  OFIELD *ofield = scinew OFIELD(field->get_typed_mesh());
  ofield->resize_fdata();

  typename FIELD::mesh_handle_type mesh = field->get_typed_mesh();
  typename FIELD::mesh_type::Node::array_type nodearray;

  if (field->basis_order() > 0)
  {
    typename FIELD::mesh_type::Elem::iterator it, eit;

    mesh->begin(it);
    mesh->end(eit);

    if ((method == "Interpolate")||(method == "Average"))
    {
      while (it != eit)
      {
        mesh->get_nodes(nodearray, *it);
        int nsize = nodearray.size();
        typename FIELD::value_type val;
        val = 0;
        for (size_t p = 0; p < nsize; p++)
        {
          val += field->value(nodearray[p]);
        }
        val = val * static_cast<double>((1.0/static_cast<double>(nsize)));
        ofield->set_value(val,*it);
        ++it;         
      }
    }
    
    if (method == "Max")
    {
      while (it != eit)
      {
        mesh->get_nodes(nodearray, *it);
        int nsize = nodearray.size();
        typename FIELD::value_type val = 0;
        typename FIELD::value_type tval = 0;
        if (nsize > 0)
        {
          val = field->value(nodearray[0]);
          for (size_t p = 1; p < nsize; p++)
          {
            tval = field->value(nodearray[p]);
            if (val < tval) val = tval;
          }
        }
        ofield->set_value(val,*it);
        ++it;                 
      }
    }
    
    if (method == "Min")
    {
      while (it != eit)
      {
        mesh->get_nodes(nodearray, *it);
        int nsize = nodearray.size();
        typename FIELD::value_type val = 0;
        typename FIELD::value_type tval = 0;
        if (nsize > 0)
        {
          val = field->value(nodearray[0]);
          for (size_t p = 1; p < nsize; p++)
          {
            tval = field->value(nodearray[p]);
            if (tval < val) val = tval;
          }
        }
        ofield->set_value(val,*it);
        ++it;        
      } 
    }

    if (method == "Sum")
    {
      while (it != eit)
      {
        mesh->get_nodes(nodearray, *it);
        int nsize = nodearray.size();
        typename FIELD::value_type val = 0;
        for (size_t p = 0; p < nsize; p++)
        {
          val += field->value(nodearray[p]);
        }
        ofield->set_value(val,*it);
        ++it;                
      }
    }

    if (method == "Median")
    {
      while (it != eit)
      {
        mesh->get_nodes(nodearray, *it);
        int nsize = nodearray.size();
        std::vector<typename FIELD::value_type> valarray(nsize);
        for (size_t p = 0; p < nsize; p++)
        {
          valarray[p] = field->value(nodearray[p]);
        }
        sort(valarray.begin(),valarray.end());
        int idx = static_cast<int>((valarray.size()/2));
        ofield->set_value(valarray[idx],*it);        
        ++it;
      }
    }
  }
  else if (field->basis_order() == 0)
  {
    pr->warning("FieldDataNodeToElem: Data is already on element");
    output = input;
    return(true);
  }
  else
  {    
    pr->warning("FieldDataNodeToElem: There is no data on input field");
    output = input;
    return(true);
  }

  output = dynamic_cast<SCIRun::Field *>(ofield);
  return(true);
}

} // namespace ModelCreation

#endif
