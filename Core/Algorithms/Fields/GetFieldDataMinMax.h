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

#ifndef CORE_ALGORITHMS_FIELDS_GETFIELDDATAMINMAX_H
#define CORE_ALGORITHMS_FIELDS_GETFIELDDATAMINMAX_H 1


#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <float.h>


namespace SCIRunAlgo {

using namespace SCIRun;

class GetFieldDataMinMaxAlgo : public SCIRun::DynamicAlgoBase
{
  public:
    virtual bool GetFieldDataMinMax(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, double& min, double& max);  
};


template <class FIELD>
class GetFieldDataMinMaxAlgoT : public GetFieldDataMinMaxAlgo
{
  public:
    virtual bool GetFieldDataMinMax(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, double& min, double& max);  
};


template <class FIELD>
bool GetFieldDataMinMaxAlgoT<FIELD>::GetFieldDataMinMax(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, double& min, double& max)
{
  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());

  if (field == 0)
  {
    pr->error("GetFieldDataMinMax: No input field");
    return (false);
  }
  
  if (input->basis_order() == -1)
  {
    pr->error("GetFieldDataMinMax: No data present in field");
    min = 0.0;
    max = 0.0;
    return (false);
  }
  else if (input->basis_order() == 0)
  {
    typename FIELD::mesh_type* mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Elem::iterator it, it_end;
    typename FIELD::value_type minval, maxval, val;

    minval = -DBL_MAX;
    maxval = DBL_MIN;
    
    mesh->begin(it);
    mesh->end(it_end);
    
    while (it != it_end)
    {
      val = field->value(*it);
      if (val < minval) minval = val;
      if (val > maxval) maxval = val;
      ++it;
    }

    min = static_cast<double>(minval);
    max = static_cast<double>(maxval);
    return (true);
  }
  else 
  {
    typename FIELD::mesh_type* mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Node::iterator it, it_end;
    typename FIELD::value_type minval, maxval, val;

    minval = -DBL_MAX;
    maxval = DBL_MIN;
    
    mesh->begin(it);
    mesh->end(it_end);
    
    while (it != it_end)
    {
      val = field->value(*it);
      if (val < minval) minval = val;
      if (val > maxval) maxval = val;
      ++it;
    }

    min = static_cast<double>(minval);
    max = static_cast<double>(maxval);

    return (true);
  }
}

} // end namespace SCIRunAlgo

#endif
