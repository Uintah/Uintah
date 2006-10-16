/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#ifndef CORE_ALGORITHMS_FIELDS_GETFIELDMEASURE_H
#define CORE_ALGORITHMS_FIELDS_GETFIELDMEASURE_H 1


#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <float.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class GetFieldMeasureAlgo : public DynamicAlgoBase
{
  public:
    virtual bool GetFieldMeasure(ProgressReporter *pr,FieldHandle input, std::string method, double& measure);  
};


template <class FIELD>
class GetFieldMeasureAlgoT : public GetFieldMeasureAlgo
{
  public:
    virtual bool GetFieldMeasure(ProgressReporter *pr,FieldHandle input, std::string method, double& measure);  
};


template <class FIELD>
bool GetFieldMeasureAlgoT<FIELD>::GetFieldMeasure(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, std::string method, double& measure)
{
  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());

  if (field == 0)
  {
    pr->error("GetFieldMeasure: No input field");
    return (false);
  }
  
  if (input->basis_order() == -1)
  {
    pr->error("GetFieldMeasure: No data present in field");
    return (false);
  }
  else if (input->basis_order() == 0)
  {
    typename FIELD::mesh_type* mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Elem::iterator it, it_end;
    typename FIELD::value_type tval, val;
    
    mesh->begin(it);
    mesh->end(it_end);
    
    if ((method == "minimum")||(method == "Minimum")||(method=="min")||(method=="Min"))
    {
      tval = DBL_MAX;
      while (it != it_end)
      {
        val = field->value(*it);
        if (val < tval) tval = val;
        ++it;
      }
      measure = tval;
    }
    else if ((method == "maximum")||(method == "Maximum")||(method=="max")||(method=="Max"))
    {
      tval = -DBL_MAX;
      while (it != it_end)
      {
        val = field->value(*it);
        if (val > tval) tval = val;
        ++it;
      }
      measure = tval;
    }
    else if ((method == "sum")||(method=="Sum"))
    {
      tval = 0;
      while (it != it_end)
      {
        tval += field->value(*it);
        ++it;
      }
      measure = tval;
    }
    else if ((method == "average")||(method=="Average")||(method=="avr")||(method=="Avr"))
    {
      tval = 0;
      int cnt = 0;
      while (it != it_end)
      {
        tval += field->value(*it);
        ++it;
        cnt++;
      }
      measure = tval*(1.0/cnt);
    }
    else
    {
      pr->error("GetFieldMeasure: This method has not yet been implemeneted");
      return (false);
    }
    return (true);    
  }
  else if (input->basis_order() == 1)
  {
    typename FIELD::mesh_type* mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Node::iterator it, it_end;
    typename FIELD::value_type tval, val;
    
    mesh->begin(it);
    mesh->end(it_end);
    
    if ((method == "minimum")||(method == "Minimum")||(method=="min")||(method=="Min"))
    {
      tval = 0;
      if (it != it_end) tval = field->value(*it);
      while (it != it_end)
      {
        val = field->value(*it);
        if (val < tval) tval = val;
        ++it;
      }
      measure = tval;
    }
    else if ((method == "maximum")||(method == "Maximum")||(method=="max")||(method=="Max"))
    {
      tval = 0;
      if (it != it_end) tval = field->value(*it);
      while (it != it_end)
      {
        val = field->value(*it);
        if (val > tval) tval = val;
        ++it;
      }
      measure = tval;
    }
    else if ((method == "sum")||(method=="Sum"))
    {
      tval = 0;
      while (it != it_end)
      {
        tval += field->value(*it);
        ++it;
      }
      measure = tval;
    }
    else if ((method == "average")||(method=="Average")||(method=="avr")||(method=="Avr"))
    {
      tval = 0;
      int cnt = 0;
      while (it != it_end)
      {
        tval += field->value(*it);
        ++it;
        cnt++;
      }
      measure = tval*(1.0/cnt);
    }
    else
    {
      pr->error("GetFieldMeasure: This method has not yet been implemeneted");
      return (false);
    }
    return (true);    
  }
  else
  {
    pr->error("GetFieldMeasure: This function has not been implemented for higher order elements");
    return (false);
  }
}

} // end namespace SCIRunAlgo

#endif
