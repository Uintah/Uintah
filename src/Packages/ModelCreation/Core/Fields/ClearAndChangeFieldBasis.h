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


#ifndef MODELCREATION_CORE_FIELDS_CLEARANDCHANGEFIELDBASIS_H
#define MODELCREATION_CORE_FIELDS_CLEARANDCHANGEFIELDBASIS_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace ModelCreation {

using namespace SCIRun;

class ClearAndChangeFieldBasisAlgo;

class ClearAndChangeFieldBasisAlgo : public DynamicAlgoBase
{
public:
  virtual bool ClearAndChangeFieldBasis(ProgressReporter *pr, FieldHandle input, FieldHandle& output,std::string newbasis);
  virtual bool testinput(FieldHandle input);

  static AlgoList<ClearAndChangeFieldBasisAlgo> precompiled_;
};


template <class FSRC, class FDST>
class ClearAndChangeFieldBasisAlgoT : public ClearAndChangeFieldBasisAlgo
{
public:
  virtual bool ClearAndChangeFieldBasis(ProgressReporter *pr, FieldHandle input, FieldHandle& output,std::string newbasis);
  virtual bool testinput(FieldHandle input);
};


template <class FSRC, class FDST>
bool ClearAndChangeFieldBasisAlgoT<FSRC, FDST>::ClearAndChangeFieldBasis(ProgressReporter *pr, FieldHandle input, FieldHandle& output,std::string newbasis)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("ClearAndChangeFieldBasis: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("ClearAndChangeFieldBasis: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("ClearAndChangeFieldBasis: Could not create output field");
    return (false);
  }

  FDST *ofield = scinew FDST(omesh);
  output = dynamic_cast<Field*>(ofield);

  if (ofield->basis_order() == 0)
  {
    typename FDST::mesh_type::Elem::iterator it, it_end;
    typename FDST::value_type val = 0;
    
    omesh->begin(it);
    omesh->end(it_end);
    while (it != it_end)
    {
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  else if (ofield->basis_order() == 1)
  {
    typename FDST::mesh_type::Node::iterator it, it_end;
    typename FDST::value_type val = 0;
    
    omesh->begin(it);
    omesh->end(it_end);
    while (it != it_end)
    {
      ofield->set_value(val,*(it));
      ++it;
    }  
  }
  
	output->copy_properties(input.get_rep());
  return (true);
}

template <class FSRC, class FDST>
bool ClearAndChangeFieldBasisAlgoT<FSRC, FDST>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep()) != 0);
}


} // end namespace ModelCreation

#endif 
