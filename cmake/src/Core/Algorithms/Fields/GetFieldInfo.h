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


#ifndef CORE_ALGORITHMS_FIELDS_GETFIELDINFO_H
#define CORE_ALGORITHMS_FIELDS_GETFIELDINFO_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class GetFieldInfoAlgo : public DynamicAlgoBase
{
  public:
    virtual bool GetFieldInfo(ProgressReporter *pr, FieldHandle input, int& numnodes, int& numelems);
};

template <class FIELD>
class GetFieldInfoAlgoT : public GetFieldInfoAlgo
{
  public:
    virtual bool GetFieldInfo(ProgressReporter *pr, FieldHandle input, int& numnodes, int& numelems);
};

template <class FIELD>
bool GetFieldInfoAlgoT<FIELD>::GetFieldInfo(ProgressReporter *pr, FieldHandle input, int& numnodes, int& numelems)
{
  numnodes = 0;
  numelems = 0;
  
  FIELD* field = dynamic_cast<FIELD *>(input.get_rep());
  if (field == 0) return (false);

  typename FIELD::mesh_type::Node::size_type nnodes;
  typename FIELD::mesh_type::Elem::size_type nelems;

  field->get_typed_mesh()->size(nnodes);
  field->get_typed_mesh()->size(nelems);
  numnodes = static_cast<int>(nnodes);
  numelems = static_cast<int>(nelems);
  
  return (true);
}

} // end namespace SCIRunAlgo

#endif
