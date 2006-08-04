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

#ifndef CORE_ALGORITHMS_REGRESSION_COMPAREFIELDS_H
#define CORE_ALGORITHMS_REGRESSION_COMPAREFIELDS_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>

// Additionally we include sci_hash_map here as it is needed by the algorithm

namespace SCIRunAlgo {

using namespace SCIRun;

class CompareFieldsAlgo : public DynamicAlgoBase
{
public:
  virtual bool CompareFields(ProgressReporter *pr, FieldHandle field1, FieldHandle field2);
};

template <class FIELD>
class CompareFieldsAlgoT : public CompareFieldsAlgo
{
public:
  virtual bool CompareFields(ProgressReporter *pr, FieldHandle field1, FieldHandle field2);
};

template <class FIELD>
bool CompareFieldsAlgoT<FIELD>::CompareFields(ProgressReporter *pr, FieldHandle field1, FieldHandle field2)
{
  FIELD *ifield1 = dynamic_cast<FIELD *>(field1.get_rep());
  FIELD *ifield2 = dynamic_cast<FIELD *>(field2.get_rep());

  if (ifield1 == 0)
  { 
    pr->remark("CompareFields: field1 is empty");
    return (false);
  }

  if (ifield2 == 0)
  { 
    pr->remark("CompareFields: field2 is empty");
    return (false);
  }

  typename FIELD::mesh_handle_type imesh1 = ifield1->get_typed_mesh();
  if (imesh1.get_rep() == 0)
  {
    pr->remark("CompareFields: No mesh associated with input field 1");
    return (false);
  }


  typename FIELD::mesh_handle_type imesh2 = ifield2->get_typed_mesh();
  if (imesh2.get_rep() == 0)
  {
    pr->remark("CompareFields: No mesh associated with input field 2");
    return (false);
  }

  
  typename FIELD::mesh_type::Node::size_type numnodes1, numnodes2; 
  typename FIELD::mesh_type::Elem::size_type numelems1, numelems2; 
  imesh1->size(numnodes1);
  imesh1->size(numelems1);
  imesh2->size(numnodes2);
  imesh2->size(numelems2);
  
  if (numnodes1 != numnodes2)
  {
    pr->remark("CompareFields: Number of nodes is not equal");
    return (false);    
  }
  
  if (numelems1 != numelems2)
  {
    pr->remark("CompareFields: Number of elements is not equal");
    return (false);    
  }
  
  typename FIELD::mesh_type::Node::iterator nbi1, nei1;  
  typename FIELD::mesh_type::Node::iterator nbi2, nei2;  
  
  imesh1->begin(nbi1);
  imesh2->begin(nbi2);
  imesh1->end(nei1);
  imesh2->end(nei2);
  
  while (nbi1 != nei1)
  {
    Point p1, p2;
    imesh1->get_center(p1,*nbi1);
    imesh2->get_center(p2,*nbi2);
    
    if (p1 != p2)
    {
      pr->remark("CompareFields: The nodes are not equal");
      return (false);
    }
    ++nbi1;
    ++nbi2;
  }
  
  
  typename FIELD::mesh_type::Elem::iterator ebi1, eei1;  
  typename FIELD::mesh_type::Elem::iterator ebi2, eei2;  
  typename FIELD::mesh_type::Node::array_type nodes1, nodes2;
  
  imesh1->begin(ebi1);
  imesh2->begin(ebi2);
  imesh1->end(eei1);
  imesh2->end(eei2);
  
  while (ebi1 != eei1)
  {
    Point p1, p2;
    imesh1->get_nodes(nodes1,*ebi1);
    imesh2->get_nodes(nodes2,*ebi2);
    
    if (nodes1.size() != nodes2.size())
    {
      pr->remark("CompareFields: The number of nodes per element are not equal");
      return (false);
    }
    
    for (int p=0; p<nodes1.size(); p++)
    {
      if (nodes1[p] != nodes2[p])
      {
        pr->remark("CompareFields: The nodes that define the element are not equal");
        return (false);
      }
    }
    ++ebi1;
    ++ebi2;
  }
  
  if (ifield1->basis_order() == 0)
  {
    imesh1->begin(ebi1); 
    imesh1->end(eei1);
    imesh2->begin(ebi2); 
    imesh2->end(eei2);

    typename FIELD::value_type val1, val2;
    
    while (ebi1 != eei1)
    {
    
      ifield1->value(val1,*ebi1);
      ifield2->value(val2,*ebi2);
      
      if (val1 != val2)
      {
        pr->remark("CompareFields: The values in the fields are not equal");
        return (false);        
      }
      ++ebi1;
      ++ebi2;
    }
  }

  if (ifield1->basis_order() == 1)
  {
    imesh1->begin(nbi1); 
    imesh1->end(nei1);
    imesh2->begin(nbi2); 
    imesh2->end(nei2);

    typename FIELD::value_type val1, val2;
    
    while (nbi1 != nei1)
    {

      ifield1->value(val1,*nbi1);
      ifield2->value(val2,*nbi2);
 
      if (val1 != val2)
      {
        pr->remark("CompareFields: The values in the fields are not equal");
        return (false);        
      }
      ++nbi1;
      ++nbi2;
    }
  }

  return (true);
}

} // end namespace SCIRunAlgo

#endif 

