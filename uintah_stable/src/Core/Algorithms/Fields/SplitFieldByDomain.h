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

#ifndef CORE_ALGORITHMS_FIELDS_SPLITFIELDBYELEMENTDATA_H
#define CORE_ALGORITHMS_FIELDS_SPLITFIELDBYELEMENTDATA_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class SplitFieldByDomainAlgo : public DynamicAlgoBase
{
  public:
    virtual bool SplitFieldByDomain(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};

template<class FIELD>
class SplitFieldByDomainAlgoT: public SplitFieldByDomainAlgo
{
  public:
    virtual bool SplitFieldByDomain(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


template<class FIELD>
bool SplitFieldByDomainAlgoT<FIELD>::SplitFieldByDomain(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{
  FIELD *field = dynamic_cast<FIELD *>(input.get_rep());
  if (field == 0)
  {
    pr->error("SplitFieldByDomain: No field on input");
    return(false);
  }
  
  typename FIELD::mesh_type *mesh = dynamic_cast<typename FIELD::mesh_type *>(field->mesh().get_rep());

  typename FIELD::mesh_type::Elem::iterator bei, eei;
  typename FIELD::mesh_type::Elem::iterator bei2, eei2;

  typename FIELD::mesh_type::Node::iterator bni, eni;
  typename FIELD::mesh_type::Elem::size_type nelems;
  typename FIELD::mesh_type::Node::size_type nnodes;
  mesh->size(nelems);
  mesh->size(nnodes);

  mesh->begin(bni); mesh->end(eni);

  unsigned int maxindex = 0;

  while(bni != eni)
  {
    if (*(bni) > maxindex) maxindex = *(bni);
    ++bni;
  }  

  std::vector<typename FIELD::mesh_type::Node::index_type> idxarray(maxindex+1);
  std::vector<bool> newidxarray(maxindex+1);
  typename FIELD::mesh_type::Node::array_type nodes;
  typename FIELD::mesh_type::Node::array_type newnodes;
    
  typename FIELD::value_type val, minval;
  typename FIELD::value_type eval;
  unsigned int idx;
 
  typename FIELD::mesh_type *omesh = scinew typename FIELD::mesh_type();
  omesh->elem_reserve(nelems); // exact number
  omesh->node_reserve(nnodes); // minimum number of nodes
  
  std::vector<typename FIELD::value_type> newdata(nelems);
  
  FIELD* ofield = scinew FIELD(omesh);
  output = dynamic_cast<SCIRun::Field* >(ofield);
  if (ofield == 0)
  {
    pr->error("SplitFieldByDomain: Could not create output field");
    return(false);
  }
  
  mesh->begin(bei2);
  mesh->end(eei2);
  field->value(minval,*(bei2));
  while (bei2 != eei2)  
  {
    field->value(val,*(bei2));
    if (val < minval) minval = val;
    ++bei2; 
  }
  
  int k = 0;
  
  while(1)
  {
    val = minval;
  
    for (size_t p =0; p<(maxindex+1); p++) newidxarray[p] = true;

    mesh->begin(bei); mesh->end(eei);
    mesh->get_nodes(newnodes,*(bei));

    while (bei != eei)
    {
      field->value(eval,*(bei));
      if (eval == val)
      {
        mesh->get_nodes(nodes,*(bei));
        for (size_t p=0; p< nodes.size(); p++)
        {
          idx = nodes[p];
          if (newidxarray[idx])
          {
            Point pt;
            mesh->get_center(pt,nodes[p]);
            idxarray[idx] = omesh->add_point(pt);
            newidxarray[idx] = false;
          }
          newnodes[p] = idxarray[idx];
        }
        omesh->add_elem(newnodes);
        newdata[k++] = eval;
      }
      ++bei;
    }

    eval = val;
    bool foundminval = false;
    
    mesh->begin(bei2);
    mesh->end(eei2);
    while (bei2 != eei2)
    {
      field->value(eval,*(bei2));
      if (eval > val)
      {
        if (foundminval)
        {
          if (eval < minval) minval = eval;
        }
        else
        {
          minval = eval;
          foundminval = true;
        }
      }
      ++bei2;
    }


    if (minval > val)
    {
      val = minval;
    }
    else
    {
      break;
    }
    
  }
  
  omesh->begin(bei); omesh->end(eei);
  ofield->resize_fdata();
  k = 0;
  while (bei != eei)
  {
    ofield->set_value(newdata[k],(*bei));
    ++bei;
    k++;
  }
  
  return(true);
}

} // end namespace SCIRunAlgo

#endif

