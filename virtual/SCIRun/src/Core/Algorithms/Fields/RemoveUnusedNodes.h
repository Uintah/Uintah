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


#ifndef CORE_ALGORITHMS_FIELDS_REMOVEUNUSEDNODES_H
#define CORE_ALGORITHMS_FIELDS_REMOVEUNUSEDNODES_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class RemoveUnusedNodesAlgo : public DynamicAlgoBase
{
public:
  virtual bool RemoveUnusedNodes(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


template <class FIELD>
class RemoveUnusedNodesAlgoT : public RemoveUnusedNodesAlgo
{
public:
  virtual bool RemoveUnusedNodes(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


template <class FIELD>
bool RemoveUnusedNodesAlgoT<FIELD>::RemoveUnusedNodes(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{
  FIELD *ifield = dynamic_cast<FIELD *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("RemoveUnusedNodes: Could not obtain input field");
    return (false);
  }

  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh.get_rep() == 0)
  {
    pr->error("RemoveUnusedNodes: Could not obtain mesh from input field");
    return (false);
  }
   
  typename FIELD::mesh_handle_type omesh = scinew typename FIELD::mesh_type;
  if (omesh.get_rep() == 0)
  {
    pr->error("RemoveUnusedNodes: Could not create new mesh");
    return (false);
  }
  
  FIELD *ofield = scinew FIELD(omesh.get_rep());
  if (ofield == 0)
  {
    pr->error("RemoveUnusedNodes: Could not create output field");
    return (false);
  }  

  output = dynamic_cast<Field*>(ofield);
  
  typename FIELD::mesh_type::Elem::iterator it, it_end, it2;
  typename FIELD::mesh_type::Node::size_type nnodes;
  typename FIELD::mesh_type::Node::array_type nodes, newnodes;
  typename FIELD::mesh_type::Node::index_type n;
  Point p;
  
  imesh->size(nnodes);
  typename FIELD::mesh_type::Node::index_type noidx = static_cast<typename FIELD::mesh_type::Node::index_type>(nnodes);
  std::vector<typename FIELD::mesh_type::Node::index_type> nidx(nnodes,noidx);
  
  imesh->begin(it);
  imesh->end(it_end);
  
  // Create new mesh
  while(it != it_end)
  {
    imesh->get_nodes(nodes,*it);
    newnodes.resize(nodes.size());
    for (size_t q=0; q<nodes.size(); q++)
    {
      n = nodes[q];
      if (nidx[n] == noidx)
      {
        imesh->get_center(p,n);
        nidx[n] = omesh->add_point(p);
      }
      newnodes[q] = n;
    }
    
    omesh->add_elem(n);
    ++it;
  }

  typename FIELD::value_type val;

  ofield->resize_fdata();
  if (ofield->basis_order() == 0)
  {
    omesh->begin(it2);
    imesh->begin(it);
    imesh->end(it_end);
    while (it != it_end)
    {
      ifield->value(val,*it);
      ofield->set_value(val,*it2);
      ++it;
      ++it2;
    }
  }
  else if (ofield->basis_order() == 1)
  {
    for (size_t r=0; r<static_cast<size_t>(nnodes); r++)
    {
      if (nidx[r] < noidx)
      {
        ifield->value(val,static_cast<typename FIELD::mesh_type::Node::index_type>(r));
        ofield->set_value(val,nidx[r]);
      }
    }
  }
  
	output->copy_properties(input.get_rep());
  return (true);
}

} // end namespace SCIRunAlgo

#endif 

