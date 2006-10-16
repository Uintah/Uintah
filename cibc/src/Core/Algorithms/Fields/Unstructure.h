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


#ifndef CORE_ALGORITHMS_FIELDS_UNSTRUCTURE_H
#define CORE_ALGORITHMS_FIELDS_UNSTRUCTURE_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class UnstructureAlgo : public DynamicAlgoBase
{
public:
  virtual bool Unstructure(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


template <class FSRC, class FDST>
class UnstructureAlgoT : public UnstructureAlgo
{
public:
  virtual bool Unstructure(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


template <class FSRC, class FDST>
bool UnstructureAlgoT<FSRC, FDST>::Unstructure(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("Unstructure: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("Unstructure: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("Unstructure: Could not create output field");
    return (false);
  }

  imesh->synchronize(Mesh::NODES_E);
  if (imesh->dimensionality() == 1) imesh->synchronize(Mesh::EDGES_E);
  if (imesh->dimensionality() == 2) imesh->synchronize(Mesh::FACES_E);
  if (imesh->dimensionality() == 3) imesh->synchronize(Mesh::CELLS_E);
  
  // Copy all points.
  typename FSRC::mesh_type::Node::iterator bn, en;
  typename FSRC::mesh_type::Node::size_type numnodes;
  imesh->begin(bn); 
  imesh->end(en);
  imesh->size(numnodes);
  omesh->node_reserve(numnodes);
  
  while (bn != en) 
  {
    Point np;
    imesh->get_center(np, *bn);
    omesh->add_point(np);
    ++bn;
  }

  // Copy the elements.
  typename FSRC::mesh_type::Elem::iterator bi, ei;
  typename FSRC::mesh_type::Elem::size_type numelems;

  imesh->begin(bi); 
  imesh->end(ei);
  imesh->size(numelems);
  omesh->elem_reserve(numelems);
  
  while (bi != ei) 
  {
    typename FSRC::mesh_type::Node::array_type onodes;
    imesh->get_nodes(onodes, *bi);
    typename FDST::mesh_type::Node::array_type nnodes(onodes.size());

    for (unsigned int i=0; i<onodes.size(); i++) 
    {
      nnodes[i] = static_cast<typename FDST::mesh_type::Node::index_type>(onodes[i]);
    }

    omesh->add_elem(nnodes);
    ++bi;
  }

  FDST *ofield = scinew FDST(omesh);
  output = dynamic_cast<Field*>(ofield);

  if (ifield->basis_order() == 1) 
  {
    imesh->begin(bn);
    imesh->end(en);

    ofield->fdata().resize(numnodes);
    while (bn != en) 
    {
      typename FSRC::value_type val;
      ifield->value(val, (*bn));
      ofield->set_value(val, static_cast<typename FDST::mesh_type::Node::index_type>(*bn));
      ++bn;
    }
  }
  else if (ifield->basis_order() == 0) 
  {
    imesh->begin(bi);
    imesh->end(ei);

    ofield->fdata().resize(numelems);
    while (bi != ei) 
    {
      typename FSRC::value_type val;
      ifield->value(val, (*bi));
      ofield->set_value(val, static_cast<typename FDST::mesh_type::Elem::index_type>(*bi));
      ++bi;
    }
  } 
  else if (ifield->basis_order() == -1)
  {
    // do nothing, no data to copy
  }
  else
  {
    pr->error("Function has not been defined for higher order elements, proper iterators for higher order elements have not yet been defined in the basis classes");
    ofield = 0;
    return (false);
  }

  // copy property manager
	output->copy_properties(input.get_rep());
  return (true);
}

} // end namespace SCIRunAlgo

#endif

