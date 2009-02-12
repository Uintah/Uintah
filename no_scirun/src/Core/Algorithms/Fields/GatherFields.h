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


#ifndef CORE_ALGORITHMS_FIELDS_GATHERFIELDS_H
#define CORE_ALGORITHMS_FIELDS_GATHERFIELDS_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <list>
#include <sgi_stl_warnings_on.h>

// Simplified version of MergeFields that does not do any checking on 
// whether nodes are double.

namespace SCIRunAlgo {

class GatherFieldsAlgo : public SCIRun::DynamicAlgoBase
{
  public:
    virtual bool GatherFields(SCIRun::ProgressReporter *pr,std::list<SCIRun::FieldHandle> input, SCIRun::FieldHandle& output);  
};


template<class FIELD>
class GatherFieldsAlgoT : public GatherFieldsAlgo
{
  public:
    virtual bool GatherFields(SCIRun::ProgressReporter *pr,std::list<SCIRun::FieldHandle> input, SCIRun::FieldHandle& output);  
};


template<class FIELD>
bool GatherFieldsAlgoT<FIELD>::GatherFields(SCIRun::ProgressReporter *pr,std::list<SCIRun::FieldHandle> input, SCIRun::FieldHandle& output)
{
  std::list<SCIRun::FieldHandle>::iterator it, it_end; 

  typename FIELD::mesh_type::Node::size_type numnodes;
  typename FIELD::mesh_type::Elem::size_type numelems;

  int totnumnodes = 0;
  int totnumelems = 0;

  it = input.begin();
  it_end = input.end();
  int basisorder = 1;
  
  if (it == it_end) 
  {
    output = 0;
    return (true);
  }

  
  while (it != it_end)
  {
    FIELD* ifield = dynamic_cast<FIELD *>((*it).get_rep());
    if (ifield == 0) 
    {
      pr->error("GatherFields: Fields are not of the same type");
      return (false);
    }
    
    if (!(ifield->mesh()->is_editable()))
    {
      pr->error("GatheFields: Field type is not editable");
      return (false);
    }
    typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
    
    basisorder = ifield->basis_order();
    imesh->size(numnodes);
    imesh->size(numelems);
    totnumnodes += numnodes;
    totnumelems += numelems;
    ++it;
  }

  typename FIELD::mesh_type *omesh = dynamic_cast<typename FIELD::mesh_type*>(scinew typename FIELD::mesh_type());
  omesh->node_reserve(totnumnodes);
  omesh->elem_reserve(totnumelems);
  
  typename FIELD::mesh_handle_type meshhandle = omesh;
  FIELD *ofield = dynamic_cast<FIELD *>(scinew FIELD(meshhandle));
  output = dynamic_cast<SCIRun::Field *>(ofield);
  
  if (output.get_rep() == 0)
  {
    pr->error("GatherFields: Could not allocate output mesh");
    return (false);
  }
  
  if (basisorder == 0) ofield->fdata().resize(totnumelems);  
  if (basisorder == 1) ofield->fdata().resize(totnumnodes);  
  if (basisorder > 1)
  {
    pr->error("GatherFields: This function has not yet been implemented for higher order elements");
    return (false);
  }

  it = input.begin();
  it_end = input.end();

  unsigned int idx_offset = 0;
  unsigned int idx_offset_new = 0;

  typename FIELD::mesh_type::Node::index_type nidx;
  typename FIELD::mesh_type::Elem::index_type eidx;

  while (it != it_end)
  {
    FIELD* ifield = dynamic_cast<FIELD *>((*it).get_rep());
    typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
    typename FIELD::value_type val;
    
    typename FIELD::mesh_type::Node::iterator nit, nit_end;
    imesh->begin(nit);
    imesh->end(nit_end);
    
    SCIRun::Point p;
    while (nit != nit_end)
    {
      imesh->get_center(p,*nit);
      nidx = omesh->add_point(p);
      idx_offset_new++;
      if (basisorder == 1)
      {
        ifield->value(val,*nit);
        ofield->set_value(val,nidx);
      }
      ++nit;
    }
    
    typename FIELD::mesh_type::Elem::iterator eit, eit_end;
    typename FIELD::mesh_type::Node::array_type nodes;
    imesh->begin(eit);
    imesh->end(eit_end);
    
    while (eit != eit_end)
    {
      imesh->get_nodes(nodes,*eit);
      for (int q=0;q<nodes.size();q++) nodes[q] = nodes[q] + static_cast<typename FIELD::mesh_type::Node::index_type>(idx_offset);
      eidx = omesh->add_elem(nodes);
      if (basisorder == 0)
      {
        ifield->value(val,*eit);
        ofield->set_value(val,eidx);
      }
      ++eit;
    }

    idx_offset = idx_offset_new;
    ++it;
  }

  return (true);
}


} // end namespace SCIRunAlgo

#endif
