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

#ifndef MODELCREATION_CORE_FIELDS_SPLITFIELDBYELEMENTDATA_H
#define MODELCREATION_CORE_FIELDS_SPLITFIELDBYELEMENTDATA_H 1

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Util/ProgressReporter.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Mesh.h>

#include <sci_hash_map.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class SplitFieldByElementDataAlgo : public DynamicAlgoBase
{
  public:
    virtual bool execute(ProgressReporter *reporter, 
                         FieldHandle input, 
                         FieldHandle& output) = 0;

  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
                         
                         
};

template<class FIELD>
class SplitFieldByElementDataAlgoT: public SplitFieldByElementDataAlgo
{
  public:
    virtual bool execute(ProgressReporter *reporter, 
                         FieldHandle input, 
                         FieldHandle& output);
};


template<class FIELD>
bool SplitFieldByElementDataAlgoT<FIELD>::execute(ProgressReporter *reporter, 
                         FieldHandle input, 
                         FieldHandle& output)
{
  FIELD *field = dynamic_cast<FIELD *>(input.get_rep());
  if (field == 0)
  {
    reporter->error("SplitFieldByElementData: No field on input");
    return(false);
  }
  
  typename FIELD::mesh_type *mesh = dynamic_cast<typename FIELD::mesh_type *>(field->mesh().get_rep());

  typename FIELD::mesh_type::Elem::iterator bei, eei;
  typename FIELD::mesh_type::Node::iterator bni, eni;
  typename FIELD::mesh_typ::Elem::size_type nelems;
  typename FIELD::mesh_typ::Node::size_type nnodes;
  mesh->size(nelems);
  mesh->size(nnodes);

  mesh->begin(bni); mesh->end(eni);

  typename FIELD::mesh_type::Node::under_type maxindex = 0;
  while(bni != eni)
  {
    if (*(bni) > maxindex) maxindex = *(bni);
    ++bni;
  }  

  std::vector<typename FIELD::mesh_type::Node::index_type> idxarray(maxindex);
  std::vector<bool> newidxarray(numnodes);
  typename FIELD::mesh_type::Node::array_type nodes;
  typename FIELD::mesh_type::Node::array_type newnodes;
    
  typename FIELD::value_type val;
  typename FIELD::value_type eval;
  typename FIELD::mesh_type::Node::under_type idx;
 
  typename FIELD::mesh_type *omesh = scinew typename FIELD::mesh_type();
  omesh->elem_reserve(nelems); // exact number
  omesh->node_reserve(nnodes); // minimum number of nodes
  
  FIELD* ofield = scinew FIELD(omesh,0);
  output = dynamic_cast<SCIRun::Field* >(ofield);

  mesh->begin(bei);
  val = field->value(*(bei));
  while(1)
  {
    for (size_t p =0; p<maxindex; p++) newidxarray[p] = true;

    mesh->begin(bei); mesh->end(eei);
    while (bei != eei)
    {
      eval = field->value(*(bei));
      if (eval == val)
      {
        get_nodes(nodes,*(bei));
        for (size_t p=0; p< nodes.size(); p++)
        {
          idx = *(nodes[p]);
          if (newidxarray[idx])
          {
            Point pt;
            mesh->get_center(pt,*(nodes[p]));
            idxarray[idx] = omesh->add_point(pt);
            newidxarray[idx] = false;
          }
          newnodes[p] = idxarray[idx];
        }
        omesh->add_elem(newnodes);
        ++bei;
      }
    }

    eval = val;
    mesh->begin(bei); mesh->end(eei);
    while (bei != eei)
    {
      eval = field->value(*(bei));
      if (eval > val) break;
      ++bei;
    }

    if (eval > val)
    {
      val = eval;
    }
    else
    {
      break;
    }
  }
  
  mesh->begin(bei); mesh->end(eei);
  ofield->resize_fdata();
  while (bei != eei)
  {
    val = field->value(*(bei));
    ofield->set_value(val,(*bei));
    ++bei;
  }
  
  return(true);
}

} // end namespace

#endif
