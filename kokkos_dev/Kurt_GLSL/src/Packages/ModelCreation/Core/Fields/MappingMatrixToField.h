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

#ifndef MODELCREATION_CORE_FIELDS_MAPPINGMATRIXTOFIELD_H
#define MODELCREATION_CORE_FIELDS_MAPPINGMATRIXTOFIELD_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>

namespace ModelCreation {

using namespace SCIRun;

class MappingMatrixToFieldAlgo : public DynamicAlgoBase
{
  public:
    virtual bool MappingMatrixToField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle mapping);

};

template<class FIELD, class OFIELD>
class MappingMatrixToFieldAlgoT: public MappingMatrixToFieldAlgo
{
  public:
    virtual bool MappingMatrixToField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle mapping);
};

template<class FIELD, class OFIELD>
bool MappingMatrixToFieldAlgoT<FIELD,OFIELD>::MappingMatrixToField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle mapping)
{
  FIELD *field = dynamic_cast<FIELD *>(input.get_rep());
  if (field == 0)
  {
    pr->error("MappingMatrixToField: No field on input");
    return(false);
  }
  
  if (!mapping->is_sparse())
  {
    pr->error("MappingMatrixToField: Matrix is not a sparse matrix");
    return(false);
  }
  
  SparseRowMatrix* sm = dynamic_cast<SparseRowMatrix* >(mapping->sparse());
  int*    rr  = sm->get_row();
  int*    cc  = sm->get_col();
  double* val = sm->get_val();
  int     n   = sm->ncols();
  int     m   = sm->nrows();
  int     nnz = sm->get_nnz();
  
  for (size_t p=0; p<m+1; p++)
  {
    if (rr[p] != p)
    {
      pr->error("MappingMatrixToField: This mapping matrix does not do a one-to-one mapping");
      return(false);
    }
  }

  typename FIELD::mesh_type *mesh = dynamic_cast<typename FIELD::mesh_type *>(field->mesh().get_rep());  
  typename FIELD::mesh_typ::Elem::size_type nelems;
  typename FIELD::mesh_typ::Node::size_type nnodes;
  mesh->size(nelems);
  mesh->size(nnodes);

  if (m == nnodes)
  {
    output = dynamic_cast<Field*>(scinew OFIELD(field->mesh,0));
    OFIELD *ofield = dynamic_cast<OFIELD*>(output.get_rep());

    if (ofield == 0)
    {
      pr->error("MappingMatrixToField: Could not allocate output field");
      return(false);
    }

    typename FIELD::mesh_type::Elem::iterator bei, eei;
    typename FIELD::value_type val;
    mesh->begin(bei); mesh->end(eei); 
    
    size_t p = 0;
    while (bei != eei)
    {
      val = static_cast<typename FIELD::value_type>(cc[p++]);
      ofield->set_value(val,(*bei));    
      ++bei;
    }
  }  
  else if (m == nelems)
  {
    output = dynamic_cast<Field*>(scinew OFIELD(field->mesh,1));
    OFIELD *ofield = dynamic_cast<OFIELD*>(output.get_rep());

    if (ofield == 0)
    {
      pr->error("MappingMatrixToField: Could not allocate output field");
      return(false);
    }

    typename FIELD::mesh_type::Node::iterator bni, eni;
    typename FIELD::value_type val;
    mesh->begin(bni); mesh->end(eni); 
    
    size_t p = 0;
    while (bni != eni)
    {
      val = static_cast<typename FIELD::value_type>(cc[p++]);
      ofield->set_value(val,(*bni));    
      ++bni;
    }
  }  
  else
  {
    pr->error("MappingMatrixToField: Number of nodes or elements is not equal to the number of rows of the mapping matrix");
    return(false);
  }

  return(true);
}

} // end namespace

#endif
