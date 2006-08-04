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


#ifndef CORE_ALGORITHMS_FIELDS_APPLYMAPPINGMATRIX_H
#define CORE_ALGORITHMS_FIELDS_APPLYMAPPINGMATRIX_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class ApplyMappingMatrixAlgo : public DynamicAlgoBase
{
public:
  virtual bool ApplyMappingMatrix(ProgressReporter *pr, FieldHandle fsrc, FieldHandle fdst, FieldHandle& output,MatrixHandle mapping);
};


template <class FSRC, class FDST, class FOUT>
class ApplyMappingMatrixAlgoT : public ApplyMappingMatrixAlgo
{
public:
  virtual bool ApplyMappingMatrix(ProgressReporter *pr, FieldHandle fsrc, FieldHandle fdst, FieldHandle& output,MatrixHandle mapping);
};


template <class FSRC, class FDST, class FOUT>
bool ApplyMappingMatrixAlgoT<FSRC, FDST, FOUT>::ApplyMappingMatrix(ProgressReporter *pr, FieldHandle fsrc, FieldHandle fdst, FieldHandle& output,MatrixHandle mapping)
{
  FSRC *ifsrc = dynamic_cast<FSRC *>(fsrc.get_rep());
  if (ifsrc == 0)
  {
    pr->error("ApplyMappingMatrix: Could not obtain input field");
    return (false);
  }

  FDST *ifdst = dynamic_cast<FSRC *>(fdst.get_rep());
  if (ifdst == 0)
  {
    pr->error("ApplyMappingMatrix: Could not obtain input field");
    return (false);
  }

  if (mapping.get_rep() == 0)
  {
    pr->error("ApplyMappingMatrix: Mapping matrix is empty");
    return (false);  
  }

  if (static_cast<unsigned int>(mapping->nrows()) != ifdst->fdata().size())
  {
    pr->error("ApplyMappingMatrix: Number of rows of mapping matrix is not equal to number of elements in destination field");
    return (false);    
  }

  if (static_cast<unsigned int>(mapping->ncols()) != ifsrc->fdata().size())
  {
    pr->error("ApplyMappingMatrix: Number of columns of mapping matrix not is equal to number of elements in source field");
    return (false);    
  }
  
  typename FOUT::mesh_type *dstmesh = ifdst->get_typed_mesh();
  output = dynamic_cast<Matrix *>(scinew FOUT(dstmesh));
  FOUT* fout = dynamic_cast<FOUT*>(output.get_rep());


  if (ifdst->basis_order() == 0)
  {
    typename FDST::mesh_type::Elem::iterator dbi, dei;
    int* idx;
    double* val;
    int idxsize;
    int idxstride;

    int i;
    unsigned int counter = 0; 
    

    ifdst->get_typed_mesh()->begin(dbi);
    ifdst->get_typed_mesh()->end(dei);

    while (dbi != dei)
    {
      mapping->getRowNonzerosNoCopy(counter, idxsize, idxstride, idx, val);
      
      typename FOUT::value_type dval = 0;
      for (i = 0; i < idxsize; i++)
      {
        typename FSRC::value_type v;
        typename FSRC::index_type index;
        ifsrc->mesh()->to_index(index, idx?(unsigned int)idx[i*idxstride]:i);
        ifsrc->value(v, index);
        dval += static_cast<typename FOUT::value_type>(v * val[i*idxstride]);
      }
      
      fout->set_value(dval, *dbi);
      ++counter;
      ++dbi;
    }  
  }
  else
  {
    typename FDST::mesh_type::Node::iterator dbi, dei;
    int* idx;
    double* val;
    int idxsize;
    int idxstride;

    int i;
    unsigned int counter = 0; 
    

    ifdst->get_typed_mesh()->begin(dbi);
    ifdst->get_typed_mesh()->end(dei);

    while (dbi != dei)
    {
      mapping->getRowNonzerosNoCopy(counter, idxsize, idxstride, idx, val);
      
      typename FOUT::value_type dval = 0;
      for (i = 0; i < idxsize; i++)
      {
        typename FSRC::value_type v;
        typename FSRC::index_type index;
        ifsrc->to_index(index, idx?(unsigned int)idx[i*idxstride]:i);
        ifsrc->value(v, index);
        dval += static_cast<typename FOUT::value_type>(v * val[i*idxstride]);
      }
      
      fout->set_value(dval, *dbi);
      ++counter;
      ++dbi;
    }
  }
  
  // copy property manager
	fdst->copy_properties(output.get_rep());
  return (true);
}

} // end namespace SCIRunAlgo

#endif 

