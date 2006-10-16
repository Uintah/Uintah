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

#ifndef CORE_ALGORITHMS_FIELDS_INDICESTODATA_H
#define CORE_ALGORITHMS_FIELDS_INDICESTODATA_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class IndicesToDataAlgo : public DynamicAlgoBase
{
public:
  virtual bool IndicesToData(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle data);
};

template <class FSRC, class FDST>
class IndicesToScalarAlgoT : public IndicesToDataAlgo
{
public:
  virtual bool IndicesToData(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle data);
};

template <class FSRC, class FDST>
class IndicesToVectorAlgoT : public IndicesToDataAlgo
{
public:
  virtual bool IndicesToData(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle data);
};

template <class FSRC, class FDST>
class IndicesToTensorAlgoT : public IndicesToDataAlgo
{
public:
  virtual bool IndicesToData(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle data);
};


template <class FSRC, class FDST>
bool IndicesToScalarAlgoT<FSRC, FDST>::IndicesToData(ProgressReporter *pr, FieldHandle input, FieldHandle& output,MatrixHandle data)
{
  
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  { 
    pr->error("IndicesToData: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh.get_rep() == 0)
  {
    pr->error("IndicesToData: No mesh associated with input field");
    return (false);
  }

  typename FSRC::mesh_type::Node::iterator nbi, nei;
  typename FSRC::mesh_type::Elem::iterator ebi, eei;

  DenseMatrix* dm = data->dense();
  MatrixHandle dmh = dynamic_cast<Matrix *>(dm);
  double *dataptr = dm->get_data_pointer();

  if ((!(data->nrows()==1)) && (!(data->ncols()==1)))
  {
    pr->error("IndicesToData: Input data is not of scalar type");
    return (false);  
  }
  
  int max_index = data->nrows() * data->ncols();

  if (ifield->basis_order() == 0)
  {
    imesh->begin(ebi);
    imesh->end(eei);
    
    typename FSRC::value_type val;
    while (ebi != eei)
    {
      ifield->value(val,*ebi);
      unsigned int idx = static_cast<unsigned int>(val);
      if ((val < 0)|| (val >= max_index))
      {
        pr->error("IndicesToData: Index exceeds matrix dimensions");
        return (false);  
      }
      
      ++ebi;
    }
  }
  else if (ifield->basis_order() == 1)
  {
    imesh->begin(nbi);
    imesh->end(nei);
    
    typename FSRC::value_type val;
    while (nbi != nei)
    {
      ifield->value(val,*nbi);
      unsigned int idx = static_cast<unsigned int>(val);
      if ((val < 0)|| (val >= max_index))
      {
        pr->error("IndicesToData: Index exceeds matrix dimensions");
        return (false);  
      }
      
      ++nbi;
    }
  }
  else
  {
    pr->error("IndicesToData: Algorithm not implemented for basis order");
    return (false);    
  }
  

  typename FDST::mesh_handle_type omesh = imesh.get_rep();
  FDST* ofield = scinew FDST(omesh);
  if (ofield == 0)
  {
    pr->error("ConvertToTetVol: Could not create output field");
    return (false);  
  }
  output = dynamic_cast<Field*>(ofield);
  ofield->resize_fdata();  

  if (ifield->basis_order() == 0)
  {
    imesh->begin(ebi); 
    imesh->end(eei);
    typename FSRC::value_type val;
    typename FDST::value_type val2;
  
    
    while (ebi != eei)
    {
      ifield->value(val,*ebi);
      unsigned int idx = static_cast<unsigned int>(val);
      val2 = static_cast<typename FDST::value_type>(dataptr[idx]);
      ofield->set_value(val2,*ebi);
      ++ebi;
    }
  }
  else
  {
    imesh->begin(nbi); 
    imesh->end(nei);
    typename FSRC::value_type val;
    typename FDST::value_type val2;
  
    
    while (nbi != nei)
    {
      ifield->value(val,*nbi);
      unsigned int idx = static_cast<unsigned int>(val);
      val2 = static_cast<typename FDST::value_type>(dataptr[idx]);
      ofield->set_value(val2,*nbi);
      ++nbi;
    }  
  }

	output->copy_properties(input.get_rep());
  
  // Success:
  return (true);
}




template <class FSRC, class FDST>
bool IndicesToVectorAlgoT<FSRC, FDST>::IndicesToData(ProgressReporter *pr, FieldHandle input, FieldHandle& output,MatrixHandle data)
{
  
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  { 
    pr->error("IndicesToData: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh.get_rep() == 0)
  {
    pr->error("IndicesToData: No mesh associated with input field");
    return (false);
  }

  typename FSRC::mesh_type::Node::iterator nbi, nei;
  typename FSRC::mesh_type::Elem::iterator ebi, eei;

  DenseMatrix* dm = data->dense();
  MatrixHandle dmh = dynamic_cast<Matrix *>(dm);

  if ((!(data->nrows()==3)) && (!(data->ncols()==3)))
  {
    pr->error("IndicesToData: Input data is not of vector type");
    return (false);  
  }

  if (data->ncols() != 3)
  {
    MatrixHandle temp = dmh;
    dmh = dm->transpose();
    dm = dmh->dense();
  }

  double *dataptr = dmh->get_data_pointer();
  int max_index = dmh->nrows();

  if (ifield->basis_order() == 0)
  {
    imesh->begin(ebi);
    imesh->end(eei);
    
    typename FSRC::value_type val;
    while (ebi != eei)
    {
      ifield->value(val,*ebi);
      unsigned int idx = static_cast<unsigned int>(val);
      if ((val < 0)|| (val >= max_index))
      {
        pr->error("IndicesToData: Index exceeds matrix dimensions");
        return (false);  
      }
      
      ++ebi;
    }
  }
  else if (ifield->basis_order() == 1)
  {
    imesh->begin(nbi);
    imesh->end(nei);
    
    typename FSRC::value_type val;
    while (nbi != nei)
    {
      ifield->value(val,*nbi);
      unsigned int idx = static_cast<unsigned int>(val);
      if ((val < 0)|| (val >= max_index))
      {
        pr->error("IndicesToData: Index exceeds matrix dimensions");
        return (false);  
      }
      
      ++nbi;
    }
  }
  else
  {
    pr->error("IndicesToData: Algorithm not implemented for basis order");
    return (false);    
  }
  

  typename FDST::mesh_handle_type omesh = imesh.get_rep();
  FDST* ofield = scinew FDST(omesh);
  if (ofield == 0)
  {
    pr->error("ConvertToTetVol: Could not create output field");
    return (false);  
  }
  output = dynamic_cast<Field*>(ofield);
  ofield->resize_fdata();  

  if (ifield->basis_order() == 0)
  {
    imesh->begin(ebi); 
    imesh->end(eei);
    typename FSRC::value_type val;
    SCIRun::Vector val2;
  
    while (ebi != eei)
    {
      ifield->value(val,*ebi);
      unsigned int idx = static_cast<unsigned int>(val);
      val2 = Vector(dataptr[3*idx],dataptr[3*idx+1],dataptr[3*idx+2]);
      ofield->set_value(val2,*ebi);
      ++ebi;
    }
  }
  else
  {
    imesh->begin(nbi); 
    imesh->end(nei);
    typename FSRC::value_type val;
    SCIRun::Vector val2;
      
    while (nbi != nei)
    {
      ifield->value(val,*nbi);
      unsigned int idx = static_cast<unsigned int>(val);
      val2 = Vector(dataptr[3*idx],dataptr[3*idx+1],dataptr[3*idx+2]);
      ofield->set_value(val2,*nbi);
      ++nbi;
    }  
  }

	output->copy_properties(input.get_rep());
  
  // Success:
  return (true);
}


template <class FSRC, class FDST>
bool IndicesToTensorAlgoT<FSRC, FDST>::IndicesToData(ProgressReporter *pr, FieldHandle input, FieldHandle& output,MatrixHandle data)
{
  
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  { 
    pr->error("IndicesToData: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh.get_rep() == 0)
  {
    pr->error("IndicesToData: No mesh associated with input field");
    return (false);
  }

  typename FSRC::mesh_type::Node::iterator nbi, nei;
  typename FSRC::mesh_type::Elem::iterator ebi, eei;

  DenseMatrix* dm = data->dense();
  MatrixHandle dmh = dynamic_cast<Matrix *>(dm);

  if ((!(data->nrows()==6 || data->nrows()==9)) && (!(data->ncols()==6 || data->nrows()==9)))
  {
    pr->error("IndicesToData: Input data is not of vector type");
    return (false);  
  }

  if ((data->ncols() != 6)&&(data->ncols() != 9))
  {
    MatrixHandle temp = dmh;
    dmh = dm->transpose();
    dm = dmh->dense();
  }

  double *dataptr = dmh->get_data_pointer();
  int max_index = dmh->nrows();
  int ncols = dmh->ncols();

  if (ifield->basis_order() == 0)
  {
    imesh->begin(ebi);
    imesh->end(eei);
    
    typename FSRC::value_type val;
    while (ebi != eei)
    {
      ifield->value(val,*ebi);
      unsigned int idx = static_cast<unsigned int>(val);
      if ((val < 0)|| (val >= max_index))
      {
        pr->error("IndicesToData: Index exceeds matrix dimensions");
        return (false);  
      }
      
      ++ebi;
    }
  }
  else if (ifield->basis_order() == 1)
  {
    imesh->begin(nbi);
    imesh->end(nei);
    
    typename FSRC::value_type val;
    while (nbi != nei)
    {
      ifield->value(val,*nbi);
      unsigned int idx = static_cast<unsigned int>(val);
      if ((val < 0)|| (val >= max_index))
      {
        pr->error("IndicesToData: Index exceeds matrix dimensions");
        return (false);  
      }
      
      ++nbi;
    }
  }
  else
  {
    pr->error("IndicesToData: Algorithm not implemented for basis order");
    return (false);    
  }
  

  typename FDST::mesh_handle_type omesh = imesh.get_rep();
  FDST* ofield = scinew FDST(omesh);
  if (ofield == 0)
  {
    pr->error("ConvertToTetVol: Could not create output field");
    return (false);  
  }
  output = dynamic_cast<Field*>(ofield);
  ofield->resize_fdata();  

  if (ifield->basis_order() == 0)
  {
    imesh->begin(ebi); 
    imesh->end(eei);
    typename FSRC::value_type val;
    SCIRun::Tensor val2;
  
    while (ebi != eei)
    {
      ifield->value(val,*ebi);
      unsigned int idx = static_cast<unsigned int>(val);
      if (ncols == 6)
      {
        val2 = Tensor(dataptr[3*idx],dataptr[3*idx+1],dataptr[3*idx+2],dataptr[3*idx+3],dataptr[3*idx+4],dataptr[3*idx+5]);
      }
      else
      {
        val2 = Tensor(dataptr[3*idx],dataptr[3*idx+1],dataptr[3*idx+2],dataptr[3*idx+4],dataptr[3*idx+5],dataptr[3*idx+8]);      
      }
      ofield->set_value(val2,*ebi);
      ++ebi;
    }
  }
  else
  {
    imesh->begin(nbi); 
    imesh->end(nei);
    typename FSRC::value_type val;
    SCIRun::Tensor val2;
      
    while (nbi != nei)
    {
      ifield->value(val,*nbi);
      unsigned int idx = static_cast<unsigned int>(val);
      if (ncols == 6)
      {
        val2 = Tensor(dataptr[3*idx],dataptr[3*idx+1],dataptr[3*idx+2],dataptr[3*idx+3],dataptr[3*idx+4],dataptr[3*idx+5]);
      }
      else
      {
        val2 = Tensor(dataptr[3*idx],dataptr[3*idx+1],dataptr[3*idx+2],dataptr[3*idx+4],dataptr[3*idx+5],dataptr[3*idx+8]);      
      }
      ofield->set_value(val2,*nbi);
      ++nbi;
    }  
  }

	output->copy_properties(input.get_rep());
  
  // Success:
  return (true);
}




} // end namespace SCIRunAlgo

#endif 

