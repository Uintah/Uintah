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

#ifndef MODELCREATION_CORE_FIELDS_GETFIELDDATA_H
#define MODELCREATION_CORE_FIELDS_GETFIELDDATA_H 1


#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <Core/Datatypes/DenseMatrix.h>

namespace ModelCreation {

using namespace SCIRun;


class GetFieldDataAlgo : public SCIRun::DynamicAlgoBase
{
  public:
    virtual bool GetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::MatrixHandle& output);  
};


template <class FIELD>
class GetFieldScalarDataAlgoT : public GetFieldDataAlgo
{
  public:
    virtual bool GetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::MatrixHandle& output);  
};


template <class FIELD>
class GetFieldVectorDataAlgoT : public GetFieldDataAlgo
{
  public:
    virtual bool GetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::MatrixHandle& output);  
};

template <class FIELD>
class GetFieldTensorDataAlgoT : public GetFieldDataAlgo
{
  public:
    virtual bool GetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::MatrixHandle& output);  
};


template <class FIELD>
bool GetFieldScalarDataAlgoT<FIELD>::GetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::MatrixHandle& output)
{
  typename FIELD::value_type val;

  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());

  if (input->basis_order() == -1)
  {
    pr->warning("GetFieldData: No data present in field");
    output = dynamic_cast<Matrix *>(scinew DenseMatrix(0,0));
    return (true);
  }
  else if (input->basis_order() == 0)
  {
    typename FIELD::mesh_type* mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Elem::size_type size;
    typename FIELD::mesh_type::Elem::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    output = dynamic_cast<SCIRun::Matrix *>(scinew DenseMatrix(size,1));
    if (output.get_rep() == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }
    
    double* dataptr = output->get_data_pointer();
    if (dataptr == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }

    size_t k = 0;
    while (it != it_end)
    {
      val = field->value(*it);
      dataptr[k++] = static_cast<double>(val);
      ++it;
    }

    return (true);
  }
  else 
  {
    typename FIELD::mesh_type* mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Node::size_type size;
    typename FIELD::mesh_type::Node::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it); 
    mesh->end(it_end);
    
    output = dynamic_cast<SCIRun::Matrix *>(scinew DenseMatrix(size,1));
    if (output.get_rep() == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }
    
    double* dataptr = output->get_data_pointer();
    if (dataptr == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }

    size_t k = 0;

    while (it != it_end)
    {
      val = field->value(*it);
      dataptr[k++] = static_cast<double>(val);
      ++it;
    }

    return (true);
  }
}

template <class FIELD>
bool GetFieldVectorDataAlgoT<FIELD>::GetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::MatrixHandle& output)
{
  typename FIELD::value_type val;

  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());

  if (input->basis_order() == -1)
  {
    pr->warning("GetFieldData: No data present in field");
    output = dynamic_cast<Matrix *>(scinew DenseMatrix(0,0));
    return (true);
  }
  else if (input->basis_order() == 0)
  {
    typename FIELD::mesh_type* mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Elem::size_type size;
    typename FIELD::mesh_type::Elem::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    output = dynamic_cast<SCIRun::Matrix *>(scinew DenseMatrix(size,3));
    if (output.get_rep() == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }
    
    double* dataptr = output->get_data_pointer();
    if (dataptr == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }

    size_t k = 0;
    while (it != it_end)
    {
      val = field->value(*it);
      dataptr[k++] = static_cast<double>(val.x());
      dataptr[k++] = static_cast<double>(val.y());
      dataptr[k++] = static_cast<double>(val.z());
      ++it;
    }

    return (true);
  }
  else 
  {
    typename FIELD::mesh_type* mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Node::size_type size;
    typename FIELD::mesh_type::Node::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    output = dynamic_cast<SCIRun::Matrix *>(scinew DenseMatrix(size,3));
    if (output.get_rep() == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }
    
    double* dataptr = output->get_data_pointer();
    if (dataptr == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }

    size_t k = 0;

    while (it != it_end)
    {
      val = field->value(*it);
      dataptr[k++] = static_cast<double>(val.x());
      dataptr[k++] = static_cast<double>(val.y());      
      dataptr[k++] = static_cast<double>(val.z());
      ++it;
    }
    
    return (true);
  }
}


template <class FIELD>
bool GetFieldTensorDataAlgoT<FIELD>::GetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::MatrixHandle& output)
{
  typename FIELD::value_type val;

  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());

  if (input->basis_order() == -1)
  {
    pr->warning("GetFieldData: No data present in field");
    output = dynamic_cast<Matrix *>(scinew DenseMatrix(0,0));
    return (true);
  }
  else if (input->basis_order() == 0)
  {
    typename FIELD::mesh_type* mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Elem::size_type size;
    typename FIELD::mesh_type::Elem::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    output = dynamic_cast<SCIRun::Matrix *>(scinew DenseMatrix(size,6));
    if (output.get_rep() == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }
    
    double* dataptr = output->get_data_pointer();
    if (dataptr == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }

    size_t k = 0;

    while (it != it_end)
    {
      val = field->value(*it);
      dataptr[k++] = static_cast<double>(val.mat_[0][0]);
      dataptr[k++] = static_cast<double>(val.mat_[0][1]);
      dataptr[k++] = static_cast<double>(val.mat_[0][2]);
      dataptr[k++] = static_cast<double>(val.mat_[1][1]);
      dataptr[k++] = static_cast<double>(val.mat_[1][2]);
      dataptr[k++] = static_cast<double>(val.mat_[2][2]);

      ++it;
    }

    return (true);
  }
  else 
  {
    typename FIELD::mesh_type* mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Node::size_type size;
    typename FIELD::mesh_type::Node::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    output = dynamic_cast<SCIRun::Matrix *>(scinew DenseMatrix(size,6));
    if (output.get_rep() == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }
    
    double* dataptr = output->get_data_pointer();
    if (dataptr == 0)
    {
      pr->error("GetFieldData: Could not allocate output matrix");
      return (false);
    }

    size_t k = 0;

    while (it != it_end)
    {
      val = field->value(*it);
      dataptr[k++] = static_cast<double>(val.mat_[0][0]);
      dataptr[k++] = static_cast<double>(val.mat_[0][1]);
      dataptr[k++] = static_cast<double>(val.mat_[0][2]);
      dataptr[k++] = static_cast<double>(val.mat_[1][1]);
      dataptr[k++] = static_cast<double>(val.mat_[1][2]);
      dataptr[k++] = static_cast<double>(val.mat_[2][2]);
      ++it;
    }
    
    return (true);
  }
}

} // end namespace

#endif
