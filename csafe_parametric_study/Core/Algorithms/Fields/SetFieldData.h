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


#ifndef CORE_ALGORITHMS_FIELDS_SETFIELDDATA_H
#define CORE_ALGORITHMS_FIELDS_SETFIELDDATA_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>


namespace SCIRunAlgo {

using namespace SCIRun;


class SetFieldDataAlgo : public SCIRun::DynamicAlgoBase
{
  public:
    virtual bool SetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::FieldHandle& output, SCIRun::MatrixHandle data, bool keepscalartype);  
};


template <class FIELD>
class SetFieldScalarDataAlgoT : public SetFieldDataAlgo
{
  public:
    virtual bool SetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::FieldHandle& output, SCIRun::MatrixHandle data, bool keepscalartype);  
};

template <class FIELD>
class SetFieldVectorDataAlgoT : public SetFieldDataAlgo
{
  public:
    virtual bool SetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::FieldHandle& output, SCIRun::MatrixHandle data, bool keepscalartype);  
};

template <class FIELD>
class SetFieldTensorDataAlgoT : public SetFieldDataAlgo
{
  public:
    virtual bool SetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::FieldHandle& output, SCIRun::MatrixHandle data, bool keepscalartype);  
};



template <class FIELD>
bool SetFieldScalarDataAlgoT<FIELD>::SetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::FieldHandle& output, SCIRun::MatrixHandle data, bool keepscalartype)
{

  if (data.get_rep() == 0)
  {
    pr->error("SetFieldData: No data matrix was specified");
    return (false);  
  }

  FIELD* field = dynamic_cast<FIELD* >(scinew FIELD(dynamic_cast<typename FIELD::mesh_type *>(input->mesh().get_rep())));
  if (field == 0)
  {
    pr->error("SetFieldData: Could not allocate a new field");
    return (false);
  }
  output = dynamic_cast<SCIRun::Field*>(field);
  output->copy_properties(input.get_rep());

  if (field->basis_order() == -1)
  {
    pr->warning("SetFieldData: The Field has no basis for field data");
    return (false);
  }
  else if (field->basis_order() == 0)
  {
    typename FIELD::value_type val;
    typename FIELD::mesh_handle_type mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Elem::size_type size;
    typename FIELD::mesh_type::Elem::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    if (((data->ncols() ==1)&&(data->nrows() == size))||
        ((data->ncols() ==size)&&(data->nrows() == 1)))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val =  dataptr[k++];
        field->set_value(val,*it);
        ++it;
      }
      return (true);
    }
    else if ((data->ncols() == 1)&&(data->nrows() == 1))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      while (it != it_end)
      {
        val =  dataptr[0];
        field->set_value(val,*it);
        ++it;
      }
      return (true);    
    }
    else
    {
      pr->error("SetFieldData: Dimensions of input matrix do not match field");
      return (false);
    }
  }
  else 
  {
    typename FIELD::value_type val;
    typename FIELD::mesh_handle_type mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Node::size_type size;
    typename FIELD::mesh_type::Node::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    if (((data->ncols() ==1)&&(data->nrows() == size))||
        ((data->ncols() ==size)&&(data->nrows() == 1)))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val =  dataptr[k++];
        field->set_value(val,*it);
        ++it;
      }
      return (true);
    }
    else if ((data->ncols() == 1)&&(data->nrows() == 1))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      while (it != it_end)
      {
        val =  dataptr[0];
        field->set_value(val,*it);
        ++it;
      }
      return (true);    
    }
    else
    {
      pr->error("SetFieldData: Dimensions of input matrix do not match field");
      return (false);
    }
  }
}


template <class FIELD>
bool SetFieldVectorDataAlgoT<FIELD>::SetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::FieldHandle& output, SCIRun::MatrixHandle data, bool keepscalartype)
{

  if (data.get_rep() == 0)
  {
    pr->error("SetFieldData: No data matrix was specified");
    return (false);  
  }

  FIELD* field = dynamic_cast<FIELD* >(scinew FIELD(dynamic_cast<typename FIELD::mesh_type *>(input->mesh().get_rep())));
  if (field == 0)
  {
    pr->error("SetFieldData: Could not allocate a new field");
    return (false);
  }
  output = dynamic_cast<SCIRun::Field*>(field);
  output->copy_properties(input.get_rep());
  

  if (field->basis_order() == -1)
  {
    pr->warning("SetFieldData: The Field has no basis for field data");
    return (false);
  }
  else if (field->basis_order() == 0)
  {
    typename FIELD::value_type val;
    typename FIELD::mesh_handle_type mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Elem::size_type size;
    typename FIELD::mesh_type::Elem::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    if ((data->ncols() ==3)&&(data->nrows() == size))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val =  SCIRun::Vector(dataptr[k++],dataptr[k++],dataptr[k++]);
        field->set_value(val,*it);
        ++it;
      }
      return (true);
    }
    
    if ((data->ncols() ==size)&&(data->nrows() == 3))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val =  SCIRun::Vector(dataptr[k],dataptr[k+size],dataptr[k+2*size]);
        field->set_value(val,*it);
        ++it;
        k++;
      }
      return (true);
    }

    if (((data->ncols() == 1)&&(data->nrows() == 3))||((data->ncols() ==3)&&(data->nrows() == 1)))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      while (it != it_end)
      {
        val =  SCIRun::Vector(dataptr[0],dataptr[1],dataptr[2]);
        field->set_value(val,*it);
        ++it;
      }
      return (true);
    }
    
    
    pr->error("SetFieldData: Dimensions of input matrix do not match field");
    return (false);
  }
  else 
  {
    typename FIELD::value_type val;
    typename FIELD::mesh_handle_type mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Node::size_type size;
    typename FIELD::mesh_type::Node::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    if ((data->ncols() ==3)&&(data->nrows() == size))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val =  SCIRun::Vector(dataptr[k++],dataptr[k++],dataptr[k++]);
        field->set_value(val,*it);
        ++it;
      }
      return (true);
    }
    
    if ((data->ncols() ==size)&&(data->nrows() == 3))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val =  SCIRun::Vector(dataptr[k],dataptr[k+size],dataptr[k+2*size]);
        field->set_value(val,*it);
        ++it;
        k++;
      }
      return (true);
    }

    if (((data->ncols() == 1)&&(data->nrows() == 3))&&((data->ncols() == 3)&&(data->nrows() == 1)))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      while (it != it_end)
      {
        val =  SCIRun::Vector(dataptr[0],dataptr[1],dataptr[2]);
        field->set_value(val,*it);
        ++it;
      }
      return (true);
    }    
    
    pr->error("SetFieldData: Dimensions of input matrix do not match field");
    return (false);
  }
}


template <class FIELD>
bool SetFieldTensorDataAlgoT<FIELD>::SetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::FieldHandle& output, SCIRun::MatrixHandle data, bool keepscalartype)
{

  if (data.get_rep() == 0)
  {
    pr->error("SetFieldData: No data matrix was specified");
    return (false);  
  }

  FIELD* field = dynamic_cast<FIELD* >(scinew FIELD(dynamic_cast<typename FIELD::mesh_type *>(input->mesh().get_rep())));
  if (field == 0)
  {
    pr->error("SetFieldData: Could not allocate a new field");
    return (false);
  }
  output = dynamic_cast<SCIRun::Field*>(field);
  output->copy_properties(input.get_rep());

  if (field->basis_order() == -1)
  {
    pr->warning("SetFieldData: The Field has no basis for field data");
    return (false);
  }
  else if (field->basis_order() == 0)
  {
    typename FIELD::value_type val;
    typename FIELD::mesh_handle_type mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Elem::size_type size;
    typename FIELD::mesh_type::Elem::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    if ((data->ncols() ==6)&&(data->nrows() == size))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[k+0];
        val.mat_[0][1] = dataptr[k+1];
        val.mat_[0][2] = dataptr[k+2];
        val.mat_[1][0] = dataptr[k+1];
        val.mat_[1][1] = dataptr[k+3];
        val.mat_[1][2] = dataptr[k+4];
        val.mat_[2][0] = dataptr[k+2];
        val.mat_[2][1] = dataptr[k+4];
        val.mat_[2][2] = dataptr[k+5];
        field->set_value(val,*it);
        ++it;
        k += 6;
      }
      return (true);
    }
    
    if ((data->ncols() ==9)&&(data->nrows() == size))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[k+0];
        val.mat_[0][1] = dataptr[k+1];
        val.mat_[0][2] = dataptr[k+2];
        val.mat_[1][0] = dataptr[k+3];
        val.mat_[1][1] = dataptr[k+4];
        val.mat_[1][2] = dataptr[k+5];
        val.mat_[2][0] = dataptr[k+6];
        val.mat_[2][1] = dataptr[k+7];
        val.mat_[2][2] = dataptr[k+8];
        field->set_value(val,*it);
        ++it;
        k +=9 ;
      }
      return (true);
    }

    
    if ((data->ncols() == size)&&(data->nrows() == 6))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[k+0*size];
        val.mat_[0][1] = dataptr[k+1*size];
        val.mat_[0][2] = dataptr[k+2*size];
        val.mat_[1][0] = dataptr[k+1*size];
        val.mat_[1][1] = dataptr[k+3*size];
        val.mat_[1][2] = dataptr[k+4*size];
        val.mat_[2][0] = dataptr[k+2*size];
        val.mat_[2][1] = dataptr[k+4*size];
        val.mat_[2][2] = dataptr[k+5*size];
        field->set_value(val,*it);
        ++it;
        k++;
      }
      return (true);
    }
    
    if ((data->ncols() ==size)&&(data->nrows() == 9))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[k+0*size];
        val.mat_[0][1] = dataptr[k+1*size];
        val.mat_[0][2] = dataptr[k+2*size];
        val.mat_[1][0] = dataptr[k+3*size];
        val.mat_[1][1] = dataptr[k+4*size];
        val.mat_[1][2] = dataptr[k+5*size];
        val.mat_[2][0] = dataptr[k+6*size];
        val.mat_[2][1] = dataptr[k+7*size];
        val.mat_[2][2] = dataptr[k+8*size];
        field->set_value(val,*it);
        ++it;
        k++;
      }
      return (true);
    }

    if (((data->ncols() == 6)&&(data->nrows() == 1))||((data->ncols() == 1)&&(data->nrows() == 6)))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[0];
        val.mat_[0][1] = dataptr[1];
        val.mat_[0][2] = dataptr[2];
        val.mat_[1][0] = dataptr[1];
        val.mat_[1][1] = dataptr[3];
        val.mat_[1][2] = dataptr[4];
        val.mat_[2][0] = dataptr[2];
        val.mat_[2][1] = dataptr[4];
        val.mat_[2][2] = dataptr[5];
        field->set_value(val,*it);
        ++it;
      }
      return (true);
    }

    if (((data->ncols() == 9)&&(data->nrows() == 1))||((data->ncols() == 1)&&(data->nrows() == 9)))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[0];
        val.mat_[0][1] = dataptr[1];
        val.mat_[0][2] = dataptr[2];
        val.mat_[1][0] = dataptr[3];
        val.mat_[1][1] = dataptr[4];
        val.mat_[1][2] = dataptr[5];
        val.mat_[2][0] = dataptr[6];
        val.mat_[2][1] = dataptr[7];
        val.mat_[2][2] = dataptr[8];
        field->set_value(val,*it);
        ++it;
      }
      return (true);
    }


    pr->error("SetFieldData: Dimensions of input matrix do not match field");
    return (false);
  }
  else 
  {
    typename FIELD::value_type val;
    typename FIELD::mesh_handle_type mesh = field->get_typed_mesh().get_rep();
    typename FIELD::mesh_type::Node::size_type size;
    typename FIELD::mesh_type::Node::iterator it, it_end;
    
    mesh->size(size);
    mesh->begin(it);
    mesh->end(it_end);
    
    if ((data->ncols() ==6)&&(data->nrows() == size))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[k+0];
        val.mat_[0][1] = dataptr[k+1];
        val.mat_[0][2] = dataptr[k+2];
        val.mat_[1][0] = dataptr[k+1];
        val.mat_[1][1] = dataptr[k+3];
        val.mat_[1][2] = dataptr[k+4];
        val.mat_[2][0] = dataptr[k+2];
        val.mat_[2][1] = dataptr[k+4];
        val.mat_[2][2] = dataptr[k+5];
        field->set_value(val,*it);
        ++it;
        k += 6;
      }
      return (true);
    }
    
    if ((data->ncols() ==9)&&(data->nrows() == size))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[k+0];
        val.mat_[0][1] = dataptr[k+1];
        val.mat_[0][2] = dataptr[k+2];
        val.mat_[1][0] = dataptr[k+3];
        val.mat_[1][1] = dataptr[k+4];
        val.mat_[1][2] = dataptr[k+5];
        val.mat_[2][0] = dataptr[k+6];
        val.mat_[2][1] = dataptr[k+7];
        val.mat_[2][2] = dataptr[k+8];
        field->set_value(val,*it);
        ++it;
        k +=9 ;
      }
      return (true);
    }

    
    if ((data->ncols() == size)&&(data->nrows() == 6))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[k+0*size];
        val.mat_[0][1] = dataptr[k+1*size];
        val.mat_[0][2] = dataptr[k+2*size];
        val.mat_[1][0] = dataptr[k+1*size];
        val.mat_[1][1] = dataptr[k+3*size];
        val.mat_[1][2] = dataptr[k+4*size];
        val.mat_[2][0] = dataptr[k+2*size];
        val.mat_[2][1] = dataptr[k+4*size];
        val.mat_[2][2] = dataptr[k+5*size];
        field->set_value(val,*it);
        ++it;
        k++;
      }
      return (true);
    }
    
    if ((data->ncols() ==size)&&(data->nrows() == 9))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      size_t k = 0;
      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[k+0*size];
        val.mat_[0][1] = dataptr[k+1*size];
        val.mat_[0][2] = dataptr[k+2*size];
        val.mat_[1][0] = dataptr[k+3*size];
        val.mat_[1][1] = dataptr[k+4*size];
        val.mat_[1][2] = dataptr[k+5*size];
        val.mat_[2][0] = dataptr[k+6*size];
        val.mat_[2][1] = dataptr[k+7*size];
        val.mat_[2][2] = dataptr[k+8*size];
        field->set_value(val,*it);
        ++it;
        k++;
      }
      return (true);
    }

    if (((data->ncols() == 6)&&(data->nrows() == 1))||((data->ncols() == 1)&&(data->nrows() == 6)))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[0];
        val.mat_[0][1] = dataptr[1];
        val.mat_[0][2] = dataptr[2];
        val.mat_[1][0] = dataptr[1];
        val.mat_[1][1] = dataptr[3];
        val.mat_[1][2] = dataptr[4];
        val.mat_[2][0] = dataptr[2];
        val.mat_[2][1] = dataptr[4];
        val.mat_[2][2] = dataptr[5];
        field->set_value(val,*it);
        ++it;
      }
      return (true);
    }
    
    if (((data->ncols() == 9)&&(data->nrows() == 1))||((data->ncols() == 1)&&(data->nrows() == 9)))
    {
      double* dataptr = data->get_data_pointer();
      if (dataptr == 0)
      {
        pr->error("SetFieldData: No data in data matrix");
        return (false);
      }

      while (it != it_end)
      {
        val.mat_[0][0] = dataptr[0];
        val.mat_[0][1] = dataptr[1];
        val.mat_[0][2] = dataptr[2];
        val.mat_[1][0] = dataptr[3];
        val.mat_[1][1] = dataptr[4];
        val.mat_[1][2] = dataptr[5];
        val.mat_[2][0] = dataptr[6];
        val.mat_[2][1] = dataptr[7];
        val.mat_[2][2] = dataptr[8];
        field->set_value(val,*it);
        ++it;
      }
      return (true);
    }

    pr->error("SetFieldData: Dimensions of input matrix do not match field");
    return (false);
  }
}

} // end namespace SCIRunAlgo

#endif

