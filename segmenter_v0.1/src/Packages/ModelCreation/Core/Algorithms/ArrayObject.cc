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

#include <Packages/ModelCreation/Core/Algorithms/ArrayObject.h>

namespace ModelCreation {

void ArrayObject::clear()
{
  type_ = INVALID;
  size_ = 0;
  
  matrix_ = 0;
  field_ = 0;

  fielddataalgo_ = 0;
  fieldlocationalgo_ = 0;
  fieldcreatealgo_ = 0;
  
  data_ = 0;
  ncols_ = 0;
  idx_ = 0;
  
  xname_ = "";
  yname_ = "";
  zname_ = "";
}

bool ArrayObject::create_inputdata(SCIRun::FieldHandle field, std::string name)
{
  clear();
  
  if (field.get_rep() == 0) return(false);
  
  name_ = name;
  field_ = field;
  
  SCIRun::CompileInfoHandle ci = ArrayObjectFieldDataAlgo::get_compile_info(field);
  if (!SCIRun::DynamicCompilation::compile(ci,fielddataalgo_,false,pr_))
  {
    error("Dynamic compilation failed");
    return(false);
  }
  
  if(!(fielddataalgo_->setfield(field)))
  {
    error("Could not initiate dynamic algorithm");
    return(false);
  }
  
  size_ = fielddataalgo_->size();
  if (fielddataalgo_->isscalar()) type_ = FIELDSCALAR;
  if (fielddataalgo_->isvector()) type_ = FIELDVECTOR;
  if (fielddataalgo_->istensor()) type_ = FIELDTENSOR;  

  if (type_ == INVALID) return(false);

  return(true);
}

bool ArrayObject::create_inputdata(SCIRun::MatrixHandle matrix, std::string name)
{
  clear();
  
  if (matrix.get_rep() == 0) return(false);
  
  matrix_ = dynamic_cast<SCIRun::Matrix *>(matrix->dense());
  if (matrix_.get_rep())
  {
    name_   = name;
    ncols_  = matrix_->ncols();
    size_   = matrix_->nrows();
    data_   = matrix_->get_data_pointer();
  }
  else
  {
    return(false);
  }
  
  type_ = INVALID;
  if (ncols_ == 1) type_ = MATRIXSCALAR;
  if (ncols_ == 3) type_ = MATRIXVECTOR;
  if ((ncols_ == 6)||(ncols_ == 9)) type_ = MATRIXTENSOR;
  
  if (type_ == INVALID) return(false);    

  return(true);
}

bool ArrayObject::create_inputindex(std::string name, std::string sizename)
{
  clear();
  name_ = name;
  sizename_ = sizename;
  type_ = INDEX;
  size_ = 1;
  return(true);
}

bool ArrayObject::create_inputlocation(SCIRun::FieldHandle field, std::string locname, std::string xname, std::string yname, std::string zname)
{
  clear();
  
  if (field.get_rep() == 0) return(false);
  
  name_ =  locname;
  xname_ = xname;
  yname_ = yname;
  zname_ = zname;
  field_ = field;
  
  SCIRun::CompileInfoHandle ci = ArrayObjectFieldLocationAlgo::get_compile_info(field);
  if (!SCIRun::DynamicCompilation::compile(ci,fieldlocationalgo_,false,pr_))
  {
    error("Dynamic compilation failed");
    return(false);
  }
  
  if(!(fieldlocationalgo_->setfield(field)))
  {
    error("Could not initiate dynamic algorithm");
    return(false);
  }

  size_ = fieldlocationalgo_->size();
  type_ = LOCATION;
  
  return(true);
}

bool ArrayObject::create_inputelement(SCIRun::FieldHandle field, std::string name)
{
  clear();
  
  if (field.get_rep() == 0) return(false);
  
  name_ = name;
  SCIRun::CompileInfoHandle ci = ArrayObjectFieldElemAlgo::get_compile_info(field);
  if (!SCIRun::DynamicCompilation::compile(ci,fieldelementalgo_,false,pr_))
  {
    error("Dynamic compilation failed");
    return(false);
  }  

  if(!(fieldelementalgo_->setfield(field)))
  {
    error("Could not initiate dynamic algorithm");
    return(false);
  }

  size_ = fieldelementalgo_->size();
  type_ = ELEMENT;

  return(true);
}
    
bool ArrayObject::create_outputdata(SCIRun::FieldHandle& field, std::string datatype, std::string name, SCIRun::FieldHandle& ofield)
{
  clear();
  
  if (field.get_rep() == 0) 
  {
    error("No input data field");
    return(false);
  }
  name_ = name;
  
  SCIRun::CompileInfoHandle ci = ArrayObjectFieldCreateAlgo::get_compile_info(field,datatype);
  if (!SCIRun::DynamicCompilation::compile(ci,fieldcreatealgo_,false,pr_))
  {
    error("Dynamic compilation failed");
    return(false);
  }  

  if(!(fieldcreatealgo_->createfield(field,field_)))
  {
    error("Could not create output field");
    return(false);
  }

  SCIRun::CompileInfoHandle ci2 = ArrayObjectFieldDataAlgo::get_compile_info(field_);
  if (!SCIRun::DynamicCompilation::compile(ci2,fielddataalgo_,false,pr_))
  {
    error("Dynamic compilation failed");
    return(false);
  }  
  
  if (!(fielddataalgo_->setfield(field_)))
  {
    error("Could not link field with dynamic algorithm");
    return(false);  
  }
  
  size_ = fielddataalgo_->size();
  type_ = INVALID;
  
  if (fielddataalgo_->isscalar()) type_ = FIELDSCALAR;
  if (fielddataalgo_->isvector()) type_ = FIELDVECTOR;
  if (fielddataalgo_->istensor()) type_ = FIELDTENSOR;  
  
  if (type_ == INVALID)
   {
    error("Could not link field with dynamic algorithm");
    return(false);  
  }
   
  ofield = field_;
  return(true);
}




bool ArrayObject::create_outputlocation(SCIRun::FieldHandle& field,  std::string name, SCIRun::FieldHandle& ofield)
{
  clear();
  
  if (field.get_rep() == 0) 
  {
    error("No input data field");
    return(false);
  }
  name_ = name;
  
  field_ = field->clone();
  field_->mesh_detach();
 
  SCIRun::CompileInfoHandle ci2 = ArrayObjectFieldLocationAlgo::get_compile_info(field_);
  if (!SCIRun::DynamicCompilation::compile(ci2,fieldlocationalgo_,false,pr_))
  {
    error("Dynamic compilation failed");
    return(false);
  }  
  
  if (!(fieldlocationalgo_->setfield(field_)))
  {
    error("Could not link field with dynamic algorithm");
    return(false);  
  }
  
  size_ = fieldlocationalgo_->size();
  type_ = LOCATION;
     
  ofield = field_;
  return(true);
}




bool ArrayObject::create_outputdata(SCIRun::FieldHandle& field, std::string datatype, std::string basistype, std::string name, SCIRun::FieldHandle& ofield)
{
  clear();
  
  if (field.get_rep() == 0) 
  {
    error("No input data field");
    return(false);
  }
  name_ = name;
  
  SCIRun::CompileInfoHandle ci = ArrayObjectFieldCreateAlgo::get_compile_info(field,datatype,basistype);
  if (!SCIRun::DynamicCompilation::compile(ci,fieldcreatealgo_,false,pr_))
  {
    error("Dynamic compilation failed");
    return(false);
  }  

  if(!(fieldcreatealgo_->createfield(field,field_)))
  {
    error("Could not create output field");
    return(false);
  }

  SCIRun::CompileInfoHandle ci2 = ArrayObjectFieldDataAlgo::get_compile_info(field_);
  if (!SCIRun::DynamicCompilation::compile(ci2,fielddataalgo_,false,pr_))
  {
    error("Dynamic compilation failed");
    return(false);
  }  
  
  if (!(fielddataalgo_->setfield(field_)))
  {
    error("Could not link field with dynamic algorithm");
    return(false);  
  }
  
  size_ = fielddataalgo_->size();
  type_ = INVALID;
  
  if (fielddataalgo_->isscalar()) type_ = FIELDSCALAR;
  if (fielddataalgo_->isvector()) type_ = FIELDVECTOR;
  if (fielddataalgo_->istensor()) type_ = FIELDTENSOR;  
  
  if (type_ == INVALID)
   {
    error("Could not link field with dynamic algorithm");
    return(false);  
  }
   
  ofield = field_;
  return(true);
}



bool ArrayObject::create_outputdata(int size, std::string datatype, std::string name, SCIRun::MatrixHandle& omatrix)
{    
  clear();
  
  if (size == 0)
  {
    error("Output matrix size is 0");
    return(false);
  }
  
  name_ = name;
  size_ = size;
  idx_ = 0;
  
  if (datatype == "Scalar")  { ncols_ = 1; type_ = MATRIXSCALAR; }
  if (datatype == "Vector")  { ncols_ = 3; type_ = MATRIXVECTOR; }
  if (datatype == "Tensor")  { ncols_ = 9; type_ = MATRIXTENSOR; }
  if (datatype == "Tensor6") { ncols_ = 6; type_ = MATRIXTENSOR; }
  if (datatype == "Tensor9") { ncols_ = 9; type_ = MATRIXTENSOR; }
  
  matrix_ = dynamic_cast<SCIRun::Matrix *>(scinew SCIRun::DenseMatrix(size_,ncols_));
  if (matrix_.get_rep() == 0)
  {
    error("Could not allocate matrix");
    return(false);
  }
  
  data_ = matrix_->get_data_pointer();
  omatrix = matrix_;
}

} // end namespace

