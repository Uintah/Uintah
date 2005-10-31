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

#include <Packages/ModelCreation/Core/Algorithms/ArrayObjectFieldAlgo.h>

namespace ModelCreation {

void ArrayObjectFieldDataAlgo::getnextscalar(TensorVectorMath::Scalar& scalar)
{
  scalar = 0.0;
}

void ArrayObjectFieldDataAlgo::getnextvector(TensorVectorMath::Vector& vector)
{
  vector = TensorVectorMath::Vector(0.0,0.0,0.0);
}

void ArrayObjectFieldDataAlgo::getnexttensor(TensorVectorMath::Tensor& tensor)
{
  tensor = TensorVectorMath::Tensor(0.0);
}

void ArrayObjectFieldDataAlgo::setnextscalar(TensorVectorMath::Scalar& scalar)
{
}

void ArrayObjectFieldDataAlgo::setnextvector(TensorVectorMath::Vector& vector)
{
}

void ArrayObjectFieldDataAlgo::setnexttensor(TensorVectorMath::Tensor& tensor)
{
}

void ArrayObjectFieldDataAlgo::reset()
{
}

int ArrayObjectFieldDataAlgo::size()
{
  return(0);
}

bool ArrayObjectFieldDataAlgo::isscalar()
{
  return(false);
}

bool ArrayObjectFieldDataAlgo::isvector()
{
  return(false);
}

bool ArrayObjectFieldDataAlgo::istensor()
{
  return(false);
}

bool ArrayObjectFieldDataAlgo::setfield(SCIRun::FieldHandle handle)
{
  return(false);
}

SCIRun::CompileInfoHandle 
    ArrayObjectFieldDataAlgo::get_compile_info(SCIRun::FieldHandle& field)
{
  const SCIRun::TypeDescription *fieldtype = field->get_type_description();
  const SCIRun::TypeDescription *locationtype = field->order_type_description();
  const SCIRun::TypeDescription *datatype = field->get_type_description(1);

  // As I use my own Tensor and Vector algorithms they need to be
  // converted when reading the data, hence separate algorithms are
  // implemented for those cases
  
  std::string algo_type = "Scalar";  
  if (datatype->get_name() == "Vector") algo_type = "Vector";
  if (datatype->get_name() == "Tensor") algo_type = "Tensor";

  std::string algo_name = "ArrayObjectFieldData" + algo_type + "AlgoT";
  std::string algo_base = "ArrayObjectFieldDataAlgo";

  std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));

  SCIRun::CompileInfoHandle ci = 
    scinew SCIRun::CompileInfo(algo_name + "." +
                       fieldtype->get_filename() + "." +
                       locationtype->get_filename() + ".",
                       algo_base, 
                       algo_name, 
                       fieldtype->get_name() + "," + locationtype->get_name());

  ci->add_include(include_path);
  ci->add_namespace("ModelCreation");
  fieldtype->fill_compile_info(ci.get_rep());
  return(ci);
}
    
SCIRun::CompileInfoHandle 
    ArrayObjectFieldLocationAlgo::get_compile_info(SCIRun::FieldHandle& field)
{
  const SCIRun::TypeDescription *fieldtype = field->get_type_description();
  const SCIRun::TypeDescription *locationtype = field->order_type_description();

  // As I use my own Tensor and Vector algorithms they need to be
  // converted when reading the data, hence separate algorithms are
  // implemented for those cases

  std::string algotype ="Node";
  if (field->basis_order() == 0) algotype = "Elem";

  std::string algo_name = "ArrayObjectFieldLocation"+algotype+"AlgoT";
  std::string algo_base = "ArrayObjectFieldLocationAlgo";

  std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));

  SCIRun::CompileInfoHandle ci = 
    scinew SCIRun::CompileInfo(algo_name + "." +
                       fieldtype->get_filename() + "." +
                       locationtype->get_filename() + ".",
                       algo_base, 
                       algo_name, 
                       fieldtype->get_name());

  // Add in the include path to compile this obj
  ci->add_include(include_path);
  ci->add_namespace("ModelCreation");
  fieldtype->fill_compile_info(ci.get_rep());
  return(ci);
}

SCIRun::CompileInfoHandle 
    ArrayObjectFieldCreateAlgo::get_compile_info(SCIRun::FieldHandle field,std::string datatype, std::string basistype)
{
  if (datatype == "Scalar") datatype = "double";
  if ((datatype == "input")||(datatype == "Same as Input"))
  {
    datatype = field->get_type_description(1)->get_name();
  }

  std::string fieldtype = field->get_type_description(0)->get_name() +
        "<" + datatype + "> ";
  std::string fieldtype_filename = field->get_type_description(0)->get_name() + datatype;


  // As I use my own Tensor and Vector algorithms they need to be
  // converted when reading the data, hence separate algorithms are
  // implemented for those cases

  
  std::string algo_name = "ArrayObjectFieldCreateAlgoT";
  std::string algo_base = "ArrayObjectFieldCreateAlgo";
  if ((basistype == "linear")||(basistype == "Linear")) algo_name = "ArrayObjectFieldCreateNodeAlgoT"; 
  if ((basistype == "constant")||(basistype == "Constant")) algo_name = "ArrayObjectFieldCreateElemAlgoT"; 

  std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));

  SCIRun::CompileInfoHandle ci = 
    scinew SCIRun::CompileInfo(algo_name + "." +
                       fieldtype_filename + ".",
                       algo_base, 
                       algo_name, 
                       fieldtype);

  // Add in the include path to compile this obj
  ci->add_include(include_path);
  ci->add_namespace("ModelCreation");  
  field->get_type_description()->fill_compile_info(ci.get_rep());
  return(ci);
}

void ArrayObjectFieldElemAlgo::getcenter(TensorVectorMath::Vector& node)
{
  node = TensorVectorMath::Vector(0.0,0.0,0.0);
}

void ArrayObjectFieldElemAlgo::getsize(TensorVectorMath::Scalar& size)
{
  size = 0.0;
}

void ArrayObjectFieldElemAlgo::getlength(TensorVectorMath::Scalar& length)
{
  length = 0.0;
}

void ArrayObjectFieldElemAlgo::getarea(TensorVectorMath::Scalar& area)
{
  area = 0.0;
}

void ArrayObjectFieldElemAlgo::getvolume(TensorVectorMath::Scalar& volume)
{
  volume = 0.0;
}

bool ArrayObjectFieldElemAlgo::ispoint()
{
  return(false);
}

bool ArrayObjectFieldElemAlgo::isline()
{
  return(false);
}

bool ArrayObjectFieldElemAlgo::issurface()
{
  return(false);
}

bool ArrayObjectFieldElemAlgo::isvolume()
{
  return(false);
}

bool ArrayObjectFieldElemAlgo::setfield(SCIRun::FieldHandle handle)
{
  return(false);
}

void ArrayObjectFieldElemAlgo::reset()
{
}
 
void ArrayObjectFieldElemAlgo::next()
{
}

int ArrayObjectFieldElemAlgo::size()
{
  return(0);
}

 
void ArrayObjectFieldElemAlgo::getdimension(TensorVectorMath::Scalar& dim)
{
  dim = 0.0;
}    
          
SCIRun::CompileInfoHandle ArrayObjectFieldElemAlgo::get_compile_info(SCIRun::FieldHandle field)
{
  const SCIRun::TypeDescription *fieldtype = field->get_type_description();
  const SCIRun::TypeDescription *locationtype = field->order_type_description();

  // As I use my own Tensor and Vector algorithms they need to be
  // converted when reading the data, hence separate algorithms are
  // implemented for those cases
  
  std::string algo_type = "Point";
  
  if (field->basis_order() == 0)
  {
    SCIRun::MeshHandle mesh = field->mesh().get_rep();
    int dim = mesh->dimensionality();
    if (dim == 1) algo_type = "Line";
    if (dim == 2) algo_type = "Surf";
    if (dim == 3) algo_type = "Volume";
  }
  
  std::string algo_name = "ArrayObjectFieldElem" + algo_type + "AlgoT";
  std::string algo_base = "ArrayObjectFieldElemAlgo";

  std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));

  SCIRun::CompileInfoHandle ci = 
    scinew SCIRun::CompileInfo(algo_name + "." +
                       fieldtype->get_filename() + ".",
                       algo_base, 
                       algo_name, 
                       fieldtype->get_name() + "," + locationtype->get_name());

  ci->add_include(include_path);
  ci->add_namespace("ModelCreation");
  fieldtype->fill_compile_info(ci.get_rep());
  return(ci);
}




} // namespace ModelCreation
