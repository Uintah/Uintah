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

#include <Packages/ModelCreation/Core/Algorithms/TVMEngine.h>

namespace TensorVectorMath {


TVMArray::TVMArray()
{
  matrix_ = 0;
  name_   = "";
  data_   = 0;
  nrows_  = 0;
  ncols_  = 0;
  idx_    = 0;
}

TVMArray::TVMArray(SCIRun::MatrixHandle& matrix,std::string name)
{
  matrix_ = matrix;
  name_   = name;
  data_   = 0;
  nrows_  = 0;
  ncols_  = 0;
  idx_    = 0;
   
  matrix_ = dynamic_cast<SCIRun::Matrix *>(matrix->dense());
  if (matrix_.get_rep())
  {
    ncols_ = matrix_->ncols();
    nrows_ = matrix_->nrows();
    data_  = matrix_->get_data_pointer();
    if (nrows_ == 1) idx_ = -1; else idx_ = 0;
  }
}

TVMArray::TVMArray(int ncols,std::string name)
{
  matrix_ = 0;
  name_   = name;
  data_   = 0;
  nrows_  = 0;
  ncols_  = ncols;
  idx_    = 0;
}

void TVMArray::create(int numelems)
{
  if ((matrix_.get_rep() == 0)&&(ncols_ > 0))
  {
    matrix_ = dynamic_cast<SCIRun::Matrix* >(scinew SCIRun::DenseMatrix(numelems,ncols_));
    if (matrix_.get_rep() != 0)
    {
      data_ = matrix_->get_data_pointer();
      nrows_ = numelems;
      if (nrows_ == 1) idx_ = -1; else idx_ = 0;
    }
  }
}



TVMEngine::TVMEngine(SCIRun::Module *module) :
  module_(module)
{
}

TVMEngine::~TVMEngine()
{
}

bool TVMEngine::engine(TVMArrayList& Input, TVMArrayList& Output, std::string function, int num)
{
  // Do some up front checking on whether the input data is correct
  
  
  int n = 1;
  
  for (int p=0; p < Input.size(); p++)
  {
    if (!(Input[p].isvalid())) 
    { 
      std::ostringstream oss;
      oss << "Input matrix " << p << " is not a ScalarArray, VectorArray, or TensorArray";
      error(oss.str());
      return(false);
    }
    
    if (Input[p].size() == 0)
    {
      std::ostringstream oss;
      oss << "Input matrix " << p << " is empty";
      error(oss.str());
      return(false);
    }
    
    if (n == 1)
    {
      n = Input[p].size();
    }
    else
    {
      if ((Input[p].size() != 1)&&(Input[p].size() != n))
      {
        std::ostringstream oss;
        oss << "The number of elements in input matrix " << p << " is not equal to 1 or the number of elements in the other arrays";
        error(oss.str());
        return(false);
      }      
    }
  }

  if ((num != -1)&&(n == 1)) n = num;

  if (Output.size() < 1)
  {
    error("No output matrix is given");
    return(false);
  }

  for (int p=0; p < Output.size(); p++)
  {
    if (!Output[p].isvalid()) Output[p].create(num);
    if (!Output[p].isvalid())
    {
        std::ostringstream oss;
        oss << "Output matrix " << p << " cannot be generated";
        error(oss.str());
        return(false);     
    }
    if (Output[p].size() != num)
    {
        std::ostringstream oss;
        oss << "The size of output matrix " << p << " is not equal to the number of elements in the input matrix";
        error(oss.str());
        return(false);        
    }
  }

  // Remove white spaces from function
  while (function.size() && isspace(function[function.size()-1])) function.resize(function.size()-1);

  int hoffset = 0;
  SCIRun::Handle<TensorVectorMath::TVMEngineAlgo> algo;
    
  while (1) 
  {
    SCIRun::CompileInfoHandle ci = TVMEngineAlgo::get_compile_info(Input,Output,function,hoffset);
    if (!SCIRun::DynamicCompilation::compile(ci, algo, false, module_)) 
    {
      
      error("Function did not compile.");
      if (module_)
      {
        SCIRun::GuiInterface* gui = module_->get_gui();
        gui->eval(module_->get_id() + " compile_error "+ci->filename_);
      }
      SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
      return(false);
    }

    if (algo->identify() == function) 
    {
      break;
    }
    hoffset++;
  }

  algo->function(Input,Output,n);
  return(true);
}


SCIRun::CompileInfoHandle TVMEngineAlgo::get_compile_info(
        TVMArrayList& Input, TVMArrayList& Output,
        std::string function, int hashoffset)

{
  unsigned int hashval = SCIRun::Hash(function, 0x7fffffff) + hashoffset;

  // name of include_path
  static const std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  // name of the basis class
  static const std::string base_class_name("TVMEngineAlgo");

  // Unique filename
  std::string inputtype;
  for (int p = 0; p < Input.size(); p++)
  {
    TVMArray TA = Input[p];
    if (TA.isscalar()) { inputtype += "S"; continue; }
    if (TA.istensor()) { inputtype += "T"; continue; }
    if (TA.isvector()) { inputtype += "V"; continue; }
    inputtype += "U";
  }

  std::string outputtype;
  for (int p = 0; p < Output.size(); p++)
  {
    TVMArray TA = Output[p];
    if (TA.isscalar()) { outputtype += "S"; continue; }
    if (TA.istensor()) { outputtype += "T"; continue; }
    if (TA.isvector()) { outputtype += "V"; continue; }
    outputtype += "U";
  }

  std::string template_name("TVMEngine_" + SCIRun::to_string(hashval) + "_" + inputtype + "_" + outputtype);

  SCIRun::CompileInfo *rval = scinew SCIRun::CompileInfo(template_name+".",base_class_name,template_name,"double");

  // Code for the function.
  std::string fcn;
  
  fcn = std::string("template <class DATATYPE>\n") +
    "class " + template_name + " : public TVMEngineAlgo\n" +
    "{\nvirtual void function(TensorVectorMath::TVMArrayList& input_," + 
    "TensorVectorMath::TVMArrayList& output_, int size_)\n  {\n";
  fcn += "    DATATYPE dummy_ = 0.0; dummy_ += 1.0;\n\n";   // Make compiler happy
  
  for (int p = 0; p< Input.size(); p++)
  {
    if (Input[p].isscalar()) fcn += "    TensorVectorMath::Scalar " + Input[p].getname() + ";\n";
    if (Input[p].isvector()) fcn += "    TensorVectorMath::Vector " + Input[p].getname() + ";\n";
    if (Input[p].istensor()) fcn += "    TensorVectorMath::Tensor " + Input[p].getname() + ";\n";
  }

  fcn += "    TensorVectorMath::Scalar INDEX;\n";
  fcn += "    TensorVectorMath::Scalar SIZE=static_cast<TensorVectorMath::Scalar>(size_);\n";

  for (int p = 0; p< Output.size(); p++)
  {
    if (Output[p].isscalar()) fcn += "    TensorVectorMath::Scalar " + Output[p].getname() + ";\n";
    if (Output[p].isvector()) fcn += "    TensorVectorMath::Vector " + Output[p].getname() + ";\n";
    if (Output[p].istensor()) fcn += "    TensorVectorMath::Tensor " + Output[p].getname() + ";\n";
  }
  
  fcn += "\n";
  fcn += "    for (int p_ = 0; p_ < size_; p_++)\n    {\n";
  fcn += "      INDEX = static_cast<double>(p_);\n";
  
  for (int p = 0; p< Input.size(); p++)
  {
    std::ostringstream oss; oss << p;
    if (Input[p].isscalar()) fcn += "      input_[" + oss.str() + "].getnextscalar(" + Input[p].getname() + ");\n";
    if (Input[p].isvector()) fcn += "      input_[" + oss.str() + "].getnextvector(" + Input[p].getname() + ");\n";
    if (Input[p].istensor()) fcn += "      input_[" + oss.str() + "].getnexttensor(" + Input[p].getname() + ");\n";
  }
  
  fcn += "\n";
  fcn += "      " + function + " \n\n";

  for (int p = 0; p< Output.size(); p++)
  {
    std::ostringstream oss; oss << p;
    if (Output[p].isscalar()) fcn += "     output_[" + oss.str() + "].setnextscalar(" + Output[p].getname() + ");\n";
    if (Output[p].isvector()) fcn += "     output_[" + oss.str() + "].setnextvector(" + Output[p].getname() + ");\n";
    if (Output[p].istensor()) fcn += "     output_[" + oss.str() + "].setnexttensor(" + Output[p].getname() + ");\n";
  }
  
  fcn += std::string("    }\n  }\n\n") +
    "  virtual std::string identify()\n" +
    "  { return std::string(\"" + std::string(SCIRun::string_Cify(function)) + 
    "\"); }\n};\n\n";

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_post_include(fcn);
  rval->add_namespace("TensorVectorMath");

  return rval;
}


} // End namespace SCIRun
