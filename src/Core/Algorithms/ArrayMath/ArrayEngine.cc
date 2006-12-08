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

#include <Core/Algorithms/ArrayMath/ArrayEngine.h>

namespace SCIRunAlgo {

ArrayEngine::ArrayEngine(SCIRun::ProgressReporter *pr) :
  pr_(pr),
  free_pr_(false)
{
  if (pr_ == 0)
  {
    pr_ = scinew SCIRun::ProgressReporter;
    free_pr_ = true;
  }
}

ArrayEngine::~ArrayEngine()
{
  if (free_pr_) delete pr_;
}

bool ArrayEngine::computesize(ArrayObjectList& Input, int& size)
{
  size = 1;
  
  for (size_t p=0; p < Input.size(); p++)
  {
    if (!(Input[p].isvalid())) 
    { 
      return(false);
    }
    
    if (Input[p].size() == 0)
    {
      return(false);
    }
    
    if (size == 1)
    {
      size = Input[p].size();
    }
    else
    {
      if ((Input[p].size() != 1)&&(Input[p].size() != size))
      {
        return(false);
      }      
    }
  }

  return(true);
}


bool ArrayEngine::engine(ArrayObjectList& Input, ArrayObjectList& Output, std::string function)
{
  // Do some up front checking on whether the input data is correct
  
  int n = 1;
  
  for (size_t p=0; p < Input.size(); p++)
  {
    if (!(Input[p].isvalid())) 
    { 
      std::ostringstream oss;
      oss << "Input object " << p+1 << " does not seem to be a valid object";
      pr_->error(oss.str());
      return(false);     
    }
    
    if (Input[p].size() == 0)
    {
      std::ostringstream oss;
      oss << "Input object " << p+1 << " is empty";
      pr_->error(oss.str());
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
        oss << "The size of input object " << p+1 << " is not one or does not match the size of the other objects";
        pr_->error(oss.str());
        return(false);
      }      
    }
  }

  if (Output.size() < 1)
  {
    pr_->error("No output matrix/field is given");
    return(false);
  }

  for (size_t p=0; p < Output.size(); p++)
  {
    if (!Output[p].isvalid())
    {
        std::ostringstream oss;
        oss << "Output object " << p << " does not seem to be a valid object";
        pr_->error(oss.str());
        return(false);     
    }
    if ((n != 1)&&(Output[p].size() != n))
    {
        std::ostringstream oss;
        oss << "The size of output object " << p << " is not equal to the number of elements in input";
        pr_->error(oss.str());
        return(false);        
    }
  }

  // Remove white spaces from function
  while (function.size() && isspace(function[function.size()-1])) function.resize(function.size()-1);

  int hoffset = 0;
  SCIRun::Handle<DataArrayMath::ArrayEngineAlgo> algo;
    
  while (1) 
  {
    SCIRun::CompileInfoHandle ci = DataArrayMath::ArrayEngineAlgo::get_compile_info(Input,Output,function,hoffset);
    if (!SCIRun::DynamicCompilation::compile(ci, algo, pr_)) 
    {
      pr_->compile_error(ci->filename_);
      SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
      return(false);
    }

    if (algo->identify() == function) 
    {
      break;
    }
    hoffset++;
  }

  algo->function(Input,Output,n,pr_);
  return(true);
}

} // end namespace

namespace DataArrayMath {

SCIRun::CompileInfoHandle ArrayEngineAlgo::get_compile_info(
        SCIRunAlgo::ArrayObjectList& Input, SCIRunAlgo::ArrayObjectList& Output,
        std::string function, int hashoffset)
{
  unsigned int hashval = SCIRun::Hash(function, 0x7fffffff) + hashoffset;

  // name of include_path
  static const std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  // name of the basis class
  static const std::string base_class_name("ArrayEngineAlgo");

  // Unique filename
  std::string inputtype;
  for (size_t p = 0; p < Input.size(); p++)
  {
      if (Input[p].isfieldscalar())   { inputtype += "FS"; continue; }
      if (Input[p].isfieldtensor())   { inputtype += "FT"; continue; }
      if (Input[p].isfieldvector())   { inputtype += "FV"; continue; }    
      if (Input[p].iselement())       { inputtype += "EL"; continue; }    
      if (Input[p].islocation())      { inputtype += "LC"; continue; }    
      if (Input[p].isindex())         { inputtype += "ID"; continue; }    
      if (Input[p].iscmatrixscalar()) { inputtype += "CS"; continue; }
      if (Input[p].iscmatrixtensor()) { inputtype += "CT"; continue; }
      if (Input[p].iscmatrixvector()) { inputtype += "CV"; continue; }      
      if (Input[p].ismatrixscalar())  { inputtype += "MS"; continue; }
      if (Input[p].ismatrixtensor())  { inputtype += "MT"; continue; }
      if (Input[p].ismatrixvector())  { inputtype += "MV"; continue; }
      inputtype += "UN";
  }

  std::string outputtype;
  for (size_t p = 0; p < Output.size(); p++)
  {
    if (Output[p].ismatrixscalar())  { outputtype += "MS"; continue; }
    if (Output[p].ismatrixtensor())  { outputtype += "MT"; continue; }
    if (Output[p].ismatrixvector())  { outputtype += "MV"; continue; }
    if (Output[p].iscmatrixscalar()) { outputtype += "CS"; continue; }
    if (Output[p].iscmatrixtensor()) { outputtype += "CT"; continue; }
    if (Output[p].iscmatrixvector()) { outputtype += "CV"; continue; }
    if (Output[p].isfieldscalar())   { outputtype += "FS"; continue; }
    if (Output[p].isfieldtensor())   { outputtype += "FT"; continue; }
    if (Output[p].isfieldvector())   { outputtype += "FV"; continue; }
    if (Output[p].islocation())      { outputtype += "LC"; continue; }    
    outputtype += "UN";
  }

  std::string template_name("ALGOArrayEngine_" + SCIRun::to_string(hashval) + "_" + inputtype + "_" + outputtype);

  SCIRun::CompileInfo *ci = scinew SCIRun::CompileInfo(template_name+".",base_class_name,template_name,"double");

  // Code for the function.
  std::string fcn;
  
  fcn = std::string("namespace DataArrayMath {\ntemplate <class DATATYPE>\n") +
    "class " + template_name + " : public ArrayEngineAlgo\n" +
    "{\nvirtual void function(SCIRunAlgo::ArrayObjectList& input_," + 
    "SCIRunAlgo::ArrayObjectList& output_, int size_, SCIRun::ProgressReporter* pr_)\n  {\n";
  fcn += "    DATATYPE dummy_ = 0.0; dummy_ += 1.0;\n\n";   // Make compiler happy
  
  for (size_t p = 0; p< Input.size(); p++)
  {
    if (Input[p].ismatrixscalar())  fcn += "    DataArrayMath::Scalar " + Input[p].getname() + ";\n";
    if (Input[p].ismatrixvector())  fcn += "    DataArrayMath::Vector " + Input[p].getname() + ";\n";
    if (Input[p].iscmatrixtensor()) fcn += "    DataArrayMath::Tensor " + Input[p].getname() + ";\n";
    if (Input[p].iscmatrixscalar()) fcn += "    DataArrayMath::Scalar " + Input[p].getname() + ";\n";
    if (Input[p].iscmatrixvector()) fcn += "    DataArrayMath::Vector " + Input[p].getname() + ";\n";
    if (Input[p].ismatrixtensor())  fcn += "    DataArrayMath::Tensor " + Input[p].getname() + ";\n";
    if (Input[p].isfieldscalar())   fcn += "    DataArrayMath::Scalar " + Input[p].getname() + ";\n";
    if (Input[p].isfieldvector())   fcn += "    DataArrayMath::Vector " + Input[p].getname() + ";\n";
    if (Input[p].isfieldtensor())   fcn += "    DataArrayMath::Tensor " + Input[p].getname() + ";\n";
    if (Input[p].islocation())    { fcn += "    DataArrayMath::Scalar " + Input[p].getxname() + ";\n";
                                    fcn += "    DataArrayMath::Scalar " + Input[p].getyname() + ";\n";
                                    fcn += "    DataArrayMath::Scalar " + Input[p].getzname() + ";\n";
                                    fcn += "    DataArrayMath::Vector " + Input[p].getname() + ";\n"; }
    if (Input[p].isindex())       { fcn += "    DataArrayMath::Scalar " + Input[p].getname() + ";\n";                                   
                                    fcn += "    DataArrayMath::Scalar " + Input[p].getsizename() + " = static_cast<DataArrayMath::Scalar>(size_);\n"; }
    if (Input[p].iselement())     { std::ostringstream oss; oss << p;
                                    fcn += "    DataArrayMath::Element " + Input[p].getname() + ";\n";
                                    fcn += "    input_["+oss.str()+"].getelement("+Input[p].getname() +");\n";  }                                  
  }

  for (size_t p = 0; p< Output.size(); p++)
  {
    if (Output[p].iscmatrixscalar()) fcn += "    DataArrayMath::Scalar " + Output[p].getname() + ";\n";
    if (Output[p].iscmatrixvector()) fcn += "    DataArrayMath::Vector " + Output[p].getname() + ";\n";
    if (Output[p].iscmatrixtensor()) fcn += "    DataArrayMath::Tensor " + Output[p].getname() + ";\n";
    if (Output[p].ismatrixscalar())  fcn += "    DataArrayMath::Scalar " + Output[p].getname() + ";\n";
    if (Output[p].ismatrixvector())  fcn += "    DataArrayMath::Vector " + Output[p].getname() + ";\n";
    if (Output[p].ismatrixtensor())  fcn += "    DataArrayMath::Tensor " + Output[p].getname() + ";\n";
    if (Output[p].isfieldscalar())   fcn += "    DataArrayMath::Scalar " + Output[p].getname() + ";\n";
    if (Output[p].isfieldvector())   fcn += "    DataArrayMath::Vector " + Output[p].getname() + ";\n";
    if (Output[p].isfieldtensor())   fcn += "    DataArrayMath::Tensor " + Output[p].getname() + ";\n";
    if (Output[p].islocation())      fcn += "    DataArrayMath::Vector " + Output[p].getname() + ";\n";
  }
  
  fcn += "\n\n";
  fcn += "int cnt_ = 0;";
  fcn += "    for (int p_ = 0; p_ < size_; p_++)\n    {\n";
  
  for (size_t p = 0; p< Input.size(); p++)
  {
    std::ostringstream oss; oss << p;
    if (Input[p].ismatrixscalar())  fcn += "      input_[" + oss.str() + "].getnextmatrixscalar(" + Input[p].getname() + ");\n";
    if (Input[p].ismatrixvector())  fcn += "      input_[" + oss.str() + "].getnextmatrixvector(" + Input[p].getname() + ");\n";
    if (Input[p].ismatrixtensor())  fcn += "      input_[" + oss.str() + "].getnextmatrixtensor(" + Input[p].getname() + ");\n";
    if (Input[p].iscmatrixscalar()) fcn += "      input_[" + oss.str() + "].getmatrixscalar(" + Input[p].getname() + ");\n";
    if (Input[p].iscmatrixvector()) fcn += "      input_[" + oss.str() + "].getmatrixvector(" + Input[p].getname() + ");\n";
    if (Input[p].iscmatrixtensor()) fcn += "      input_[" + oss.str() + "].getmatrixtensor(" + Input[p].getname() + ");\n";
    if (Input[p].isfieldscalar())   fcn += "      input_[" + oss.str() + "].getnextfieldscalar(" + Input[p].getname() + ");\n";
    if (Input[p].isfieldvector())   fcn += "      input_[" + oss.str() + "].getnextfieldvector(" + Input[p].getname() + ");\n";
    if (Input[p].isfieldtensor())   fcn += "      input_[" + oss.str() + "].getnextfieldtensor(" + Input[p].getname() + ");\n";
    if (Input[p].islocation())    { fcn += "      input_[" + oss.str() + "].getnextfieldlocation(" + Input[p].getname() + ");\n"; 
                                    fcn += "      " + Input[p].getxname() + "=" + Input[p].getname() + ".x();\n";
                                    fcn += "      " + Input[p].getyname() + "=" + Input[p].getname() + ".y();\n";
                                    fcn += "      " + Input[p].getzname() + "=" + Input[p].getname() + ".z();\n"; }
    if (Input[p].isindex())         fcn += "      " + Input[p].getname() + "= static_cast<double>(p_);\n";
  }
  
  fcn += "\n\n";
  fcn += "      " + function + " \n\n\n";

  for (size_t p = 0; p< Input.size(); p++)
  {
    std::ostringstream oss; oss << p;
    if (Input[p].iselement())        fcn += "      " + Input[p].getname() + ".next();\n";
  }

  for (size_t p = 0; p< Output.size(); p++)
  {
    std::ostringstream oss; oss << p;
    if (Output[p].ismatrixscalar())  fcn += "     output_[" + oss.str() + "].setnextmatrixscalar(" + Output[p].getname() + ");\n";
    if (Output[p].ismatrixvector())  fcn += "     output_[" + oss.str() + "].setnextmatrixvector(" + Output[p].getname() + ");\n";
    if (Output[p].ismatrixtensor())  fcn += "     output_[" + oss.str() + "].setnextmatrixtensor(" + Output[p].getname() + ");\n";
    if (Output[p].iscmatrixscalar()) fcn += "     output_[" + oss.str() + "].setmatrixscalar(" + Output[p].getname() + ");\n";
    if (Output[p].iscmatrixvector()) fcn += "     output_[" + oss.str() + "].setmatrixvector(" + Output[p].getname() + ");\n";
    if (Output[p].iscmatrixtensor()) fcn += "     output_[" + oss.str() + "].setmatrixtensor(" + Output[p].getname() + ");\n";
    if (Output[p].isfieldscalar())   fcn += "     output_[" + oss.str() + "].setnextfieldscalar(" + Output[p].getname() + ");\n";
    if (Output[p].isfieldvector())   fcn += "     output_[" + oss.str() + "].setnextfieldvector(" + Output[p].getname() + ");\n";
    if (Output[p].isfieldtensor())   fcn += "     output_[" + oss.str() + "].setnextfieldtensor(" + Output[p].getname() + ");\n";
    if (Output[p].islocation())      fcn += "     output_[" + oss.str() + "].setnextfieldlocation(" + Output[p].getname() + ");\n"; 

  }
  
  fcn += std::string("    cnt_++; if (cnt_ == 200) { cnt_ = 0; pr_->update_progress(p_,size_); }\n");
  
  fcn += std::string("    }\n  }\n\n") +
    "  virtual std::string identify()\n" +
    "  { return std::string(\"" + std::string(SCIRun::string_Cify(function)) + 
    "\"); }\n};\n\n}\n";

  // Add in the include path to compile this obj
  ci->add_include(include_path);
  ci->add_post_include(fcn);
  ci->add_namespace("DataArrayMath");

  return(ci);
}


} // End namespace SCIRun
