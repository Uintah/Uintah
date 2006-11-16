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

// Get all the class definitions. 
#include <Core/Algorithms/Fields/ConvertMeshToPointCloud.h>


// As we already included DynamicAlgorithm.h in the header we do not need to
// include it again here.

namespace SCIRunAlgo {

using namespace SCIRun;

// Implementation of the actual access point to the algorithm

bool ConvertMeshToPointCloudAlgo::ConvertMeshToPointCloud(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{

  // Step 0:
  // Safety test:
  // Test whether we received actually a field. A handle can point to no object.
  // Using a null handle will cause the program to crash. Hence it is a good
  // policy to check all incoming handles and to see whether they point to actual
  // objects.
  
  // Handle: the function get_rep() returns the pointer contained in the handle
  if (input.get_rep() == 0)
  {
    // If we encounter a null pointer we return an error message and return to
    // the program to deal with this error. 
    pr->error("ConvertMeshToPointCloud: No input field");
    return (false);
  }

  // Step 1: determine the type of the input fields and determine what type the
  // output field should be.
  
  // FieldInformation is a helper class that will store all the names of all the
  // components a field is made of. It takes a handle to a field and then
  // determines what the actual type is of the field.
  
  // As the current Field class has a variety of functions to query for its type
  // the FieldInformation object will do this for you and will contain a summary of all
  // the type information.
  
  // As the output field will be a variation on the input field we initialize 
  // both with the input handle.
  FieldInformation fi(input);
  FieldInformation fo(input);
  
  // Recent updates to the software allow for quadratic and cubic hermitian 
  // representations. However these methods have not fully been exposed yet.
  // Hence the iterators in the field will not consider the information needed
  // to define these non-linear elements. And hence although the algorithm may
  // provide output for these cases and may not fail, the output is mathematically
  // improper and hence for a proper implementation we have to wait until the
  // mesh and field classes are fully completed.
  
  // Here we test whether the class is part of any of these newly defined 
  // non-linear classes. If so we return an error.
  if (fi.is_nonlinear())
  {
    pr->error("ConvertMeshToPointCloud: This function has not yet been defined for non-linear elements yet");
    return (false);
  }

  // If the mesh is already a PointCloud, we only need to copy the input to the
  // output. No algorithm is needed in this case.
  if (fi.get_mesh_type() == "PointCloudMesh")
  {
    output = input;
    return (true);
  }
  
  // Define the output type of the data
  fo.set_mesh_type("PointCloudMesh");
  fo.set_mesh_basis_type("ConstantBasis");
  
  if (fi.get_basis_type() == "ConstantBasis") 
  {
    // Inform the user of some possible unintented consequence:
    pr->remark("Data is defined at the elements: hence removing the data from the field");
    // Though we can still perform the operation
    fo.set_basis_type("NoDataBasis");
  }
  
  // Step 3: Build information structure for the dynamic compilation
  
  // The only object we need to build to perform a dynamic compilation is the
  // CompileInfo. This object is created and we use the handle to the object
  // to pass the data structure around. 
  
  // CompileInfo object:
  // The constructor needs the following information:
  //  1) an unique filename descriptor which can used in the on-the-fly-libs
  //     directory. The FieldInformation object has a function that renders an
  //     unique name for each field type.
  //  2) The name of the base class
  //  3) The name of the templated class without template descriptors
  //  4) The template descriptors separated by commas
   
  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOConvertMeshToPointCloud."+fi.get_field_filename()+"."+fo.get_field_filename()+".",
    "ConvertMeshToPointCloudAlgo","ConvertMeshToPointCloudAlgoT",
    fi.get_field_name() + "," + fo.get_field_name());

  // The dynamic algorithm will be created by writing a small piece of code in
  // a .cc file. This file needs to know which file to include for the definitions
  // of this algorithm. 
  ci->add_include(TypeDescription::cc_to_h(__FILE__));

  // This function is defined in the namespace SCIRunAlgo, add a statement
  // 'using namespace SCIRunAlgo' to the dynamic file to be created.
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  // In order to be able to compile the dynamic code it needs to include the
  // descriptions of the mesh/field classes. The following two statements will
  // add the proper include files for both the input and output field types 
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;    
  
  // Step 3: Build the dynamic algorithm 
  
  // Create an access point to the dynamically compiled algorithm
  // Note: this is currently a handle to the base class algorithm.
  SCIRun::Handle<ConvertMeshToPointCloudAlgo> algo;
  
  // Dynamically compile the algorithm. 
  // If the function is a success: algo will point to the dynamically
  // compiled algorithm. Since the access function is virtual executing it will
  // invoke the dynamic version.
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    // In case we detect an error: we forward the error to the user
    // The current system will take the filename of the file that failed to compile
    // It will display the error and dynamic file to the user, in the hope it
    // will tell something on what went wrong
    pr->compile_error(ci->filename_);
    
    // If compilation failed: remove file from on-the-fly-libs directory
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  // Finally step 4: invoke dynamic algorithm
  
  // Depending on whether dynamic algorithm fails or succeeds, false or true
  // is returned. 
  // As error messages are reportered to the ProgressReporter we do not need to
  // handle any error messages here, they automatically are forwarded to the user. 
  return(algo->ConvertMeshToPointCloud(pr,input,output));
}

} // End namespace SCIRunAlgo

