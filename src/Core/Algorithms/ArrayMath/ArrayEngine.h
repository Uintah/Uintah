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

#ifndef CORE_ALGORITHMS_ARRAYMATH_ARRAYENGINE_H
#define CORE_ALGORITHMS_ARRAYMATH_ARRAYENGINE_H 1

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Containers/HashTable.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>

#include <Core/Algorithms/ArrayMath/ArrayObject.h>
#include <Core/Algorithms/ArrayMath/ArrayObjectFieldAlgo.h>
#include <Core/Algorithms/ArrayMath/share.h>

namespace SCIRunAlgo {


// Engine to combine scalar, vector, and tensor data from both matrices and
// fields.

// To use the engine initialize the class of then engine with the pointer to
// the progress reporter or the module it self.
//
// The function needs a list of input and output objects defined as ArrayObjects
// These objects can be either fields or matrices.

class SCISHARE ArrayEngine {
  
  public:
    // Constructor 
    ArrayEngine(SCIRun::ProgressReporter *pr_ = 0);  
    
    // Destructor
    ~ArrayEngine();

    // Do any arbitrary function with matrices
    bool computesize(ArrayObjectList& Input, int& size);
    bool engine(ArrayObjectList& Input, ArrayObjectList& Output, std::string function);      

  private:
    SCIRun::ProgressReporter *pr_;
    bool free_pr_;
};

}


namespace TensorVectorMath {

class SCISHARE ArrayEngineAlgo : public SCIRun::DynamicAlgoBase
{
public:
  virtual std::string identify() = 0;
  static  SCIRun::CompileInfoHandle get_compile_info(SCIRunAlgo::ArrayObjectList& Input, 
                                             SCIRunAlgo::ArrayObjectList& Output,
                                             std::string function,
                                             int hashoffset);
  virtual void function(SCIRunAlgo::ArrayObjectList& Input,SCIRunAlgo::ArrayObjectList& Output,int n) = 0;                                             
};

} // end namespace

#endif
