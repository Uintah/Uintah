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


#ifndef CORE_DATATYPES_SELECTIONMASK_H
#define CORE_DATATYPES_SELECTIONMASK_H 1

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Module.h>

namespace ModelCreation {

class SelectionMask {
  public:
    // Create an empty SelectionMask
    SelectionMask();
    
    // Create an empty SelectionMask of a certain size
    SelectionMask(size_t size);
    
    // Create a SelectionMask based on a boolean matrix
    SelectionMask(SCIRun::MatrixHandle& handle);
    
    // Create a SelectionMask based on indices
    SelectionMask(SCIRun::MatrixHandle& handle, size_t size);
    
    // Destructor
    virtual ~SelectionMask();
    
    // Copy constructor
    SelectionMask(const SelectionMask& mask);
    
    // Create a SelectionMask of a certain length
    bool create(size_t size);

    // Create a SelectionMask from a boolean matrix
    bool create(SCIRun::MatrixHandle& matrix);
    
    // Create selection from index vector
    bool create(SCIRun::MatrixHandle& handle, size_t size); 
    
    // Clear() the vector
    void clear();
    
    // get properties
    inline size_t size();
    inline bool   isvalid();
    
    //operators so it works as vector
    inline double& operator[](int idx);    
    inline double* getdataptr();
    
    // get the handle of the underlying object
    inline SCIRun::MatrixHandle gethandle();

    // general functions
    SelectionMask AND(SelectionMask& mask);
    SelectionMask OR(SelectionMask& mask);
    SelectionMask XOR(SelectionMask& mask);
    SelectionMask NOT();
    
    bool get_indices(SCIRun::MatrixHandle& handle);

  private:
    // We keep a handle to the object so it will not get destroyed
    SCIRun::MatrixHandle handle_;
    
    // We also get the physical pointer of where the data is stored
    double* data_;
    size_t  size_;
    
    // In case we do not have a SelectionMask we have something to return
    double  zero_;
};

inline size_t SelectionMask::size() 
{ 
  return(size_); 
}

inline bool SelectionMask::isvalid() 
{ 
  if(size_ == 0) return(false); return(true); 
}

inline double& SelectionMask::operator[](int idx) 
{ 
  if (!data_) return(zero_);  return (data_[idx]); 
}

inline double* SelectionMask::getdataptr() 
{ 
  return(data_); 
}

inline SCIRun::MatrixHandle SelectionMask::gethandle() 
{ 
  return(handle_); 
}

}

#endif
