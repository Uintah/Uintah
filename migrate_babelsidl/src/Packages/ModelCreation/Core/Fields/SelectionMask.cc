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



/*
 * SelectionMask.cc
 */

#include <Packages/ModelCreation/Core/Fields/SelectionMask.h>


namespace ModelCreation 
{

using namespace std;

SelectionMask::SelectionMask() :
  handle_(0), data_(0), size_(0), zero_(0)
{
}

SelectionMask::SelectionMask(size_t size) :
  handle_(0), data_(0), size_(0), zero_(0.0)
{
  create(size);
}

bool SelectionMask::create(size_t size)
{
  handle_ = 0;
  data_ = 0;
  size_ = 0;
  
  handle_ = dynamic_cast<SCIRun::Matrix *>(scinew SCIRun::DenseMatrix(size,1));
  if (handle_.get_rep() == 0) return(false);
  
  data_ = handle_->get_data_pointer(); 
  size_ = size;
  for (size_t j=0;j<size;j++) data_[j] = 0.0;
  
  return(true);
}

SelectionMask::SelectionMask(SCIRun::MatrixHandle& handle) :
  handle_(0), data_(0), size_(0), zero_(0.0)
{
  create(handle);
}

SelectionMask::SelectionMask(SCIRun::MatrixHandle& handle,size_t size)
{
  create(handle,size);
}

bool SelectionMask::create(SCIRun::MatrixHandle& handle,size_t size)
{
  handle_ = 0;
  data_ = 0;
  size_ = 0;
 
   if (handle.get_rep() == 0) return(false);

  handle = handle->dense();
  if (handle->get_data_pointer() == 0) return(false);

  int s = handle->get_data_size();
  double* data = handle->get_data_pointer();

  if (size == -1)
  {
    double maxind = 0.0;
    for (int i=0; i<s; i++)
    {
      if (data[i] > maxind) maxind = data[i];
    }
    size = static_cast<int>(maxind)+1;
  }
  
  create(static_cast<size_t>(size));
  
  int idx;  
  for (int i=0; i<s; i++)
  {
      idx = static_cast<int>(data[i]);
      if ((idx>=0)&&(idx<(size+1))) data_[idx] = 1.0;
  }
  
  return(true);
}


// Function uses the same NRRD if data is in the proper format, if not it
// will do an internal conversion and add a new NRRD in the proper format
bool SelectionMask::create(SCIRun::MatrixHandle& handle)
{
  handle_ = 0;
  data_ = 0;
  size_ = 0;
 
  if (handle.get_rep() == 0) return(false);
  if (handle->ncols() != 1) return(false);
  
  handle_ = handle->dense();
  data_   = handle->get_data_pointer();
  size_   = handle->get_data_size();
  
  return(true);
}

SelectionMask::~SelectionMask()
{
  handle_ = 0;
  data_ = 0;
  size_ = 0;
}

SelectionMask::SelectionMask(const SelectionMask& mask)
{
  handle_ = mask.handle_;
  data_ = mask.data_;
  size_ = mask.size_;
}

void SelectionMask::clear()
{
  handle_ = 0;
  data_ = 0;
  size_ = 0;
}

SelectionMask SelectionMask::AND(SelectionMask& mask)
{
  SelectionMask newmask;
  
  if (!isvalid()||(!mask.isvalid())) return(newmask);
  if ((size() != mask.size())&&(size()!=1)&&(mask.size()!=1)) return(newmask);
  
  if (size() > mask.size())
  {
    newmask.create(size());
  }
  else
  {
    newmask.create(mask.size());
  }
    
  if (!newmask.isvalid()) return(newmask);  
    
  double* data = mask.getdataptr();
  double* newdata = newmask.getdataptr();
  
  if (size() == mask.size())
  {
    for (size_t j = 0; j <size();j++) 
    {
      if( data_[j] && data[j] ) newdata[j] = 1.0; else newdata[j] = 0.0;
    }
  }
  
  if ((size() == 1)&&(mask.size() > 1))
  {
    for (size_t j = 0; j <mask.size();j++) 
    {
      if( data_[0] && data[j] ) newdata[j] = 1.0; else newdata[j] = 0.0;
    }  
  }

  if ((mask.size() == 1)&&(size() > 1))
  {
    for (size_t j = 0; j <size();j++) 
    {
      if( data_[j] && data[0] ) newdata[j] = 1.0; else newdata[j] = 0.0;
    }  
  }
  
  return(newmask);
}


SelectionMask SelectionMask::OR(SelectionMask& mask)
{
  SelectionMask newmask;
  
  if (!isvalid()||(!mask.isvalid())) return(newmask);
  if ((size() != mask.size())&&(size()!=1)&&(mask.size()!=1)) return(newmask);
  
  if (size() > mask.size())
  {
    newmask.create(size());
  }
  else
  {
    newmask.create(mask.size());
  }
    
  if (!newmask.isvalid()) return(newmask);  
    
  double* data = mask.getdataptr();
  double* newdata = newmask.getdataptr();
  
  if (size() == mask.size())
  {
    for (size_t j = 0; j <size();j++) 
    {
      if( data_[j] || data[j] ) newdata[j] = 1.0; else newdata[j] = 0.0;
    }
  }
  
  if ((size() == 1)&&(mask.size() > 1))
  {
    for (size_t j = 0; j <mask.size();j++) 
    {
      if( data_[0] || data[j] ) newdata[j] = 1.0; else newdata[j] = 0.0;
    }  
  }

  if ((mask.size() == 1)&&(size() > 1))
  {
    for (size_t j = 0; j <size();j++) 
    {
      if( data_[j] || data[0] ) newdata[j] = 1.0; else newdata[j] = 0.0;
    }  
  }
  
  return(newmask);
}

SelectionMask SelectionMask::XOR(SelectionMask& mask)
{
  SelectionMask newmask;
  
  if (!isvalid()||(!mask.isvalid())) return(newmask);
  if ((size() != mask.size())&&(size()!=1)&&(mask.size()!=1)) return(newmask);
  
  if (size() > mask.size())
  {
    newmask.create(size());
  }
  else
  {
    newmask.create(mask.size());
  }
    
  if (!newmask.isvalid()) return(newmask);  
    
  double* data = mask.getdataptr();
  double* newdata = newmask.getdataptr();
  
  if (size() == mask.size())
  {
    for (size_t j = 0; j <size();j++) 
    {
      if( (data_[j]!=0.0) xor (data[j]!=0.0) ) newdata[j] = 1.0; else newdata[j] = 0.0;
    }
  }
  
  if ((size() == 1)&&(mask.size() > 1))
  {
    for (size_t j = 0; j <mask.size();j++) 
    {
      if( (data_[0]!=0.0) xor (data[j]!=0.0) ) newdata[j] = 1.0; else newdata[j] = 0.0;
    }  
  }

  if ((mask.size() == 1)&&(size() > 1))
  {
    for (size_t j = 0; j <size();j++) 
    {
      if( (data_[j]!=0.0) xor (data[0]!=0.0) ) newdata[j] = 1.0; else newdata[j] = 0.0;
    }  
  }
  
  return(newmask);}

SelectionMask SelectionMask::NOT()
{
  SelectionMask newmask;
  if ((!isvalid())||(size() == 0)) return(newmask);
  newmask.create(size());
  if (!newmask.isvalid()) return(newmask);
  
  double* newdata = newmask.getdataptr();
  for (size_t j=0;j<size_; j++) 
  {
    if (data_[j]) {newdata[j] = 0.0;} else {newdata[j] = 1.0;}
  }
  return(newmask);
}  


bool SelectionMask::get_indices(SCIRun::MatrixHandle& handle)
{
  if (!isvalid()) return(false);

  int numindices = 0;
  for (int i=0; i < size_; i++) if (data_[i]) numindices++;
  
  handle = dynamic_cast<SCIRun::Matrix *>(scinew SCIRun::DenseMatrix(numindices,1));
  if (handle.get_rep() == 0) { return(false); }

  double *data = handle->get_data_pointer();
  if (data == 0) { handle = 0; return(false); }

  numindices = 0;
  for (int i=0; i < size_; i++) 
  {
    if (data_[i]) data[numindices++] = static_cast<double>(i);
  }
  
  return(true);
}

} // end namespace
