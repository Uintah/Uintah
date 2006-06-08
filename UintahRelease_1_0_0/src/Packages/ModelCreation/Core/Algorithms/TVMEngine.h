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


#ifndef CORE_ALGORITHMS_TVMENGINE_H
#define CORE_ALGORITHMS_TVMENGINE_H 1

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
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Module.h>

#include <Packages/ModelCreation/Core/Algorithms/TVMMath.h>

// TVM stands for Tensor Vector Array Math

namespace TensorVectorMath {

// This class converts a normal Matrix into a Tensor/Vector/Scalar array 

class TVMArray;
class TVMEngine;
class TVMEngineAlgo;
typedef std::vector<TVMArray> TVMArrayList;


class TVMEngine {
  
  public:
    // Constructor 
    TVMEngine(SCIRun::Module *module_ = 0);  
    
    // Destructor
    virtual ~TVMEngine();

    // Do any arbitrary function with matrices
    bool engine(TVMArrayList& Input, TVMArrayList& Output, std::string function, int n = -1);      

  private:
    SCIRun::Module *module_;
    
    inline void error(std::string error);
    inline void warning(std::string warning);
};

class TVMArray {

  public:
    TVMArray();
    TVMArray(SCIRun::MatrixHandle& matrix,std::string name);
    TVMArray(int ncols,std::string name);
    
    inline bool    isvalid();
    inline bool    isscalar();
    inline bool    istensor();
    inline bool    isvector();
    inline bool    isarray();
    inline int     size();
    
    inline void    getnextscalar(double& d);
    inline void    getnexttensor(Tensor& t);
    inline void    getnextvector(Vector& v);
    inline void    setnextscalar(double& d);
    inline void    setnexttensor(Tensor& t);
    inline void    setnextvector(Vector& v);
    
    inline std::string getname();
    inline void        reset();
    
    void create(int numelems);
    
  private:
    SCIRun::MatrixHandle  matrix_;
    std::string   name_;
    double*       data_;
    int           nrows_;
    int           ncols_;
    int           idx_;
};



inline bool TVMArray::isvalid()
{
  return((matrix_.get_rep() != 0));
}

inline bool TVMArray::isscalar()
{
  return((ncols_  == 1));
}

inline bool TVMArray::isvector()
{
  return((ncols_  == 3));
}

inline bool TVMArray::istensor()
{
  return((ncols_  == 6)||(ncols_ == 9));
}

inline bool TVMArray::isarray()
{
  return((nrows_ > 1));
}

inline int TVMArray::size()
{
  return(nrows_);
}

inline std::string TVMArray::getname()
{
  return(name_);
}

inline void TVMArray::reset()
{
  if (nrows_ == 1) idx_ = -1; else idx_ = 0;
}


inline void TVMArray::getnextscalar(double& d)
{
  if (idx_ != -1) d = data_[idx_++]; else d = data_[0];
}

inline void TVMArray::getnexttensor(Tensor& t)
{
  if (ncols_ == 6)
  {
    if (idx_ != -1) { t = Tensor(data_[idx_],data_[idx_+1],data_[idx_+2],data_[idx_+3],data_[idx_+4],data_[idx_+5]); idx_+=6; } 
    else { t = Tensor(data_[0],data_[1],data_[2],data_[3],data_[4],data_[5]); }
  }
  else
  {
    if (idx_ != -1) { t = Tensor(data_[idx_],data_[idx_+1],data_[idx_+2],data_[idx_+4],data_[idx_+5],data_[idx_+8]); idx_+=9; } 
    else { t = Tensor(data_[0],data_[1],data_[2],data_[4],data_[5],data_[8]); }  
  }
}

inline void TVMArray::getnextvector(Vector& t)
{
  if (idx_ != -1) { t = Vector(data_[idx_],data_[idx_+1],data_[idx_+2]); idx_+=3; } 
  else { t = Vector(data_[0],data_[1],data_[2]); }
}

inline void TVMArray::setnextscalar(double& d)
{
  if (idx_ != -1) data_[idx_++] = d; else {data_[0] = d;}
}

inline void TVMArray::setnexttensor(Tensor& t)
{
  if (idx_ != -1)
  {
    if (ncols_ == 6)
    {
      data_[idx_++] = t.xx();  
      data_[idx_++] = t.xy();  
      data_[idx_++] = t.xz();  
      data_[idx_++] = t.yy();  
      data_[idx_++] = t.yz();  
      data_[idx_++] = t.zz();  
    }
    else
    {
      data_[idx_++] = t.xx();  
      data_[idx_++] = t.xy();  
      data_[idx_++] = t.xz();  
      data_[idx_++] = t.xy();  
      data_[idx_++] = t.yy();  
      data_[idx_++] = t.yz();
      data_[idx_++] = t.xz();  
      data_[idx_++] = t.yz();        
      data_[idx_++] = t.zz();  
    }  
  }
  else
    {
    if (ncols_ == 6)
    {
      data_[0] = t.xx();  
      data_[1] = t.xy();  
      data_[2] = t.xz();  
      data_[3] = t.yy();  
      data_[4] = t.yz();  
      data_[5] = t.zz();  
    }
    else
    {
      data_[0] = t.xx();  
      data_[1] = t.xy();  
      data_[2] = t.xz();  
      data_[3] = t.xy();  
      data_[4] = t.yy();  
      data_[5] = t.yz();
      data_[6] = t.xz();  
      data_[7] = t.yz();        
      data_[8] = t.zz();  
    }  
  }
}

inline void TVMArray::setnextvector(Vector& v)
{
  if (idx_ != -1)
  {
    data_[idx_++] = v.x();
    data_[idx_++] = v.y();
    data_[idx_++] = v.z();
  }
  else
  {
    data_[0] = v.x();
    data_[1] = v.y();
    data_[2] = v.z();  
  }
}


inline void TVMEngine::error(std::string error)
{
  if(module_) 
  {
    module_->error(error);
  }
  else
  {
    std::cerr << "ERROR: " << error << std::endl;
  }
}

inline void TVMEngine::warning(std::string warning)
{
  if(module_) 
  {
    module_->warning(warning);
  }
  else
  {
    std::cerr << "WARNING: " << warning << std::endl;
  }
}

class TVMEngineAlgo : public SCIRun::DynamicAlgoBase
{
public:
  virtual std::string identify() = 0;
  static  SCIRun::CompileInfoHandle get_compile_info(TVMArrayList& Input, 
                                             TVMArrayList& Output,
                                             std::string function,
                                             int hashoffset);
  virtual void function(TVMArrayList& Input,TVMArrayList& Output,int n) = 0;                                             
};

} // end namespace 

#endif 
