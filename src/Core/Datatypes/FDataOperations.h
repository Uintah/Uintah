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


#ifndef CORE_DATATYPES_FDATAOPERATIONS_H
#define CORE_DATATYPES_FDATAOPERATIONS_H 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/Field.h>
#include <string>
#include <vector>
#include <float.h>

namespace SCIRun {


// Instantiations of operations do to on the fdata

template<class FDATA>
inline bool FData_operation(const std::string& op,const FDATA& fdata, 
                                                          FDataResult& result)
{
  return (false);
}

template<class T, class MESH>
inline bool FData_operation(const std::string& op,const FData2d<T,MESH>& fdata, 
                                                          FDataResult& result)
{
  return(FData_operation(op,static_cast<const Array2<T>& >(fdata),result));
}

template<class T, class MESH>
inline bool FData_operation(const std::string& op,const FData3d<T,MESH>& fdata, 
                                                          FDataResult& result)
{
  return(FData_operation(op,static_cast<const Array3<T>& >(fdata),result));
}

bool FData_operation(const std::string& op,const std::vector<char>& fdata, 
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<unsigned char>& fdata, 
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<short>& fdata, 
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<unsigned short>& fdata, 
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<int>& fdata,
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<unsigned int>& fdata, 
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<long>& fdata, 
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<unsigned long>& fdata, 
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<float>& fdata, 
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<double>& fdata, 
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<Vector>& fdata, 
                                                          FDataResult& result);
bool FData_operation(const std::string& op,const std::vector<Tensor>& fdata, 
                                                          FDataResult& result);

bool FData_operation(const std::string& op,const Array2<char>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<unsigned char>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<short>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<unsigned short>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<int>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<unsigned int>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<long>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<unsigned long>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<float>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<double>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<Vector>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array2<Tensor>& fdata, FDataResult& result);

bool FData_operation(const std::string& op,const Array3<char>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<unsigned char>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<short>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<unsigned short>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<int>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<unsigned int>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<long>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<unsigned long>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<float>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<double>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<Vector>& fdata, FDataResult& result);
bool FData_operation(const std::string& op,const Array3<Tensor>& fdata, FDataResult& result);



template<class FDATA>
bool FData_operation_scalar(const std::string& op,const FDATA& fdata, FDataResult& result)
{
  if (op == "min")
  {
    result.scalar.resize(1);
    result.index.resize(1);
    
    double val = -DBL_MAX;
    double val2;
    size_t idx = 0;
    size_t sz = fdata.size();
    for (size_t i=0; i<sz;i++)
    {
      val2 = static_cast<typename FDATA::value_type>(fdata[i]);
      if (val2 < val) { val = val2; idx = i; }
    }
  
    result.scalar[0] = val;
    result.index[0] = static_cast<Mesh::index_type>(idx);
    return (true);
  }

  if (op == "max")
  {
    result.scalar.resize(1);
    result.index.resize(1);
    
    double val = DBL_MAX;
    double val2;
    size_t idx = 0;
    size_t sz = fdata.size();
    for (size_t i=0; i<sz;i++)
    {
      val2 = static_cast<typename FDATA::value_type>(fdata[i]);
      if (val2 > val) { val = val2; idx = i; }
    }
  
    result.scalar[0] = val;
    result.index[0] = static_cast<Mesh::index_type>(idx);
    return (true);
  }

  if (op == "minmax")
  {
    result.scalar.resize(2);
    result.index.resize(2);
    
    double val1 = -DBL_MAX;
    double val2 = DBL_MAX;
    double val;
    size_t idx1 = 0;
    size_t idx2 = 0;
    size_t sz = fdata.size();
    for (size_t i=0; i<sz;i++)
    {
      val = static_cast<typename FDATA::value_type>(fdata[i]);
      if (val < val2) { val1 = val; idx1 = i; }
      if (val > val2) { val2 = val; idx2 = i; }
    }
  
    result.scalar[0] = val1;
    result.index[0] = static_cast<Mesh::index_type>(idx1);
    result.scalar[1] = val2;
    result.index[1] = static_cast<Mesh::index_type>(idx2);
    return (true);
  }

  //! Test whether we have an interface, if we get this far
  //! we have the interface.
  if (op == "test")
  {
    return (true);
  }
  
  return (false);
}


template<class FDATA>
bool FData_operation_vector(const std::string& op,const FDATA& fdata, FDataResult& result)
{

  //! Test whether we have an interface, if we get this far
  //! we have the interface.
  if (op == "test")
  {
    return (true);
  }
  
  return (false);
}


template<class FDATA>
bool FData_operation_tensor(const std::string& op,const FDATA& fdata, FDataResult& result)
{

  //! Test whether we have an interface, if we get this far
  //! we have the interface.
  if (op == "test")
  {
    return (true);
  }

  return (false);
}

} // end namespace

#endif



