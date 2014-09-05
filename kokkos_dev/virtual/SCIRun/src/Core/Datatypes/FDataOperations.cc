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

#include <Core/Datatypes/FDataOperations.h>

namespace SCIRun {

bool FData_operation(const std::string& op,const std::vector<char>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const std::vector<unsigned char>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const std::vector<short>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const std::vector<unsigned short>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }
                                                          
bool FData_operation(const std::string& op,const std::vector<int>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const std::vector<unsigned int>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }
                                                          
bool FData_operation(const std::string& op,const std::vector<long>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const std::vector<unsigned long>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const std::vector<float>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const std::vector<double>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const std::vector<Vector>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_vector(op,fdata,result)); }
                                                          
bool FData_operation(const std::string& op,const std::vector<Tensor>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_tensor(op,fdata,result)); }



bool FData_operation(const std::string& op,const Array2<char>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array2<unsigned char>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array2<short>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array2<unsigned short>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }
                                                          
bool FData_operation(const std::string& op,const Array2<int>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array2<unsigned int>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }
                                                          
bool FData_operation(const std::string& op,const Array2<long>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array2<unsigned long>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array2<float>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array2<double>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array2<Vector>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_vector(op,fdata,result)); }
                                                          
bool FData_operation(const std::string& op,const Array2<Tensor>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_tensor(op,fdata,result)); }



bool FData_operation(const std::string& op,const Array3<char>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array3<unsigned char>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array3<short>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array3<unsigned short>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }
                                                          
bool FData_operation(const std::string& op,const Array3<int>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array3<unsigned int>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }
                                                          
bool FData_operation(const std::string& op,const Array3<long>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array3<unsigned long>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array3<float>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array3<double>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_scalar(op,fdata,result)); }

bool FData_operation(const std::string& op,const Array3<Vector>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_vector(op,fdata,result)); }
                                                          
bool FData_operation(const std::string& op,const Array3<Tensor>& fdata, 
                                                          FDataResult& result)
{ return(FData_operation_tensor(op,fdata,result)); }


}
