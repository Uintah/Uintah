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

#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/CurveMesh.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

using namespace SCIRun;

typedef TriLinearLgn<Tensor>                FDTensorBasis;
typedef TriLinearLgn<Vector>                FDVectorBasis;
typedef TriLinearLgn<double>                FDdoubleBasis;
typedef TriLinearLgn<float>                 FDfloatBasis;
typedef TriLinearLgn<int>                   FDintBasis;
typedef TriLinearLgn<short>                 FDshortBasis;
typedef TriLinearLgn<char>                  FDcharBasis;
typedef TriLinearLgn<unsigned int>          FDuintBasis;
typedef TriLinearLgn<unsigned short>        FDushortBasis;
typedef TriLinearLgn<unsigned char>         FDucharBasis;
typedef TriLinearLgn<unsigned long>         FDulongBasis;

typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;
template class GenericField<TSMesh, FDTensorBasis, vector<Tensor> >;       
template class GenericField<TSMesh, FDVectorBasis, vector<Vector> >;       
template class GenericField<TSMesh, FDdoubleBasis, vector<double> >;       
template class GenericField<TSMesh, FDfloatBasis,  vector<float> >;        
template class GenericField<TSMesh, FDintBasis,    vector<int> >;          
template class GenericField<TSMesh, FDshortBasis,  vector<short> >;        
template class GenericField<TSMesh, FDcharBasis,   vector<char> >;         
template class GenericField<TSMesh, FDuintBasis,   vector<unsigned int> >; 
template class GenericField<TSMesh, FDushortBasis, vector<unsigned short> >;
template class GenericField<TSMesh, FDucharBasis,  vector<unsigned char> >;
template class GenericField<TSMesh, FDulongBasis,  vector<unsigned long> >;

typedef CrvLinearLgn<Tensor>                CFDTensorBasis;
typedef CrvLinearLgn<Vector>                CFDVectorBasis;
typedef CrvLinearLgn<double>                CFDdoubleBasis;
typedef CrvLinearLgn<float>                 CFDfloatBasis;
typedef CrvLinearLgn<int>                   CFDintBasis;
typedef CrvLinearLgn<short>                 CFDshortBasis;
typedef CrvLinearLgn<char>                  CFDcharBasis;
typedef CrvLinearLgn<unsigned int>          CFDuintBasis;
typedef CrvLinearLgn<unsigned short>        CFDushortBasis;
typedef CrvLinearLgn<unsigned char>         CFDucharBasis;
typedef CrvLinearLgn<unsigned long>         CFDulongBasis;

typedef CurveMesh<CrvLinearLgn<Point> > CMesh;
template class GenericField<CMesh, CFDTensorBasis, vector<Tensor> >;       
template class GenericField<CMesh, CFDVectorBasis, vector<Vector> >;       
template class GenericField<CMesh, CFDdoubleBasis, vector<double> >;       
template class GenericField<CMesh, CFDfloatBasis,  vector<float> >;        
template class GenericField<CMesh, CFDintBasis,    vector<int> >;          
template class GenericField<CMesh, CFDshortBasis,  vector<short> >;        
template class GenericField<CMesh, CFDcharBasis,   vector<char> >;         
template class GenericField<CMesh, CFDuintBasis,   vector<unsigned int> >; 
template class GenericField<CMesh, CFDushortBasis, vector<unsigned short> >;
template class GenericField<CMesh, CFDucharBasis,  vector<unsigned char> >;
template class GenericField<CMesh, CFDulongBasis,  vector<unsigned long> >;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
