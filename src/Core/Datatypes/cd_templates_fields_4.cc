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

PersistentTypeID backwards_compat_TSFT("TriSurfField<Tensor>", "Field",
				       GenericField<TSMesh, FDTensorBasis, 
				       vector<Tensor> >::maker, true);
PersistentTypeID backwards_compat_TSFV("TriSurfField<Vector>", "Field",
				       GenericField<TSMesh, FDVectorBasis, 
				       vector<Vector> >::maker, true);
PersistentTypeID backwards_compat_TSFd("TriSurfField<double>", "Field",
				       GenericField<TSMesh, FDdoubleBasis, 
				       vector<double> >::maker, true);
PersistentTypeID backwards_compat_TSFf("TriSurfField<float>", "Field",
				       GenericField<TSMesh, FDfloatBasis, 
				       vector<float> >::maker, true);
PersistentTypeID backwards_compat_TSFi("TriSurfField<int>", "Field",
				       GenericField<TSMesh, FDintBasis, 
				       vector<int> >::maker, true);
PersistentTypeID backwards_compat_TSFs("TriSurfField<short>", "Field",
				       GenericField<TSMesh, FDshortBasis, 
				       vector<short> >::maker, true);
PersistentTypeID backwards_compat_TSFc("TriSurfField<char>", "Field",
				       GenericField<TSMesh, FDcharBasis, 
				       vector<char> >::maker, true);
PersistentTypeID backwards_compat_TSFui("TriSurfField<unsigned_int>", "Field",
				       GenericField<TSMesh, FDuintBasis, 
				       vector<unsigned int> >::maker, true);
PersistentTypeID backwards_compat_TSFus("TriSurfField<unsigned_short>", "Field",
				       GenericField<TSMesh, FDushortBasis, 
				       vector<unsigned short> >::maker, true);
PersistentTypeID backwards_compat_TSFuc("TriSurfField<unsigned_char>", "Field",
				       GenericField<TSMesh, FDucharBasis, 
				       vector<unsigned char> >::maker, true);
PersistentTypeID backwards_compat_TSFul("TriSurfField<unsigned_long>", "Field",
				       GenericField<TSMesh, FDulongBasis, 
				       vector<unsigned long> >::maker, true);

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

PersistentTypeID backwards_compat_CFT("CurveField<Tensor>", "Field",
				       GenericField<CMesh, CFDTensorBasis, 
				       vector<Tensor> >::maker, true);
PersistentTypeID backwards_compat_CFV("CurveField<Vector>", "Field",
				       GenericField<CMesh, CFDVectorBasis, 
				       vector<Vector> >::maker, true);
PersistentTypeID backwards_compat_CFd("CurveField<double>", "Field",
				       GenericField<CMesh, CFDdoubleBasis, 
				       vector<double> >::maker, true);
PersistentTypeID backwards_compat_CFf("CurveField<float>", "Field",
				       GenericField<CMesh, CFDfloatBasis, 
				       vector<float> >::maker, true);
PersistentTypeID backwards_compat_CFi("CurveField<int>", "Field",
				       GenericField<CMesh, CFDintBasis, 
				       vector<int> >::maker, true);
PersistentTypeID backwards_compat_CFs("CurveField<short>", "Field",
				       GenericField<CMesh, CFDshortBasis, 
				       vector<short> >::maker, true);
PersistentTypeID backwards_compat_CFc("CurveField<char>", "Field",
				       GenericField<CMesh, CFDcharBasis, 
				       vector<char> >::maker, true);
PersistentTypeID backwards_compat_CFui("CurveField<unsigned_int>", "Field",
				       GenericField<CMesh, CFDuintBasis, 
				       vector<unsigned int> >::maker, true);
PersistentTypeID backwards_compat_CFus("CurveField<unsigned_short>", "Field",
				       GenericField<CMesh, CFDushortBasis, 
				       vector<unsigned short> >::maker, true);
PersistentTypeID backwards_compat_CFuc("CurveField<unsigned_char>", "Field",
				       GenericField<CMesh, CFDucharBasis, 
				       vector<unsigned char> >::maker, true);
PersistentTypeID backwards_compat_CFul("CurveField<unsigned_long>", "Field",
				       GenericField<CMesh, CFDulongBasis, 
				       vector<unsigned long> >::maker, true);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
