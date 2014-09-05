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
#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

using namespace SCIRun;
typedef ImageMesh<QuadBilinearLgn<Point> > IMesh;
PersistentTypeID backwards_compat_IM("ImageMesh", "Mesh",
				      IMesh::maker, IMesh::maker);

//NoData
typedef NoDataBasis<double>               NDBasis;

//Constant
typedef ConstantBasis<Tensor>             CFDTensorBasis;
typedef ConstantBasis<Vector>             CFDVectorBasis;
typedef ConstantBasis<double>             CFDdoubleBasis;
typedef ConstantBasis<float>              CFDfloatBasis;
typedef ConstantBasis<int>                CFDintBasis;
typedef ConstantBasis<short>              CFDshortBasis;
typedef ConstantBasis<char>               CFDcharBasis;
typedef ConstantBasis<unsigned int>       CFDuintBasis;
typedef ConstantBasis<unsigned short>     CFDushortBasis;
typedef ConstantBasis<unsigned char>      CFDucharBasis;
typedef ConstantBasis<unsigned long>      CFDulongBasis;

//Linear
typedef QuadBilinearLgn<Tensor>             FDTensorBasis;
typedef QuadBilinearLgn<Vector>             FDVectorBasis;
typedef QuadBilinearLgn<double>             FDdoubleBasis;
typedef QuadBilinearLgn<float>              FDfloatBasis;
typedef QuadBilinearLgn<int>                FDintBasis;
typedef QuadBilinearLgn<short>              FDshortBasis;
typedef QuadBilinearLgn<char>               FDcharBasis;
typedef QuadBilinearLgn<unsigned int>       FDuintBasis;
typedef QuadBilinearLgn<unsigned short>     FDushortBasis;
typedef QuadBilinearLgn<unsigned char>      FDucharBasis;
typedef QuadBilinearLgn<unsigned long>      FDulongBasis;

//NoData
template class GenericField<IMesh, NDBasis, FData2d<double, IMesh> >;

//Constant
template class GenericField<IMesh, CFDTensorBasis, FData2d<Tensor, IMesh> >;
template class GenericField<IMesh, CFDVectorBasis, FData2d<Vector, IMesh> >;
template class GenericField<IMesh, CFDdoubleBasis, FData2d<double, IMesh> >;
template class GenericField<IMesh, CFDfloatBasis,  FData2d<float, IMesh> >;
template class GenericField<IMesh, CFDintBasis,    FData2d<int, IMesh> >;
template class GenericField<IMesh, CFDshortBasis,  FData2d<short, IMesh> >;
template class GenericField<IMesh, CFDcharBasis,   FData2d<char, IMesh> >;
template class GenericField<IMesh, CFDuintBasis,   FData2d<unsigned int, IMesh> >;
template class GenericField<IMesh, CFDushortBasis, FData2d<unsigned short, IMesh> >;
template class GenericField<IMesh, CFDucharBasis,  FData2d<unsigned char, IMesh> >;
template class GenericField<IMesh, CFDulongBasis,  FData2d<unsigned long, IMesh> >;

//Linear
template class GenericField<IMesh, FDTensorBasis, FData2d<Tensor, IMesh> >;
template class GenericField<IMesh, FDVectorBasis, FData2d<Vector, IMesh> >;
template class GenericField<IMesh, FDdoubleBasis, FData2d<double, IMesh> >;
template class GenericField<IMesh, FDfloatBasis,  FData2d<float, IMesh> >;
template class GenericField<IMesh, FDintBasis,    FData2d<int, IMesh> >;
template class GenericField<IMesh, FDshortBasis,  FData2d<short, IMesh> >;
template class GenericField<IMesh, FDcharBasis,   FData2d<char, IMesh> >;
template class GenericField<IMesh, FDuintBasis,   FData2d<unsigned int, IMesh> >;
template class GenericField<IMesh, FDushortBasis, FData2d<unsigned short, IMesh> >;
template class GenericField<IMesh, FDucharBasis,  FData2d<unsigned char, IMesh> >;
template class GenericField<IMesh, FDulongBasis,  FData2d<unsigned long, IMesh> >;

PersistentTypeID 
backwards_compat_IFT("ImageField<Tensor>", "Field",
		     GenericField<IMesh, FDTensorBasis, 
		     FData2d<Tensor, IMesh> >::maker, 
		     GenericField<IMesh, CFDTensorBasis, 
		     FData2d<Tensor, IMesh> >::maker);
PersistentTypeID 
backwards_compat_IFV("ImageField<Vector>", "Field",
		     GenericField<IMesh, FDVectorBasis, 
		     FData2d<Vector, IMesh> >::maker, 
		     GenericField<IMesh, CFDVectorBasis, 
		     FData2d<Vector, IMesh> >::maker);
PersistentTypeID 
backwards_compat_IFd("ImageField<double>", "Field",
		     GenericField<IMesh, FDdoubleBasis, 
		     FData2d<double, IMesh> >::maker, 
		     GenericField<IMesh, CFDdoubleBasis, 
		     FData2d<double, IMesh> >::maker,
		     GenericField<IMesh, NDBasis, 
		     FData2d<double, IMesh> >::maker);
PersistentTypeID 
backwards_compat_IFf("ImageField<float>", "Field",
		     GenericField<IMesh, FDfloatBasis, 
		     FData2d<float, IMesh> >::maker, 
		     GenericField<IMesh, CFDfloatBasis, 
		     FData2d<float, IMesh> >::maker);
PersistentTypeID 
backwards_compat_IFi("ImageField<int>", "Field",
		     GenericField<IMesh, FDintBasis, 
		     FData2d<int, IMesh> >::maker, 
		     GenericField<IMesh, CFDintBasis, 
		     FData2d<int, IMesh> >::maker);
PersistentTypeID 
backwards_compat_IFs("ImageField<short>", "Field",
		     GenericField<IMesh, FDshortBasis, 
		     FData2d<short, IMesh> >::maker, 
		     GenericField<IMesh, CFDshortBasis, 
		     FData2d<short, IMesh> >::maker);
PersistentTypeID 
backwards_compat_IFc("ImageField<char>", "Field",
		     GenericField<IMesh, FDcharBasis, 
		     FData2d<char, IMesh> >::maker, 
		     GenericField<IMesh, CFDcharBasis, 
		     FData2d<char, IMesh> >::maker);
PersistentTypeID 
backwards_compat_IFui("ImageField<unsigned_int>", "Field",
		      GenericField<IMesh, FDuintBasis, 
		      FData2d<unsigned int, IMesh> >::maker, 
		      GenericField<IMesh, CFDuintBasis, 
		      FData2d<unsigned int, IMesh> >::maker);
PersistentTypeID 
backwards_compat_IFus("ImageField<unsigned_short>", "Field",
		      GenericField<IMesh, FDushortBasis, 
		      FData2d<unsigned short, IMesh> >::maker, 
		      GenericField<IMesh, CFDushortBasis, 
		      FData2d<unsigned short, IMesh> >::maker);
PersistentTypeID 
backwards_compat_IFuc("ImageField<unsigned_char>", "Field",
		      GenericField<IMesh, FDucharBasis, 
		      FData2d<unsigned char, IMesh> >::maker, 
		      GenericField<IMesh, CFDucharBasis, 
		      FData2d<unsigned char, IMesh> >::maker);
PersistentTypeID 
backwards_compat_IFul("ImageField<unsigned_long>", "Field",
		      GenericField<IMesh, FDulongBasis, 
		      FData2d<unsigned long, IMesh> >::maker, 
		      GenericField<IMesh, CFDulongBasis, 
		      FData2d<unsigned long, IMesh> >::maker);


typedef QuadSurfMesh<QuadBilinearLgn<Point> > QSMesh;
PersistentTypeID backwards_compat_QSM("QuadSurfMesh", "Mesh",
				      QSMesh::maker, QSMesh::maker);

//NoData
template class GenericField<QSMesh, NDBasis, vector<double> >;

//Constant
template class GenericField<QSMesh, CFDTensorBasis, vector<Tensor> >;       
template class GenericField<QSMesh, CFDVectorBasis, vector<Vector> >;       
template class GenericField<QSMesh, CFDdoubleBasis, vector<double> >;       
template class GenericField<QSMesh, CFDfloatBasis,  vector<float> >;        
template class GenericField<QSMesh, CFDintBasis,    vector<int> >;          
template class GenericField<QSMesh, CFDshortBasis,  vector<short> >;        
template class GenericField<QSMesh, CFDcharBasis,   vector<char> >;         
template class GenericField<QSMesh, CFDuintBasis,   vector<unsigned int> >; 
template class GenericField<QSMesh, CFDushortBasis, vector<unsigned short> >;
template class GenericField<QSMesh, CFDucharBasis,  vector<unsigned char> >;
template class GenericField<QSMesh, CFDulongBasis,  vector<unsigned long> >;

//Linear
template class GenericField<QSMesh, FDTensorBasis, vector<Tensor> >;       
template class GenericField<QSMesh, FDVectorBasis, vector<Vector> >;       
template class GenericField<QSMesh, FDdoubleBasis, vector<double> >;       
template class GenericField<QSMesh, FDfloatBasis,  vector<float> >;        
template class GenericField<QSMesh, FDintBasis,    vector<int> >;          
template class GenericField<QSMesh, FDshortBasis,  vector<short> >;        
template class GenericField<QSMesh, FDcharBasis,   vector<char> >;         
template class GenericField<QSMesh, FDuintBasis,   vector<unsigned int> >; 
template class GenericField<QSMesh, FDushortBasis, vector<unsigned short> >;
template class GenericField<QSMesh, FDucharBasis,  vector<unsigned char> >;
template class GenericField<QSMesh, FDulongBasis,  vector<unsigned long> >;


PersistentTypeID 
backwards_compat_QSFT("QuadSurfField<Tensor>", "Field",
		      GenericField<QSMesh, FDTensorBasis, 
		      vector<Tensor> >::maker, 
		      GenericField<QSMesh, CFDTensorBasis, 
		      vector<Tensor> >::maker);
PersistentTypeID 
backwards_compat_QSFV("QuadSurfField<Vector>", "Field",
		      GenericField<QSMesh, FDVectorBasis, 
		      vector<Vector> >::maker, 
		      GenericField<QSMesh, CFDVectorBasis, 
		      vector<Vector> >::maker);
PersistentTypeID 
backwards_compat_QSFd("QuadSurfField<double>", "Field",
		      GenericField<QSMesh, FDdoubleBasis, 
		      vector<double> >::maker, 
		      GenericField<QSMesh, CFDdoubleBasis, 
		      vector<double> >::maker,
		      GenericField<QSMesh, NDBasis, 
		      vector<double> >::maker);
PersistentTypeID 
backwards_compat_QSFf("QuadSurfField<float>", "Field",
		      GenericField<QSMesh, FDfloatBasis, 
		      vector<float> >::maker, 
		      GenericField<QSMesh, CFDfloatBasis, 
		      vector<float> >::maker);
PersistentTypeID 
backwards_compat_QSFi("QuadSurfField<int>", "Field",
		      GenericField<QSMesh, FDintBasis, 
		      vector<int> >::maker, 
		      GenericField<QSMesh, CFDintBasis, 
		      vector<int> >::maker);
PersistentTypeID 
backwards_compat_QSFs("QuadSurfField<short>", "Field",
		      GenericField<QSMesh, FDshortBasis, 
		      vector<short> >::maker, 
		      GenericField<QSMesh, CFDshortBasis, 
		      vector<short> >::maker);
PersistentTypeID 
backwards_compat_QSFc("QuadSurfField<char>", "Field",
		      GenericField<QSMesh, FDcharBasis, 
		      vector<char> >::maker, 
		      GenericField<QSMesh, CFDcharBasis, 
		      vector<char> >::maker);
PersistentTypeID 
backwards_compat_QSFui("QuadSurfField<unsigned_int>", "Field",
		       GenericField<QSMesh, FDuintBasis, 
		       vector<unsigned int> >::maker, 
		       GenericField<QSMesh, CFDuintBasis, 
		       vector<unsigned int> >::maker);
PersistentTypeID 
backwards_compat_QSFus("QuadSurfField<unsigned_short>", "Field",
		       GenericField<QSMesh, FDushortBasis, 
		       vector<unsigned short> >::maker, 
		       GenericField<QSMesh, CFDushortBasis, 
		       vector<unsigned short> >::maker);
PersistentTypeID 
backwards_compat_QSFuc("QuadSurfField<unsigned_char>", "Field",
		       GenericField<QSMesh, FDucharBasis, 
		       vector<unsigned char> >::maker, 
		       GenericField<QSMesh, CFDucharBasis, 
		       vector<unsigned char> >::maker);
PersistentTypeID 
backwards_compat_QSFul("QuadSurfField<unsigned_long>", "Field",
		       GenericField<QSMesh, FDulongBasis, 
		       vector<unsigned long> >::maker, 
		       GenericField<QSMesh, CFDulongBasis, 
		       vector<unsigned long> >::maker);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif









