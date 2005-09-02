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
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>

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

PersistentTypeID backwards_compat_IFT("ImageField<Tensor>", "Field",
				       GenericField<IMesh, FDTensorBasis, 
				       FData2d<Tensor, IMesh> >::maker, true);
PersistentTypeID backwards_compat_IFV("ImageField<Vector>", "Field",
				       GenericField<IMesh, FDVectorBasis, 
				       FData2d<Vector, IMesh> >::maker, true);
PersistentTypeID backwards_compat_IFd("ImageField<double>", "Field",
				       GenericField<IMesh, FDdoubleBasis, 
				       FData2d<double, IMesh> >::maker, true);
PersistentTypeID backwards_compat_IFf("ImageField<float>", "Field",
				       GenericField<IMesh, FDfloatBasis, 
				       FData2d<float, IMesh> >::maker, true);
PersistentTypeID backwards_compat_IFi("ImageField<int>", "Field",
				       GenericField<IMesh, FDintBasis, 
				       FData2d<int, IMesh> >::maker, true);
PersistentTypeID backwards_compat_IFs("ImageField<short>", "Field",
				       GenericField<IMesh, FDshortBasis, 
				       FData2d<short, IMesh> >::maker, true);
PersistentTypeID backwards_compat_IFc("ImageField<char>", "Field",
				       GenericField<IMesh, FDcharBasis, 
				       FData2d<char, IMesh> >::maker, true);
PersistentTypeID backwards_compat_IFui("ImageField<unsigned_int>", "Field",
				       GenericField<IMesh, FDuintBasis, 
				       FData2d<unsigned int, IMesh> >::maker, true);
PersistentTypeID backwards_compat_IFus("ImageField<unsigned_short>", "Field",
				       GenericField<IMesh, FDushortBasis, 
				       FData2d<unsigned short, IMesh> >::maker, true);
PersistentTypeID backwards_compat_IFuc("ImageField<unsigned_char>", "Field",
				       GenericField<IMesh, FDucharBasis, 
				       FData2d<unsigned char, IMesh> >::maker, true);
PersistentTypeID backwards_compat_IFul("ImageField<unsigned_long>", "Field",
				       GenericField<IMesh, FDulongBasis, 
				       FData2d<unsigned long, IMesh> >::maker, true);


typedef QuadSurfMesh<QuadBilinearLgn<Point> > QSMesh;
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


PersistentTypeID backwards_compat_QSFT("QuadSurfField<Tensor>", "Field",
				       GenericField<QSMesh, FDTensorBasis, 
				       vector<Tensor> >::maker, true);
PersistentTypeID backwards_compat_QSFV("QuadSurfField<Vector>", "Field",
				       GenericField<QSMesh, FDVectorBasis, 
				       vector<Vector> >::maker, true);
PersistentTypeID backwards_compat_QSFd("QuadSurfField<double>", "Field",
				       GenericField<QSMesh, FDdoubleBasis, 
				       vector<double> >::maker, true);
PersistentTypeID backwards_compat_QSFf("QuadSurfField<float>", "Field",
				       GenericField<QSMesh, FDfloatBasis, 
				       vector<float> >::maker, true);
PersistentTypeID backwards_compat_QSFi("QuadSurfField<int>", "Field",
				       GenericField<QSMesh, FDintBasis, 
				       vector<int> >::maker, true);
PersistentTypeID backwards_compat_QSFs("QuadSurfField<short>", "Field",
				       GenericField<QSMesh, FDshortBasis, 
				       vector<short> >::maker, true);
PersistentTypeID backwards_compat_QSFc("QuadSurfField<char>", "Field",
				       GenericField<QSMesh, FDcharBasis, 
				       vector<char> >::maker, true);
PersistentTypeID backwards_compat_QSFui("QuadSurfField<unsigned_int>", "Field",
				       GenericField<QSMesh, FDuintBasis, 
				       vector<unsigned int> >::maker, true);
PersistentTypeID backwards_compat_QSFus("QuadSurfField<unsigned_short>", "Field",
				       GenericField<QSMesh, FDushortBasis, 
				       vector<unsigned short> >::maker, true);
PersistentTypeID backwards_compat_QSFuc("QuadSurfField<unsigned_char>", "Field",
				       GenericField<QSMesh, FDucharBasis, 
				       vector<unsigned char> >::maker, true);
PersistentTypeID backwards_compat_QSFul("QuadSurfField<unsigned_long>", "Field",
				       GenericField<QSMesh, FDulongBasis, 
				       vector<unsigned long> >::maker, true);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif









