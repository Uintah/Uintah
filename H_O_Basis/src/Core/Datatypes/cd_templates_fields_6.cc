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
 * Manual template instantiations
 */


/*
 * These aren't used by Datatypes directly, but since they are used in
 * a lot of different modules, we instantiate them here to avoid bloat
 *
 * Find the bloaters with:
find . -name "*.ii" -print | xargs cat | sort | uniq -c | sort -nr | more
 */

#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>


using namespace SCIRun;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif

#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/StructCurveMesh.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <Core/Datatypes/StructHexVolMesh.h>

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

typedef StructCurveMesh<CrvLinearLgn<Point> > CMesh;
PersistentTypeID backwards_compat_SCM("StructCurveMesh", "Mesh",
				      CMesh::maker, true);

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

PersistentTypeID backwards_compat_SCFT("StructCurveField<Tensor>", "Field",
				       GenericField<CMesh, CFDTensorBasis, 
				       vector<Tensor> >::maker, true);
PersistentTypeID backwards_compat_SCFV("StructCurveField<Vector>", "Field",
				       GenericField<CMesh, CFDVectorBasis, 
				       vector<Vector> >::maker, true);
PersistentTypeID backwards_compat_SCFd("StructCurveField<double>", "Field",
				       GenericField<CMesh, CFDdoubleBasis, 
				       vector<double> >::maker, true);
PersistentTypeID backwards_compat_SCFf("StructCurveField<float>", "Field",
				       GenericField<CMesh, CFDfloatBasis, 
				       vector<float> >::maker, true);
PersistentTypeID backwards_compat_SCFi("StructCurveField<int>", "Field",
				       GenericField<CMesh, CFDintBasis, 
				       vector<int> >::maker, true);
PersistentTypeID backwards_compat_SCFs("StructCurveField<short>", "Field",
				       GenericField<CMesh, CFDshortBasis, 
				       vector<short> >::maker, true);
PersistentTypeID backwards_compat_SCFc("StructCurveField<char>", "Field",
				       GenericField<CMesh, CFDcharBasis, 
				       vector<char> >::maker, true);
PersistentTypeID backwards_compat_SCFui("StructCurveField<unsigned_int>", "Field",
				       GenericField<CMesh, CFDuintBasis, 
				       vector<unsigned int> >::maker, true);
PersistentTypeID backwards_compat_SCFus("StructCurveField<unsigned_short>", "Field",
				       GenericField<CMesh, CFDushortBasis, 
				       vector<unsigned short> >::maker, true);
PersistentTypeID backwards_compat_SCFuc("StructCurveField<unsigned_char>", "Field",
				       GenericField<CMesh, CFDucharBasis, 
				       vector<unsigned char> >::maker, true);
PersistentTypeID backwards_compat_SCFul("StructCurveField<unsigned_long>", "Field",
				       GenericField<CMesh, CFDulongBasis, 
				       vector<unsigned long> >::maker, true);

typedef QuadBilinearLgn<Tensor>                QFDTensorBasis;
typedef QuadBilinearLgn<Vector>                QFDVectorBasis;
typedef QuadBilinearLgn<double>                QFDdoubleBasis;
typedef QuadBilinearLgn<float>                 QFDfloatBasis;
typedef QuadBilinearLgn<int>                   QFDintBasis;
typedef QuadBilinearLgn<short>                 QFDshortBasis;
typedef QuadBilinearLgn<char>                  QFDcharBasis;
typedef QuadBilinearLgn<unsigned int>          QFDuintBasis;
typedef QuadBilinearLgn<unsigned short>        QFDushortBasis;
typedef QuadBilinearLgn<unsigned char>         QFDucharBasis;
typedef QuadBilinearLgn<unsigned long>         QFDulongBasis;

typedef StructQuadSurfMesh<QuadBilinearLgn<Point> > SQMesh;
PersistentTypeID backwards_compat_SQM("StructQuadSurfMesh", "Mesh",
				      SQMesh::maker, true);

template class GenericField<SQMesh, QFDTensorBasis, FData2d<Tensor,SQMesh> >;
template class GenericField<SQMesh, QFDVectorBasis, FData2d<Vector,SQMesh> >;
template class GenericField<SQMesh, QFDdoubleBasis, FData2d<double,SQMesh> >;
template class GenericField<SQMesh, QFDfloatBasis,  FData2d<float,SQMesh> >;
template class GenericField<SQMesh, QFDintBasis,    FData2d<int,SQMesh> >;
template class GenericField<SQMesh, QFDshortBasis,  FData2d<short,SQMesh> >;
template class GenericField<SQMesh, QFDcharBasis,   FData2d<char,SQMesh> >;
template class GenericField<SQMesh, QFDuintBasis,   FData2d<unsigned int,
							    SQMesh> >; 
template class GenericField<SQMesh, QFDushortBasis, FData2d<unsigned short,
							    SQMesh> >;
template class GenericField<SQMesh, QFDucharBasis,  FData2d<unsigned char,
							    SQMesh> >;
template class GenericField<SQMesh, QFDulongBasis,  FData2d<unsigned long,
							    SQMesh> >;

PersistentTypeID backwards_compat_SQSFT("StructQuadSurfField<Tensor>", "Field",
				       GenericField<SQMesh, QFDTensorBasis, 
				       FData2d<Tensor, SQMesh> >::maker, true);
PersistentTypeID backwards_compat_SQSFV("StructQuadSurfField<Vector>", "Field",
				       GenericField<SQMesh, QFDVectorBasis, 
				       FData2d<Vector, SQMesh> >::maker, true);
PersistentTypeID backwards_compat_SQSFd("StructQuadSurfField<double>", "Field",
				       GenericField<SQMesh, QFDdoubleBasis, 
				       FData2d<double, SQMesh> >::maker, true);
PersistentTypeID backwards_compat_SQSFf("StructQuadSurfField<float>", "Field",
				       GenericField<SQMesh, QFDfloatBasis, 
				       FData2d<float, SQMesh> >::maker, true);
PersistentTypeID backwards_compat_SQSFi("StructQuadSurfField<int>", "Field",
				       GenericField<SQMesh, QFDintBasis, 
				       FData2d<int, SQMesh> >::maker, true);
PersistentTypeID backwards_compat_SQSFs("StructQuadSurfField<short>", "Field",
				       GenericField<SQMesh, QFDshortBasis, 
				       FData2d<short, SQMesh> >::maker, true);
PersistentTypeID backwards_compat_SQSFc("StructQuadSurfField<char>", "Field",
				       GenericField<SQMesh, QFDcharBasis, 
				       FData2d<char, SQMesh> >::maker, true);
PersistentTypeID backwards_compat_SQSFui("StructQuadSurfField<unsigned_int>", "Field",
				       GenericField<SQMesh, QFDuintBasis, 
				       FData2d<unsigned int, SQMesh> >::maker, true);
PersistentTypeID backwards_compat_SQSFus("StructQuadSurfField<unsigned_short>", "Field",
				       GenericField<SQMesh, QFDushortBasis, 
				       FData2d<unsigned short, SQMesh> >::maker, true);
PersistentTypeID backwards_compat_SQSFuc("StructQuadSurfField<unsigned_char>", "Field",
				       GenericField<SQMesh, QFDucharBasis, 
				       FData2d<unsigned char, SQMesh> >::maker, true);
PersistentTypeID backwards_compat_SQSFul("StructQuadSurfField<unsigned_long>", "Field",
				       GenericField<SQMesh, QFDulongBasis, 
				       FData2d<unsigned long, SQMesh> >::maker, true);


typedef HexTrilinearLgn<Tensor>                HFDTensorBasis;
typedef HexTrilinearLgn<Vector>                HFDVectorBasis;
typedef HexTrilinearLgn<double>                HFDdoubleBasis;
typedef HexTrilinearLgn<float>                 HFDfloatBasis;
typedef HexTrilinearLgn<int>                   HFDintBasis;
typedef HexTrilinearLgn<short>                 HFDshortBasis;
typedef HexTrilinearLgn<char>                  HFDcharBasis;
typedef HexTrilinearLgn<unsigned int>          HFDuintBasis;
typedef HexTrilinearLgn<unsigned short>        HFDushortBasis;
typedef HexTrilinearLgn<unsigned char>         HFDucharBasis;
typedef HexTrilinearLgn<unsigned long>         HFDulongBasis;

typedef StructHexVolMesh<HexTrilinearLgn<Point> > SHMesh;
PersistentTypeID backwards_compat_SHVM("StructHexVolMesh", "Mesh",
				       SHMesh::maker, true);

template class GenericField<SHMesh, HFDTensorBasis, FData3d<Tensor,SHMesh> >;
template class GenericField<SHMesh, HFDVectorBasis, FData3d<Vector,SHMesh> >;
template class GenericField<SHMesh, HFDdoubleBasis, FData3d<double,SHMesh> >;
template class GenericField<SHMesh, HFDfloatBasis,  FData3d<float,SHMesh> >;
template class GenericField<SHMesh, HFDintBasis,    FData3d<int,SHMesh> >;
template class GenericField<SHMesh, HFDshortBasis,  FData3d<short,SHMesh> >;
template class GenericField<SHMesh, HFDcharBasis,   FData3d<char,SHMesh> >;
template class GenericField<SHMesh, HFDuintBasis,   FData3d<unsigned int,
							    SHMesh> >;
template class GenericField<SHMesh, HFDushortBasis, FData3d<unsigned short,
							    SHMesh> >;
template class GenericField<SHMesh, HFDucharBasis,  FData3d<unsigned char,
							    SHMesh> >;
template class GenericField<SHMesh, HFDulongBasis,  FData3d<unsigned long,
							    SHMesh> >;


PersistentTypeID backwards_compat_SHVFT("StructHexVolField<Tensor>", "Field",
				       GenericField<SHMesh, HFDTensorBasis, 
				       FData3d<Tensor, SHMesh> >::maker, true);
PersistentTypeID backwards_compat_SHVFV("StructHexVolField<Vector>", "Field",
				       GenericField<SHMesh, HFDVectorBasis, 
				       FData3d<Vector, SHMesh> >::maker, true);
PersistentTypeID backwards_compat_SHVFd("StructHexVolField<double>", "Field",
				       GenericField<SHMesh, HFDdoubleBasis, 
				       FData3d<double, SHMesh> >::maker, true);
PersistentTypeID backwards_compat_SHVFf("StructHexVolField<float>", "Field",
				       GenericField<SHMesh, HFDfloatBasis, 
				       FData3d<float, SHMesh> >::maker, true);
PersistentTypeID backwards_compat_SHVFi("StructHexVolField<int>", "Field",
				       GenericField<SHMesh, HFDintBasis, 
				       FData3d<int, SHMesh> >::maker, true);
PersistentTypeID backwards_compat_SHVFs("StructHexVolField<short>", "Field",
				       GenericField<SHMesh, HFDshortBasis, 
				       FData3d<short, SHMesh> >::maker, true);
PersistentTypeID backwards_compat_SHVFc("StructHexVolField<char>", "Field",
				       GenericField<SHMesh, HFDcharBasis, 
				       FData3d<char, SHMesh> >::maker, true);
PersistentTypeID backwards_compat_SHVFui("StructHexVolField<unsigned_int>", "Field",
				       GenericField<SHMesh, HFDuintBasis, 
				       FData3d<unsigned int, SHMesh> >::maker, true);
PersistentTypeID backwards_compat_SHVFus("StructHexVolField<unsigned_short>", "Field",
				       GenericField<SHMesh, HFDushortBasis, 
				       FData3d<unsigned short, SHMesh> >::maker, true);
PersistentTypeID backwards_compat_SHVFuc("StructHexVolField<unsigned_char>", "Field",
				       GenericField<SHMesh, HFDucharBasis, 
				       FData3d<unsigned char, SHMesh> >::maker, true);
PersistentTypeID backwards_compat_SHVFul("StructHexVolField<unsigned_long>", "Field",
				       GenericField<SHMesh, HFDulongBasis, 
				       FData3d<unsigned long, SHMesh> >::maker, true);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif


