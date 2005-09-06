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
#include <Core/Basis/HexTricubicHmt.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/HexVolMesh.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

using namespace SCIRun;

typedef HexTrilinearLgn<Tensor>                FDTensorBasis;
typedef HexTrilinearLgn<Vector>                FDVectorBasis;
typedef HexTrilinearLgn<double>                FDdoubleBasis;
typedef HexTrilinearLgn<float>                 FDfloatBasis;
typedef HexTrilinearLgn<int>                   FDintBasis;
typedef HexTrilinearLgn<short>                 FDshortBasis;
typedef HexTrilinearLgn<char>                  FDcharBasis;
typedef HexTrilinearLgn<unsigned int>          FDuintBasis;
typedef HexTrilinearLgn<unsigned short>        FDushortBasis;
typedef HexTrilinearLgn<unsigned char>         FDucharBasis;
typedef HexTrilinearLgn<unsigned long>         FDulongBasis;

typedef HexVolMesh<HexTrilinearLgn<Point> > HVMesh;
PersistentTypeID backwards_compat_HVM("HexVolMesh", "Mesh",
				      HVMesh::maker, true);

template class GenericField<HVMesh, FDTensorBasis, vector<Tensor> >;       
template class GenericField<HVMesh, FDVectorBasis, vector<Vector> >;       
template class GenericField<HVMesh, FDdoubleBasis, vector<double> >;       
template class GenericField<HVMesh, FDfloatBasis,  vector<float> >;        
template class GenericField<HVMesh, FDintBasis,    vector<int> >;          
template class GenericField<HVMesh, FDshortBasis,  vector<short> >;        
template class GenericField<HVMesh, FDcharBasis,   vector<char> >;         
template class GenericField<HVMesh, FDuintBasis,   vector<unsigned int> >; 
template class GenericField<HVMesh, FDushortBasis, vector<unsigned short> >;
template class GenericField<HVMesh, FDucharBasis,  vector<unsigned char> >;
template class GenericField<HVMesh, FDulongBasis,  vector<unsigned long> >;


PersistentTypeID backwards_compat_HVFT("HexVolField<Tensor>", "Field",
				       GenericField<HVMesh, FDTensorBasis, 
				       vector<Tensor> >::maker, true);
PersistentTypeID backwards_compat_HVFV("HexVolField<Vector>", "Field",
				       GenericField<HVMesh, FDVectorBasis, 
				       vector<Vector> >::maker, true);
PersistentTypeID backwards_compat_HVFd("HexVolField<double>", "Field",
				       GenericField<HVMesh, FDdoubleBasis, 
				       vector<double> >::maker, true);
PersistentTypeID backwards_compat_HVFf("HexVolField<float>", "Field",
				       GenericField<HVMesh, FDfloatBasis, 
				       vector<float> >::maker, true);
PersistentTypeID backwards_compat_HVFi("HexVolField<int>", "Field",
				       GenericField<HVMesh, FDintBasis, 
				       vector<int> >::maker, true);
PersistentTypeID backwards_compat_HVFs("HexVolField<short>", "Field",
				       GenericField<HVMesh, FDshortBasis, 
				       vector<short> >::maker, true);
PersistentTypeID backwards_compat_HVFc("HexVolField<char>", "Field",
				       GenericField<HVMesh, FDcharBasis, 
				       vector<char> >::maker, true);
PersistentTypeID backwards_compat_HVFui("HexVolField<unsigned_int>", "Field",
				       GenericField<HVMesh, FDuintBasis, 
				       vector<unsigned int> >::maker, true);
PersistentTypeID backwards_compat_HVFus("HexVolField<unsigned_short>", "Field",
				       GenericField<HVMesh, FDushortBasis, 
				       vector<unsigned short> >::maker, true);
PersistentTypeID backwards_compat_HVFuc("HexVolField<unsigned_char>", "Field",
				       GenericField<HVMesh, FDucharBasis, 
				       vector<unsigned char> >::maker, true);
PersistentTypeID backwards_compat_HVFul("HexVolField<unsigned_long>", "Field",
				       GenericField<HVMesh, FDulongBasis, 
				       vector<unsigned long> >::maker, true);


typedef HexTricubicHmt<double>             HTCdoubleBasis;
typedef NoDataBasis<double>                NDdoubleBasis;

typedef HexVolMesh<HexTricubicHmt<Point> > HVCubMesh;
template class GenericField<HVCubMesh, NDdoubleBasis, vector<double> >; 
template class GenericField<HVCubMesh, HTCdoubleBasis, vector<double> >; 

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
