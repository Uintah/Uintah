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
#include <Core/Basis/HexTricubicHmt.h>
#include <Core/Basis/HexTricubicHmtScaleFactors.h>
#include <Core/Basis/HexTricubicHmtScaleFactorsEdges.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>

using namespace SCIRun;

//NoData
typedef NoDataBasis<double>                NDBasis;

//Linear
typedef ConstantBasis<Tensor>                CFDTensorBasis;
typedef ConstantBasis<Vector>                CFDVectorBasis;
typedef ConstantBasis<double>                CFDdoubleBasis;
typedef ConstantBasis<float>                 CFDfloatBasis;
typedef ConstantBasis<int>                   CFDintBasis;
typedef ConstantBasis<short>                 CFDshortBasis;
typedef ConstantBasis<char>                  CFDcharBasis;
typedef ConstantBasis<unsigned int>          CFDuintBasis;
typedef ConstantBasis<unsigned short>        CFDushortBasis;
typedef ConstantBasis<unsigned char>         CFDucharBasis;
typedef ConstantBasis<unsigned long>         CFDulongBasis;

//Linear
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
				      HVMesh::maker, HVMesh::maker);

//NoData
template class GenericField<HVMesh, NDBasis, vector<double> >;

//Constant
template class GenericField<HVMesh, CFDTensorBasis, vector<Tensor> >;       
template class GenericField<HVMesh, CFDVectorBasis, vector<Vector> >;       
template class GenericField<HVMesh, CFDdoubleBasis, vector<double> >;       
template class GenericField<HVMesh, CFDfloatBasis,  vector<float> >;        
template class GenericField<HVMesh, CFDintBasis,    vector<int> >;          
template class GenericField<HVMesh, CFDshortBasis,  vector<short> >;        
template class GenericField<HVMesh, CFDcharBasis,   vector<char> >;         
template class GenericField<HVMesh, CFDuintBasis,   vector<unsigned int> >; 
template class GenericField<HVMesh, CFDushortBasis, vector<unsigned short> >;
template class GenericField<HVMesh, CFDucharBasis,  vector<unsigned char> >;
template class GenericField<HVMesh, CFDulongBasis,  vector<unsigned long> >;

//Linear
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


PersistentTypeID 
backwards_compat_HVFT("HexVolField<Tensor>", "Field",
		      GenericField<HVMesh, FDTensorBasis, 
		      vector<Tensor> >::maker,
		      GenericField<HVMesh, CFDTensorBasis, 
		      vector<Tensor> >::maker);
PersistentTypeID 
backwards_compat_HVFV("HexVolField<Vector>", "Field",
		      GenericField<HVMesh, FDVectorBasis, 
		      vector<Vector> >::maker,
		      GenericField<HVMesh, CFDVectorBasis, 
		      vector<Vector> >::maker);
PersistentTypeID 
backwards_compat_HVFd("HexVolField<double>", "Field",
		      GenericField<HVMesh, FDdoubleBasis, 
		      vector<double> >::maker,
		      GenericField<HVMesh, CFDdoubleBasis, 
		      vector<double> >::maker,
		      GenericField<HVMesh, NDBasis, 
		      vector<double> >::maker);
PersistentTypeID 
backwards_compat_HVFf("HexVolField<float>", "Field",
		      GenericField<HVMesh, FDfloatBasis, 
		      vector<float> >::maker,
		      GenericField<HVMesh, CFDfloatBasis, 
		      vector<float> >::maker);
PersistentTypeID 
backwards_compat_HVFi("HexVolField<int>", "Field",
		      GenericField<HVMesh, FDintBasis, 
		      vector<int> >::maker,
		      GenericField<HVMesh, CFDintBasis, 
		      vector<int> >::maker);
PersistentTypeID 
backwards_compat_HVFs("HexVolField<short>", "Field",
		      GenericField<HVMesh, FDshortBasis, 
		      vector<short> >::maker,
		      GenericField<HVMesh, CFDshortBasis, 
		      vector<short> >::maker);
PersistentTypeID 
backwards_compat_HVFc("HexVolField<char>", "Field",
		      GenericField<HVMesh, FDcharBasis, 
		      vector<char> >::maker,
		      GenericField<HVMesh, CFDcharBasis, 
		      vector<char> >::maker);
PersistentTypeID 
backwards_compat_HVFui("HexVolField<unsigned_int>", "Field",
		       GenericField<HVMesh, FDuintBasis, 
		       vector<unsigned int> >::maker,
		       GenericField<HVMesh, CFDuintBasis, 
		       vector<unsigned int> >::maker);
PersistentTypeID 
backwards_compat_HVFus("HexVolField<unsigned_short>", "Field",
		       GenericField<HVMesh, FDushortBasis, 
		       vector<unsigned short> >::maker,
		       GenericField<HVMesh, CFDushortBasis, 
		       vector<unsigned short> >::maker);
PersistentTypeID 
backwards_compat_HVFuc("HexVolField<unsigned_char>", "Field",
		       GenericField<HVMesh, FDucharBasis, 
		       vector<unsigned char> >::maker,
		       GenericField<HVMesh, CFDucharBasis, 
		       vector<unsigned char> >::maker);
PersistentTypeID 
backwards_compat_HVFul("HexVolField<unsigned_long>", "Field",
		       GenericField<HVMesh, FDulongBasis, 
		       vector<unsigned long> >::maker,
		       GenericField<HVMesh, CFDulongBasis, 
		       vector<unsigned long> >::maker);


typedef HexTricubicHmt<double>             HTCdoubleBasis;

typedef HexVolMesh<HexTricubicHmt<Point> > HVCubMesh;
template class GenericField<HVCubMesh, NDBasis, vector<double> >; 
template class GenericField<HVCubMesh, HTCdoubleBasis, vector<double> >; 

typedef HexTricubicHmtScaleFactors<double>             HTCSFdoubleBasis;
typedef HexTricubicHmtScaleFactors<Vector>             HTCSFVectorBasis;

typedef HexVolMesh<HexTricubicHmtScaleFactors<Point> > HVCubSFMesh;
template class GenericField<HVCubSFMesh, NDBasis, vector<double> >; 
template class GenericField<HVCubSFMesh, HTCSFdoubleBasis, vector<double> >; 
template class GenericField<HVCubSFMesh, HTCSFVectorBasis, vector<Vector> >; 

typedef HexVolMesh<HexTricubicHmtScaleFactorsEdges<Point> > HVCubSFEMesh;
template class GenericField<HVCubSFEMesh, NDBasis, vector<double> >; 

