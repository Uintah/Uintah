/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/NoData.h>
#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/GenericField.h>

using namespace SCIRun;
//NoData
typedef NoDataBasis<double>                  NDBasis;

//Constant
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
typedef CrvLinearLgn<Tensor>                FDTensorBasis;
typedef CrvLinearLgn<Vector>                FDVectorBasis;
typedef CrvLinearLgn<double>                FDdoubleBasis;
typedef CrvLinearLgn<float>                 FDfloatBasis;
typedef CrvLinearLgn<int>                   FDintBasis;
typedef CrvLinearLgn<short>                 FDshortBasis;
typedef CrvLinearLgn<char>                  FDcharBasis;
typedef CrvLinearLgn<unsigned int>          FDuintBasis;
typedef CrvLinearLgn<unsigned short>        FDushortBasis;
typedef CrvLinearLgn<unsigned char>         FDucharBasis;
typedef CrvLinearLgn<unsigned long>         FDulongBasis;

typedef ScanlineMesh<CrvLinearLgn<Point> > SLMesh;
PersistentTypeID backwards_compat_SLM("ScanlineMesh", "Mesh",
				      SLMesh::maker, SLMesh::maker);
//NoData
template class GenericField<SLMesh, NDBasis, vector<double> >;

//Linear
template class GenericField<SLMesh, CFDTensorBasis, vector<Tensor> >;       
template class GenericField<SLMesh, CFDVectorBasis, vector<Vector> >;       
template class GenericField<SLMesh, CFDdoubleBasis, vector<double> >;       
template class GenericField<SLMesh, CFDfloatBasis,  vector<float> >;        
template class GenericField<SLMesh, CFDintBasis,    vector<int> >;          
template class GenericField<SLMesh, CFDshortBasis,  vector<short> >;        
template class GenericField<SLMesh, CFDcharBasis,   vector<char> >;         
template class GenericField<SLMesh, CFDuintBasis,   vector<unsigned int> >; 
template class GenericField<SLMesh, CFDushortBasis, vector<unsigned short> >;
template class GenericField<SLMesh, CFDucharBasis,  vector<unsigned char> >;
template class GenericField<SLMesh, CFDulongBasis,  vector<unsigned long> >;

//Linear
template class GenericField<SLMesh, FDTensorBasis, vector<Tensor> >;       
template class GenericField<SLMesh, FDVectorBasis, vector<Vector> >;       
template class GenericField<SLMesh, FDdoubleBasis, vector<double> >;       
template class GenericField<SLMesh, FDfloatBasis,  vector<float> >;        
template class GenericField<SLMesh, FDintBasis,    vector<int> >;          
template class GenericField<SLMesh, FDshortBasis,  vector<short> >;        
template class GenericField<SLMesh, FDcharBasis,   vector<char> >;         
template class GenericField<SLMesh, FDuintBasis,   vector<unsigned int> >; 
template class GenericField<SLMesh, FDushortBasis, vector<unsigned short> >;
template class GenericField<SLMesh, FDucharBasis,  vector<unsigned char> >;
template class GenericField<SLMesh, FDulongBasis,  vector<unsigned long> >;

PersistentTypeID 
backwards_compat_SLFT("ScanlineField<Tensor>", "Field",
		      GenericField<SLMesh, FDTensorBasis, 
		      vector<Tensor> >::maker, 
		      GenericField<SLMesh, CFDTensorBasis, 
		      vector<Tensor> >::maker);
PersistentTypeID 
backwards_compat_SLFV("ScanlineField<Vector>", "Field",
		      GenericField<SLMesh, FDVectorBasis, 
		      vector<Vector> >::maker, 
		      GenericField<SLMesh, CFDVectorBasis, 
		      vector<Vector> >::maker);
PersistentTypeID 
backwards_compat_SLFd("ScanlineField<double>", "Field",
		      GenericField<SLMesh, FDdoubleBasis, 
		      vector<double> >::maker, 
		      GenericField<SLMesh, CFDdoubleBasis, 
		      vector<double> >::maker,
		      GenericField<SLMesh, NDBasis, 
		      vector<double> >::maker);
PersistentTypeID 
backwards_compat_SLFf("ScanlineField<float>", "Field",
		      GenericField<SLMesh, FDfloatBasis, 
		      vector<float> >::maker, 
		      GenericField<SLMesh, CFDfloatBasis, 
		      vector<float> >::maker);
PersistentTypeID 
backwards_compat_SLFi("ScanlineField<int>", "Field",
		      GenericField<SLMesh, FDintBasis, 
		      vector<int> >::maker, 
		      GenericField<SLMesh, CFDintBasis, 
		      vector<int> >::maker);
PersistentTypeID 
backwards_compat_SLFs("ScanlineField<short>", "Field",
		      GenericField<SLMesh, FDshortBasis, 
		      vector<short> >::maker, 
		      GenericField<SLMesh, CFDshortBasis, 
		      vector<short> >::maker);
PersistentTypeID 
backwards_compat_SLFc("ScanlineField<char>", "Field",
		      GenericField<SLMesh, FDcharBasis, 
		      vector<char> >::maker, 
		      GenericField<SLMesh, CFDcharBasis, 
		      vector<char> >::maker);
PersistentTypeID 
backwards_compat_SLFui("ScanlineField<unsigned_int>", "Field",
		       GenericField<SLMesh, FDuintBasis, 
		       vector<unsigned int> >::maker, 
		       GenericField<SLMesh, CFDuintBasis, 
		       vector<unsigned int> >::maker);
PersistentTypeID 
backwards_compat_SLFus("ScanlineField<unsigned_short>", "Field",
		       GenericField<SLMesh, FDushortBasis, 
		       vector<unsigned short> >::maker, 
		       GenericField<SLMesh, CFDushortBasis, 
		       vector<unsigned short> >::maker);
PersistentTypeID 
backwards_compat_SLFuc("ScanlineField<unsigned_char>", "Field",
		       GenericField<SLMesh, FDucharBasis, 
		       vector<unsigned char> >::maker, 
		       GenericField<SLMesh, CFDucharBasis, 
		       vector<unsigned char> >::maker);
PersistentTypeID 
backwards_compat_SLFul("ScanlineField<unsigned_long>", "Field",
		       GenericField<SLMesh, FDulongBasis, 
		       vector<unsigned long> >::maker, 
		       GenericField<SLMesh, CFDulongBasis, 
		       vector<unsigned long> >::maker);

typedef PointCloudMesh<ConstantBasis<Point> > PCMesh;
PersistentTypeID backwards_compat_PCM("PointCloudMesh", "Mesh",
				      PCMesh::maker, PCMesh::maker);
//NoData
template class GenericField<PCMesh, NDBasis, vector<double> >;  

//Constant
template class GenericField<PCMesh, CFDTensorBasis, vector<Tensor> >;       
template class GenericField<PCMesh, CFDVectorBasis, vector<Vector> >;       
template class GenericField<PCMesh, CFDdoubleBasis, vector<double> >;       
template class GenericField<PCMesh, CFDfloatBasis,  vector<float> >;        
template class GenericField<PCMesh, CFDintBasis,    vector<int> >;          
template class GenericField<PCMesh, CFDshortBasis,  vector<short> >;        
template class GenericField<PCMesh, CFDcharBasis,   vector<char> >;         
template class GenericField<PCMesh, CFDuintBasis,   vector<unsigned int> >; 
template class GenericField<PCMesh, CFDushortBasis, vector<unsigned short> >;
template class GenericField<PCMesh, CFDucharBasis,  vector<unsigned char> >;
template class GenericField<PCMesh, CFDulongBasis,  vector<unsigned long> >;

PersistentTypeID 
backwards_compat_PCFT("PointCloudField<Tensor>", "Field",
		      GenericField<PCMesh, CFDTensorBasis, 
		      vector<Tensor> >::maker, 
		      GenericField<PCMesh, CFDTensorBasis, 
		      vector<Tensor> >::maker);
PersistentTypeID 
backwards_compat_PCFV("PointCloudField<Vector>", "Field",
		      GenericField<PCMesh, CFDVectorBasis, 
		      vector<Vector> >::maker, 
		      GenericField<PCMesh, CFDVectorBasis, 
		      vector<Vector> >::maker);
PersistentTypeID 
backwards_compat_PCFd("PointCloudField<double>", "Field",
		      GenericField<PCMesh, CFDdoubleBasis, 
		      vector<double> >::maker, 
		      GenericField<PCMesh, NDBasis, 
		      vector<double> >::maker);
PersistentTypeID 
backwards_compat_PCFf("PointCloudField<float>", "Field",
		      GenericField<PCMesh, CFDfloatBasis, 
		      vector<float> >::maker, 
		      GenericField<PCMesh, CFDfloatBasis, 
		      vector<float> >::maker);
PersistentTypeID 
backwards_compat_PCFi("PointCloudField<int>", "Field",
		      GenericField<PCMesh, CFDintBasis, 
		      vector<int> >::maker, 
		      GenericField<PCMesh, CFDintBasis, 
		      vector<int> >::maker);
PersistentTypeID 
backwards_compat_PCFs("PointCloudField<short>", "Field",
		      GenericField<PCMesh, CFDshortBasis, 
		      vector<short> >::maker, 
		      GenericField<PCMesh, CFDshortBasis, 
		      vector<short> >::maker);
PersistentTypeID 
backwards_compat_PCFc("PointCloudField<char>", "Field",
		      GenericField<PCMesh, CFDcharBasis, 
		      vector<char> >::maker, 
		      GenericField<PCMesh, CFDcharBasis, 
		      vector<char> >::maker);
PersistentTypeID 
backwards_compat_PCFui("PointCloudField<unsigned_int>", "Field",
		       GenericField<PCMesh, CFDuintBasis, 
		       vector<unsigned int> >::maker, 
		       GenericField<PCMesh, CFDuintBasis, 
		       vector<unsigned int> >::maker);
PersistentTypeID 
backwards_compat_PCFus("PointCloudField<unsigned_short>", "Field",
		       GenericField<PCMesh, CFDushortBasis, 
		       vector<unsigned short> >::maker, 
		       GenericField<PCMesh, CFDushortBasis, 
		       vector<unsigned short> >::maker);
PersistentTypeID 
backwards_compat_PCFuc("PointCloudField<unsigned_char>", "Field",
		       GenericField<PCMesh, CFDucharBasis, 
		       vector<unsigned char> >::maker, 
		       GenericField<PCMesh, CFDucharBasis, 
		       vector<unsigned char> >::maker);
PersistentTypeID 
backwards_compat_PCFul("PointCloudField<unsigned_long>", "Field",
		       GenericField<PCMesh, CFDulongBasis, 
		       vector<unsigned long> >::maker, 
		       GenericField<PCMesh, CFDulongBasis, 
		       vector<unsigned long> >::maker);
