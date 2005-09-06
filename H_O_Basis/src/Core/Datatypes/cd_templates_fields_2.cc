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
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Basis/NoData.h>

using namespace SCIRun;


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
				      SLMesh::maker, true);
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

PersistentTypeID backwards_compat_SLFT("ScanlineField<Tensor>", "Field",
				       GenericField<SLMesh, FDTensorBasis, 
				       vector<Tensor> >::maker, true);
PersistentTypeID backwards_compat_SLFV("ScanlineField<Vector>", "Field",
				       GenericField<SLMesh, FDVectorBasis, 
				       vector<Vector> >::maker, true);
PersistentTypeID backwards_compat_SLFd("ScanlineField<double>", "Field",
				       GenericField<SLMesh, FDdoubleBasis, 
				       vector<double> >::maker, true);
PersistentTypeID backwards_compat_SLFf("ScanlineField<float>", "Field",
				       GenericField<SLMesh, FDfloatBasis, 
				       vector<float> >::maker, true);
PersistentTypeID backwards_compat_SLFi("ScanlineField<int>", "Field",
				       GenericField<SLMesh, FDintBasis, 
				       vector<int> >::maker, true);
PersistentTypeID backwards_compat_SLFs("ScanlineField<short>", "Field",
				       GenericField<SLMesh, FDshortBasis, 
				       vector<short> >::maker, true);
PersistentTypeID backwards_compat_SLFc("ScanlineField<char>", "Field",
				       GenericField<SLMesh, FDcharBasis, 
				       vector<char> >::maker, true);
PersistentTypeID backwards_compat_SLFui("ScanlineField<unsigned_int>", "Field",
				       GenericField<SLMesh, FDuintBasis, 
				       vector<unsigned int> >::maker, true);
PersistentTypeID backwards_compat_SLFus("ScanlineField<unsigned_short>", "Field",
				       GenericField<SLMesh, FDushortBasis, 
				       vector<unsigned short> >::maker, true);
PersistentTypeID backwards_compat_SLFuc("ScanlineField<unsigned_char>", "Field",
				       GenericField<SLMesh, FDucharBasis, 
				       vector<unsigned char> >::maker, true);
PersistentTypeID backwards_compat_SLFul("ScanlineField<unsigned_long>", "Field",
				       GenericField<SLMesh, FDulongBasis, 
				       vector<unsigned long> >::maker, true);


typedef ConstantBasis<Tensor>                FDCTensorBasis;
typedef ConstantBasis<Vector>                FDCVectorBasis;
typedef ConstantBasis<double>                FDCdoubleBasis;
typedef ConstantBasis<float>                 FDCfloatBasis;
typedef ConstantBasis<int>                   FDCintBasis;
typedef ConstantBasis<short>                 FDCshortBasis;
typedef ConstantBasis<char>                  FDCcharBasis;
typedef ConstantBasis<unsigned int>          FDCuintBasis;
typedef ConstantBasis<unsigned short>        FDCushortBasis;
typedef ConstantBasis<unsigned char>         FDCucharBasis;
typedef ConstantBasis<unsigned long>         FDCulongBasis;

typedef PointCloudMesh<ConstantBasis<Point> > PCMesh;
PersistentTypeID backwards_compat_PCM("PointCloudMesh", "Mesh",
				      PCMesh::maker, true);

template class GenericField<PCMesh, FDCTensorBasis, vector<Tensor> >;       
template class GenericField<PCMesh, FDCVectorBasis, vector<Vector> >;       
template class GenericField<PCMesh, FDCdoubleBasis, vector<double> >;       
template class GenericField<PCMesh, FDCfloatBasis,  vector<float> >;        
template class GenericField<PCMesh, FDCintBasis,    vector<int> >;          
template class GenericField<PCMesh, FDCshortBasis,  vector<short> >;        
template class GenericField<PCMesh, FDCcharBasis,   vector<char> >;         
template class GenericField<PCMesh, FDCuintBasis,   vector<unsigned int> >; 
template class GenericField<PCMesh, FDCushortBasis, vector<unsigned short> >;
template class GenericField<PCMesh, FDCucharBasis,  vector<unsigned char> >;
template class GenericField<PCMesh, FDCulongBasis,  vector<unsigned long> >;
template class GenericField<PCMesh, NoDataBasis<double>, vector<double> >; 

PersistentTypeID backwards_compat_PCFT("PointCloudField<Tensor>", "Field",
				       GenericField<PCMesh, FDCTensorBasis, 
				       vector<Tensor> >::maker, true);
PersistentTypeID backwards_compat_PCFV("PointCloudField<Vector>", "Field",
				       GenericField<PCMesh, FDCVectorBasis, 
				       vector<Vector> >::maker, true);
PersistentTypeID backwards_compat_PCFd("PointCloudField<double>", "Field",
				       GenericField<PCMesh, FDCdoubleBasis, 
				       vector<double> >::maker, true);
PersistentTypeID backwards_compat_PCFf("PointCloudField<float>", "Field",
				       GenericField<PCMesh, FDCfloatBasis, 
				       vector<float> >::maker, true);
PersistentTypeID backwards_compat_PCFi("PointCloudField<int>", "Field",
				       GenericField<PCMesh, FDCintBasis, 
				       vector<int> >::maker, true);
PersistentTypeID backwards_compat_PCFs("PointCloudField<short>", "Field",
				       GenericField<PCMesh, FDCshortBasis, 
				       vector<short> >::maker, true);
PersistentTypeID backwards_compat_PCFc("PointCloudField<char>", "Field",
				       GenericField<PCMesh, FDCcharBasis, 
				       vector<char> >::maker, true);
PersistentTypeID backwards_compat_PCFui("PointCloudField<unsigned_int>", "Field",
				       GenericField<PCMesh, FDCuintBasis, 
				       vector<unsigned int> >::maker, true);
PersistentTypeID backwards_compat_PCFus("PointCloudField<unsigned_short>", "Field",
				       GenericField<PCMesh, FDCushortBasis, 
				       vector<unsigned short> >::maker, true);
PersistentTypeID backwards_compat_PCFuc("PointCloudField<unsigned_char>", "Field",
				       GenericField<PCMesh, FDCucharBasis, 
				       vector<unsigned char> >::maker, true);
PersistentTypeID backwards_compat_PCFul("PointCloudField<unsigned_long>", "Field",
				       GenericField<PCMesh, FDCulongBasis, 
				       vector<unsigned long> >::maker, true);
