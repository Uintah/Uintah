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
#include <Core/Basis/Constant.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/GenericField.h>

using namespace SCIRun;

//NoData
typedef NoDataBasis<double>                  NDBasis;

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
PersistentTypeID backwards_compat_TSM("TriSurfMesh", "Mesh",
				      TSMesh::maker, TSMesh::maker);

//noData
template class GenericField<TSMesh, NDBasis, vector<double> >;   

//Constant
template class GenericField<TSMesh, CFDTensorBasis, vector<Tensor> >;       
template class GenericField<TSMesh, CFDVectorBasis, vector<Vector> >;       
template class GenericField<TSMesh, CFDdoubleBasis, vector<double> >;       
template class GenericField<TSMesh, CFDfloatBasis,  vector<float> >;        
template class GenericField<TSMesh, CFDintBasis,    vector<int> >;          
template class GenericField<TSMesh, CFDshortBasis,  vector<short> >;        
template class GenericField<TSMesh, CFDcharBasis,   vector<char> >;         
template class GenericField<TSMesh, CFDuintBasis,   vector<unsigned int> >; 
template class GenericField<TSMesh, CFDushortBasis, vector<unsigned short> >;
template class GenericField<TSMesh, CFDucharBasis,  vector<unsigned char> >;
template class GenericField<TSMesh, CFDulongBasis,  vector<unsigned long> >;

//Linear
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

PersistentTypeID 
backwards_compat_TSFT("TriSurfField<Tensor>", "Field",
		      GenericField<TSMesh, FDTensorBasis, 
		      vector<Tensor> >::maker,
		      GenericField<TSMesh, CFDTensorBasis, 
		      vector<Tensor> >::maker);
PersistentTypeID 
backwards_compat_TSFV("TriSurfField<Vector>", "Field",
		      GenericField<TSMesh, FDVectorBasis, 
		      vector<Vector> >::maker,
		      GenericField<TSMesh, CFDVectorBasis, 
		      vector<Vector> >::maker);
PersistentTypeID 
backwards_compat_TSFd("TriSurfField<double>", "Field",
		      GenericField<TSMesh, FDdoubleBasis, 
		      vector<double> >::maker,
		      GenericField<TSMesh, CFDdoubleBasis, 
		      vector<double> >::maker);
PersistentTypeID 
backwards_compat_TSFf("TriSurfField<float>", "Field",
		      GenericField<TSMesh, FDfloatBasis, 
		      vector<float> >::maker,
		      GenericField<TSMesh, CFDfloatBasis, 
		      vector<float> >::maker);
PersistentTypeID 
backwards_compat_TSFi("TriSurfField<int>", "Field",
		      GenericField<TSMesh, FDintBasis, 
		      vector<int> >::maker,
		      GenericField<TSMesh, CFDintBasis, 
		      vector<int> >::maker);
PersistentTypeID 
backwards_compat_TSFs("TriSurfField<short>", "Field",
		      GenericField<TSMesh, FDshortBasis, 
		      vector<short> >::maker,
		      GenericField<TSMesh, CFDshortBasis, 
		      vector<short> >::maker);
PersistentTypeID 
backwards_compat_TSFc("TriSurfField<char>", "Field",
		      GenericField<TSMesh, FDcharBasis, 
		      vector<char> >::maker,
		      GenericField<TSMesh, CFDcharBasis, 
		      vector<char> >::maker);
PersistentTypeID 
backwards_compat_TSFui("TriSurfField<unsigned_int>", "Field",
		       GenericField<TSMesh, FDuintBasis, 
		       vector<unsigned int> >::maker,
		       GenericField<TSMesh, CFDuintBasis, 
		       vector<unsigned int> >::maker);
PersistentTypeID 
backwards_compat_TSFus("TriSurfField<unsigned_short>", "Field",
		       GenericField<TSMesh, FDushortBasis, 
		       vector<unsigned short> >::maker,
		       GenericField<TSMesh, CFDushortBasis, 
		       vector<unsigned short> >::maker);
PersistentTypeID 
backwards_compat_TSFuc("TriSurfField<unsigned_char>", "Field",
		       GenericField<TSMesh, FDucharBasis, 
		       vector<unsigned char> >::maker,
		       GenericField<TSMesh, CFDucharBasis, 
		       vector<unsigned char> >::maker);
PersistentTypeID 
backwards_compat_TSFul("TriSurfField<unsigned_long>", "Field",
		       GenericField<TSMesh, FDulongBasis, 
		       vector<unsigned long> >::maker,
		       GenericField<TSMesh, CFDulongBasis, 
		       vector<unsigned long> >::maker);

//Linear
typedef CrvLinearLgn<Tensor>                CrFDTensorBasis;
typedef CrvLinearLgn<Vector>                CrFDVectorBasis;
typedef CrvLinearLgn<double>                CrFDdoubleBasis;
typedef CrvLinearLgn<float>                 CrFDfloatBasis;
typedef CrvLinearLgn<int>                   CrFDintBasis;
typedef CrvLinearLgn<short>                 CrFDshortBasis;
typedef CrvLinearLgn<char>                  CrFDcharBasis;
typedef CrvLinearLgn<unsigned int>          CrFDuintBasis;
typedef CrvLinearLgn<unsigned short>        CrFDushortBasis;
typedef CrvLinearLgn<unsigned char>         CrFDucharBasis;
typedef CrvLinearLgn<unsigned long>         CrFDulongBasis;

typedef CurveMesh<CrvLinearLgn<Point> > CMesh;
PersistentTypeID backwards_compat_CM("CurveMesh", "Mesh",
				      CMesh::maker, CMesh::maker);

//NoData
template class GenericField<CMesh, NDBasis,  vector<double> >;  

//Constant
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

//Linear
template class GenericField<CMesh, CrFDTensorBasis, vector<Tensor> >;       
template class GenericField<CMesh, CrFDVectorBasis, vector<Vector> >;       
template class GenericField<CMesh, CrFDdoubleBasis, vector<double> >;       
template class GenericField<CMesh, CrFDfloatBasis,  vector<float> >;        
template class GenericField<CMesh, CrFDintBasis,    vector<int> >;          
template class GenericField<CMesh, CrFDshortBasis,  vector<short> >;        
template class GenericField<CMesh, CrFDcharBasis,   vector<char> >;         
template class GenericField<CMesh, CrFDuintBasis,   vector<unsigned int> >; 
template class GenericField<CMesh, CrFDushortBasis, vector<unsigned short> >;
template class GenericField<CMesh, CrFDucharBasis,  vector<unsigned char> >;
template class GenericField<CMesh, CrFDulongBasis,  vector<unsigned long> >;

PersistentTypeID 
backwards_compat_CFT("CurveField<Tensor>", "Field",
		     GenericField<CMesh, CrFDTensorBasis, 
		     vector<Tensor> >::maker,
		     GenericField<CMesh, CFDTensorBasis, 
		     vector<Tensor> >::maker);
PersistentTypeID 
backwards_compat_CFV("CurveField<Vector>", "Field",
		     GenericField<CMesh, CrFDVectorBasis, 
		     vector<Vector> >::maker,
		     GenericField<CMesh, CFDVectorBasis, 
		     vector<Vector> >::maker);
PersistentTypeID 
backwards_compat_CFd("CurveField<double>", "Field",
		     GenericField<CMesh, CrFDdoubleBasis, 
		     vector<double> >::maker,
		     GenericField<CMesh, CFDdoubleBasis, 
		     vector<double> >::maker,
		     GenericField<CMesh, NDBasis, 
		     vector<double> >::maker);
PersistentTypeID 
backwards_compat_CFf("CurveField<float>", "Field",
		     GenericField<CMesh, CrFDfloatBasis, 
		     vector<float> >::maker,
		     GenericField<CMesh, CFDfloatBasis, 
		     vector<float> >::maker);
PersistentTypeID 
backwards_compat_CFi("CurveField<int>", "Field",
		     GenericField<CMesh, CrFDintBasis, 
		     vector<int> >::maker,
		     GenericField<CMesh, CFDintBasis, 
		     vector<int> >::maker);
PersistentTypeID 
backwards_compat_CFs("CurveField<short>", "Field",
		     GenericField<CMesh, CrFDshortBasis, 
		     vector<short> >::maker,
		     GenericField<CMesh, CFDshortBasis, 
		     vector<short> >::maker);
PersistentTypeID 
backwards_compat_CFc("CurveField<char>", "Field",
		     GenericField<CMesh, CrFDcharBasis, 
		     vector<char> >::maker,
		     GenericField<CMesh, CFDcharBasis, 
		     vector<char> >::maker);
PersistentTypeID 
backwards_compat_CFui("CurveField<unsigned_int>", "Field",
		      GenericField<CMesh, CrFDuintBasis, 
		      vector<unsigned int> >::maker,
		      GenericField<CMesh, CFDuintBasis, 
		      vector<unsigned int> >::maker);
PersistentTypeID 
backwards_compat_CFus("CurveField<unsigned_short>", "Field",
		      GenericField<CMesh, CrFDushortBasis, 
		      vector<unsigned short> >::maker,
		      GenericField<CMesh, CFDushortBasis, 
		      vector<unsigned short> >::maker);
PersistentTypeID 
backwards_compat_CFuc("CurveField<unsigned_char>", "Field",
		      GenericField<CMesh, CrFDucharBasis, 
		      vector<unsigned char> >::maker,
		      GenericField<CMesh, CFDucharBasis, 
		      vector<unsigned char> >::maker);
PersistentTypeID 
backwards_compat_CFul("CurveField<unsigned_long>", "Field",
		      GenericField<CMesh, CrFDulongBasis, 
		      vector<unsigned long> >::maker,
		      GenericField<CMesh, CFDulongBasis, 
		      vector<unsigned long> >::maker);

