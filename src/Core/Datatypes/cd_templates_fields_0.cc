/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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
#include <Core/Basis/Constant.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/MaskedLatVolMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/MultiLevelField.h>

//#include <Core/Datatypes/MaskedLatVolField.h>


using namespace SCIRun;
typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
PersistentTypeID backwards_compat_LVM("LatVolMesh", "Mesh",
				      LVMesh::maker,  
				      LVMesh::maker);

typedef HexTrilinearLgn<Tensor>             FDTensorBasis;
typedef HexTrilinearLgn<Vector>             FDVectorBasis;
typedef HexTrilinearLgn<double>             FDdoubleBasis;
typedef HexTrilinearLgn<float>              FDfloatBasis;
typedef HexTrilinearLgn<int>                FDintBasis;
typedef HexTrilinearLgn<short>              FDshortBasis;
typedef HexTrilinearLgn<char>               FDcharBasis;
typedef HexTrilinearLgn<unsigned int>       FDuintBasis;
typedef HexTrilinearLgn<unsigned short>     FDushortBasis;
typedef HexTrilinearLgn<unsigned char>      FDucharBasis;
typedef HexTrilinearLgn<unsigned long>      FDulongBasis;

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

typedef NoDataBasis<double>             NDBasis;
//NoData
template class GenericField<LVMesh, NDBasis,  FData3d<double, LVMesh> >;

//Constant
template class GenericField<LVMesh, CFDTensorBasis,  FData3d<Tensor, LVMesh> >;
template class GenericField<LVMesh, CFDVectorBasis,  FData3d<Vector, LVMesh> >;
template class GenericField<LVMesh, CFDdoubleBasis,  FData3d<double, LVMesh> >;
template class GenericField<LVMesh, CFDfloatBasis,   FData3d<float, LVMesh> >;
template class GenericField<LVMesh, CFDintBasis,     FData3d<int, LVMesh> >;
template class GenericField<LVMesh, CFDshortBasis,   FData3d<short, LVMesh> >;
template class GenericField<LVMesh, CFDcharBasis,    FData3d<char, LVMesh> >;
template class GenericField<LVMesh, CFDuintBasis,    
			    FData3d<unsigned int, LVMesh> >;
template class GenericField<LVMesh, CFDushortBasis,  
			    FData3d<unsigned short, LVMesh> >;
template class GenericField<LVMesh, CFDucharBasis,   
			    FData3d<unsigned char, LVMesh> >;
template class GenericField<LVMesh, CFDulongBasis,   
			    FData3d<unsigned long, LVMesh> >;

//Linear
template class GenericField<LVMesh, FDTensorBasis,  FData3d<Tensor, LVMesh> >;
template class GenericField<LVMesh, FDVectorBasis,  FData3d<Vector, LVMesh> >;
template class GenericField<LVMesh, FDdoubleBasis,  FData3d<double, LVMesh> >;
template class GenericField<LVMesh, FDfloatBasis,   FData3d<float, LVMesh> >;
template class GenericField<LVMesh, FDintBasis,     FData3d<int, LVMesh> >;
template class GenericField<LVMesh, FDshortBasis,   FData3d<short, LVMesh> >;
template class GenericField<LVMesh, FDcharBasis,    FData3d<char, LVMesh> >;
template class GenericField<LVMesh, FDuintBasis,    
			    FData3d<unsigned int, LVMesh> >;
template class GenericField<LVMesh, FDushortBasis,  
			    FData3d<unsigned short, LVMesh> >;
template class GenericField<LVMesh, FDucharBasis,   
			    FData3d<unsigned char, LVMesh> >;
template class GenericField<LVMesh, FDulongBasis,   
			    FData3d<unsigned long, LVMesh> >;

PersistentTypeID 
backwards_compat_LVFT("LatVolField<Tensor>", "Field",
		      GenericField<LVMesh, FDTensorBasis, 
		      FData3d<Tensor, LVMesh> >::maker, 
		      GenericField<LVMesh, CFDTensorBasis, 
		      FData3d<Tensor, LVMesh> >::maker);
PersistentTypeID 
backwards_compat_LVFV("LatVolField<Vector>", "Field",
		      GenericField<LVMesh, FDVectorBasis, 
		      FData3d<Vector, LVMesh> >::maker, 
		      GenericField<LVMesh, CFDVectorBasis, 
		      FData3d<Vector, LVMesh> >::maker);
PersistentTypeID 
backwards_compat_LVFd("LatVolField<double>", "Field",
		      GenericField<LVMesh, FDdoubleBasis, 
		      FData3d<double, LVMesh> >::maker, 
		      GenericField<LVMesh, CFDdoubleBasis, 
		      FData3d<double, LVMesh> >::maker,
		      GenericField<LVMesh, NDBasis, 
		      FData3d<double, LVMesh> >::maker);
PersistentTypeID 
backwards_compat_LVFf("LatVolField<float>", "Field",
		      GenericField<LVMesh, FDfloatBasis, 
		      FData3d<float, LVMesh> >::maker, 
		      GenericField<LVMesh, CFDfloatBasis, 
		      FData3d<float, LVMesh> >::maker);
PersistentTypeID 
backwards_compat_LVFi("LatVolField<int>", "Field",
		      GenericField<LVMesh, FDintBasis, 
		      FData3d<int, LVMesh> >::maker, 
		      GenericField<LVMesh, CFDintBasis, 
		      FData3d<int, LVMesh> >::maker);
PersistentTypeID 
backwards_compat_LVFs("LatVolField<short>", "Field",
		      GenericField<LVMesh, FDshortBasis, 
		      FData3d<short, LVMesh> >::maker, 
		      GenericField<LVMesh, CFDshortBasis, 
		      FData3d<short, LVMesh> >::maker);
PersistentTypeID 
backwards_compat_LVFc("LatVolField<char>", "Field",
		      GenericField<LVMesh, FDcharBasis, 
		      FData3d<char, LVMesh> >::maker, 
		      GenericField<LVMesh, CFDcharBasis, 
		      FData3d<char, LVMesh> >::maker);
PersistentTypeID 
backwards_compat_LVFui("LatVolField<unsigned_int>", "Field",
		       GenericField<LVMesh, FDuintBasis, 
		       FData3d<unsigned int, LVMesh> >::maker, 
		       GenericField<LVMesh, CFDuintBasis, 
		       FData3d<unsigned int, LVMesh> >::maker);
PersistentTypeID 
backwards_compat_LVFus("LatVolField<unsigned_short>", "Field",
		       GenericField<LVMesh, FDushortBasis, 
		       FData3d<unsigned short, LVMesh> >::maker,
		       GenericField<LVMesh, CFDushortBasis, 
		       FData3d<unsigned short, LVMesh> >::maker);
PersistentTypeID 
backwards_compat_LVFuc("LatVolField<unsigned_char>", "Field",
		       GenericField<LVMesh, FDucharBasis, 
		       FData3d<unsigned char, LVMesh> >::maker, 
		       GenericField<LVMesh, CFDucharBasis, 
		       FData3d<unsigned char, LVMesh> >::maker);
PersistentTypeID 
backwards_compat_LVFul("LatVolField<unsigned_long>", "Field",
		       GenericField<LVMesh, FDulongBasis, 
		       FData3d<unsigned long, LVMesh> >::maker, 
		       GenericField<LVMesh, CFDulongBasis, 
		       FData3d<unsigned long, LVMesh> >::maker);


typedef MaskedLatVolMesh<HexTrilinearLgn<Point> > MLVMesh;
PersistentTypeID backwards_compat_MLVM("MaskedLatVolMesh", "Mesh",
				      MLVMesh::maker, MLVMesh::maker);
//NoData
template class GenericField<MLVMesh, NDBasis, FData3d<double, MLVMesh> >;

//Contsant
template class GenericField<MLVMesh, CFDTensorBasis, 
			    FData3d<Tensor, MLVMesh> >;
template class GenericField<MLVMesh, CFDVectorBasis, 
			    FData3d<Vector, MLVMesh> >;
template class GenericField<MLVMesh, CFDdoubleBasis, 
			    FData3d<double, MLVMesh> >;
template class GenericField<MLVMesh, CFDfloatBasis,  
			    FData3d<float, MLVMesh> >;
template class GenericField<MLVMesh, CFDintBasis,    
			    FData3d<int, MLVMesh> >;
template class GenericField<MLVMesh, CFDshortBasis,  
			    FData3d<short, MLVMesh> >;
template class GenericField<MLVMesh, CFDcharBasis,   
			    FData3d<char, MLVMesh> >;
template class GenericField<MLVMesh, CFDuintBasis,   
			    FData3d<unsigned int, MLVMesh> >;
template class GenericField<MLVMesh, CFDushortBasis,  
			    FData3d<unsigned short, MLVMesh> >;
template class GenericField<MLVMesh, CFDucharBasis,   
			    FData3d<unsigned char, MLVMesh> >;
template class GenericField<MLVMesh, CFDulongBasis,   
			    FData3d<unsigned long, MLVMesh> >;

//Linear
template class GenericField<MLVMesh, FDTensorBasis, FData3d<Tensor, MLVMesh> >;
template class GenericField<MLVMesh, FDVectorBasis, FData3d<Vector, MLVMesh> >;
template class GenericField<MLVMesh, FDdoubleBasis, FData3d<double, MLVMesh> >;
template class GenericField<MLVMesh, FDfloatBasis,  FData3d<float, MLVMesh> >;
template class GenericField<MLVMesh, FDintBasis,    FData3d<int, MLVMesh> >;
template class GenericField<MLVMesh, FDshortBasis,  FData3d<short, MLVMesh> >;
template class GenericField<MLVMesh, FDcharBasis,   FData3d<char, MLVMesh> >;
template class GenericField<MLVMesh, FDuintBasis,   
			    FData3d<unsigned int, MLVMesh> >;
template class GenericField<MLVMesh, FDushortBasis,  
			    FData3d<unsigned short, MLVMesh> >;
template class GenericField<MLVMesh, FDucharBasis,   
			    FData3d<unsigned char, MLVMesh> >;
template class GenericField<MLVMesh, FDulongBasis,   
			    FData3d<unsigned long, MLVMesh> >;

PersistentTypeID 
backwards_compat_MLVFT("MaskedLatVolField<Tensor>", "Field",
		       GenericField<MLVMesh, FDTensorBasis, 
		       FData3d<Tensor, MLVMesh> >::maker,
		       GenericField<MLVMesh, CFDTensorBasis, 
		       FData3d<Tensor, MLVMesh> >::maker);
PersistentTypeID 
backwards_compat_MLVFV("MaskedLatVolField<Vector>", "Field",
		       GenericField<MLVMesh, FDVectorBasis, 
		       FData3d<Vector, MLVMesh> >::maker,
		       GenericField<MLVMesh, CFDVectorBasis, 
		       FData3d<Vector, MLVMesh> >::maker);
PersistentTypeID 
backwards_compat_MLVFd("MaskedLatVolField<double>", "Field",
		       GenericField<MLVMesh, FDdoubleBasis, 
		       FData3d<double, MLVMesh> >::maker,
		       GenericField<MLVMesh, CFDdoubleBasis, 
		       FData3d<double, MLVMesh> >::maker);
PersistentTypeID 
backwards_compat_MLVFf("MaskedLatVolField<float>", "Field",
		       GenericField<MLVMesh, FDfloatBasis, 
		       FData3d<float, MLVMesh> >::maker,
		       GenericField<MLVMesh, CFDfloatBasis, 
		       FData3d<float, MLVMesh> >::maker);
PersistentTypeID 
backwards_compat_MLVFi("MaskedLatVolField<int>", "Field",
		       GenericField<MLVMesh, FDintBasis, 
		       FData3d<int, MLVMesh> >::maker,
		       GenericField<MLVMesh, CFDintBasis, 
		       FData3d<int, MLVMesh> >::maker);
PersistentTypeID 
backwards_compat_MLVFs("MaskedLatVolField<short>", "Field",
		       GenericField<MLVMesh, FDshortBasis, 
		       FData3d<short, MLVMesh> >::maker,
		       GenericField<MLVMesh, CFDshortBasis, 
		       FData3d<short, MLVMesh> >::maker);
PersistentTypeID 
backwards_compat_MLVFc("MaskedLatVolField<char>", "Field",
		       GenericField<MLVMesh, FDcharBasis, 
		       FData3d<char, MLVMesh> >::maker,
		       GenericField<MLVMesh, CFDcharBasis, 
		       FData3d<char, MLVMesh> >::maker);
PersistentTypeID 
backwards_compat_MLVFui("MaskedLatVolField<unsigned_int>", "Field",
			GenericField<MLVMesh, FDuintBasis, 
			FData3d<unsigned int, MLVMesh> >::maker,
			GenericField<MLVMesh, CFDuintBasis, 
			FData3d<unsigned int, MLVMesh> >::maker);
PersistentTypeID 
backwards_compat_MLVFus("MaskedLatVolField<unsigned_short>", "Field",
			GenericField<MLVMesh, FDushortBasis, 
			FData3d<unsigned short, MLVMesh> >::maker,
			GenericField<MLVMesh, CFDushortBasis, 
			FData3d<unsigned short, MLVMesh> >::maker);
PersistentTypeID 
backwards_compat_MLVFuc("MaskedLatVolField<unsigned_char>", "Field",
			GenericField<MLVMesh, FDucharBasis, 
			FData3d<unsigned char, MLVMesh> >::maker,
			GenericField<MLVMesh, CFDucharBasis, 
			FData3d<unsigned char, MLVMesh> >::maker);
PersistentTypeID 
backwards_compat_MLVFul("MaskedLatVolField<unsigned_long>", "Field",
			GenericField<MLVMesh, FDulongBasis, 
			FData3d<unsigned long, MLVMesh> >::maker,
			GenericField<MLVMesh, CFDulongBasis, 
			FData3d<unsigned long, MLVMesh> >::maker);



// const TypeDescription* get_type_description(MultiLevelField<Tensor> *);
// const TypeDescription* get_type_description(MultiLevelField<Vector> *);
// const TypeDescription* get_type_description(MultiLevelField<double> *);
// const TypeDescription* get_type_description(MultiLevelField<float> *);
// const TypeDescription* get_type_description(MultiLevelField<int> *);
// const TypeDescription* get_type_description(MultiLevelField<short> *);
// const TypeDescription* get_type_description(MultiLevelField<char> *);
// const TypeDescription* get_type_description(MultiLevelField<unsigned int> *);
// const TypeDescription* get_type_description(MultiLevelField<unsigned short> *);
// const TypeDescription* get_type_description(MultiLevelField<unsigned char> *);

//NoData
template class MultiLevelField<LVMesh, NDBasis,  FData3d<double, LVMesh> >;

//Constant
template class MultiLevelField<LVMesh, CFDTensorBasis,  
                             FData3d<Tensor, LVMesh> >;
template class MultiLevelField<LVMesh, CFDVectorBasis,  
                             FData3d<Vector, LVMesh> >;
template class MultiLevelField<LVMesh, CFDdoubleBasis,  
                             FData3d<double, LVMesh> >;
template class MultiLevelField<LVMesh, CFDfloatBasis,   FData3d<float, LVMesh> >;
template class MultiLevelField<LVMesh, CFDintBasis,     FData3d<int, LVMesh> >;
template class MultiLevelField<LVMesh, CFDshortBasis,   FData3d<short, LVMesh> >;
template class MultiLevelField<LVMesh, CFDcharBasis,    FData3d<char, LVMesh> >;
template class MultiLevelField<LVMesh, CFDuintBasis,    
			    FData3d<unsigned int, LVMesh> >;
template class MultiLevelField<LVMesh, CFDushortBasis,  
			    FData3d<unsigned short, LVMesh> >;
template class MultiLevelField<LVMesh, CFDucharBasis,   
			    FData3d<unsigned char, LVMesh> >;
template class MultiLevelField<LVMesh, CFDulongBasis,   
			    FData3d<unsigned long, LVMesh> >;

//Linear
template class MultiLevelField<LVMesh, FDTensorBasis,  FData3d<Tensor, LVMesh> >;
template class MultiLevelField<LVMesh, FDVectorBasis,  FData3d<Vector, LVMesh> >;
template class MultiLevelField<LVMesh, FDdoubleBasis,  FData3d<double, LVMesh> >;
template class MultiLevelField<LVMesh, FDfloatBasis,   FData3d<float, LVMesh> >;
template class MultiLevelField<LVMesh, FDintBasis,     FData3d<int, LVMesh> >;
template class MultiLevelField<LVMesh, FDshortBasis,   FData3d<short, LVMesh> >;
template class MultiLevelField<LVMesh, FDcharBasis,    FData3d<char, LVMesh> >;
template class MultiLevelField<LVMesh, FDuintBasis,    
			    FData3d<unsigned int, LVMesh> >;
template class MultiLevelField<LVMesh, FDushortBasis,  
			    FData3d<unsigned short, LVMesh> >;
template class MultiLevelField<LVMesh, FDucharBasis,   
			    FData3d<unsigned char, LVMesh> >;
template class MultiLevelField<LVMesh, FDulongBasis,   
			    FData3d<unsigned long, LVMesh> >;



