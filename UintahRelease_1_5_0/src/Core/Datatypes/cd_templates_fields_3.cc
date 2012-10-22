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
#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/TetQuadraticLgn.h>
#include <Core/Basis/HexTriquadraticLgn.h>
#include <Core/Basis/PrismLinearLgn.h>
#include <Core/Datatypes/PrismVolMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
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
typedef PrismLinearLgn<Tensor>                PFDTensorBasis;
typedef PrismLinearLgn<Vector>                PFDVectorBasis;
typedef PrismLinearLgn<double>                PFDdoubleBasis;
typedef PrismLinearLgn<float>                 PFDfloatBasis;
typedef PrismLinearLgn<int>                   PFDintBasis;
typedef PrismLinearLgn<short>                 PFDshortBasis;
typedef PrismLinearLgn<char>                  PFDcharBasis;
typedef PrismLinearLgn<unsigned int>          PFDuintBasis;
typedef PrismLinearLgn<unsigned short>        PFDushortBasis;
typedef PrismLinearLgn<unsigned char>         PFDucharBasis;
typedef PrismLinearLgn<unsigned long>         PFDulongBasis;

typedef PrismVolMesh<PrismLinearLgn<Point> > PVMesh;
PersistentTypeID backwards_compat_PVM("PrismVolMesh", "Mesh",
				      PVMesh::maker, PVMesh::maker);

//NoData
template class GenericField<PVMesh, NDBasis, vector<double> >;  

//Constant
template class GenericField<PVMesh, CFDTensorBasis, vector<Tensor> >;       
template class GenericField<PVMesh, CFDVectorBasis, vector<Vector> >;       
template class GenericField<PVMesh, CFDdoubleBasis, vector<double> >;       
template class GenericField<PVMesh, CFDfloatBasis,  vector<float> >;        
template class GenericField<PVMesh, CFDintBasis,    vector<int> >;          
template class GenericField<PVMesh, CFDshortBasis,  vector<short> >;        
template class GenericField<PVMesh, CFDcharBasis,   vector<char> >;         
template class GenericField<PVMesh, CFDuintBasis,   vector<unsigned int> >; 
template class GenericField<PVMesh, CFDushortBasis, vector<unsigned short> >;
template class GenericField<PVMesh, CFDucharBasis,  vector<unsigned char> >;
template class GenericField<PVMesh, CFDulongBasis,  vector<unsigned long> >;

//Linear
template class GenericField<PVMesh, PFDTensorBasis, vector<Tensor> >;       
template class GenericField<PVMesh, PFDVectorBasis, vector<Vector> >;       
template class GenericField<PVMesh, PFDdoubleBasis, vector<double> >;       
template class GenericField<PVMesh, PFDfloatBasis,  vector<float> >;        
template class GenericField<PVMesh, PFDintBasis,    vector<int> >;          
template class GenericField<PVMesh, PFDshortBasis,  vector<short> >;        
template class GenericField<PVMesh, PFDcharBasis,   vector<char> >;         
template class GenericField<PVMesh, PFDuintBasis,   vector<unsigned int> >; 
template class GenericField<PVMesh, PFDushortBasis, vector<unsigned short> >;
template class GenericField<PVMesh, PFDucharBasis,  vector<unsigned char> >;
template class GenericField<PVMesh, PFDulongBasis,  vector<unsigned long> >;

PersistentTypeID 
backwards_compat_PVFT("PrismVolField<Tensor>", "Field",
		      GenericField<PVMesh, PFDTensorBasis, 
		      vector<Tensor> >::maker, 
		      GenericField<PVMesh, CFDTensorBasis, 
		      vector<Tensor> >::maker);
PersistentTypeID 
backwards_compat_PVFV("PrismVolField<Vector>", "Field",
		      GenericField<PVMesh, PFDVectorBasis, 
		      vector<Vector> >::maker,
		      GenericField<PVMesh, CFDVectorBasis, 
		      vector<Vector> >::maker);
PersistentTypeID 
backwards_compat_PVFd("PrismVolField<double>", "Field",
		      GenericField<PVMesh, PFDdoubleBasis, 
		      vector<double> >::maker,
		      GenericField<PVMesh, CFDdoubleBasis, 
		      vector<double> >::maker,
		      GenericField<PVMesh, NDBasis, 
		      vector<double> >::maker);
PersistentTypeID 
backwards_compat_PVFf("PrismVolField<float>", "Field",
		      GenericField<PVMesh, PFDfloatBasis, 
		      vector<float> >::maker,
		      GenericField<PVMesh, CFDfloatBasis, 
		      vector<float> >::maker);
PersistentTypeID 
backwards_compat_PVFi("PrismVolField<int>", "Field",
		      GenericField<PVMesh, PFDintBasis, 
		      vector<int> >::maker,
		      GenericField<PVMesh, CFDintBasis, 
		      vector<int> >::maker);
PersistentTypeID 
backwards_compat_PVFs("PrismVolField<short>", "Field",
		      GenericField<PVMesh, PFDshortBasis, 
		      vector<short> >::maker,
		      GenericField<PVMesh, CFDshortBasis, 
		      vector<short> >::maker);
PersistentTypeID 
backwards_compat_PVFc("PrismVolField<char>", "Field",
		      GenericField<PVMesh, PFDcharBasis, 
		      vector<char> >::maker,
		      GenericField<PVMesh, CFDcharBasis, 
		      vector<char> >::maker);
PersistentTypeID 
backwards_compat_PVFui("PrismVolField<unsigned_int>", "Field",
		       GenericField<PVMesh, PFDuintBasis, 
		       vector<unsigned int> >::maker,
		       GenericField<PVMesh, CFDuintBasis, 
		       vector<unsigned int> >::maker);
PersistentTypeID 
backwards_compat_PVFus("PrismVolField<unsigned_short>", "Field",
		       GenericField<PVMesh, PFDushortBasis, 
		       vector<unsigned short> >::maker,
		       GenericField<PVMesh, CFDushortBasis, 
		       vector<unsigned short> >::maker);
PersistentTypeID 
backwards_compat_PVFuc("PrismVolField<unsigned_char>", "Field",
		       GenericField<PVMesh, PFDucharBasis, 
		       vector<unsigned char> >::maker,
		       GenericField<PVMesh, CFDucharBasis, 
		       vector<unsigned char> >::maker);
PersistentTypeID 
backwards_compat_PVFul("PrismVolField<unsigned_long>", "Field",
		       GenericField<PVMesh, PFDulongBasis, 
		       vector<unsigned long> >::maker,
		       GenericField<PVMesh, CFDulongBasis, 
		       vector<unsigned long> >::maker);

typedef TetLinearLgn<Tensor>                TFDTensorBasis;
typedef TetLinearLgn<Vector>                TFDVectorBasis;
typedef TetLinearLgn<double>                TFDdoubleBasis;
typedef TetLinearLgn<float>                 TFDfloatBasis;
typedef TetLinearLgn<int>                   TFDintBasis;
typedef TetLinearLgn<short>                 TFDshortBasis;
typedef TetLinearLgn<char>                  TFDcharBasis;
typedef TetLinearLgn<unsigned int>          TFDuintBasis;
typedef TetLinearLgn<unsigned short>        TFDushortBasis;
typedef TetLinearLgn<unsigned char>         TFDucharBasis;
typedef TetLinearLgn<unsigned long>         TFDulongBasis;

typedef TetVolMesh<TetLinearLgn<Point> > TVMesh;
PersistentTypeID backwards_compat_TVM("TetVolMesh", "Mesh",
				      TVMesh::maker, TVMesh::maker);

//NoData
template class GenericField<TVMesh, NDBasis, vector<double> >;  

//Constant
template class GenericField<TVMesh, CFDTensorBasis, vector<Tensor> >;       
template class GenericField<TVMesh, CFDVectorBasis, vector<Vector> >;       
template class GenericField<TVMesh, CFDdoubleBasis, vector<double> >;       
template class GenericField<TVMesh, CFDfloatBasis,  vector<float> >;        
template class GenericField<TVMesh, CFDintBasis,    vector<int> >;          
template class GenericField<TVMesh, CFDshortBasis,  vector<short> >;        
template class GenericField<TVMesh, CFDcharBasis,   vector<char> >;         
template class GenericField<TVMesh, CFDuintBasis,   vector<unsigned int> >; 
template class GenericField<TVMesh, CFDushortBasis, vector<unsigned short> >;
template class GenericField<TVMesh, CFDucharBasis,  vector<unsigned char> >;
template class GenericField<TVMesh, CFDulongBasis,  vector<unsigned long> >;

//Linear
template class GenericField<TVMesh, TFDTensorBasis, vector<Tensor> >;       
template class GenericField<TVMesh, TFDVectorBasis, vector<Vector> >;       
template class GenericField<TVMesh, TFDdoubleBasis, vector<double> >;       
template class GenericField<TVMesh, TFDfloatBasis,  vector<float> >;        
template class GenericField<TVMesh, TFDintBasis,    vector<int> >;          
template class GenericField<TVMesh, TFDshortBasis,  vector<short> >;        
template class GenericField<TVMesh, TFDcharBasis,   vector<char> >;         
template class GenericField<TVMesh, TFDuintBasis,   vector<unsigned int> >; 
template class GenericField<TVMesh, TFDushortBasis, vector<unsigned short> >;
template class GenericField<TVMesh, TFDucharBasis,  vector<unsigned char> >;
template class GenericField<TVMesh, TFDulongBasis,  vector<unsigned long> >;

PersistentTypeID 
backwards_compat_TVFT("TetVolField<Tensor>", "Field",
		      GenericField<TVMesh, TFDTensorBasis, 
		      vector<Tensor> >::maker,
		      GenericField<TVMesh, CFDTensorBasis, 
		      vector<Tensor> >::maker);
PersistentTypeID 
backwards_compat_TVFV("TetVolField<Vector>", "Field",
		      GenericField<TVMesh, TFDVectorBasis, 
		      vector<Vector> >::maker,
		      GenericField<TVMesh, CFDVectorBasis, 
		      vector<Vector> >::maker);
PersistentTypeID 
backwards_compat_TVFd("TetVolField<double>", "Field",
		      GenericField<TVMesh, TFDdoubleBasis, 
		      vector<double> >::maker,
		      GenericField<TVMesh, CFDdoubleBasis, 
		      vector<double> >::maker,
		      GenericField<TVMesh, NDBasis, 
		      vector<double> >::maker);
PersistentTypeID 
backwards_compat_TVFf("TetVolField<float>", "Field",
		      GenericField<TVMesh, TFDfloatBasis, 
		      vector<float> >::maker,
		      GenericField<TVMesh, CFDfloatBasis, 
		      vector<float> >::maker);
PersistentTypeID 
backwards_compat_TVFi("TetVolField<int>", "Field",
		      GenericField<TVMesh, TFDintBasis, 
		      vector<int> >::maker,
		      GenericField<TVMesh, CFDintBasis, 
		      vector<int> >::maker);
PersistentTypeID 
backwards_compat_TVFs("TetVolField<short>", "Field",
		      GenericField<TVMesh, TFDshortBasis, 
		      vector<short> >::maker,
		      GenericField<TVMesh, CFDshortBasis, 
		      vector<short> >::maker);
PersistentTypeID 
backwards_compat_TVFc("TetVolField<char>", "Field",
		      GenericField<TVMesh, TFDcharBasis, 
		      vector<char> >::maker,
		      GenericField<TVMesh, CFDcharBasis, 
		      vector<char> >::maker);
PersistentTypeID 
backwards_compat_TVFui("TetVolField<unsigned_int>", "Field",
		       GenericField<TVMesh, TFDuintBasis, 
		       vector<unsigned int> >::maker,
		       GenericField<TVMesh, CFDuintBasis, 
		       vector<unsigned int> >::maker);
PersistentTypeID 
backwards_compat_TVFus("TetVolField<unsigned_short>", "Field",
		       GenericField<TVMesh, TFDushortBasis, 
		       vector<unsigned short> >::maker,
		       GenericField<TVMesh, CFDushortBasis, 
		       vector<unsigned short> >::maker);
PersistentTypeID 
backwards_compat_TVFuc("TetVolField<unsigned_char>", "Field",
		       GenericField<TVMesh, TFDucharBasis, 
		       vector<unsigned char> >::maker,
		       GenericField<TVMesh, CFDucharBasis, 
		       vector<unsigned char> >::maker);
PersistentTypeID 
backwards_compat_TVFul("TetVolField<unsigned_long>", "Field",
		       GenericField<TVMesh, TFDulongBasis, 
		       vector<unsigned long> >::maker,
		       GenericField<TVMesh, CFDulongBasis, 
		       vector<unsigned long> >::maker);


typedef TetQuadraticLgn<Tensor>                TQFDTensorBasis;
typedef TetQuadraticLgn<Vector>                TQFDVectorBasis;
typedef TetQuadraticLgn<double>                TQFDdoubleBasis;
typedef TetQuadraticLgn<float>                 TQFDfloatBasis;
typedef TetQuadraticLgn<int>                   TQFDintBasis;
typedef TetQuadraticLgn<short>                 TQFDshortBasis;
typedef TetQuadraticLgn<char>                  TQFDcharBasis;
typedef TetQuadraticLgn<unsigned int>          TQFDuintBasis;
typedef TetQuadraticLgn<unsigned short>        TQFDushortBasis;
typedef TetQuadraticLgn<unsigned char>         TQFDucharBasis;
typedef TetQuadraticLgn<unsigned long>         TQFDulongBasis;

typedef TetVolMesh<TetQuadraticLgn<Point> > QTVMesh;
template class GenericField<QTVMesh, TQFDTensorBasis, vector<Tensor> >;       
template class GenericField<QTVMesh, TQFDVectorBasis, vector<Vector> >;       
template class GenericField<QTVMesh, TQFDdoubleBasis, vector<double> >;       
template class GenericField<QTVMesh, TQFDfloatBasis,  vector<float> >;        
template class GenericField<QTVMesh, TQFDintBasis,    vector<int> >;          
template class GenericField<QTVMesh, TQFDshortBasis,  vector<short> >;        
template class GenericField<QTVMesh, TQFDcharBasis,   vector<char> >;         
template class GenericField<QTVMesh, TQFDuintBasis,   vector<unsigned int> >; 
template class GenericField<QTVMesh, TQFDushortBasis, vector<unsigned short> >;
template class GenericField<QTVMesh, TQFDucharBasis,  vector<unsigned char> >;
template class GenericField<QTVMesh, TQFDulongBasis,  vector<unsigned long> >;


typedef HexTriquadraticLgn<Tensor>                QHFDTensorBasis;
typedef HexTriquadraticLgn<Vector>                QHFDVectorBasis;
typedef HexTriquadraticLgn<double>                QHFDdoubleBasis;
typedef HexTriquadraticLgn<float>                 QHFDfloatBasis;
typedef HexTriquadraticLgn<int>                   QHFDintBasis;
typedef HexTriquadraticLgn<short>                 QHFDshortBasis;
typedef HexTriquadraticLgn<char>                  QHFDcharBasis;
typedef HexTriquadraticLgn<unsigned int>          QHFDuintBasis;
typedef HexTriquadraticLgn<unsigned short>        QHFDushortBasis;
typedef HexTriquadraticLgn<unsigned char>         QHFDucharBasis;
typedef HexTriquadraticLgn<unsigned long>         QHFDulongBasis;

// typedef LatVolMesh<HexTriquadraticLgn<Point> > HQVMesh;
// template class GenericField<HQVMesh, QHFDTensorBasis, vector<Tensor> >;
// template class GenericField<HQVMesh, QHFDVectorBasis, vector<Vector> >;
// template class GenericField<HQVMesh, QHFDdoubleBasis, vector<double> >;
// template class GenericField<HQVMesh, QHFDfloatBasis,  vector<float> >;
// template class GenericField<HQVMesh, QHFDintBasis,    vector<int> >;
// template class GenericField<HQVMesh, QHFDshortBasis,  vector<short> >;
// template class GenericField<HQVMesh, QHFDcharBasis,   vector<char> >;
// template class GenericField<HQVMesh, QHFDuintBasis,   vector<unsigned int> >;
// template class GenericField<HQVMesh, QHFDushortBasis, vector<unsigned short> >;
// template class GenericField<HQVMesh, QHFDucharBasis,  vector<unsigned char> >;
// template class GenericField<HQVMesh, QHFDulongBasis,  vector<unsigned long> >;


