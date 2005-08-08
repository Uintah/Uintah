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
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/TetQuadraticLgn.h>
#include <Core/Basis/HexQuadraticLgn.h>
#include <Core/Basis/PrismLinearLgn.h>
#include <Core/Datatypes/PrismVolMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Basis/NoData.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

using namespace SCIRun;

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

typedef NoDataBasis<double>                 NDBasis;
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
template class GenericField<TVMesh, TFDTensorBasis, vector<Tensor> >;       
template class GenericField<TVMesh, TFDVectorBasis, vector<Vector> >;       
template class GenericField<TVMesh, TFDdoubleBasis, vector<double> >;       
template class GenericField<TVMesh, NDBasis, vector<double> >;  
template class GenericField<TVMesh, TFDfloatBasis,  vector<float> >;        
template class GenericField<TVMesh, TFDintBasis,    vector<int> >;          
template class GenericField<TVMesh, TFDshortBasis,  vector<short> >;        
template class GenericField<TVMesh, TFDcharBasis,   vector<char> >;         
template class GenericField<TVMesh, TFDuintBasis,   vector<unsigned int> >; 
template class GenericField<TVMesh, TFDushortBasis, vector<unsigned short> >;
template class GenericField<TVMesh, TFDucharBasis,  vector<unsigned char> >;
template class GenericField<TVMesh, TFDulongBasis,  vector<unsigned long> >;

PersistentTypeID backwards_compat_TVFd("TetVolField<double>", "Field",
				       GenericField<TVMesh, TFDdoubleBasis, 
				       vector<double> >::maker);




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


typedef HexQuadraticLgn<Tensor>                QHFDTensorBasis;
typedef HexQuadraticLgn<Vector>                QHFDVectorBasis;
typedef HexQuadraticLgn<double>                QHFDdoubleBasis;
typedef HexQuadraticLgn<float>                 QHFDfloatBasis;
typedef HexQuadraticLgn<int>                   QHFDintBasis;
typedef HexQuadraticLgn<short>                 QHFDshortBasis;
typedef HexQuadraticLgn<char>                  QHFDcharBasis;
typedef HexQuadraticLgn<unsigned int>          QHFDuintBasis;
typedef HexQuadraticLgn<unsigned short>        QHFDushortBasis;
typedef HexQuadraticLgn<unsigned char>         QHFDucharBasis;
typedef HexQuadraticLgn<unsigned long>         QHFDulongBasis;

typedef LatVolMesh<HexQuadraticLgn<Point> > HQVMesh;
template class GenericField<HQVMesh, QHFDTensorBasis, vector<Tensor> >;
template class GenericField<HQVMesh, QHFDVectorBasis, vector<Vector> >;
template class GenericField<HQVMesh, QHFDdoubleBasis, vector<double> >;
template class GenericField<HQVMesh, QHFDfloatBasis,  vector<float> >;
template class GenericField<HQVMesh, QHFDintBasis,    vector<int> >;
template class GenericField<HQVMesh, QHFDshortBasis,  vector<short> >;
template class GenericField<HQVMesh, QHFDcharBasis,   vector<char> >;
template class GenericField<HQVMesh, QHFDuintBasis,   vector<unsigned int> >;
template class GenericField<HQVMesh, QHFDushortBasis, vector<unsigned short> >;
template class GenericField<HQVMesh, QHFDucharBasis,  vector<unsigned char> >;
template class GenericField<HQVMesh, QHFDulongBasis,  vector<unsigned long> >;


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
