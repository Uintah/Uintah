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

