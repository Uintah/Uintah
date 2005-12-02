//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : FEFields.h
//    Author : Martin Cole
//    Date   : Fri Oct  7 12:35:17 2005

#if !defined(FEFields_h)
#define FEFields_h

#include <Core/Geometry/Tensor.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>

namespace BioPSE {
using namespace SCIRun;

typedef TriSurfMesh<TriLinearLgn<Point> >               TSMesh;
typedef ConstantBasis<int>                              TSIBasis;
typedef ConstantBasis<Tensor>                           TSTBasis;
typedef GenericField<TSMesh, TSIBasis,    vector<int> > TSFieldI;  
typedef GenericField<TSMesh, TSTBasis, vector<Tensor> > TSFieldT;  

typedef TetVolMesh<TetLinearLgn<Point> >                   TVMesh;
typedef ConstantBasis<int>                                 TVIBasis;
typedef ConstantBasis<Tensor>                              TVTBasis;
typedef GenericField<TVMesh, TVIBasis,    vector<int> >    TVFieldI;   
typedef GenericField<TVMesh, TVTBasis,    vector<Tensor> > TVFieldT; 

typedef HexVolMesh<HexTrilinearLgn<Point> >             HVMesh;
typedef ConstantBasis<int>                              HVIBasis;
typedef ConstantBasis<Tensor>                           HVTBasis;
typedef GenericField<HVMesh, HVIBasis,    vector<int> > HVFieldI;  
typedef GenericField<HVMesh, HVTBasis, vector<Tensor> > HVFieldT; 


} // end of namespace BioPSE

#endif
