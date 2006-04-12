
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

#include <Packages/ModelCreation/Core/Fields/Precompile.h>

#ifdef PRECOMPILE_ALGOS
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>

#include <Core/Basis/Constant.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/HexTrilinearLgn.h>

#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/HexVolMesh.h>

#include <Core/Containers/FData.h>

#include <Core/Datatypes/GenericField.h>
#include <Packages/ModelCreation/Core/Fields/ToPointCloud.h>
#include <Packages/ModelCreation/Core/Fields/Unstructure.h>
#include <Packages/ModelCreation/Core/Fields/ConvertToTetVol.h>
#include <Packages/ModelCreation/Core/Fields/ConvertToTriSurf.h>

#endif

namespace ModelCreation {

#ifdef PRECOMPILE_ALGOS

using namespace SCIRun;

typedef NoDataBasis<double>                    nodata;
typedef NoDataBasis<Vector>                    ndvector;
typedef NoDataBasis<Tensor>                    ndtensor;
typedef ConstantBasis<int>                     cstint;
typedef ConstantBasis<double>                  cstdouble;
typedef ConstantBasis<Vector>                  cstvector;
typedef ConstantBasis<Tensor>                  csttensor;

typedef CrvLinearLgn<double>                   crvdouble;
typedef CrvLinearLgn<int>                      crvint;
typedef CrvLinearLgn<Vector>                   crvvector;
typedef CrvLinearLgn<Tensor>                   crvtensor;
typedef TriLinearLgn<int>                      triint;
typedef TriLinearLgn<double>                   tridouble;
typedef TriLinearLgn<Vector>                   trivector;
typedef TriLinearLgn<Tensor>                   tritensor;
typedef QuadBilinearLgn<double>                quaddouble;
typedef QuadBilinearLgn<int>                   quadint;
typedef QuadBilinearLgn<Vector>                quadvector;
typedef QuadBilinearLgn<Tensor>                quadtensor;
typedef TetLinearLgn<double>                   tetdouble;
typedef TetLinearLgn<int>                      tetint;
typedef TetLinearLgn<Vector>                   tetvector;
typedef TetLinearLgn<Tensor>                   tettensor;
typedef HexTrilinearLgn<Tensor>                hexdouble;
typedef HexTrilinearLgn<Vector>                hexvector;
typedef HexTrilinearLgn<double>                hextensor;
typedef HexTrilinearLgn<int>                   hexint;

typedef std::vector<double>                 cdouble;
typedef std::vector<int>                    cint;
typedef std::vector<Vector>                 cvector;
typedef std::vector<Tensor>                 ctensor;


// Only interesting meshes
typedef ImageMesh<QuadBilinearLgn<Point> >     IMMesh;
typedef LatVolMesh<HexTrilinearLgn<Point> >    LVMesh;
typedef PointCloudMesh<ConstantBasis<Point> >  PCMesh;
typedef CurveMesh<CrvLinearLgn<Point> >        CVMesh;
typedef TriSurfMesh<TriLinearLgn<Point> >      TSMesh;
typedef QuadSurfMesh<QuadBilinearLgn<Point> >  QSMesh;
typedef TetVolMesh<TetLinearLgn<Point> >       TVMesh;
typedef HexVolMesh<HexTrilinearLgn<Point> >    HVMesh;


typedef FData2d<double,IMMesh>              idouble;
typedef FData2d<int,IMMesh>                 iint;
typedef FData2d<Vector,IMMesh>              ivector;
typedef FData2d<Tensor,IMMesh>              itensor;

typedef FData3d<double,LVMesh>              ldouble;
typedef FData3d<int,LVMesh>                 lint;
typedef FData3d<Vector,LVMesh>              lvector;
typedef FData3d<Tensor,LVMesh>              ltensor;


// Define some commonly used fields

// Common PointCloud Fields
typedef GenericField<PCMesh,nodata,cdouble>     PC_ND;
typedef GenericField<PCMesh,ndvector,cvector>   PC_ND_VCTR;
typedef GenericField<PCMesh,ndtensor,ctensor>   PC_ND_TNSR;
typedef GenericField<PCMesh,cstdouble,cdouble>  PC_CD_DBL;
typedef GenericField<PCMesh,cstvector,cvector>  PC_CD_VCTR;
typedef GenericField<PCMesh,csttensor,ctensor>  PC_CD_TNSR;

// Common Curve Fields
typedef GenericField<CVMesh,nodata,cdouble>     CV_ND;
typedef GenericField<CVMesh,cstdouble,cdouble>  CV_CD_DBL;
typedef GenericField<CVMesh,cstvector,cvector>  CV_CD_VCTR;
typedef GenericField<CVMesh,csttensor,ctensor>  CV_CD_TNSR;
typedef GenericField<CVMesh,crvdouble,cdouble>  CV_LD_DBL;
typedef GenericField<CVMesh,crvvector,cvector>  CV_LD_VCTR;
typedef GenericField<CVMesh,crvtensor,ctensor>  CV_LD_TNSR;

// Common TriSurf Fields
typedef GenericField<TSMesh,nodata,cdouble>     TS_ND;
typedef GenericField<TSMesh,cstdouble,cdouble>  TS_CD_DBL;
typedef GenericField<TSMesh,cstvector,cvector>  TS_CD_VCTR;
typedef GenericField<TSMesh,csttensor,ctensor>  TS_CD_TNSR;
typedef GenericField<TSMesh,tridouble,cdouble>  TS_LD_DBL;
typedef GenericField<TSMesh,trivector,cvector>  TS_LD_VCTR;
typedef GenericField<TSMesh,tritensor,ctensor>  TS_LD_TNSR;

// Common QuadSurf Fields
typedef GenericField<QSMesh,nodata,cdouble>     QS_ND;
typedef GenericField<QSMesh,cstdouble,cdouble>  QS_CD_DBL;
typedef GenericField<QSMesh,cstvector,cvector>  QS_CD_VCTR;
typedef GenericField<QSMesh,csttensor,ctensor>  QS_CD_TNSR;
typedef GenericField<QSMesh,quaddouble,cdouble> QS_LD_DBL;
typedef GenericField<QSMesh,quadvector,cvector> QS_LD_VCTR;
typedef GenericField<QSMesh,quadtensor,ctensor> QS_LD_TNSR;

// Common TetVol Fields
typedef GenericField<TVMesh,nodata,cdouble>     TV_ND;
typedef GenericField<TVMesh,cstdouble,cdouble>  TV_CD_DBL;
typedef GenericField<TVMesh,cstvector,cvector>  TV_CD_VCTR;
typedef GenericField<TVMesh,csttensor,ctensor>  TV_CD_TNSR;
typedef GenericField<TVMesh,tetdouble,cdouble>  TV_LD_DBL;
typedef GenericField<TVMesh,tetvector,cvector>  TV_LD_VCTR;
typedef GenericField<TVMesh,tettensor,ctensor>  TV_LD_TNSR;

// Common HexVol Fields
typedef GenericField<HVMesh,nodata,cdouble>     HV_ND;
typedef GenericField<HVMesh,cstdouble,cdouble>  HV_CD_DBL;
typedef GenericField<HVMesh,cstvector,cvector>  HV_CD_VCTR;
typedef GenericField<HVMesh,csttensor,ctensor>  HV_CD_TNSR;
typedef GenericField<HVMesh,hexdouble,cdouble>  HV_LD_DBL;
typedef GenericField<HVMesh,hexvector,cvector>  HV_LD_VCTR;
typedef GenericField<HVMesh,hextensor,ctensor>  HV_LD_TNSR;

// Common LatVol Fields
typedef GenericField<LVMesh,nodata,ldouble>     LV_ND;
typedef GenericField<LVMesh,cstdouble,ldouble>  LV_CD_DBL;
typedef GenericField<LVMesh,cstvector,lvector>  LV_CD_VCTR;
typedef GenericField<LVMesh,csttensor,ltensor>  LV_CD_TNSR;
typedef GenericField<LVMesh,hexdouble,ldouble>  LV_LD_DBL;
typedef GenericField<LVMesh,hexvector,lvector>  LV_LD_VCTR;
typedef GenericField<LVMesh,hextensor,ltensor>  LV_LD_TNSR;


// Common Image Fields
typedef GenericField<IMMesh,nodata,idouble>     IM_ND;
typedef GenericField<IMMesh,cstdouble,idouble>  IM_CD_DBL;
typedef GenericField<IMMesh,cstvector,ivector>  IM_CD_VCTR;
typedef GenericField<IMMesh,csttensor,itensor>  IM_CD_TNSR;
typedef GenericField<IMMesh,quaddouble,idouble> IM_LD_DBL;
typedef GenericField<IMMesh,quadvector,ivector> IM_LD_VCTR;
typedef GenericField<IMMesh,quadtensor,itensor> IM_LD_TNSR;

#endif


void Precompile_FieldsAlgo()
{

#ifdef PRECOMPILE_ALGOS

/*
  // ToPointCloud precompiles
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<QS_ND,PC_ND>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<QS_CD_DBL,PC_ND>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<QS_CD_VCTR,PC_ND_VCTR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<QS_CD_TNSR,PC_ND_TNSR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<QS_LD_DBL,PC_CD_DBL>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<QS_LD_VCTR,PC_CD_VCTR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<QS_LD_TNSR,PC_CD_TNSR>);

  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TS_ND,PC_ND>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TS_CD_DBL,PC_ND>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TS_CD_VCTR,PC_ND_VCTR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TS_CD_TNSR,PC_ND_TNSR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TS_LD_DBL,PC_CD_DBL>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TS_LD_VCTR,PC_CD_VCTR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TS_LD_TNSR,PC_CD_TNSR>);

  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TV_ND,PC_ND>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TV_CD_DBL,PC_ND>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TV_CD_VCTR,PC_ND_VCTR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TV_CD_TNSR,PC_ND_TNSR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TV_LD_DBL,PC_CD_DBL>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TV_LD_VCTR,PC_CD_VCTR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<TV_LD_TNSR,PC_CD_TNSR>);

  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<HV_ND,PC_ND>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<HV_CD_DBL,PC_ND>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<HV_CD_VCTR,PC_ND_VCTR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<HV_CD_TNSR,PC_ND_TNSR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<HV_LD_DBL,PC_CD_DBL>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<HV_LD_VCTR,PC_CD_VCTR>);
  ToPointCloudAlgo::precompiled_.add(scinew ToPointCloudAlgoT<HV_LD_TNSR,PC_CD_TNSR>);

  // Unstructure precompiles
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<IM_ND,QS_ND>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<IM_CD_DBL,QS_CD_DBL>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<IM_CD_VCTR,QS_CD_VCTR>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<IM_CD_TNSR,QS_CD_TNSR>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<IM_LD_DBL,QS_LD_DBL>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<IM_LD_VCTR,QS_LD_VCTR>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<IM_LD_TNSR,QS_LD_TNSR>);

  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<LV_ND,HV_ND>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<LV_CD_DBL,HV_CD_DBL>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<LV_CD_VCTR,HV_CD_VCTR>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<LV_CD_TNSR,HV_CD_TNSR>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<LV_LD_DBL,HV_LD_DBL>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<LV_LD_VCTR,HV_LD_VCTR>);
  UnstructureAlgo::precompiled_.add(scinew UnstructureAlgoT<LV_LD_TNSR,HV_LD_TNSR>);

  // ConvertToTetVol precompiles
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertHexVolToTetVolAlgoT<HV_ND,TV_ND>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertHexVolToTetVolAlgoT<HV_CD_DBL,TV_CD_DBL>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertHexVolToTetVolAlgoT<HV_CD_VCTR,TV_CD_VCTR>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertHexVolToTetVolAlgoT<HV_CD_TNSR,TV_CD_TNSR>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertHexVolToTetVolAlgoT<HV_LD_DBL,TV_LD_DBL>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertHexVolToTetVolAlgoT<HV_LD_VCTR,TV_LD_VCTR>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertHexVolToTetVolAlgoT<HV_LD_TNSR,TV_LD_TNSR>);

  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertLatVolToTetVolAlgoT<LV_ND,TV_ND>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertLatVolToTetVolAlgoT<LV_CD_DBL,TV_CD_DBL>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertLatVolToTetVolAlgoT<LV_CD_VCTR,TV_CD_VCTR>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertLatVolToTetVolAlgoT<LV_CD_TNSR,TV_CD_TNSR>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertLatVolToTetVolAlgoT<LV_LD_DBL,TV_LD_DBL>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertLatVolToTetVolAlgoT<LV_LD_VCTR,TV_LD_VCTR>);
  ConvertToTetVolAlgo::precompiled_.add(scinew ConvertLatVolToTetVolAlgoT<LV_LD_TNSR,TV_LD_TNSR>);
*/

#endif

}

} // end namespace#include <Core/Datatypes/HexVolMesh.h>
