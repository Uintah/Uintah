// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/PointCloudField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/QuadraticTetVolField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/DirectInterpolate.h"
using namespace SCIRun;

extern "C" {
DirectInterpAlgo* maker() {
  return scinew DirectInterpAlgoT<QuadraticTetVolField<double> , TetVolMesh::Node, PointCloudField<double> , PointCloudMesh::Node>;
}
}
