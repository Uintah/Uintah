// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/PointCloudField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/TransformScalarData.h"
using namespace SCIRun;

extern "C" {
TransformScalarDataAlgo* maker() {
  return scinew TransformScalarDataAlgoT<PointCloudField<double> , PointCloudMesh::Node>;
}
}
