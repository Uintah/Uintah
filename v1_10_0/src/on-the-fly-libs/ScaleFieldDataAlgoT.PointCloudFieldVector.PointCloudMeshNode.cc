// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Geometry/Vector.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/PointCloudField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/ScaleFieldData.h"
using namespace SCIRun;

extern "C" {
ScaleFieldDataAlgo* maker() {
  return scinew ScaleFieldDataAlgoT<PointCloudField<Vector> , PointCloudMesh::Node>;
}
}
