// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Geometry/Vector.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/PointCloudField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/AttractNormals.h"
using namespace SCIRun;

extern "C" {
AttractNormalsAlgo* maker() {
  return scinew AttractNormalsAlgoT<PointCloudMesh, PointCloudMesh::Node, PointCloudField<Vector> >;
}
}
