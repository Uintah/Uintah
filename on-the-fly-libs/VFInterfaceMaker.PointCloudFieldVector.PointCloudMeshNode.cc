// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/PointCloudMesh.h"
#include "/usr/local/SCIRun/src/Core/Geometry/Vector.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/PointCloudField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/FieldInterfaceAux.h"
using namespace SCIRun;

extern "C" {
VectorFieldInterfaceMaker* maker() {
  return scinew VFInterfaceMaker<PointCloudField<Vector> , PointCloudMesh::Node>;
}
}
