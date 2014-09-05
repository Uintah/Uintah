// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/QuadSurfMesh.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/QuadSurfField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/FieldInterfaceAux.h"
using namespace SCIRun;

extern "C" {
ScalarFieldInterfaceMaker* maker() {
  return scinew SFInterfaceMaker<QuadSurfField<double> , QuadSurfMesh::Node>;
}
}
