// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/CurveMesh.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/CurveField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/FieldInterfaceAux.h"
using namespace SCIRun;

extern "C" {
ScalarFieldInterfaceMaker* maker() {
  return scinew SFInterfaceMaker<CurveField<double> , CurveMesh::Node>;
}
}
