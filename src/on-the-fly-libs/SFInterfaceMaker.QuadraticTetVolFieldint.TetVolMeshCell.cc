// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/TetVolMesh.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/QuadraticTetVolField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/FieldInterfaceAux.h"
using namespace SCIRun;

extern "C" {
ScalarFieldInterfaceMaker* maker() {
  return scinew SFInterfaceMaker<QuadraticTetVolField<int> , TetVolMesh::Cell>;
}
}
