// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/LatVolMesh.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/LatVolField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/FieldInterfaceAux.h"
using namespace SCIRun;

extern "C" {
ScalarFieldInterfaceMaker* maker() {
  return scinew SFInterfaceMaker<LatVolField<unsigned short> , LatVolMesh::Node>;
}
}
