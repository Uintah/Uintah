// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/ImageMesh.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/ImageField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/FieldInterfaceAux.h"
using namespace SCIRun;

extern "C" {
ScalarFieldInterfaceMaker* maker() {
  return scinew SFInterfaceMaker<ImageField<double> , ImageMesh::Node>;
}
}
