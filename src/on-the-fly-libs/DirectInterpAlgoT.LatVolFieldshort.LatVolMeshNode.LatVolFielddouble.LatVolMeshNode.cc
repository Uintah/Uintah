// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/LatVolField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/LatVolField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/DirectInterpolate.h"
using namespace SCIRun;

extern "C" {
DirectInterpAlgo* maker() {
  return scinew DirectInterpAlgoT<LatVolField<short> , LatVolMesh::Node, LatVolField<short> , LatVolMesh::Node>;
}
}
