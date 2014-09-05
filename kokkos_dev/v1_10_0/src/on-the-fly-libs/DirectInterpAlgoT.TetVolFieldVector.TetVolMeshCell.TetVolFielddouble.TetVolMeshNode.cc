// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/TetVolField.h"
#include "/usr/local/SCIRun/src/Core/Geometry/Vector.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/TetVolField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/DirectInterpolate.h"
using namespace SCIRun;

extern "C" {
DirectInterpAlgo* maker() {
  return scinew DirectInterpAlgoT<TetVolField<Vector> , TetVolMesh::Cell, TetVolField<Vector> , TetVolMesh::Node>;
}
}
