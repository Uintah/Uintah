// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Geometry/Vector.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/TetVolField.h"
#include "/usr/local/SCIRun/src/Packages/Teem/Dataflow/Modules/DataIO/ConvertToNrrd.h"
using namespace SCIRun;

extern "C" {
ConvertToNrrdBase* maker() {
  return scinew ConvertToNrrd<TetVolField<Vector> >;
}
}
