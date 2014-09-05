// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/TetVolMesh.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/TetVolMesh.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/FieldMeasures.h"
using namespace SCIRun;

extern "C" {
FieldMeasuresAlgo* maker() {
  return scinew FieldMeasuresAlgoT<TetVolMesh, TetVolMesh::Cell>;
}
}
