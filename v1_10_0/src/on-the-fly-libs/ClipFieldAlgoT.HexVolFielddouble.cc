// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/HexVolField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/ClipField.h"
using namespace SCIRun;

extern "C" {
ClipFieldAlgo* maker() {
  return scinew ClipFieldAlgoT<HexVolField<double> >;
}
}
