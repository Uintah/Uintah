// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/QuadraticTetVolField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/Gradient.h"
using namespace SCIRun;

extern "C" {
GradientAlgo* maker() {
  return scinew GradientAlgoT<QuadraticTetVolField, double>;
}
}
