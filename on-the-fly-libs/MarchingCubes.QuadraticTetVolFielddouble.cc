// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/QuadraticTetVolField.h"
#include "/usr/local/SCIRun/src/Core/Algorithms/Visualization/TetMC.h"
#include "/usr/local/SCIRun/src/Core/Algorithms/Visualization/MarchingCubes.h"
using namespace SCIRun;

extern "C" {
MarchingCubesAlg* maker() {
  return scinew MarchingCubes<TetMC<QuadraticTetVolField<double> > >;
}
}
