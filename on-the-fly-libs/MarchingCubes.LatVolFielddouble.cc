// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/LatVolField.h"
#include "/usr/local/SCIRun/src/Core/Algorithms/Visualization/HexMC.h"
#include "/usr/local/SCIRun/src/Core/Algorithms/Visualization/MarchingCubes.h"
using namespace SCIRun;

extern "C" {
MarchingCubesAlg* maker() {
  return scinew MarchingCubes<HexMC<LatVolField<double> > >;
}
}
