// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/TriSurfField.h"
#include "/usr/local/SCIRun/src/Core/Algorithms/Visualization/TriMC.h"
#include "/usr/local/SCIRun/src/Core/Algorithms/Visualization/Noise.h"
using namespace SCIRun;

extern "C" {
NoiseAlg* maker() {
  return scinew Noise<TriMC<TriSurfField<double> > >;
}
}
