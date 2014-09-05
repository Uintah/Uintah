// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/TetVolField.h"
#include "/usr/local/SCIRun/src/Core/Geometry/Vector.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/TetVolField.h"
#include "/usr/local/SCIRun/src/Core/Algorithms/Visualization/RenderField.h"
using namespace SCIRun;

extern "C" {
RenderVectorFieldBase* maker() {
  return scinew RenderVectorField<TetVolField<Vector> , TetVolField<double> , TetVolMesh::Cell>;
}
}
