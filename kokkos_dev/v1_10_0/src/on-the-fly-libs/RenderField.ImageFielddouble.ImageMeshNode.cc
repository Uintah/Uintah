// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/ImageField.h"
#include "/usr/local/SCIRun/src/Core/Algorithms/Visualization/RenderField.h"
using namespace SCIRun;

extern "C" {
RenderFieldBase* maker() {
  return scinew RenderField<ImageField<double> , ImageMesh::Node>;
}
}
