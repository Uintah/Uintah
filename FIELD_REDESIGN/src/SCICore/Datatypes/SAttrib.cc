//  SAttrib.cc - Scalar Attribute
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/SAttrib.h>

namespace SCICore{
namespace Datatypes{

SAttrib::SAttrib():Attrib(),
  has_minmax(0){
}

SAttrib::SAttrib(const SAttrib& copy):
  Attrib(copy), min(copy.min),
  max(copy.max), has_minmax(copy.has_minmax){
    
}

SAttrib::SAttrib(int x, int y, int z):Attrib(x, y, z),
  has_minmax(0){
}

SAttrib::SAttrib(int x, int y):Attrib(x, y),
  has_minmax(0){
}

SAttrib::SAttrib(int x):Attrib(x),
  has_minmax(0){
}

bool SAttrib::get_minmax(double& imin, double& imax){
  compute_minmax();
  imin = min;
  imax = max;
  return true;
}

}
}
