#ifndef __CELL_H__
#define __CELL_H__

#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>

#include "CrackFace.h"

#include <vector>
#include <list>

namespace Uintah {

using namespace SCIRun;

class Cell {
public:
  std::vector<particleIndex> particles;
  std::list<CrackFace> crackFaces;
  
  void  insert(const particleIndex& p);
  void  insert(const CrackFace& crackFace);

private:
};

} // End namespace Uintah

#endif //__CELL_H__

