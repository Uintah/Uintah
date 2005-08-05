#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

#include "Param.h"
#include <vector>

class Level;
class Patch;

class Hierarchy {
  /*_____________________________________________________________________
    class Hierarchy:
    A sequence of Levels. Level 0 is the coarsest, Level 1 is the next
    finer level, and so on. Hierarchy is the same for all procs.
    _____________________________________________________________________*/
public:

  Hierarchy( const Param* param ) : _param(param) {}

  void make();
  void printPatchBoundaries();

  std::vector<Level*>      _levels;
  const Param*             _param;

  std::vector<Patch*> finePatchesOverMe(const Patch& patch);

 private:
  void getPatchesFromOtherProcs();

};

#endif // __HIERARCHY_H__
