#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

#include <vector>

#include "Param.h"

class Level;

class Hierarchy {
  /*_____________________________________________________________________
    class Hierarchy:
    A sequence of Levels. Level 0 is the coarsest, Level 1 is the next
    finer level, and so on. Hierarchy is the same for all procs.
    _____________________________________________________________________*/
public:

  Hierarchy( const Param & param ) : _param(param) {}

  void make();
  void printPatchBoundaries();

  std::vector<Level*> _levels;
  const Param & _param;

 private:
  void getPatchesFromOtherProcs();

};

#endif // __HIERARCHY_H__
