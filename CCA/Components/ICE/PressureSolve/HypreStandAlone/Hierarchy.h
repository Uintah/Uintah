#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

#include <vector>

class Level;

class Hierarchy {
  /*_____________________________________________________________________
    class Hierarchy:
    A sequence of Levels. Level 0 is the coarsest, Level 1 is the next
    finer level, and so on. Hierarchy is the same for all procs.
    _____________________________________________________________________*/
public:
  std::vector<Level*> _levels;
};

#endif // __HIERARCHY_H__
