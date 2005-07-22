#ifndef __PARAM_H__
#define __PARAM_H__

class Param {
  /*_____________________________________________________________________
    class Param:
    a structure of input parameters.
    _____________________________________________________________________*/
public:
  int         numDims;      // # dimensions
  int         numLevels;    // # of levels in a static MR hierarchy
  int         numProcs;     // # of processors
  int         n;            // Resolution of Level 0 in all dimensions
  int         solverID;     // Solver ID, 30 = AMG, 99 = FAC
  bool        printSystem;  // Debugging flag for linear system file dump
};

#endif // __PARAM_H__
