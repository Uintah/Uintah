#ifndef __PARAM_H__
#define __PARAM_H__

#include "mydriver.h"
#include <vector>
#include <string>
using std::string;
using std::vector;

class Param {
  /*_____________________________________________________________________
    class Param:
    a structure of input parameters.
    _____________________________________________________________________*/
public:

  /* Types */

  enum ProblemType {
    Linear = 0,
    Quad1,
    Quad2,
    SinSin,
    GaussianSource,
    JumpLinear,
    JumpQuad,
    DiffusionQuadLinear,
    DiffusionQuadQuad,
    LShaped
  };
  
  enum OutputType {
    Screen, File, Both
  };

  enum RefPattern {
    centralHalf
  };

  /* Problem parameters */
  ProblemType problemType;    // Type of problem
  int         numDims;        // # dimensions
  int         numProcs;       // # of processors
  string      longTitle;      // Title of this test case
  vector<int> supportedDims;  // Which dims this test case is designed for

  /* log files, output types */
  string      outputDir;      // Directory of output files
  string      logFile;        // File logging run flow
  OutputType  outputType;     // Output to log file/screen/both
  
  /* Domain geometry */
  Location    domainSize;     // Size of domain in all dimensions
  
  /* AMR hierarchy */
  int         numLevels;      // # of levels in a static MR hierarchy
  int         baseResolution; // Resolution of Level 0 in all dimensions
  RefPattern  twoLevelType;   // Refinement pattern for refining level 0 -> 1
  RefPattern  threeLevelType; // Refinement pattern for refining level 1 -> 2
  
  /* Debugging and control flags */
  int         solverID;       // Solver ID, 30 = AMG, 99 = FAC
  bool        printSystem;    // Linear system dump to file
  bool        timing;         // Time results
  bool        saveResults;    // Dump the solution, error to files
  int         verboseLevel;   // Verbosity level of debug printouts
};

#endif // __PARAM_H__
