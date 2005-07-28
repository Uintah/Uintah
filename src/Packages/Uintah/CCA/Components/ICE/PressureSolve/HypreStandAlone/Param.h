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
    a base struct of input parameters for a test case.
    _____________________________________________________________________*/
public:
  
  /* Types */

  enum OutputType {
    Screen, File, Both
  };

  enum RefPattern {
    CentralHalf
  };


  Param(void)
    /* Constructor: initialize default parameters for all test cases */
    {
      outputType = Screen;     // Output to log file/screen/both

      domainSize.resize(numDims);
      for (int d = 0; d < numDims; d++)
        domainSize[d] = 1.0;

      baseResolution = 8;

      numLevels = 2;
      twoLevelType = CentralHalf;
      threeLevelType = CentralHalf;
      
      solverID = 30;
      printSystem = 1;
      timing = true;
      saveResults = true;
      verboseLevel = 1;
    }

  virtual ~Param(void) {}

  /*======================= Data Members =============================*/

  /* Problem parameters */
  int             numProcs;       // # of processors, from argv/mpirun
  Counter         numDims;        // # dimensions
  string          longTitle;      // Title of this test case

  /* log files, output types */
  string          outputDir;      // Directory of output files
  string          logFile;        // File logging run flow
  OutputType      outputType;     // Output to log file/screen/both
  
  /* Domain geometry & coarsest grid */
  Location        domainSize;     // Size of domain in all dimensions
  // TODO: replace with Level 0, including boundary conditions (Dirichlet/N)
  // types for all boundaries
  // TODO: multiple boxes define domain (e.g. for L-shaped)
  
  /* AMR hierarchy */
  int             baseResolution; // Resolution of Level 0 in all dimensions

  int             numLevels;      // # of levels in a static MR hierarchy
  RefPattern      twoLevelType;   // Refinement pattern for refining level 0 -> 1
  RefPattern      threeLevelType; // Refinement pattern for refining level 1 -> 2
  
  /* Debugging and control flags */
  int             solverID;       // Solver ID, 30 = AMG, 99 = FAC
  bool            printSystem;    // Linear system dump to file
  bool            timing;         // Time results
  bool            saveResults;    // Dump the solution, error to files
  int             verboseLevel;   // Verbosity level of debug printouts

  /* Input functions to be defined in derived test cases */

  virtual double diffusion(const Location& x) = 0;  // Diffusion coefficient
  virtual double rhs(const Location& x) = 0;        // Right-hand-side of PDE
  virtual double rhsBC(const Location& x) = 0;      // RHS of B.C.
  virtual double exactSolution(const Location& x) = 0; // Exact solution
};

#endif // __PARAM_H__
