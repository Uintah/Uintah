#ifndef __PARAM_H__
#define __PARAM_H__

#include "Patch.h"
#include "Level.h"

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

  enum SolverType {
    AMG, FAC
  };

  Param(void)
    /* Constructor: initialize default parameters for all test cases */
    {
      outputType = Screen;     // Output to log file/screen/both     
      solverType = AMG;
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
  //  Level*          domain;         // Domain as a union of boxes
  Counter         baseResolution;
  // TODO: replace with Level 0, including boundary conditions (Dirichlet/N)
  // types for all boundaries
  // TODO: multiple boxes define domain (e.g. for L-shaped)
  
  /* AMR hierarchy */

  int             numLevels;      // # of levels in a static MR hierarchy
  RefPattern      twoLevelType;   // Refinement pattern for refining level 0 -> 1
  RefPattern      threeLevelType; // Refinement pattern for refining level 1 -> 2
  
  /* Debugging and control flags */
  SolverType      solverType;     // Hypre Solver Type
  bool            printSystem;    // Linear system dump to file
  bool            timing;         // Time results
  bool            saveResults;    // Dump the solution, error to files
  int             verboseLevel;   // Verbosity level of debug printouts

  /* Input functions to be defined in derived test cases */

  virtual double harmonicAvg(const Location& x,
                             const Location& y,
                             const Location& z) const;
  virtual double diffusion(const Location& x) const = 0;// Diffusion coef
  virtual double rhs(const Location& x) const = 0;   // Right-hand-side of PDE
  virtual double rhsBC(const Location& x) const = 0;      // RHS of B.C.
  virtual double exactSolution(const Location& x) const = 0; // Exact solution

 protected:
  void setNumDims(const Counter d);
  //  virtual void setDomain(const Counter baseResolution,
  //                         const vector<Patch::BoundaryCondition>& bc);
};

#endif // __PARAM_H__
