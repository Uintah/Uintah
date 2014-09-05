#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverSStruct.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

// hypre includes
//#define HYPRE_TIMING
#include <utilities.h>
#include <HYPRE_struct_ls.h>
#include <krylov.h>
#ifndef HYPRE_TIMING
#ifndef hypre_ClearTiming
// This isn't in utilities.h for some reason...
#define hypre_ClearTiming()
#endif
#endif

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

namespace Uintah {

  HypreGenericSolver::HypreGenericSolver(const HypreInterface& interface,
                                         const ProcessorGroup* pg,
                                         const HypreSolverParams* params,
                                         const int acceptableInterface)
    : _interface(interface), _pg(pg), _params(params)
  {
    assertInterface(acceptableInterface);

    //    _solverType = getSolverType(solverTitle);
    _results.numIterations = 0;
    _results.finalResNorm  = 1.23456e+30; // Large number
  }
  
  void
  HypreGenericSolver::assertInterface(const int acceptableInterface)
  { 
    if (acceptableInterface & _interface) {
      return;
    }
    throw InternalError("Solver does not support Hypre interface: "
                        +_interface,__FILE__, __LINE__);
  }

  void
  HypreGenericSolver::setup(void)
    //-----------------------------------------------------------
    // Solver setup phase
    //-----------------------------------------------------------
  {
    cerr << "Solver setup phase" << "\n";
    int time_index = hypre_InitializeTiming("Solver Setup");
    hypre_BeginTiming(time_index);
    this->setup(); // The derived Solver setup()
    hypre_EndTiming(time_index);
    hypre_PrintTiming("Setup phase time", MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_index);
    hypre_ClearTiming();
    cerr << "Setup phase time = " << time_index << "\n";
  } // end setup()

  void
  HypreGenericSolver::solve(const HypreDriver* hypreDriver)
  {
    //-----------------------------------------------------------
    // Solver solve phase
    //-----------------------------------------------------------
    cerr << "Solver solve phase" << "\n";
    int time_index = hypre_InitializeTiming("Solver Setup");
    hypre_BeginTiming(time_index);
    this->solve(); // The derived Solver solve()

    //-----------------------------------------------------------
    //Gather the solution vector
    //-----------------------------------------------------------
    //TODO: SolverSStruct is derived from Solver; implement the
    //following in SolverSStruct. For SolverStruct (PFMG), another
    //gather vector required.
    cerr << "Gather the solution vector" << "\n";
    if (_interface == HypreStruct) {
      const HypreDriverStruct* structDriver =
        dynamic_cast<const HypreDriverStruct*>(hypreDriver);
      if (!structDriver) {
        throw InternalError("interface = Struct but HypreDriver is not!",
                            __FILE__, __LINE__);
      }
      HYPRE_SStructVectorGather(hypreDriver->_HX);
    }
    cerr << "Solve phase time = " << time_index << "\n";
  } //end solve()

  void
  Solver::printMatrix(const string& fileName /* = "solver" */)
  {
    Print("Solver::printMatrix() begin\n");
    if (!_param->printSystem) return;
    HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _A, 0);
    if (_requiresPar) {
      HYPRE_ParCSRMatrixPrint(_parA, (fileName + ".par").c_str());
      /* Print CSR matrix in IJ format, base 1 for rows and cols */
      HYPRE_ParCSRMatrixPrintIJ(_parA, 1, 1, (fileName + ".ij").c_str());
    }
    Print("Solver::printMatrix() end\n");
  }

  void
  Solver::printRHS(const string& fileName /* = "solver" */)
  {
    if (!_param->printSystem) return;
    HYPRE_SStructVectorPrint(fileName.c_str(), _b, 0);
    if (_requiresPar) {
      HYPRE_ParVectorPrint(_parB, (fileName + ".par").c_str());
    }
  }

  void
  Solver::printSolution(const string& fileName /* = "solver" */)
  {
    if (!_param->printSystem) return;
    HYPRE_SStructVectorPrint(fileName.c_str(), _x, 0);
    if (_requiresPar) {
      HYPRE_ParVectorPrint(_parX, (fileName + ".par").c_str());
    }
  }

  void
  Solver::printValues(const Patch* patch,
                      const int stencilSize,
                      const int numCells,
                      const double* values /* = 0 */,
                      const double* rhsValues /* = 0 */,
                      const double* solutionValues /* = 0 */)
    /* Print values, rhsValues vectors */
  {
#if DRIVER_DEBUG
    Print("--- Printing values,rhsValues,solutionValues arrays ---\n");
    for (Counter cell = 0; cell < numCells; cell++) {
      int offsetValues    = stencilSize * cell;
      int offsetRhsValues = cell;
      Print("cell = %4d\n",cell);
      if (values) {
        for (Counter entry = 0; entry < stencilSize; entry++) {
          Print("values   [%5d] = %+.3f\n",
                offsetValues + entry,values[offsetValues + entry]);
        }
      }
      if (rhsValues) {
        Print("rhsValues[%5d] = %+.3f\n",
              offsetRhsValues,rhsValues[offsetRhsValues]);
      }
      if (solutionValues) {
        Print("solutionValues[%5d] = %+.3f\n",
              offsetRhsValues,solutionValues[offsetRhsValues]);
      }
      Print("-------------------------------\n");
    } // end for cell
#endif
  } // end printValues()


  HypreGenericSolver::SolverType
  HypreGenericSolver::solverFromTitle(const string& solverTitle)
  {
    /* Determine solver type from title */
    if ((solverTitle == "SMG") ||
        (solverTitle == "smg")) {
      return SMG;
    } else if ((solverTitle == "PFMG") ||
               (solverTitle == "pfmg")) {
      return PFMG;
    } else if ((solverTitle == "SparseMSG") ||
               (solverTitle == "sparsemsg")) {
      return SparseMSG;
    } else if ((solverTitle == "CG") ||
               (solverTitle == "cg") ||
               (solverTitle == "PCG") ||
               (solverTitle == "conjugategradient")) {
      return CG;
    } else if ((solverTitle == "Hybrid") ||
               (solverTitle == "hybrid")) {
      return Hybrid;
    } else if ((solverTitle == "GMRES") ||
               (solverTitle == "gmres")) {
      return GMRES;
    } else if ((solverTitle == "AMG") ||
               (solverTitle == "amg") ||
               (solverTitle == "BoomerAMG") ||
               (solverTitle == "boomeramg")) {
      return AMG;
    } else if ((solverTitle == "FAC") ||
               (solverTitle == "fac")) {
      return FAC;
    } else {
      throw InternalError("Unknown solver type: "+solverTitle,
                          __FILE__, __LINE__);
    } // end "switch" (solverTitle)
  } // end solverFromTitle()

  HypreGenericSolver::precondType
  HypreGenericSolver::precondFromTitle(const string& precondTitle)
  {
    /* Determine preconditioner type from title */
    if ((precondTitle == "SMG") ||
        (precondTitle == "smg")) {
      return PrecondSMG;
    } else if ((precondTitle == "PFMG") ||
               (precondTitle == "pfmg")) {
      return PrecondPFMG;
    } else if ((precondTitle == "SparseMSG") ||
               (precondTitle == "sparsemsg")) {
      return PrecondSparseMSG;
    } else if ((precondTitle == "Jacobi") ||
               (precondTitle == "jacobi")) {
      return PrecondJacobi;
    } else if ((precondTitle == "Diagonal") ||
               (precondTitle == "diagonal")) {
      return PrecondDiagonal;
    } else if ((precondTitle == "AMG") ||
               (precondTitle == "amg") ||
               (precondTitle == "BoomerAMG") ||
               (precondTitle == "boomeramg")) {
      return PrecondAMG;
    } else if ((precondTitle == "FAC") ||
               (precondTitle == "fac")) {
      return PrecondFAC;
    } else {
      throw InternalError("Unknown preconditionertype: "+precondTitle,
                          __FILE__, __LINE__);
    } // end "switch" (precondTitle)
  } // end precondFromTitle()

  HypreDriver::Interface
  HypreGenericSolver::solverInterface(const SolverType& solverType)
    /* Determine the Hypre interface this solver uses */
  {
    switch (solverType) {
    case SMG:
      {
        return HypreDriver::Struct;
      }
    case PFMG:
      {
        return HypreDriver::Struct;
      }
    case SparseMSG:
      {
        return HypreDriver::Struct;
      }
    case CG:
      {
        return HypreDriver::Struct;
      }
    case Hybrid: 
      {
        return HypreDriver::Struct;
      }
    case GMRES:
      {
        return HypreDriver::Struct;
      }
    case FAC:
      {
        return HypreDriver::SStruct;
      }
    case AMG:
      {
        return HypreDriver::ParCSR;
      }
    default:
      throw InternalError("Unsupported solver type: "+solverType,
                          __FILE__, __LINE__);
    } // switch (solverType)
  } // end solverInterface()


  // TODO: include all derived classes here.
  HypreGenericSolver*
  HypreGenericSolver::newSolver(const SolverType& solverType)
    /* Create a new solver object of specific solverType solver type
       but a generic solver pointer type. */
  {
    switch (solverType) {
    case HypreSolverParams::SMG:
      {
        return new HypreSolverSMG(hypreData);
      }
    case HypreSolverParams::PFMG:
      {
        return new HypreSolverPFMG(hypreData);
      }
    case HypreSolverParams::SparseMSG:
      {
        return new HypreSolverSparseMSG(hypreData);
      }
    case HypreSolverParams::CG:
      {
        return new HypreSolverCG(hypreData);
      }
    case HypreSolverParams::Hybrid: 
      {
        return new HypreSolverHybrid(hypreData);
      }
    case HypreSolverParams::GMRES:
      {
        return new HypreSolverGMRES(hypreData);
      }
    default:
      throw InternalError("Unsupported solver type: "+params->solverTitle,
                          __FILE__, __LINE__);
    } // switch (solverType)
    return 0;
  }

} // end namespace Uintah
