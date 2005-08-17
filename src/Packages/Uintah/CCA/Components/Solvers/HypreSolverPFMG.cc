#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverSStruct.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

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
  HypreGenericSolver::setup(HypreDriver* hypreDriver)
    //-----------------------------------------------------------
    // Solver setup phase
    //-----------------------------------------------------------
  {
    cerr << "Solver setup phase" << "\n";
    int time_index = hypre_InitializeTiming("Solver Setup");
    hypre_BeginTiming(time_index);
    this->setup(hypreDriver); // The derived Solver setup()
    hypre_EndTiming(time_index);
    hypre_PrintTiming("Setup phase time", MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_index);
    hypre_ClearTiming();
    cerr << "Setup phase time = " << time_index << "\n";
  } // end setup()

  void
  HypreGenericSolver::solve(HypreDriver* hypreDriver)
  {
    //-----------------------------------------------------------
    // Solver solve phase
    //-----------------------------------------------------------
    cerr << "Solver solve phase" << "\n";
    int time_index = hypre_InitializeTiming("Solver Setup");
    hypre_BeginTiming(time_index);
    this->solve(hypreDriver); // The derived Solver solve()

    //-----------------------------------------------------------
    //Gather the solution vector
    //-----------------------------------------------------------
    //TODO: SolverSStruct is derived from Solver; implement the
    //following in SolverSStruct. For SolverStruct (PFMG), another
    //gather vector required.
    cerr << "Gather the solution vector" << "\n";
    switch (_interface) {
    case HypreStruct:
      {
        // It seems it is not necessary to gather the solution vector
        // for the Struct interface.
        /*
        HypreDriverStruct* structDriver =
          dynamic_cast<HypreDriverStruct*>(hypreDriver);
        if (!structDriver) {
          throw InternalError("interface = Struct but HypreDriver is not!",
                              __FILE__, __LINE__);
        }
        HYPRE_StructVectorGather(structDriver->getX());
        */
      }
#if 0
    case HypreSStruct:
      {
        HypreDriverSStruct* sstructDriver =
          dynamic_cast<HypreDriverSStruct*>(hypreDriver);
        if (!sstructDriver) {
          throw InternalError("interface = SStruct but HypreDriver is not!",
                              __FILE__, __LINE__);
        }
        HYPRE_SStructVectorGather(sstructDriver->getX());
      }
#endif
    default:
      throw InternalError("Unsupported Hypre Interface: "+_interface,
                          __FILE__, __LINE__);
    } // end switch (interface)
    
    cerr << "Solve phase time = " << time_index << "\n";
  } //end solve()

  // TODO: include all derived classes here.
  HypreGenericSolver*
  newHypreGenericSolver(const SolverType& solverType)
    /* Create a new solver object of specific solverType solver type
       but a generic solver pointer type. */
  {
    switch (solverType) {
    case SMG:
      {
        return new HypreSolverSMG(hypreData);
      }
    case PFMG:
      {
        return new HypreSolverPFMG(hypreData);
      }
    case SparseMSG:
      {
        return new HypreSolverSparseMSG(hypreData);
      }
    case CG:
      {
        return new HypreSolverCG(hypreData);
      }
    case Hybrid: 
      {
        return new HypreSolverHybrid(hypreData);
      }
    case GMRES:
      {
        return new HypreSolverGMRES(hypreData);
      }
    default:
      throw InternalError("Unsupported solver type: "+params->solverTitle,
                          __FILE__, __LINE__);
    } // switch (solverType)
    return 0;
  }

  SolverType
  getSolverType(const string& solverTitle)
  {
    // Determine solver type from title
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


  HypreInterface
  getSolverInterface(const SolverType& solverType)
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

} // end namespace Uintah
