/*--------------------------------------------------------------------------
CLASS
   HypreInterface
   
   A generic Hypre system interface.

GENERAL INFORMATION

   File: HypreInterface.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreInterface, HypreSolverParams, HYPRE_Struct, HYPRE_SStruct, HYPRE_ParCSR,
   HypreGenericSolver.

DESCRIPTION
   Class HypreInterface allows access to multiple Hypre interfaces:
   Struct (structured grid), SStruct (composite AMR grid), ParCSR (parallel
   compressed sparse row representation of the matrix).
   Only one of them is activated per solver, by running makeLinearSystem(), thereby
   creating the objects (usually A,b,x) of the specific interface.
   The solver can then be constructed from the interface data. If required by the
   solver, HypreInterface converts the data from one Hypre interface type to another.
   HypreInterface is also responsible for deleting all Hypre objects.
  
WARNING
   * If we intend to use other Hypre system interfaces (e.g., IJ interface),
   their data types (Matrix, Vector, etc.) should be added to the data
   members of this class. Specific solvers deriving from HypreSolverGeneric
   should have a specific Hypre interface they work with that exists in
   HypreInterface.
   * Each new interface requires its own makeLinearSystem() -- construction of the
   linear system objects, and getSolution() -- getting the solution vector back to
   Uintah.
   * This interface is written for Hypre 1.9.0b (released 2005). However,
   it may still work with the Struct solvers in earlier Hypre versions (e.g., 
   1.7.7).
   --------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreInterface_h
#define Packages_Uintah_CCA_Components_Solvers_HypreInterface_h

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

// hypre includes
#include <utilities.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_sstruct_ls.h>
#include <krylov.h>

namespace Uintah {

  /* Forward declarations */
  class HypreSolverParams;

  class HypreInterface {
    
    /*========================== PUBLIC SECTION ==========================*/
  public:
    /*---------- Types ----------*/

    enum InterfaceType {       // Hypre system interface for the solver
      Struct, SStruct, ParCSR
    };

    HypreInterface(const HypreSolverParams* params)
      : _params(params)
      {
        /* No interfaces are currently active */
        isActive_Struct  = false;
        isActive_SStruct = false;
        isActive_Par     = false;
      }
    
    virtual ~HypreInterface(void) {}

    /* Generate A,b,x for "Types" elliptic equation and "Interface" interface */
    template<class Types>
      void makeLinearSystem(const InterfaceType& interface);
    template<class Types>
      void getSolution(const InterfaceType& interface);
    
    void convertData(const InterfaceType& fromInterface,
                     const InterfaceType& toInterface);

    /*========================== PRIVATE SECTION ==========================*/
  private:
    /*---------- Data members ----------*/

    const HypreSolverParams* _params;
    InterfaceType            _activeInterface;        // Which makeLinearSystem was done

    /* Hypre Struct interface objects */
    bool                     isActive_Struct;         // Are these objects created?
    HYPRE_StructMatrix       _A_Struct;               // Left-hand-side matrix
    HYPRE_StructVector       _b_Struct;               // Right-hand-side vector
    HYPRE_StructVector       _x_Struct;               // Solution vector

    /* Hypre SStruct interface objects */
    bool                     isActive_SStruct;        // Are these objects created?
    HYPRE_SStructMatrix      _A_SStruct;               // Left-hand-side matrix
    HYPRE_SStructVector      _b_SStruct;               // Right-hand-side vector
    HYPRE_SStructVector      _x_SStruct;               // Solution vector
    HYPRE_SStructGraph       _graph_SStruct;           // Unstructured connections graph

    /* Hypre ParCSR interface objects */
    bool                     isActive_Par;             // Are these objects created?
    HYPRE_ParCSRMatrix       _A_Par;                   // Left-hand-side matrix
    HYPRE_ParVector          _b_Par;                   // Right-hand-side vector
    HYPRE_ParVector          _x_Par;                   // Solution vector

    /* Interface-dependent linear system construction */
    template<class Types>
      void makeLinearSystemStruct(const InterfaceType& interface);
    template<class Types>
      void makeLinearSystemSStruct(const InterfaceType& interface);
    template<class Types>
      void getSolutionStruct(const InterfaceType& interface);
    template<class Types>
      void getSolutionSStruct(const InterfaceType& interface);

  }; // end class HypreInterface
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreInterface_h
