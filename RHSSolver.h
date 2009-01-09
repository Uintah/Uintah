
#ifndef Uintah_Components_Arches_RHSSolver_h
#define Uintah_Components_Arches_RHSSolver_h

/**************************************
CLASS
   RHSSolver
   
   Class RHSSolver is a "right hand side" solver
   doing a final "solve" for new (timestep N+1) values 

GENERAL INFORMATION
   RHSSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION

WARNING
   none

****************************************/
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {

using namespace SCIRun;

class RHSSolver{

public:
  
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of a RHSSolver.
  RHSSolver();

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual Destructor
  virtual ~RHSSolver();


  ////////////////////////////////////////////////////////////////////////
  // Velocity Solve
  void calculateHatVelocity(const ProcessorGroup* ,
                            const Patch* patch,
                            double delta_t,
                            CellInformation* cellinfo,
                            ArchesVariables* vars,
                            ArchesConstVariables* constvars);

  void calculateVelocity(const Patch* patch,
                         double delta_t,
                         CellInformation* cellinfo,
                         ArchesVariables* vars,
                         constCCVariable<double>&,
                         constCCVariable<double>&);

  ////////////////////////////////////////////////////////////////////////
  // Scalar Solve
  void scalarLisolve(const ProcessorGroup* pc,
                     const Patch* patch,
                     double delta_t,
                     ArchesVariables* vars,
                     ArchesConstVariables* constvars,
                     CellInformation* cellinfo);

  ////////////////////////////////////////////////////////////////////////
  // Scalar Solve
  void enthalpyLisolve(const ProcessorGroup* pc,
                       const Patch* patch,
                       double delta_t,
                       ArchesVariables* vars,
                       ArchesConstVariables* constvars,
                       CellInformation* cellinfo);

  void scalarExplicitUpdate(const ProcessorGroup*,
                              const Patch* patch,
                              double delta_t,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars,
                              CellInformation* cellinfo, 
                              bool doingMM, int intrusionVal);

  template<class T_mtrx, class T_varmod, class T_varconst> void
  explicitUpdate(CellIterator iter, 
                          T_mtrx& A,
                          T_varconst source, 
                          constCCVariable<double> old_den, 
                          T_varconst old_phi,
                          T_varmod& new_phi,  
                          constCCVariable<int>  cellType,
                          CellInformation* cellinfo,
                          double delta_t, 
                          bool doingMM, int intrusionVal);

  inline void setMMS(bool doMMS) {
    d_doMMS=doMMS;
  }
  inline bool getMMS() const {
    return d_doMMS;
  }

protected:

private:

  //mms variables
  bool d_doMMS;


}; // End class RHSSolver.h

} // End namespace Uintah

#endif  
  
