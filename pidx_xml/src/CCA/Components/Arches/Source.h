/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Uintah_Components_Arches_Source_h
#define Uintah_Components_Arches_Source_h

/**************************************
CLASS
   Source
   
   Class Source computes source terms for 
   N-S equations.  

GENERAL INFORMATION
   Source.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   

KEYWORDS


DESCRIPTION
   Class Source computes source terms for 
   N-S equations.  

WARNING
   none

****************************************/

#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>

namespace Uintah {
  class ProcessorGroup;
class PhysicalConstants;
class BoundaryCondition;

class Source {

public:
  
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of a Source.
  // PRECONDITIONS
  // POSTCONDITIONS
  Source();

  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of a Source.
  // PRECONDITIONS
  // POSTCONDITIONS
  Source(PhysicalConstants* phys_const);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
  ~Source();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Set up the problem specification database
  void problemSetup(const ProblemSpecP& params);

  // GROUP:  Action Methods
  ////////////////////////////////////////////////////////////////////////
  // Set source terms. Will need more parameters...like velocity and
  // scalars
  void calculatePressureSourcePred(const ProcessorGroup* pc,
                                   const Patch* patch,
                                   double delta_t,
                                   ArchesVariables* vars,
                                   ArchesConstVariables* constvars); 
  ////////////////////////////////////////////////////////////////////////
  // Set source terms. Will need more parameters...like velocity and
  // scalars
  void calculateVelocitySource(const Patch* patch,
                               double delta_t, 
                               CellInformation* cellinfo,
                               ArchesVariables* vars,
                               ArchesConstVariables* constvars);

  template<class T>
  void compute_massSource(CellIterator iter, IntVector D, 
                           constCCVariable<double> volFraction, 
                           const T& vel,
                           StencilMatrix<T>& velCoeff,
                           T& velNonLinearSrc,
                           StencilMatrix<T>& velConvectCoeff);
                           
  void modifyVelMassSource(const Patch* patch,
                           constCCVariable<double> volFraction, 
                           ArchesVariables* vars,
                           ArchesConstVariables* constvars);

  ////////////////////////////////////////////////////////////////////////
  // Add multimaterial source term
  void computemmMomentumSource(const ProcessorGroup* pc,
                               const Patch* patch,
                               CellInformation* cellinfo,
                               ArchesVariables* vars,
                               ArchesConstVariables* constvars);

                                   
private:

  PhysicalConstants* d_physicalConsts;
  std::string d_mms;
  double d_airDensity, d_heDensity;
  Vector d_gravity;
  double d_viscosity;
  double d_turbPrNo;

  // linear mms
  double cu, cv, cw, cp, phi0;
  // sine mms
  double amp;

}; // end Class Source

} // End namespace Uintah
#endif  
  
