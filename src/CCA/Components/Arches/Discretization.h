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


#ifndef Uintah_Components_Arches_Discretization_h
#define Uintah_Components_Arches_Discretization_h

//#include <CCA/Components/Arches/StencilMatrix.h>
//#include <Core/Grid/Variables/CCVariable.h>
//#include <Core/Grid/FCVariable.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/Filter.h>


namespace Uintah {
  class ProcessorGroup;
  class PhysicalConstants;

/**************************************

CLASS
   Discretization

   Class Discretization is a class
   that computes stencil weights for linearized
   N-S equations.

GENERAL INFORMATION
   Discretization.h - declaration of the class

   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)

   Creation Date:   Mar 1, 2000

   C-SAFE


KEYWORDS


DESCRIPTION
   Class Discretization is an abstract base class
   that computes stencil weights for linearized
   N-S equations.

WARNING
   none

****************************************/

class Discretization {

public:

  enum MOMCONV { UPWIND, WALLUPWIND, HYBRID, CENTRAL, OLD };

  Discretization(PhysicalConstants* physConst);

  virtual ~Discretization();

  void calculateVelocityCoeff(const Patch* patch,
                              double delta_t,
                              bool lcentral,
                              CellInformation* cellinfo,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars,
                              constCCVariable<double>* volFraction,
                              SFCXVariable<double>* conv_scheme_x,
                              SFCYVariable<double>* conv_scheme_y,
                              SFCZVariable<double>* conv_scheme_z,
                              MOMCONV scheme, double re_limit );

  template<class T>
  void compute_Ap(CellIterator iter,
                  CCVariable<Stencil7>& A,
                  T& source);

   template<class T>
   void compute_Ap_stencilMatrix(CellIterator iter,
                                 StencilMatrix<T>& A,
                                 T& source);


  void calculateVelDiagonal(const Patch* patch,
                            ArchesVariables* vars);

  void calculatePressDiagonal(const Patch* patch,
                              ArchesVariables* vars);

  inline void setFilter(Filter* filter) {
    d_filter = filter;
  }

  inline void setTurbulentPrandtlNumber(double turbPrNo) {
    d_turbPrNo = turbPrNo;
  }

protected:

private:

      Filter* d_filter;
      double d_turbPrNo;
      PhysicalConstants* d_physicalConsts;

}; // end class Discretization

} // End namespace Uintah

#endif
