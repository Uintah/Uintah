/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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
#ifdef PetscFilter
#include <CCA/Components/Arches/Filter.h>
#endif

#include <Core/Containers/Array1.h>
namespace Uintah {
  class ProcessorGroup;

using namespace SCIRun;

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

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of a Discretization.
  // PRECONDITIONS
  // POSTCONDITIONS
  // Default constructor.
  Discretization();

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual Destructor
  virtual ~Discretization();

  // GROUP:  Action Methods
  ////////////////////////////////////////////////////////////////////////
  // Set stencil weights. (Velocity)
  // It uses second order hybrid differencing for computing
  // coefficients
  void calculateVelocityCoeff(const Patch* patch,
                              double delta_t,
                              bool lcentral,
                              CellInformation* cellinfo,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars);

  ////////////////////////////////////////////////////////////////////////
  // Set stencil weights. (Scalars)
  // It uses second order hybrid differencing for computing
  // coefficients
  void calculateScalarCoeff(const Patch* patch,
                            CellInformation* cellinfo,
                            ArchesVariables* vars,
                            ArchesConstVariables* constvars,
                            int conv_scheme);
  template<class T>
  void compute_Ap(CellIterator iter,
                  CCVariable<Stencil7>& A,
                  T& source);
                  
   template<class T>
   void compute_Ap_stencilMatrix(CellIterator iter,
                                 StencilMatrix<T>& A,
                                 T& source);   
  
                  
  ////////////////////////////////////////////////////////////////////////
  // Documentation here
  void calculateVelDiagonal(const Patch* patch,
                            ArchesVariables* vars);

  ////////////////////////////////////////////////////////////////////////
  // Documentation here
  void calculatePressDiagonal(const Patch* patch, 
                              ArchesVariables* vars);

  ////////////////////////////////////////////////////////////////////////
  // Documentation here
  void calculateScalarDiagonal(const Patch* patch,
                               ArchesVariables* vars);
  void calculateScalarDiagonal__new(const Patch* patch,
                               ArchesVariables* vars);

  ////////////////////////////////////////////////////////////////////////
  // Documentation here
  void calculateScalarFluxLimitedConvection(const Patch* patch,
                                            CellInformation* cellinfo,
                                            ArchesVariables* vars,
                                            ArchesConstVariables* constvars,
                                            const int wall_celltypeval,
                                            int limiter_type,
                                            int boundary_limiter_type,
                                            bool central_limiter);


  void computeDivergence(const ProcessorGroup*,
                         const Patch* patch,
                         DataWarehouse* new_dw,
                         ArchesVariables* vars,
                         ArchesConstVariables* constvars,
                         const bool filter_divergence,
                         const bool periodic);

#ifdef PetscFilter
  inline void setFilter(Filter* filter) {
    d_filter = filter;
  }
#endif
  inline void setTurbulentPrandtlNumber(double turbPrNo) {
    d_turbPrNo = turbPrNo;
  }

  inline void setMMS(bool doMMS) {
    d_doMMS=doMMS;
  }
  inline bool getMMS() const {
    return d_doMMS;
  }
protected:

private:
   
      // Stencil weights.
      // Array of size NDIM and of depth determined by stencil coefficients
      //StencilMatrix<CCVariable<double> >* d_press_stencil_matrix;
      // stores coefficients for all the velocity components
      // coefficients should be saved on staggered grid
      //StencilMatrix<FCVariable<double> >* d_mom_stencil_matrix;
      // coefficients for all the scalar components
      //StencilMatrix<CCVariable<double> >* d_scalar_stencil_matrix;

#ifdef PetscFilter
      Filter* d_filter;
#endif
      double d_turbPrNo;
      bool d_doMMS;
}; // end class Discretization

} // End namespace Uintah

#endif  
  
