/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



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
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <Core/Grid/Patch.h>

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
  void calculateHatVelocity(const Patch* patch,
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
  template<class T> void
  explicitUpdate_stencilMatrix(CellIterator iter, 
                               IntVector shift,                         
                               const T& source,                               
                               const T& old_phi,                              
                               T& new_phi,                              
                               constCCVariable<double> density,         
                               StencilMatrix<T>& A,
                               const OffsetArray1<double>& sew,
                               const OffsetArray1<double>& sns,
                               const OffsetArray1<double>& stb,              
                               double delta_t);              

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
  
