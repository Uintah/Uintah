/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

//----- Models_RadiationModel.h --------------------------------------------------

#ifndef Uintah_Component_Models_RadiationModel_h
#define Uintah_Component_Models_RadiationModel_h

/***************************************************************************
CLASS
    Models_RadiationModel
       Sets up the Models_RadiationModel
       
GENERAL INFORMATION
    Models_RadiationModel.h - Declaration of Models_RadiationModel class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    Creation Date : 05-30-2000

    Modified: for Incorporation into Models Infrastructure, 
              Seshadri Kumar (skumar@crsim.utah.edu)
    
    Modification (start of) Date: April 11, 2005

    C-SAFE
    
    
KEYWORDS
    
DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/Models/Radiation/Models_CellInformation.h>
#include <CCA/Components/Models/Radiation/RadiationVariables.h>
#include <CCA/Components/Models/Radiation/RadiationConstVariables.h>

#include <vector>

namespace Uintah {

class Models_RadiationSolver;
class Models_RadiationModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      Models_RadiationModel();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for mixing model
      //
      virtual ~Models_RadiationModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& params) = 0;

      virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
 
      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      virtual void computeRadiationProps(const ProcessorGroup*,
                                         const Patch* patch,
                                         Models_CellInformation* cellinfo,
                                         RadiationVariables* vars,
                                         RadiationConstVariables* constvars) = 0;


      /////////////////////////////////////////////////////////////////////////
      //
      virtual void boundaryCondition(const ProcessorGroup*,
                                     const Patch* patch,
                                     RadiationVariables* vars,
                                     RadiationConstVariables* constvars)  = 0;

      /////////////////////////////////////////////////////////////////////////
      //
      virtual void intensitysolve(const ProcessorGroup*,
                                  const Patch* patch,
                                  Models_CellInformation* cellinfo,
                                  RadiationVariables* vars,
                                  RadiationConstVariables* constvars)  = 0;

      Models_RadiationSolver* d_linearSolver;
 protected:
      void computeOpticalLength();
      double d_opl; // optical length
 private:

}; // end class Models_RadiationModel

} // end namespace Uintah

#endif




