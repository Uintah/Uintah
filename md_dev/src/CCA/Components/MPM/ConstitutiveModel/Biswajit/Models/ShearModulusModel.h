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


#ifndef __BB_SHEAR_MODULUS_MODEL_H__
#define __BB_SHEAR_MODULUS_MODEL_H__

#include "ModelState.h"
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>

namespace UintahBB {

  /*! \class ShearModulusModel
   *  \brief A generic wrapper for various shear modulus models
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2004 Container Dynamics Group
   *
   * Provides an abstract base class for various shear modulus models
  */
  class ShearModulusModel {

  protected:
    double d_shear;  // the initial shear modulus

  public:
         
    //! Construct a shear modulus model.  
    /*! This is an abstract base class. */
    ShearModulusModel();

    //! Destructor of shear modulus model.  
    /*! Virtual to ensure correct behavior */
    virtual ~ShearModulusModel();

    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps) = 0;
         
    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the shear modulus
    */
    /////////////////////////////////////////////////////////////////////////
    virtual double computeInitialShearModulus() = 0;
    virtual double computeShearModulus(const ModelState* state) = 0;
    virtual double computeShearModulus(const ModelState* state) const = 0;

    /*! Compute the shear strain energy */
    virtual double computeStrainEnergy(const ModelState* state)  = 0;

    /////////////////////////////////////////////////////////////////////////
    /* 
      Compute q = 3 mu epse_s
         where mu = shear modulus
               epse_s = sqrt{2/3} ||ee||
               ee = deviatoric part of elastic strain = epse - 1/3 epse_v I
               epse = total elastic strain
               epse_v = tr(epse)
    */
    /////////////////////////////////////////////////////////////////////////
    virtual double computeQ(const ModelState* state) const = 0;

    /////////////////////////////////////////////////////////////////////////
    /* 
      Compute dq/depse_s 
    */
    /////////////////////////////////////////////////////////////////////////
    virtual double computeDqDepse_s(const ModelState* state) const = 0;

    /////////////////////////////////////////////////////////////////////////
    /* 
      Compute dq/depse_v 
    */
    /////////////////////////////////////////////////////////////////////////
    virtual double computeDqDepse_v(const ModelState* state) const = 0;
  };
} // End namespace Uintah
      
#endif  // __BB_SHEAR_MODULUS_MODEL_H__

