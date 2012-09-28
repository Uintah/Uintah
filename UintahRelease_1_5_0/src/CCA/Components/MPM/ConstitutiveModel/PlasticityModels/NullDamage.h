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


#ifndef __NULL_DAMAGE_MODEL_H__
#define __NULL_DAMAGE_MODEL_H__


#include "DamageModel.h"        
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class NullDamage
    \brief Default Damage Model (no damage)
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2007 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class NullDamage : public DamageModel {

  public:

  private:

    // Prevent copying of this class
    // copy constructor
    //NullDamage(const NullDamage &cm);
    NullDamage& operator=(const NullDamage &cm);

  public:
    // constructors
    NullDamage(); 
    NullDamage(ProblemSpecP& ps); 
    NullDamage(const NullDamage* cm);
         
    // destructor 
    virtual ~NullDamage();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    //////////////////////////////////////////////////////////////////////////
    /*! 
      Initialize the damage parameter in the calling function
    */
    //////////////////////////////////////////////////////////////////////////
    double initialize();

    //////////////////////////////////////////////////////////////////////////
    /*! 
      Determine if damage has crossed cut off
    */
    //////////////////////////////////////////////////////////////////////////
    bool hasFailed(double damage);
    
    //////////
    // Calculate the scalar damage parameter 
    virtual double computeScalarDamage(const double& plasticStrainRate,
                                       const Matrix3& stress,
                                       const double& temperature,
                                       const double& delT,
                                       const MPMMaterial* matl,
                                       const double& tolerance,
                                       const double& damage_old);
  
  };

} // End namespace Uintah

#endif  // __NULL_DAMAGE_MODEL_H__ 
