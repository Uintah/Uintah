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


#ifndef __JOHNSONCOOK_DAMAGE_MODEL_H__
#define __JOHNSONCOOK_DAMAGE_MODEL_H__


#include "DamageModel.h"        
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class JohnsonCookDamage
    \brief Johnson-Cook Damage Model
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2002 University of Utah

    References:
    1) Johnson and Cook, 1985, Int. J. Eng. Fracture Mech., 21, 31-48.

    The damage evolution rule is given by \n
    \f$
    \dot{D} = \dot{\epsilon_p}/\epsilon_p^f
    \f$ \n
    where \n
    \f$ D \f$ = damage variable \n
    where \f$ D \f$ = 0 for virgin material, 
    \f$ D \f$ = 1 for fracture \n
    \f$ \epsilon_p^f\f$  = value of fracture strain given by \n
    \f$ 
    \epsilon_p^f = (D1 + D2 \exp (D3 \sigma*)][1+\dot{p}^*]^(D4)[1+D5 T^*
    \f$ \n 
    where \f$ \sigma^*= 1/3*trace(\sigma)/\sigma_{eq} \f$ \n
    \f$  D1, D2, D3, D4, D5\f$  are constants \n
    \f$  T^* = (T-T_{room})/(T_{melt}-T_{room}) \f$
  */
  /////////////////////////////////////////////////////////////////////////////

  class JohnsonCookDamage : public DamageModel {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double D0; /*< Initial damage */
      double Dc; /*< Critical damage */
      double D1;
      double D2;
      double D3;
      double D4;
      double D5;
      double spallStress;
    };   

  private:

    CMData d_initialData;
         
    // Prevent copying of this class
    // copy constructor
    //JohnsonCookDamage(const JohnsonCookDamage &cm);
    JohnsonCookDamage& operator=(const JohnsonCookDamage &cm);

  public:
    // constructors
    JohnsonCookDamage(ProblemSpecP& ps); 
    JohnsonCookDamage(const JohnsonCookDamage* cm);
         
    // destructor 
    virtual ~JohnsonCookDamage();

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

#endif  // __JOHNSONCOOK_DAMAGE_MODEL_H__ 
