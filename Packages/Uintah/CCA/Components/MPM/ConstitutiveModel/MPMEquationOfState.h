#ifndef __EQUATION_OF_STATE_H__
#define __EQUATION_OF_STATE_H__

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityState.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>


namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class MPMEquationOfState
    \brief Abstract base class for solid equations of state
    \author Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering, \n
    University of Utah \n
    Copyright (C) 2002-2003 University of Utah

  */
  ////////////////////////////////////////////////////////////////////////////

  class MPMEquationOfState {

  public:
	 
    MPMEquationOfState();
    virtual ~MPMEquationOfState();
	 
    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the hydrostatic component of stress (pressure)
        using an equation of state */
    ////////////////////////////////////////////////////////////////////////
    virtual double computePressure(const MPMMaterial* matl,
                                   const PlasticityState* state,
				   const Matrix3& deformGrad,
				   const Matrix3& rateOfDeformation,
				   const double& delT) = 0;

  };
} // End namespace Uintah
      


#endif  // __EQUATION_OF_STATE_H__

