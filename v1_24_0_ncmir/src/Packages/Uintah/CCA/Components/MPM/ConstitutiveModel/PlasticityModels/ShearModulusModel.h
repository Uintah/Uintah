#ifndef __SHEAR_MODULUS_MODEL_H__
#define __SHEAR_MODULUS_MODEL_H__

#include "PlasticityState.h"
namespace Uintah {

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

  public:
	 
    //! Construct a shear modulus model.  
    /*! This is an abstract base class. */
    ShearModulusModel();

    //! Destructor of shear modulus model.  
    /*! Virtual to ensure correct behavior */
    virtual ~ShearModulusModel();
	 
    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the shear modulus
    */
    /////////////////////////////////////////////////////////////////////////
    virtual double computeShearModulus(const PlasticityState* state) = 0;
  };
} // End namespace Uintah
      
#endif  // __SHEAR_MODULUS_MODEL_H__

