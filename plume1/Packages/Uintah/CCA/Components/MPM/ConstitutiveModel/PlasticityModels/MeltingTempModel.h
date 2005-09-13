#ifndef __MELTING_TEMP_MODEL_H__
#define __MELTING_TEMP_MODEL_H__

#include "PlasticityState.h"

namespace Uintah {

  /*! \class MeltingTempModel
   *  \brief A generic wrapper for various melting temp models
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2003 Container Dynamics Group
   *
   * Provides an abstract base class for various melting temp models used
   * in the plasticity and damage models
  */
  class MeltingTempModel {

  public:
	 
    //! Construct a melting temp model.  
    /*! This is an abstract base class. */
    MeltingTempModel();

    //! Destructor of melting temp model.  
    /*! Virtual to ensure correct behavior */
    virtual ~MeltingTempModel();
	 
    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the melting temperature
    */
    /////////////////////////////////////////////////////////////////////////
    virtual double computeMeltingTemp(const PlasticityState* state) = 0;

  };
} // End namespace Uintah
      
#endif  // __MELTING_TEMP_MODEL_H__

