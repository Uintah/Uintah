#ifndef __MPM_FLAGS_H__
#define __MPM_FLAGS_H__

#include <string>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class MPMFlags
    \brief A structure that store the plasticity state data
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2004 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class MPMFlags {

  public:

    int         d_8or27;// Number of nodes a particle can interact with
    std::string d_integrator_type; // Explicit or implicit time integration

    bool        d_artificial_viscosity; // Turn artificial viscosity on/off
    bool        d_accStrainEnergy; // Flag for accumulating strain energy
    bool        d_useLoadCurves; // Flag for using load curves
    bool        d_createNewParticles; // Flag to decide whether to create
                                         // new particles after failure
    bool        d_doErosion; // Flag to decide whether to erode or not
    bool        d_with_color;         // to turn on the color variable
    
    std::string d_erosionAlgorithm; // Algorithm to erode material points

    double      d_adiabaticHeating; // Flag adiabatic plastic heating on/off
    double      d_artificialDampCoeff;
    double      d_forceIncrementFactor;

    MPMFlags();

    ~MPMFlags();

  private:

    MPMFlags(const MPMFlags& state);
    MPMFlags& operator=(const MPMFlags& state);
    
  };

} // End namespace Uintah

#endif  // __MPM_FLAGS_H__ 
