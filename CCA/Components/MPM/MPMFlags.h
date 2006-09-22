#ifndef __MPM_FLAGS_H__
#define __MPM_FLAGS_H__

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/ParticleInterpolator.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/CCA/Components/MPM/share.h>
namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class MPMFlags
    \brief A structure that store the flags used for a MPM simulation
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2004 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////


  class SCISHARE MPMFlags {

  public:

    enum IntegratorType {
      Explicit,
      Implicit,
      Fracture
    };

    int         d_8or27;// Number of nodes a particle can interact with
    double      d_ref_temp; // Reference temperature for thermal stress  
    std::string d_integrator_type; // Explicit or implicit time integration
    IntegratorType d_integrator;

    bool        d_artificial_viscosity; // Turn artificial viscosity on/off
    bool        d_accStrainEnergy; // Flag for accumulating strain energy
    bool        d_useLoadCurves; // Flag for using load curves
    bool        d_createNewParticles; // Flag to decide whether to create
                                         // new particles after failure
    bool        d_addNewMaterial; // Flag to decide whether to create
    bool        d_doErosion; // Flag to decide whether to erode or not
    bool        d_doThermalExpansion; // Decide whether to do thermExp or not
    bool        d_with_color;         // to turn on the color variable
    bool        d_fracture;         // to turn on fracture

    int         d_minGridLevel; // Only do MPM on this grid level
    int         d_maxGridLevel; // Only do MPM on this grid level
    bool        doMPMOnLevel(int level, int numLevels) const;
    
    std::string d_erosionAlgorithm; // Algorithm to erode material points

    bool        d_adiabaticHeatingOn;
    double      d_adiabaticHeating; // Flag adiabatic plastic heating on/off
    double      d_artificialDampCoeff;
    double      d_artificialViscCoeff1; // Artificial viscosity coefficient 1
    double      d_artificialViscCoeff2; // Artificial viscosity coefficient 2
    double      d_forceIncrementFactor;
    bool        d_canAddMPMMaterial;
    bool        d_do_contact_friction;
    double      d_addFrictionWork;     // 1 == add , 0 == do not add

    int         d_extraSolverFlushes;  // Have PETSc do more flushes to save memory
    bool        d_doImplicitHeatConduction;
    bool        d_doTransientImplicitHeatConduction;
    bool        d_doGridReset;   // Default is true, standard MPM
    double      d_min_part_mass; // Minimum particle mass before deletion  
    double      d_max_vel;       // Maxmimum particle velocity before  deletion
    bool        d_usingSoilFoam_CM;   // if using soil and foam CM
    bool        d_with_ice;


    ParticleInterpolator* d_interpolator;

    MPMFlags();

    ~MPMFlags();

    void readMPMFlags(ProblemSpecP& ps);
    void outputProblemSpec(ProblemSpecP& ps);

  private:

    MPMFlags(const MPMFlags& state);
    MPMFlags& operator=(const MPMFlags& state);
    
  };

} // End namespace Uintah

#endif  // __MPM_FLAGS_H__ 
