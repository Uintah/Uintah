/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef __MPM_FLAGS_H__
#define __MPM_FLAGS_H__
#include <CCA/Ports/Output.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/ParticleInterpolator.h>
#include <string>
#include <vector>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class MPMFlags
    \brief A structure that store the flags used for a MPM simulation
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
  */
  /////////////////////////////////////////////////////////////////////////////


  class MPMFlags {

  public:

    enum IntegratorType {
      Explicit,
      Implicit,
      Fracture
    };

    Vector      d_gravity;
    int         d_8or27;                // Number of nodes a particle can interact
                                        //   with.

    std::string d_interpolator_type;    // Type of particle-grid interaction
    double      d_cpdi_lcrit;           // Cpdi interpolator maximum fractional
                                        //   cell size for a particle

    bool            d_AMR;              // Do AMR?
    bool            d_axisymmetric;     // Use axisymmetric?
    std::string     d_integrator_type;  // Explicit or implicit time integration
    IntegratorType  d_integrator;


    bool        d_artificial_viscosity;         // Turn artificial viscosity
                                                //   on/off
    bool        d_artificial_viscosity_heating; // Include heating due to AV

    bool        d_useLoadCurves;                // Flag for using load curves
    bool        d_useCBDI;                      // Flag for using CBDI boundary
                                                //   condition treatment
    bool        d_useCPTI;                      // Flag for using CPTI
                                                //   interpolator

    bool        d_useCohesiveZones;             // Use cohesive zones
    bool        d_doErosion;                    // Whether to erode or not
    bool        d_deleteRogueParticles;         // Delete rogue particles
    bool        d_doThermalExpansion;           // Allow thermal expansion
    bool        d_with_color;                   // Enable color tracking
    bool        d_fracture;                     // Enable fracture

    int         d_minGridLevel;                 // Minimum grid level index on
                                                //   which AMR/MPM can occur
    int         d_maxGridLevel;                 // Maximum grid level index on
                                                //   which AMR/MPM can occur

    bool        doMPMOnLevel(int level,         // Does MPM occur on level?
                             int numLevels) const;

    bool        d_refineParticles;              // Refine particles, step
                                                //   toward AMR
   
    std::string d_erosionAlgorithm;             // Algorithm name used to erode
                                                //   material points
 
    double      d_artificialDampCoeff;
    double      d_PICalpha;
    double      d_artificialViscCoeff1;         // Artificial viscosity coeff 1
    double      d_artificialViscCoeff2;         // Artificial viscosity coeff 2
    double      d_forceIncrementFactor;
    bool        d_do_contact_friction;
    double      d_addFrictionWork;              // 1 == include
                                                // 0 == do not include

    int         d_extraSolverFlushes;           // Have PETSc flush more to save
                                                //   memory

    bool        d_doImplicitHeatConduction;
    bool        d_doTransientImplicitHeatConduction;
    bool        d_doExplicitHeatConduction;
    bool        d_doPressureStabilization;
    bool        d_computeNodalHeatFlux;         // compute the auxilary nodal
                                                //   heat flux
    bool        d_computeScaleFactor;           // compute the scale factor for
                                                //   viz
    bool        d_doGridReset;                  // Default true, standard MPM
    double      d_min_part_mass;                // Minimum particle mass before
                                                //   deletion
    int         d_min_subcycles_for_F;          // Minimum number of subcycles
                                                //   to use in computing the
                                                //   deformation gradient
    double      d_min_mass_for_acceleration;    // Minimum mass to allow
                                                //   division by in computing
                                                //   acceleration
    double      d_max_vel;                      // Maxmimum particle velocity
                                                //   before deletion
    bool        d_prescribeDeformation;         // Prescribe deformation via a
                                                //   table of U and R
    std::string d_prescribedDeformationFile;    // File containing prescribed
                                                //   deformations
    bool        d_exactDeformation;             // Set steps exactly to match
                                                //   times in prescribed
                                                //   deformation file
    bool        d_insertParticles;              // Activate particles according
                                                //   to color
    std::string d_insertParticlesFile;          // File containing activation
                                                //   plan
    bool        d_GEVelProj;                    // Use the velocity gradient in
                                                //   projecting particle
                                                //   velocity to grid

    //********* Thermodynamic reaciton components **************
    bool        d_thermodynamicReactions;       // Enable tracking of
                                                //   thermodynamic reactions
    bool        d_changeReactedColor;           // Use color to track reaction
                                                //   evolution

    //********* Coupling with other components ******************
    bool        d_with_ice;
    bool        d_with_arches;
    bool        d_use_momentum_form;
    std::string d_mms_type;                     // MMS Flag

    //********** Start Reactive Flow Section *********************
    bool        d_doScalarDiffusion;     //for diffusion component found in ReactiveFlow
    bool        d_doAutoCycleBC;
    bool        d_autoCycleUseMinMax;
    double      d_autoCycleMax;
    double      d_autoCycleMin;
    //********** End Reactive Flow Section *********************

    // flags for turning on/off the reduction variable calculations
    struct reductionVars
    {
      bool mass;
      bool momentum;
      bool thermalEnergy;
      bool strainEnergy;
      bool accStrainEnergy;
      bool avgStressXX;
      bool avgStressYY;
      bool avgStressZZ;
      bool KE;
      bool volDeformed;
      bool centerOfMass;
    };
    
    const ProcessorGroup            * d_myworld;
          reductionVars             * d_reductionVars;
          std::vector<std::string>    d_bndy_face_txt_list;
          ParticleInterpolator      * d_interpolator;

             MPMFlags(const ProcessorGroup* myworld);
    virtual ~MPMFlags();

    virtual void readMPMFlags(ProblemSpecP& ps, Output* dataArchive);
    virtual void outputProblemSpec(ProblemSpecP& ps);

  private:
    MPMFlags(const MPMFlags& state);
    MPMFlags& operator=(const MPMFlags& state);
    
  };

} // End namespace Uintah

#endif  // __MPM_FLAGS_H__ 
