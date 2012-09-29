/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __ANGIO_FLAGS_H__
#define __ANGIO_FLAGS_H__

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/ParticleInterpolator.h>
#include <string>
#include <vector>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class AngioFlags
    \brief A structure that store the options used for an Angio simulation
    \author Jim Guilkey \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
  */
  /////////////////////////////////////////////////////////////////////////////


  class AngioFlags {

  public:

    enum IntegratorType {
      Explicit,
      Implicit,
      Fracture
    };

    int         d_8or27;// Number of nodes a particle can interact with
    std::string d_interpolator_type; // Type of particle-grid interaction
    std::string d_integrator_type; // Explicit or implicit time integration
    IntegratorType d_integrator;

    double      d_Grow_a;  //  Sigmoidal Growth parameter 
    double      d_Grow_b;  //  Sigmoidal Growth parameter 
    double      d_Grow_y0; //  Sigmoidal Growth parameter 
    double      d_Grow_x0; //  Sigmoidal Growth parameter 

    double      d_Branch_a1;  //  Branching model parameter 
    double      d_Branch_a2;  //  Branching model parameter 
    double      d_Branch_a3;  //  Branching model parameter 

/*
    bool        d_AMR;  // Do AMR?

    bool        d_artificial_viscosity; // Turn artificial viscosity on/off
    bool        d_accStrainEnergy; // Flag for accumulating strain energy
    bool        d_useLoadCurves; // Flag for using load curves
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
    bool        d_doExplicitHeatConduction;
    bool        d_computeNodalHeatFlux;  // compute the auxilary nodal heat flux 
    bool        d_doGridReset;   // Default is true, standard MPM
    double      d_min_part_mass; // Minimum particle mass before deletion  
    double      d_max_vel;       // Maxmimum particle velocity before  deletion

    bool        d_with_ice;
    bool        d_with_arches;

    std::vector<std::string> d_bndy_face_txt_list; 
*/

    ParticleInterpolator* d_interpolator;

    AngioFlags();

    virtual ~AngioFlags();

    virtual void readAngioFlags(ProblemSpecP& ps);
    virtual void outputProblemSpec(ProblemSpecP& ps);

  private:

    AngioFlags(const AngioFlags& state);
    AngioFlags& operator=(const AngioFlags& state);
    
  };

} // End namespace Uintah

#endif  // __ANGIO_FLAGS_H__ 
