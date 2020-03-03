/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

//----- ArchesLabel.h -----------------------------------------------

#ifndef Uintah_Components_Arches_ArchesLabel_h
#define Uintah_Components_Arches_ArchesLabel_h

#include <Core/Grid/MaterialManagerP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/Handle.h>
#include <sci_defs/uintah_defs.h>

#include <map>

/**************************************
CLASS
   ArchesLabel

   Class ArchesLabel creates and stores the VarLabels that are used in Arches

GENERAL INFORMATION
   ArchesLabel.h - declaration of the class

   Author: Biswajit Banerjee (bbanerje@crsim.utah.edu)

   Creation Date:   July 18, 2000

   C-SAFE


KEYWORDS

DESCRIPTION

WARNING
   none

************************************************************************/

namespace Uintah {
  class VarLabel;
    class ArchesLabel {
    public:

      enum VARID { BADVALUE, TEMPERATURE, DENSITY, SOOT, H2O, CO2, ENTHALPY, SPECIFICHEAT, MIXTUREFRACTION };

      ArchesLabel();
      ~ArchesLabel();
      void setMaterialManager(MaterialManagerP& materialManager);
      void problemSetup( const ProblemSpecP& db );

      typedef std::map<VARID,const std::string> RLMAP;

      /** @brief Retrieve a label based on its CFD role **/
      const VarLabel* getVarlabelByRole( VARID role );

      /** @brief Set a label to have a specific role **/
      void setVarlabelToRole( const std::string label, const std::string role );

      const std::string getRoleString( VARID role );

      // recompile task graph flag
      bool recompile_taskgraph;

      MaterialManagerP d_materialManager;

      // material subset for stencils
      MaterialSubset* d_stencilMatl;

      MaterialSubset* d_vectorMatl;
      MaterialSubset* d_tensorMatl;
      MaterialSubset* d_symTensorMatl;

      // Delta t label
      const VarLabel* d_timeStepLabel;
      const VarLabel* d_simulationTimeLabel;
      const VarLabel* d_delTLabel;

      // Cell Information
      // for old_dw, perpatch var
      const VarLabel* d_cellInfoLabel;

      // Cell type
      const VarLabel* d_cellTypeLabel;//computed for old_dw in cellTypeInit

      // Labels for inlet and flow rate
      const VarLabel* d_totalflowINLabel;
      const VarLabel* d_totalflowOUTLabel;
      const VarLabel* d_netflowOUTBCLabel;

      // net outlet area, mass balance, and overall outlet velocity
      const VarLabel* d_totalAreaOUTLabel;
      const VarLabel* d_denAccumLabel;
      // Density Labels

      const VarLabel* d_densityCPLabel;
      const VarLabel* d_densityGuessLabel;
      const VarLabel* d_densityTempLabel;

      // filtered drhodt
      const VarLabel* d_filterdrhodtLabel;

      // Viscosity Labels
      // for old_dw in computeTurbModel
      const VarLabel* d_viscosityCTSLabel;
      const VarLabel* d_turbViscosLabel;

      // Pressure Labels

      // for old_dw in computePressureBC
      const VarLabel* d_pressurePSLabel;
      const VarLabel* d_pressureGuessLabel;
      const VarLabel* d_pressureExtraProjectionLabel;

      // Pressure Coeff Labels
      // for new_dw in pressuresolver::linearizeMatrix
      const VarLabel* d_presCoefPBLMLabel;

      // Pressure Non Linear Src Labels
      // in pressureSolver::linearizeMatrix
      const VarLabel* d_presNonLinSrcPBLMLabel;

      // U-Velocity Labels
      const VarLabel* d_uVelocitySPBCLabel;
      // V-Velocity Labels
      const VarLabel* d_vVelocitySPBCLabel;
      // W-Velocity Labels
      const VarLabel* d_wVelocitySPBCLabel;

      // Duplicate Labels to get Kokkos Arches to work.
      // These are just copies of uVelocitySPBC, etc.
      // U-Velocity Labels
      const VarLabel* d_uVelocityLabel;
      // V-Velocity Labels
      const VarLabel* d_vVelocityLabel;
      // W-Velocity Labels
      const VarLabel* d_wVelocityLabel;

      // UMom Labels
      const VarLabel* d_uMomLabel;
      // VMom Labels
      const VarLabel* d_vMomLabel;
      // WMom Labels
      const VarLabel* d_wMomLabel;
      // conv_scheme Label
      const VarLabel* d_conv_scheme_x_Label;
      const VarLabel* d_conv_scheme_y_Label;
      const VarLabel* d_conv_scheme_z_Label;

      // labels for scalesimilaritymodels
      const VarLabel* d_stressTensorCompLabel;
      const VarLabel* d_stressSFCXdivLabel;
      const VarLabel* d_stressSFCYdivLabel;
      const VarLabel* d_stressSFCZdivLabel;
      const VarLabel* d_stressCCXdivLabel;
      const VarLabel* d_stressCCYdivLabel;
      const VarLabel* d_stressCCZdivLabel;
      const VarLabel* d_strainTensorCompLabel;
      const VarLabel* d_betaIJCompLabel;
      const VarLabel* d_cbetaIJCompLabel;
      const VarLabel* d_LIJCompLabel;

      // labels for dynamic procedure
      const VarLabel* d_strainMagnitudeLabel;
      const VarLabel* d_strainMagnitudeMLLabel;
      const VarLabel* d_strainMagnitudeMMLabel;
      const VarLabel* d_LalphaLabel;
      const VarLabel* d_cbetaHATalphaLabel;
      const VarLabel* d_alphaalphaLabel;
      const VarLabel* d_CsLabel;
      const VarLabel* d_deltaCsLabel;

      //odt model
      const VarLabel* d_odtDataLabel;

      // ref_density and pressure
      const VarLabel* d_refDensity_label;
      const VarLabel* d_refDensityPred_label;
      const VarLabel* d_refPressurePred_label;
      const VarLabel* d_refPressure_label;

      // labels for nonlinear residuals

      // For storing the interpolated CC Velocity Variables
      const VarLabel* d_CCVelocityLabel;
      const VarLabel* d_CCUVelocityLabel;
      const VarLabel* d_CCVVelocityLabel;
      const VarLabel* d_CCWVelocityLabel;

      // for multimaterial
      const VarLabel* d_mmcellTypeLabel;
      const VarLabel* d_mmgasVolFracLabel;

      const VarLabel* d_densityMicroLabel;
      const VarLabel* d_densityMicroINLabel;
      const VarLabel* d_pressPlusHydroLabel;

      // labels for pressure solver
      const VarLabel* d_uVelRhoHatLabel;
      const VarLabel* d_vVelRhoHatLabel;
      const VarLabel* d_wVelRhoHatLabel;

      const VarLabel* d_uVelRhoHat_CCLabel;
      const VarLabel* d_vVelRhoHat_CCLabel;
      const VarLabel* d_wVelRhoHat_CCLabel;

      const VarLabel* d_pressurePredLabel;

      // for radiation
      const VarLabel* d_fvtfiveINLabel;
      const VarLabel* d_tfourINLabel;
      const VarLabel* d_tfiveINLabel;
      const VarLabel* d_tnineINLabel;
      const VarLabel* d_qrgINLabel;
      const VarLabel* d_qrsINLabel;
      const VarLabel* d_absorpINLabel;
      const VarLabel* d_abskgINLabel;
      const VarLabel* d_radiationSRCINLabel;
      const VarLabel* d_radiationFluxEINLabel;
      const VarLabel* d_radiationFluxWINLabel;
      const VarLabel* d_radiationFluxNINLabel;
      const VarLabel* d_radiationFluxSINLabel;
      const VarLabel* d_radiationFluxTINLabel;
      const VarLabel* d_radiationFluxBINLabel;
      const VarLabel* d_radiationVolqINLabel;

      // runge-kutta 3d order properties labels
      const VarLabel* d_refDensityInterm_label;
      const VarLabel* d_refPressureInterm_label;

      // runge-kutta 3d order pressure and momentum labels
      const VarLabel* d_pressureIntermLabel;
      const VarLabel* d_velocityDivergenceLabel;
      const VarLabel* d_vorticityXLabel;
      const VarLabel* d_vorticityYLabel;
      const VarLabel* d_vorticityZLabel;
      const VarLabel* d_vorticityLabel;
      const VarLabel* d_velDivResidualLabel;
      const VarLabel* d_velocityDivergenceBCLabel;
      const VarLabel* d_continuityResidualLabel;

      const VarLabel* d_negativeDensityGuess_label;
      const VarLabel* d_negativeDensityGuessPred_label;
      const VarLabel* d_negativeDensityGuessInterm_label;
      const VarLabel* d_densityLag_label;
      const VarLabel* d_densityLagPred_label;
      const VarLabel* d_densityLagInterm_label;
      const VarLabel* d_densityLagAfterAverage_label;
      const VarLabel* d_densityLagAfterIntermAverage_label;

      // kinetic energy
      const VarLabel* d_kineticEnergyLabel;
      const VarLabel* d_totalKineticEnergyLabel;

      const VarLabel* d_oldDeltaTLabel;

       // test filtered terms for variable density dynamic Smagorinsky model
      const VarLabel* d_filterRhoULabel;
      const VarLabel* d_filterRhoVLabel;
      const VarLabel* d_filterRhoWLabel;
      const VarLabel* d_filterRhoLabel;
      const VarLabel* d_filterRhoFLabel;
      const VarLabel* d_filterRhoELabel;
      const VarLabel* d_filterRhoRFLabel;
      const VarLabel* d_filterScalarGradientCompLabel;
      const VarLabel* d_filterEnthalpyGradientCompLabel;
      const VarLabel* d_filterReactScalarGradientCompLabel;
      const VarLabel* d_filterStrainTensorCompLabel;
      const VarLabel* d_filterVolumeLabel;
      const VarLabel* d_ShFLabel;
      const VarLabel* d_ShELabel;
      const VarLabel* d_ShRFLabel;

      // DQMOM Variables:

      // Particle velocity map ( populated in Arches.cc::registerDQMOMEqns() )
      typedef std::map<int, const VarLabel* > PartVelMap;
      PartVelMap partVel;

      // Particle masses ( populated in Arches.cc::registerDQMOMEqns() )
      typedef std::map<int, const VarLabel* > ParticleMassMap;
      ParticleMassMap particleMasses;

      // DQMOM moments ( populated in Arches.cc::registerDQMOMEqns() )
      typedef std::vector<int> MomentVector;
      typedef std::map<const MomentVector, const VarLabel* > MomentMap;
      MomentMap DQMOMMoments;

      //CQMOM Variables:
      typedef std::map<int, const VarLabel* > WeightMap;
      WeightMap CQMOMWeights;

      typedef std::map<int, const VarLabel* > AbscissaMap;
      AbscissaMap CQMOMAbscissas;

      const VarLabel* d_areaFractionLabel;
      const VarLabel* d_areaFractionFXLabel;
      const VarLabel* d_areaFractionFYLabel;
      const VarLabel* d_areaFractionFZLabel;
      const VarLabel* d_volFractionLabel;

    private:

      RLMAP d_r_to_l;

    }; // End class ArchesLabel
} // End namespace Uintah


#endif
