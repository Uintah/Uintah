/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


//----- ArchesLabel.h -----------------------------------------------

#ifndef Uintah_Components_Arches_ArchesLabel_h
#define Uintah_Components_Arches_ArchesLabel_h

#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/Handle.h>
#include <map>

namespace Uintah {

/**
  @class    ArchesLabel
  @author   Biswajit Banerjee
  @date     July 18, 2000

  @brief    Creates and stores a large number of the VarLabels used in Arches
  
  @details
  The idea behind ArchesLabel is to create a publicly-accessible list of variable labels accessible to any class that may need them.
  Many objects' constructors have an instance of ArchesLabel in their initialization list.

  Some common prefixes/suffixes include:

  CTS = Compute Turbulent Subgrid model
  CP  = Compute Properties
  EKT = "Echt Konservativer Transport" (German for "Fully Conservative Transport") (thanks to Stanislav Borodai for the complete lack of documentation or even an explanation of what EKT means)

  @seealso For EKT: Direct and large-eddy simulation V: proceedings of the fifth international ERCOFTAC Workshop on Direct and Large-eddy simulation, held at the Munich University of Technology, August 27-29, 2003, Volume 2003
*/

  class VarLabel;
    class ArchesLabel {
    public:

      ArchesLabel();
      ~ArchesLabel();
      void setSharedState(SimulationStateP& sharedState);
     
      SimulationStateP d_sharedState;

      // material subset for stencils
      MaterialSubset* d_stencilMatl;

      MaterialSubset* d_vectorMatl;
      MaterialSubset* d_tensorMatl;
      MaterialSubset* d_symTensorMatl;

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
      const VarLabel* d_densityEKTLabel;
      const VarLabel* d_densityGuessLabel;
      const VarLabel* d_densityTempLabel;
      const VarLabel* d_densityOldOldLabel;

      // filtered drhodt
      const VarLabel* d_filterdrhodtLabel;
      // for computing divergence constraint
      const VarLabel* d_drhodfCPLabel;

      // Viscosity Labels
      // for old_dw in computeTurbModel
      const VarLabel* d_viscosityCTSLabel;            ///< Turbulent subgrid eddy viscosity
      const VarLabel* d_scalarDiffusivityLabel;       ///< 
      const VarLabel* d_enthalpyDiffusivityLabel;     ///< 
      const VarLabel* d_reactScalarDiffusivityLabel;  ///< 


      // Pressure Labels

      // for old_dw in computePressureBC
      const VarLabel* d_pressurePSLabel;
      const VarLabel* d_pressureExtraProjectionLabel;

      // Pressure Coeff Labels
      // for new_dw in pressuresolver::linearizeMatrix
      const VarLabel* d_presCoefPBLMLabel;

      // Pressure Non Linear Src Labels
      // in pressureSolver::linearizeMatrix
      const VarLabel* d_presNonLinSrcPBLMLabel;

      const VarLabel* d_uVelocitySPBCLabel; ///< U velocity labels
      const VarLabel* d_vVelocitySPBCLabel; ///< V velocity labels
      const VarLabel* d_wVelocitySPBCLabel; ///< W velocity labels

      const VarLabel* d_uVelocityEKTLabel;
      const VarLabel* d_vVelocityEKTLabel;
      const VarLabel* d_wVelocityEKTLabel;

      // Scalar Labels
      //computed for old_dw in paramInit
      // computed for new_dw in setInitialGuess

      const VarLabel* d_scalarSPLabel;
      const VarLabel* d_scalarEKTLabel;
      const VarLabel* d_scalarTempLabel;
      const VarLabel* d_scalarFELabel;

      // Scalar variance labels

      // computed for new_dw in Smagorinsky Model
      const VarLabel* d_scalarVarSPLabel;

      // computed for new_dw in Smagorinsky Model for flamelet
      const VarLabel* d_scalarDissSPLabel;

      // Scalar Coef
      const VarLabel* d_scalCoefSBLMLabel;

      // scalar diffusion coeffs, required for divergence constraint
      const VarLabel* d_scalDiffCoefLabel;
      const VarLabel* d_scalDiffCoefSrcLabel;
      const VarLabel* d_enthDiffCoefLabel;

      // new scalar coeffs:
      const VarLabel* d_scalarTotCoefLabel; 

      // Scalar NonLinear Src
      const VarLabel* d_scalNonLinSrcSBLMLabel;


      // reactive scalars
      const VarLabel* d_reactscalarSPLabel;
      const VarLabel* d_reactscalarEKTLabel;
      const VarLabel* d_reactscalarTempLabel;
      const VarLabel* d_reactscalarFELabel;

      // Reactscalar variance labels
      // computed for new_dw in Smagorinsky Model
      const VarLabel* d_reactscalarVarSPLabel;

      // Reactscalar Coef
      const VarLabel* d_reactscalCoefSBLMLabel;

      // Reactscalar Diffusion Coef
      const VarLabel* d_reactscalDiffCoefLabel;
      
      // Reactscalar NonLinear Src
      const VarLabel* d_reactscalNonLinSrcSBLMLabel;

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

      const VarLabel* d_scalarFluxCompLabel;
      
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
      const VarLabel* d_refPressure_label;

      // labels for nonlinear residuals

      const VarLabel* d_oldCCVelocityLabel;       ///< Stores old interpolated velocity
      const VarLabel* d_newCCVelocityLabel;       ///< Stores new interpolated velocity
      const VarLabel* d_newCCVelMagLabel;         ///< Stores new interpolated velocity magnitude
      const VarLabel* d_newCCUVelocityLabel;      ///< Stores new interpolated U velocity
      const VarLabel* d_newCCVVelocityLabel;      ///< Stores new interpolated V velocity
      const VarLabel* d_newCCWVelocityLabel;      ///< Stores new interpolated W velocity

      // for multimaterial
      const VarLabel* d_mmcellTypeLabel;
      const VarLabel* d_mmgasVolFracLabel;

      // for reacting flows
      const VarLabel* d_dummyTLabel;
      const VarLabel* d_tempINLabel;
      const VarLabel* d_tempFxLabel; 
      const VarLabel* d_tempFyLabel;
      const VarLabel* d_tempFzLabel; 
      const VarLabel* d_cpINLabel;
      const VarLabel* d_co2INLabel;
      const VarLabel* d_h2oINLabel;
      const VarLabel* d_normalizedScalarVarLabel;
      const VarLabel* d_heatLossLabel;

      const VarLabel* d_h2sINLabel;
      const VarLabel* d_so2INLabel;
      const VarLabel* d_so3INLabel;
      const VarLabel* d_sulfurINLabel;

      const VarLabel* d_mixMWLabel; 

      const VarLabel* d_s2INLabel;
      const VarLabel* d_shINLabel;
      const VarLabel* d_soINLabel;
      const VarLabel* d_hso2INLabel;

      const VarLabel* d_hosoINLabel;
      const VarLabel* d_hoso2INLabel;
      const VarLabel* d_snINLabel;
      const VarLabel* d_csINLabel;

      const VarLabel* d_ocsINLabel;
      const VarLabel* d_hsoINLabel;
      const VarLabel* d_hosINLabel;
      const VarLabel* d_hsohINLabel;

      const VarLabel* d_h2soINLabel;
      const VarLabel* d_hoshoINLabel;
      const VarLabel* d_hs2INLabel;
      const VarLabel* d_h2s2INLabel;

      const VarLabel* d_coINLabel;
      const VarLabel* d_c2h2INLabel;
      const VarLabel* d_ch4INLabel;
      const VarLabel* d_denRefArrayLabel;
      const VarLabel* d_densityMicroLabel;
      const VarLabel* d_densityMicroINLabel;
      const VarLabel* d_pressPlusHydroLabel;

      const VarLabel* d_uvwoutLabel;                ///< For outlet BC

      // labels for pressure solver
      const VarLabel* d_uVelRhoHatLabel;            ///< Labels for pressure solver
      const VarLabel* d_vVelRhoHatLabel;            ///< Labels for pressure solver 
      const VarLabel* d_wVelRhoHatLabel;            ///< Labels for pressure solver 

      const VarLabel* d_uVelRhoHat_CCLabel;
      const VarLabel* d_vVelRhoHat_CCLabel;
      const VarLabel* d_wVelRhoHat_CCLabel;

      const VarLabel* d_divConstraintLabel;         ///< Divergence constraint

      const VarLabel* d_pressurePredLabel;

      const VarLabel* d_enthalpySPLabel;            ///< Enthalpy equation  
      const VarLabel* d_enthalpyEKTLabel;           ///< Enthalpy equation  
      const VarLabel* d_enthalpyTempLabel;          ///< Enthalpy equation    
      const VarLabel* d_enthalpyFELabel;            ///< Enthalpy equation  
      
      const VarLabel* d_enthalpyRXNLabel;           ///< Validation


      const VarLabel* d_enthCoefSBLMLabel;          ///< Enthalpy Coef

      const VarLabel* d_enthNonLinSrcSBLMLabel;     ///< Enthalpy NonLinear Src

      // for radiation
      const VarLabel* d_fvtfiveINLabel;             ///< Radiation
      const VarLabel* d_tfourINLabel;               ///< Radiation            
      const VarLabel* d_tfiveINLabel;               ///< Radiation            
      const VarLabel* d_tnineINLabel;               ///< Radiation            
      const VarLabel* d_qrgINLabel;                 ///< Radiation          
      const VarLabel* d_qrsINLabel;                 ///< Radiation          
      const VarLabel* d_absorpINLabel;              ///< Radiation
      const VarLabel* d_sootFVINLabel;              ///< Radiation
      const VarLabel* d_abskgINLabel;               ///< Radiation
      const VarLabel* d_radiationSRCINLabel;        ///< Radiation      
      const VarLabel* d_radiationFluxEINLabel;      ///< Radiation        
      const VarLabel* d_radiationFluxWINLabel;      ///< Radiation        
      const VarLabel* d_radiationFluxNINLabel;      ///< Radiation        
      const VarLabel* d_radiationFluxSINLabel;      ///< Radiation        
      const VarLabel* d_radiationFluxTINLabel;      ///< Radiation        
      const VarLabel* d_radiationFluxBINLabel;      ///< Radiation        
      const VarLabel* d_radiationVolqINLabel;       ///< Radiation      
 
      const VarLabel* d_reactscalarSRCINLabel;      ///< Reactive scalar source term from properties
      
      const VarLabel* d_refDensityInterm_label;     ///< Runge-Kutta 3rd order properties labels
      
      const VarLabel* d_pressureIntermLabel;        ///< Runge-Kutta 3rd order pressure and momentum labels        
      const VarLabel* d_velocityDivergenceLabel;    ///< Runge-Kutta 3rd order pressure and momentum labels            
      const VarLabel* d_vorticityXLabel;            ///< Runge-Kutta 3rd order pressure and momentum labels    
      const VarLabel* d_vorticityYLabel;            ///< Runge-Kutta 3rd order pressure and momentum labels    
      const VarLabel* d_vorticityZLabel;            ///< Runge-Kutta 3rd order pressure and momentum labels    
      const VarLabel* d_vorticityLabel;             ///< Runge-Kutta 3rd order pressure and momentum labels  
      const VarLabel* d_velDivResidualLabel;        ///< Runge-Kutta 3rd order pressure and momentum labels        
      const VarLabel* d_velocityDivergenceBCLabel;  ///< Runge-Kutta 3rd order pressure and momentum labels              
      const VarLabel* d_continuityResidualLabel;    ///< Runge-Kutta 3rd order pressure and momentum labels            

      const VarLabel* d_InitNormLabel;
      const VarLabel* d_ScalarClippedLabel;
      const VarLabel* d_ReactScalarClippedLabel;
      const VarLabel* d_uVelNormLabel;
      const VarLabel* d_vVelNormLabel;
      const VarLabel* d_wVelNormLabel;
      const VarLabel* d_rhoNormLabel;
      const VarLabel* d_negativeDensityGuess_label;
      const VarLabel* d_negativeDensityGuessPred_label;
      const VarLabel* d_negativeDensityGuessInterm_label;
      const VarLabel* d_negativeEKTDensityGuess_label;
      const VarLabel* d_negativeEKTDensityGuessPred_label;
      const VarLabel* d_negativeEKTDensityGuessInterm_label;
      const VarLabel* d_densityLag_label;
      const VarLabel* d_densityLagPred_label;
      const VarLabel* d_densityLagInterm_label;
      const VarLabel* d_densityLagAfterAverage_label;
      const VarLabel* d_densityLagAfterIntermAverage_label;
      
      // kinetic energy
      const VarLabel* d_kineticEnergyLabel;
      const VarLabel* d_totalKineticEnergyLabel;
      const VarLabel* d_totalKineticEnergyPredLabel;
      const VarLabel* d_totalKineticEnergyIntermLabel;
      
      const VarLabel* d_smmsLnErrorLabel;             ///< Scalar MMS Ln error
      const VarLabel* d_totalsmmsLnErrorLabel;        ///< Scalar MMS Ln error      
      const VarLabel* d_totalsmmsLnErrorPredLabel;    ///< Scalar MMS Ln error          
      const VarLabel* d_totalsmmsLnErrorIntermLabel;  ///< Scalar MMS Ln error            
      const VarLabel* d_totalsmmsExactSolLabel;       ///< Scalar MMS Ln error      
      const VarLabel* d_totalsmmsExactSolPredLabel;   ///< Scalar MMS Ln error          
      const VarLabel* d_totalsmmsExactSolIntermLabel; ///< Scalar MMS Ln error          
      
      // grad P mms Ln error
      const VarLabel* d_gradpmmsLnErrorLabel;
      const VarLabel* d_totalgradpmmsLnErrorLabel;
      const VarLabel* d_totalgradpmmsLnErrorPredLabel;
      const VarLabel* d_totalgradpmmsLnErrorIntermLabel;
      const VarLabel* d_totalgradpmmsExactSolLabel;
      const VarLabel* d_totalgradpmmsExactSolPredLabel;
      const VarLabel* d_totalgradpmmsExactSolIntermLabel;
      
      const VarLabel* d_ummsLnErrorLabel;             ///< U velocity MMS Ln error 
      const VarLabel* d_totalummsLnErrorLabel;        ///< U velocity MMS Ln error 
      const VarLabel* d_totalummsLnErrorPredLabel;    ///< U velocity MMS Ln error 
      const VarLabel* d_totalummsLnErrorIntermLabel;  ///< U velocity MMS Ln error 
      const VarLabel* d_totalummsExactSolLabel;       ///< U velocity MMS Ln error 
      const VarLabel* d_totalummsExactSolPredLabel;   ///< U velocity MMS Ln error 
      const VarLabel* d_totalummsExactSolIntermLabel; ///< U velocity MMS Ln error 
      
      const VarLabel* d_vmmsLnErrorLabel;             ///< V velocity MMS Ln error  
      const VarLabel* d_totalvmmsLnErrorLabel;        ///< V velocity MMS Ln error    
      const VarLabel* d_totalvmmsLnErrorPredLabel;    ///< V velocity MMS Ln error    
      const VarLabel* d_totalvmmsLnErrorIntermLabel;  ///< V velocity MMS Ln error    
      const VarLabel* d_totalvmmsExactSolLabel;       ///< V velocity MMS Ln error  
      const VarLabel* d_totalvmmsExactSolPredLabel;   ///< V velocity MMS Ln error  
      const VarLabel* d_totalvmmsExactSolIntermLabel; ///< V velocity MMS Ln error  

      const VarLabel* d_wmmsLnErrorLabel;             ///< W velocity MMS Ln error
      const VarLabel* d_totalwmmsLnErrorLabel;        ///< W velocity MMS Ln error 
      const VarLabel* d_totalwmmsLnErrorPredLabel;    ///< W velocity MMS Ln error 
      const VarLabel* d_totalwmmsLnErrorIntermLabel;  ///< W velocity MMS Ln error 
      const VarLabel* d_totalwmmsExactSolLabel;       ///< W velocity MMS Ln error 
      const VarLabel* d_totalwmmsExactSolPredLabel;   ///< W velocity MMS Ln error 
      const VarLabel* d_totalwmmsExactSolIntermLabel; ///< W velocity MMS Ln error 

      // mass balance labels for RK
      const VarLabel* d_totalflowINPredLabel;
      const VarLabel* d_totalflowOUTPredLabel;
      const VarLabel* d_denAccumPredLabel;
      const VarLabel* d_netflowOUTBCPredLabel;
      const VarLabel* d_totalAreaOUTPredLabel;
      const VarLabel* d_totalflowINIntermLabel;
      const VarLabel* d_totalflowOUTIntermLabel;
      const VarLabel* d_denAccumIntermLabel;
      const VarLabel* d_netflowOUTBCIntermLabel;
      const VarLabel* d_totalAreaOUTIntermLabel;
 

      // Timestep labels
      const VarLabel* d_oldDeltaTLabel;
      const VarLabel* d_MinDQMOMTimestepLabel;  ///< VarLabel holding minimum timestep required for stability for DQMOM transport equations
      const VarLabel* d_MinScalarTimestepLabel; ///< VarLabel holding minimum timestep required for stability for scalar transport equations


       // test filtered terms for variable density dynamic Smagorinsky model
      const VarLabel* d_filterRhoULabel;
      const VarLabel* d_filterRhoVLabel;
      const VarLabel* d_filterRhoWLabel;
      const VarLabel* d_filterRhoLabel;
      const VarLabel* d_filterRhoFLabel;
      const VarLabel* d_filterRhoELabel;
      const VarLabel* d_filterRhoRFLabel;
      const VarLabel* d_scalarGradientCompLabel;
      const VarLabel* d_filterScalarGradientCompLabel;
      const VarLabel* d_enthalpyGradientCompLabel;
      const VarLabel* d_filterEnthalpyGradientCompLabel;
      const VarLabel* d_reactScalarGradientCompLabel;
      const VarLabel* d_filterReactScalarGradientCompLabel;
      const VarLabel* d_filterStrainTensorCompLabel;
      const VarLabel* d_scalarNumeratorLabel; 
      const VarLabel* d_scalarDenominatorLabel; 
      const VarLabel* d_enthalpyNumeratorLabel; 
      const VarLabel* d_enthalpyDenominatorLabel; 
      const VarLabel* d_reactScalarNumeratorLabel; 
      const VarLabel* d_reactScalarDenominatorLabel; 
      const VarLabel* d_ShFLabel;
      const VarLabel* d_ShELabel;
      const VarLabel* d_ShRFLabel;
      const VarLabel* d_CO2FlowRateLabel;
      const VarLabel* d_SO2FlowRateLabel;
      const VarLabel* d_carbonEfficiencyLabel;
      const VarLabel* d_sulfurEfficiencyLabel;
      const VarLabel* d_scalarFlowRateLabel;
      const VarLabel* d_scalarEfficiencyLabel;
      const VarLabel* d_enthalpyFlowRateLabel;
      const VarLabel* d_enthalpyEfficiencyLabel;
      const VarLabel* d_totalRadSrcLabel;
      const VarLabel* d_normTotalRadSrcLabel;

      //mms force term labels
      const VarLabel* d_uFmmsLabel;
      const VarLabel* d_vFmmsLabel;
      const VarLabel* d_wFmmsLabel;
      
      //Helper variable
      const VarLabel* d_zerosrcVarLabel;

      //rate Labels
      const VarLabel* d_co2RateLabel;
      const VarLabel* d_so2RateLabel;

      //source term labels for intrusion (non-zero) boundary conditions
      const VarLabel* d_scalarBoundarySrcLabel;
      const VarLabel* d_enthalpyBoundarySrcLabel;
      const VarLabel* d_umomBoundarySrcLabel;
      const VarLabel* d_vmomBoundarySrcLabel;
      const VarLabel* d_wmomBoundarySrcLabel;
      
      const VarLabel* d_areaFractionLabel;  ///< Cell area fraction
      const VarLabel* d_volFractionLabel;   ///< Cell volume fraction

      std::vector<std::string> model_req_species; ///< Vector containing all species required by models

      /** @brief    Add species to the list of species required by models */
      inline void add_species( std::string s ) { 
        model_req_species.push_back( s ); };

      /** @brief    Get the list of species required by models */
      inline std::vector<std::string> get_species( ) { 
        return model_req_species; 
      }; 

    }; // End class ArchesLabel
} // End namespace Uintah


#endif

