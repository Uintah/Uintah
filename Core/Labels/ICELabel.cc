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


#include <Core/Labels/ICELabel.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>

using namespace Uintah;

ICELabel::ICELabel()
{
   // shortcuts
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  const TypeDescription* CC_Vector = CCVariable<Vector>::getTypeDescription();
  
  const TypeDescription* SFCX_double = SFCXVariable<double>::getTypeDescription();
  const TypeDescription* SFCY_double = SFCYVariable<double>::getTypeDescription();
  const TypeDescription* SFCZ_double = SFCZVariable<double>::getTypeDescription();
  
  const TypeDescription* SFCX_Vector = SFCXVariable<Vector>::getTypeDescription();
  const TypeDescription* SFCY_Vector = SFCYVariable<Vector>::getTypeDescription();
  const TypeDescription* SFCZ_Vector = SFCZVariable<Vector>::getTypeDescription();
  
  const TypeDescription* sum_variable = sum_vartype::getTypeDescription();
  const TypeDescription* max_variable = max_vartype::getTypeDescription();

  delTLabel = VarLabel::create( "delT", delt_vartype::getTypeDescription() );

  NeedAddIceMaterialLabel
            = VarLabel::create("NeedAddIceMaterial", sum_variable);
    
  
  //__________________________________
  // Cell Centered variables
  TMV_CCLabel             = VarLabel::create("TMV_CC",        CC_double);
  press_CCLabel           = VarLabel::create("press_CC",      CC_double);
  press_equil_CCLabel     = VarLabel::create("press_equil_CC",CC_double);
  delP_DilatateLabel      = VarLabel::create("delP_Dilatate", CC_double);
  delP_MassXLabel         = VarLabel::create("delP_MassX",    CC_double); 
  sum_rho_CCLabel         = VarLabel::create("sum_rho_CC",    CC_double);
  compressibilityLabel    = VarLabel::create("compressiblity",CC_double);
  sumKappaLabel           = VarLabel::create("sumKappa",      CC_double);
  rho_CCLabel             = VarLabel::create("rho_CC",        CC_double);
  temp_CCLabel            = VarLabel::create("temp_CC",       CC_double);
  vel_CCLabel             = VarLabel::create("vel_CC",        CC_Vector);
  rho_micro_CCLabel       = VarLabel::create("rho_micro_CC",  CC_double);
  sp_vol_CCLabel          = VarLabel::create("sp_vol_CC",     CC_double);
  DLabel                  = VarLabel::create("D",             CC_Vector); 
  speedSound_CCLabel      = VarLabel::create("speedSound_CC", CC_double);
  vol_frac_CCLabel        = VarLabel::create("vol_frac_CC",   CC_double);
  mom_source_CCLabel      = VarLabel::create("mom_source_CC", CC_Vector);
  int_eng_source_CCLabel  = VarLabel::create("intE_source_CC",CC_double);
  heatCond_src_CCLabel    = VarLabel::create("heatCond_src_CC",CC_double);
  sp_vol_src_CCLabel      = VarLabel::create("sp_vol_src_CC", CC_double);
  mom_L_CCLabel           = VarLabel::create("mom_L_CC",      CC_Vector);
  int_eng_L_CCLabel       = VarLabel::create("int_eng_L_CC",  CC_double);
  sp_vol_L_CCLabel        = VarLabel::create("sp_vol_L_CC",   CC_double);
  mass_L_CCLabel          = VarLabel::create("mass_L_CC",     CC_double);
  mom_L_ME_CCLabel        = VarLabel::create("mom_L_ME_CC",   CC_Vector);
  eng_L_ME_CCLabel        = VarLabel::create("eng_L_ME_CC",   CC_double);
  mass_advLabel           = VarLabel::create("mass_adv",      CC_double);
  mom_advLabel            = VarLabel::create("mom_adv",       CC_Vector);
  eng_advLabel            = VarLabel::create("eng_adv",       CC_double);
  sp_vol_advLabel         = VarLabel::create("sp_vol_adv",    CC_double);

  term2Label              = VarLabel::create("term2",         CC_double);
  term3Label              = VarLabel::create("term3",         CC_double);
  f_theta_CCLabel         = VarLabel::create("f_theta",       CC_double);
  Tdot_CCLabel            = VarLabel::create("Tdot",          CC_double);
  turb_viscosity_CCLabel  = VarLabel::create("turb_viscosity_CC",CC_double);
  viscosityLabel          = VarLabel::create("viscosity",     CC_double);
  thermalCondLabel        = VarLabel::create("thermalCond",   CC_double);
  gammaLabel              = VarLabel::create("gamma",         CC_double);
  specific_heatLabel      = VarLabel::create("specific_heat", CC_double);
  temp_CC_XchangeLabel    = VarLabel::create("temp_CC_Xchange",CC_double);
  vel_CC_XchangeLabel     = VarLabel::create("vel_CC_Xchange",CC_Vector);
  dTdt_CCLabel            = VarLabel::create("dTdt_CC",       CC_double);
  dVdt_CCLabel            = VarLabel::create("dVdt_CC",       CC_Vector);
 
  //__________________________________
  // Implicit Labels
  matrixLabel = 
    VarLabel::create("matrix",CCVariable<Stencil7>::getTypeDescription());      
  rhsLabel                = VarLabel::create("rhs",         CC_double);        
  initialGuessLabel       = VarLabel::create("initialGuess",CC_double);     
  imp_delPLabel           = VarLabel::create("imp_delP",    CC_double);       
  sum_imp_delPLabel       = VarLabel::create("sum_imp_delP",CC_double);       
  betaLabel               = VarLabel::create("beta",        CC_double);
  sp_volX_FCLabel         = VarLabel::create("sp_volX_FC",  SFCX_double);
  sp_volY_FCLabel         = VarLabel::create("sp_volY_FC",  SFCY_double);
  sp_volZ_FCLabel         = VarLabel::create("sp_volZ_FC",  SFCZ_double);
  vol_fracX_FCLabel       = VarLabel::create("vol_fracX_FC",SFCX_double);  
  vol_fracY_FCLabel       = VarLabel::create("vol_fracY_FC",SFCY_double);  
  vol_fracZ_FCLabel       = VarLabel::create("vol_fracZ_FC",SFCZ_double);  
    
  //__________________________________
  // Face centered variables
  uvel_FCLabel       = VarLabel::create("uvel_FC",    SFCX_double);
  vvel_FCLabel       = VarLabel::create("vvel_FC",    SFCY_double);
  wvel_FCLabel       = VarLabel::create("wvel_FC",    SFCZ_double);
  uvel_FCMELabel     = VarLabel::create("uvel_FCME",  SFCX_double);
  vvel_FCMELabel     = VarLabel::create("vvel_FCME",  SFCY_double);
  wvel_FCMELabel     = VarLabel::create("wvel_FCME",  SFCZ_double);
  pressX_FCLabel     = VarLabel::create("pressX_FC",  SFCX_double);
  pressY_FCLabel     = VarLabel::create("pressY_FC",  SFCY_double);
  pressZ_FCLabel     = VarLabel::create("pressZ_FC",  SFCZ_double);
  TempX_FCLabel      = VarLabel::create("TempX_FC",   SFCX_double);
  TempY_FCLabel      = VarLabel::create("TempY_FC",   SFCY_double);
  TempZ_FCLabel      = VarLabel::create("TempZ_FC",   SFCZ_double);
  grad_P_XFCLabel    = VarLabel::create("grad_P_XFC", SFCX_double);
  grad_P_YFCLabel    = VarLabel::create("grad_P_YFC", SFCY_double);
  grad_P_ZFCLabel    = VarLabel::create("grad_P_ZFC", SFCZ_double);
  grad_dp_XFCLabel   = VarLabel::create("grad_dp_XFC",SFCX_double);
  grad_dp_YFCLabel   = VarLabel::create("grad_dp_YFC",SFCY_double);
  grad_dp_ZFCLabel   = VarLabel::create("grad_dp_ZFC",SFCZ_double);
      
  //__________________________________  
  // Misc labels
  machLabel           = VarLabel::create("mach",       CC_double); 
  scratchLabel        = VarLabel::create("scratch",    CC_double);
  scratchVecLabel     = VarLabel::create("scratchVec", CC_Vector);
  scratch_FCXLabel    = VarLabel::create("scratch_FCX",SFCX_double);
  scratch_FCYLabel    = VarLabel::create("scratch_FCY",SFCY_double);
  scratch_FCZLabel    = VarLabel::create("scratch_FCZ",SFCZ_double);
  IveBeenHereLabel     =
    VarLabel::create("IveBeenHere",CCVariable<int>::getTypeDescription());
    
  
  //__________________________________
  // LODI Boundary Conditions
  LODI_BC_Li1Label    = VarLabel::create("Li1",   CC_Vector);
  LODI_BC_Li2Label    = VarLabel::create("Li2",   CC_Vector);
  LODI_BC_Li3Label    = VarLabel::create("Li3",   CC_Vector);
  LODI_BC_Li4Label    = VarLabel::create("Li4",   CC_Vector);
  LODI_BC_Li5Label    = VarLabel::create("Li5",   CC_Vector);    
     
 //__________________________________
 //Reduction labels (The names must be identical to those in MPMLabel.cc)
  KineticEnergyLabel  = VarLabel::create( "KineticEnergy",    sum_variable);
  TotalMassLabel      = VarLabel::create( "TotalMass",        sum_variable);  
  TotalIntEngLabel    = VarLabel::create( "TotalIntEng",      sum_variable);
  eng_exch_errorLabel = VarLabel::create( "eng_exch_error",   sum_variable);
  max_RHSLabel        = VarLabel::create( "max_RHS",          max_variable);

  maxMach_xminusLabel = VarLabel::create( "maxMach_xminus",   max_variable);
  maxMach_xplusLabel  = VarLabel::create( "maxMach_xplus",    max_variable);
  maxMach_yminusLabel =  VarLabel::create( "maxMach_yminus",  max_variable);
  maxMach_yplusLabel  = VarLabel::create( "maxMach_yplus",    max_variable);
  maxMach_zminusLabel = VarLabel::create( "maxMach_zminus",   max_variable);  
  maxMach_zplusLabel  = VarLabel::create( "maxMach_zplus",    max_variable); 

  TotalMomentumLabel = 
    VarLabel::create( "TotalMomentum",
                                      sumvec_vartype::getTypeDescription());
  mom_exch_errorLabel = 
    VarLabel::create( "mom_exch_error",
                                      sumvec_vartype::getTypeDescription());
  //__________________________________
  // Model variables
  modelMass_srcLabel  = VarLabel::create( "modelMass_src",  CC_double);
  modelMom_srcLabel   = VarLabel::create( "modelMom_src",   CC_Vector);
  modelEng_srcLabel   = VarLabel::create( "modelEng_src",   CC_double);
  modelVol_srcLabel   = VarLabel::create( "modelVol_src",   CC_double);
  
  //__________________________________
  // AMR variables
   AMR_SyncTaskgraphLabel = 
    VarLabel::create("AMR_SyncTaskgraph",CCVariable<int>::getTypeDescription()); 
  
  // magnitude of the gradient of q_CC
  mag_grad_rho_CCLabel     = VarLabel::create("mag_grad_rho_CC",     CC_double);
  mag_grad_temp_CCLabel    = VarLabel::create("mag_grad_temp_CC",    CC_double);
  mag_div_vel_CCLabel      = VarLabel::create("mag_div_vel_CC",      CC_double);
  mag_grad_vol_frac_CCLabel= VarLabel::create("mag_grad_vol_frac_CC",CC_double);
  mag_grad_press_CCLabel   = VarLabel::create("mag_grad_press_CC",   CC_double);
    
  // refluxing variables fluxes on each face
  mass_X_FC_fluxLabel      = VarLabel::create("mass_X_FC_flux",    SFCX_double);
  mass_Y_FC_fluxLabel      = VarLabel::create("mass_Y_FC_flux",    SFCY_double);
  mass_Z_FC_fluxLabel      = VarLabel::create("mass_Z_FC_flux",    SFCZ_double);
    
  mom_X_FC_fluxLabel       = VarLabel::create("mom_X_FC_flux",     SFCX_Vector); 
  mom_Y_FC_fluxLabel       = VarLabel::create("mom_Y_FC_flux",     SFCY_Vector); 
  mom_Z_FC_fluxLabel       = VarLabel::create("mom_Z_FC_flux",     SFCZ_Vector); 
    
  sp_vol_X_FC_fluxLabel    = VarLabel::create("sp_vol_X_FC_flux",  SFCX_double);
  sp_vol_Y_FC_fluxLabel    = VarLabel::create("sp_vol_Y_FC_flux",  SFCY_double);
  sp_vol_Z_FC_fluxLabel    = VarLabel::create("sp_vol_Z_FC_flux",  SFCZ_double);
    
  int_eng_X_FC_fluxLabel   = VarLabel::create("int_eng_X_FC_flux", SFCX_double);
  int_eng_Y_FC_fluxLabel   = VarLabel::create("int_eng_Y_FC_flux", SFCY_double);
  int_eng_Z_FC_fluxLabel   = VarLabel::create("int_eng_Z_FC_flux", SFCZ_double);
    
  vol_frac_X_FC_fluxLabel  = VarLabel::create("vol_frac_X_FC_flux",SFCX_double);
  vol_frac_Y_FC_fluxLabel  = VarLabel::create("vol_frac_Y_FC_flux",SFCY_double);
  vol_frac_Z_FC_fluxLabel  = VarLabel::create("vol_frac_Z_FC_flux",SFCZ_double);
  
  // correction computed by refluxing
  mass_X_FC_corrLabel      = VarLabel::create("mass_X_FC_corr",    SFCX_double);
  mass_Y_FC_corrLabel      = VarLabel::create("mass_Y_FC_corr",    SFCY_double);
  mass_Z_FC_corrLabel      = VarLabel::create("mass_Z_FC_corr",    SFCZ_double);
    
  mom_X_FC_corrLabel       = VarLabel::create("mom_X_FC_corr",     SFCX_Vector); 
  mom_Y_FC_corrLabel       = VarLabel::create("mom_Y_FC_corr",     SFCY_Vector); 
  mom_Z_FC_corrLabel       = VarLabel::create("mom_Z_FC_corr",     SFCZ_Vector); 
    
  sp_vol_X_FC_corrLabel    = VarLabel::create("sp_vol_X_FC_corr",  SFCX_double);
  sp_vol_Y_FC_corrLabel    = VarLabel::create("sp_vol_Y_FC_corr",  SFCY_double);
  sp_vol_Z_FC_corrLabel    = VarLabel::create("sp_vol_Z_FC_corr",  SFCZ_double);
    
  int_eng_X_FC_corrLabel   = VarLabel::create("int_eng_X_FC_corr", SFCX_double);
  int_eng_Y_FC_corrLabel   = VarLabel::create("int_eng_Y_FC_corr", SFCY_double);
  int_eng_Z_FC_corrLabel   = VarLabel::create("int_eng_Z_FC_corr", SFCZ_double);
    
  vol_frac_X_FC_corrLabel  = VarLabel::create("vol_frac_X_FC_corr",SFCX_double);
  vol_frac_Y_FC_corrLabel  = VarLabel::create("vol_frac_Y_FC_corr",SFCY_double);
  vol_frac_Z_FC_corrLabel  = VarLabel::create("vol_frac_Z_FC_corr",SFCZ_double);

  //__________________________________
  // Implicit AMR variable
  matrix_CFI_weightsLabel  =VarLabel::create("matrix_CFI_weights", CC_double );
}

ICELabel::~ICELabel()
{
    // Cell centered variables
    VarLabel::destroy(delTLabel);
    VarLabel::destroy(press_CCLabel);
    VarLabel::destroy(TMV_CCLabel);
    VarLabel::destroy(press_equil_CCLabel);
    VarLabel::destroy(delP_DilatateLabel);
    VarLabel::destroy(delP_MassXLabel);
    VarLabel::destroy(rho_CCLabel);
    VarLabel::destroy(sum_rho_CCLabel);
    VarLabel::destroy(compressibilityLabel);
    VarLabel::destroy(sumKappaLabel);    
    VarLabel::destroy(temp_CCLabel);
    VarLabel::destroy(temp_CC_XchangeLabel);
    VarLabel::destroy(vel_CCLabel);
    VarLabel::destroy(vel_CC_XchangeLabel);
    VarLabel::destroy(rho_micro_CCLabel);
    VarLabel::destroy(sp_vol_CCLabel);
    VarLabel::destroy(DLabel);
    VarLabel::destroy(speedSound_CCLabel);
    VarLabel::destroy(vol_frac_CCLabel);
    VarLabel::destroy(mom_source_CCLabel);
    VarLabel::destroy(int_eng_source_CCLabel);
    VarLabel::destroy(heatCond_src_CCLabel);
    VarLabel::destroy(sp_vol_src_CCLabel);
    VarLabel::destroy(mom_L_CCLabel);
    VarLabel::destroy(int_eng_L_CCLabel);
    VarLabel::destroy(sp_vol_L_CCLabel);
    VarLabel::destroy(mass_L_CCLabel);
    VarLabel::destroy(mom_L_ME_CCLabel);
    VarLabel::destroy(eng_L_ME_CCLabel);
    VarLabel::destroy(mass_advLabel);
    VarLabel::destroy(mom_advLabel);
    VarLabel::destroy(eng_advLabel);
    VarLabel::destroy(sp_vol_advLabel);    
    
    VarLabel::destroy(term2Label);
    VarLabel::destroy(term3Label);
    VarLabel::destroy(f_theta_CCLabel);
    VarLabel::destroy(Tdot_CCLabel);
    VarLabel::destroy(turb_viscosity_CCLabel);
    VarLabel::destroy(viscosityLabel);
    VarLabel::destroy(thermalCondLabel);
    VarLabel::destroy(gammaLabel);
    VarLabel::destroy(specific_heatLabel);      
    VarLabel::destroy(dTdt_CCLabel);
    VarLabel::destroy(dVdt_CCLabel);
    
    // Implicit Labels
    VarLabel::destroy(matrixLabel);
    VarLabel::destroy(rhsLabel); 
    VarLabel::destroy(initialGuessLabel);
    VarLabel::destroy(betaLabel);
    VarLabel::destroy(imp_delPLabel);  
    VarLabel::destroy(sum_imp_delPLabel);  
    VarLabel::destroy(sp_volX_FCLabel); 
    VarLabel::destroy(sp_volY_FCLabel);
    VarLabel::destroy(sp_volZ_FCLabel);
    VarLabel::destroy(vol_fracX_FCLabel); 
    VarLabel::destroy(vol_fracY_FCLabel);
    VarLabel::destroy(vol_fracZ_FCLabel); 
    
    // Face centered variables
    VarLabel::destroy(uvel_FCLabel);
    VarLabel::destroy(vvel_FCLabel);
    VarLabel::destroy(wvel_FCLabel);
    VarLabel::destroy(uvel_FCMELabel);
    VarLabel::destroy(vvel_FCMELabel);
    VarLabel::destroy(wvel_FCMELabel);
    VarLabel::destroy(pressX_FCLabel);
    VarLabel::destroy(pressY_FCLabel);
    VarLabel::destroy(pressZ_FCLabel);
    VarLabel::destroy(TempX_FCLabel);
    VarLabel::destroy(TempY_FCLabel);
    VarLabel::destroy(TempZ_FCLabel);
    VarLabel::destroy(grad_P_XFCLabel);
    VarLabel::destroy(grad_P_YFCLabel);
    VarLabel::destroy(grad_P_ZFCLabel);
    VarLabel::destroy(grad_dp_XFCLabel);
    VarLabel::destroy(grad_dp_YFCLabel);
    VarLabel::destroy(grad_dp_ZFCLabel);
    // Misc labels
    VarLabel::destroy(IveBeenHereLabel);
    VarLabel::destroy(machLabel);
    VarLabel::destroy(scratchLabel);
    VarLabel::destroy(scratchVecLabel);
    VarLabel::destroy(scratch_FCXLabel);
    VarLabel::destroy(scratch_FCYLabel);
    VarLabel::destroy(scratch_FCZLabel);
    
    // LODI Boundary condition variables
    VarLabel::destroy(LODI_BC_Li1Label);
    VarLabel::destroy(LODI_BC_Li2Label);
    VarLabel::destroy(LODI_BC_Li3Label);
    VarLabel::destroy(LODI_BC_Li4Label);
    VarLabel::destroy(LODI_BC_Li5Label);    
    
    // Reduction Variables
    VarLabel::destroy(TotalMassLabel);
    VarLabel::destroy(KineticEnergyLabel);
    VarLabel::destroy(TotalMomentumLabel);
    VarLabel::destroy(TotalIntEngLabel); 
    VarLabel::destroy(eng_exch_errorLabel);   
    VarLabel::destroy(mom_exch_errorLabel);  
    VarLabel::destroy(max_RHSLabel);
    
    //   --- max Mach Number
    VarLabel::destroy(maxMach_xminusLabel);
    VarLabel::destroy(maxMach_xplusLabel);
    VarLabel::destroy(maxMach_yminusLabel);
    VarLabel::destroy(maxMach_yplusLabel);
    VarLabel::destroy(maxMach_zminusLabel);
    VarLabel::destroy(maxMach_zplusLabel);
    
    VarLabel::destroy(NeedAddIceMaterialLabel);

    // Model variables
    VarLabel::destroy(modelMass_srcLabel);
    VarLabel::destroy(modelMom_srcLabel);
    VarLabel::destroy(modelEng_srcLabel);
    VarLabel::destroy(modelVol_srcLabel);
    
    // AMR variables
    VarLabel::destroy(AMR_SyncTaskgraphLabel);
    // magnitude of the gradient of ()
    VarLabel::destroy(mag_grad_rho_CCLabel);
    VarLabel::destroy(mag_grad_temp_CCLabel);
    VarLabel::destroy(mag_div_vel_CCLabel);
    VarLabel::destroy(mag_grad_vol_frac_CCLabel);
    VarLabel::destroy(mag_grad_press_CCLabel);
    
    // refluxing variables
    VarLabel::destroy(mass_X_FC_fluxLabel);
    VarLabel::destroy(mass_Y_FC_fluxLabel);
    VarLabel::destroy(mass_Z_FC_fluxLabel);
    
    VarLabel::destroy(mom_X_FC_fluxLabel);
    VarLabel::destroy(mom_Y_FC_fluxLabel);
    VarLabel::destroy(mom_Z_FC_fluxLabel);
    
    VarLabel::destroy(sp_vol_X_FC_fluxLabel);
    VarLabel::destroy(sp_vol_Y_FC_fluxLabel);
    VarLabel::destroy(sp_vol_Z_FC_fluxLabel);
    
    VarLabel::destroy(int_eng_X_FC_fluxLabel);
    VarLabel::destroy(int_eng_Y_FC_fluxLabel);
    VarLabel::destroy(int_eng_Z_FC_fluxLabel);
    
    VarLabel::destroy(vol_frac_X_FC_fluxLabel);
    VarLabel::destroy(vol_frac_Y_FC_fluxLabel);
    VarLabel::destroy(vol_frac_Z_FC_fluxLabel);
    
    // corrections computed by refluxing variables
    VarLabel::destroy(mass_X_FC_corrLabel);
    VarLabel::destroy(mass_Y_FC_corrLabel);
    VarLabel::destroy(mass_Z_FC_corrLabel);
    
    VarLabel::destroy(mom_X_FC_corrLabel);
    VarLabel::destroy(mom_Y_FC_corrLabel);
    VarLabel::destroy(mom_Z_FC_corrLabel);
    
    VarLabel::destroy(sp_vol_X_FC_corrLabel);
    VarLabel::destroy(sp_vol_Y_FC_corrLabel);
    VarLabel::destroy(sp_vol_Z_FC_corrLabel);
    
    VarLabel::destroy(int_eng_X_FC_corrLabel);
    VarLabel::destroy(int_eng_Y_FC_corrLabel);
    VarLabel::destroy(int_eng_Z_FC_corrLabel);
    
    VarLabel::destroy(vol_frac_X_FC_corrLabel);
    VarLabel::destroy(vol_frac_Y_FC_corrLabel);
    VarLabel::destroy(vol_frac_Z_FC_corrLabel);

    // Implicit AMR labels
    VarLabel::destroy(matrix_CFI_weightsLabel);
}
