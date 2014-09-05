
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Grid/Variables/Stencil7.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>

using namespace Uintah;

ICELabel::ICELabel()
{
  delTLabel = VarLabel::create( "delT", delt_vartype::getTypeDescription() );
  doMechLabel
    = VarLabel::create("doMech",    delt_vartype::getTypeDescription());
  NeedAddIceMaterialLabel
    = VarLabel::create("NeedAddIceMaterial", sum_vartype::getTypeDescription());

  //__________________________________
  // Cell Centered variables
  TMV_CCLabel     =
    VarLabel::create("TMV_CC",    CCVariable<double>::getTypeDescription());
  press_CCLabel     =
    VarLabel::create("press_CC",    CCVariable<double>::getTypeDescription());
  matl_press_CCLabel     =
    VarLabel::create("matl_press_CC",CCVariable<double>::getTypeDescription());
  press_equil_CCLabel   =
    VarLabel::create("press_equil_CC",CCVariable<double>::getTypeDescription());
  delP_DilatateLabel  =
    VarLabel::create("delP_Dilatate",CCVariable<double>::getTypeDescription());
  delP_MassXLabel  =
    VarLabel::create("delP_MassX",   CCVariable<double>::getTypeDescription()); 
  sum_rho_CCLabel       = 
    VarLabel::create("sum_rho_CC",   CCVariable<double>::getTypeDescription());
  compressibilityLabel   = 
    VarLabel::create("compressiblity",CCVariable<double>::getTypeDescription());
  sumKappaLabel   = 
    VarLabel::create("sumKappa",     CCVariable<double>::getTypeDescription());
  rho_CCLabel       = 
    VarLabel::create("rho_CC",       CCVariable<double>::getTypeDescription());
  temp_CCLabel      = 
    VarLabel::create("temp_CC",      CCVariable<double>::getTypeDescription());
  vel_CCLabel       = 
    VarLabel::create("vel_CC",       CCVariable<Vector>::getTypeDescription());
  rho_micro_CCLabel = 
    VarLabel::create("rho_micro_CC", CCVariable<double>::getTypeDescription());
  sp_vol_CCLabel =
    VarLabel::create("sp_vol_CC",    CCVariable<double>::getTypeDescription());
  DLabel =
    VarLabel::create("D",            CCVariable<Vector>::getTypeDescription()); 
  speedSound_CCLabel =
    VarLabel::create("speedSound_CC",CCVariable<double>::getTypeDescription());
  vol_frac_CCLabel =
    VarLabel::create("vol_frac_CC",  CCVariable<double>::getTypeDescription());
  press_force_CCLabel =
    VarLabel::create("press_force",  CCVariable<Vector>::getTypeDescription());
  mom_source_CCLabel = 
    VarLabel::create("mom_source_CC",CCVariable<Vector>::getTypeDescription());
  int_eng_source_CCLabel = 
    VarLabel::create("intE_source_CC",CCVariable<double>::getTypeDescription());
  heatCond_src_CCLabel = 
    VarLabel::create("heatCond_src_CC",CCVariable<double>::getTypeDescription());
  sp_vol_src_CCLabel = 
   VarLabel::create("sp_vol_src_CC", CCVariable<double>::getTypeDescription());
  mom_L_CCLabel = 
    VarLabel::create("mom_L_CC",     CCVariable<Vector>::getTypeDescription());
  int_eng_L_CCLabel = 
    VarLabel::create("int_eng_L_CC", CCVariable<double>::getTypeDescription());
  sp_vol_L_CCLabel = 
    VarLabel::create("sp_vol_L_CC",  CCVariable<double>::getTypeDescription());
  mass_L_CCLabel = 
    VarLabel::create("mass_L_CC",    CCVariable<double>::getTypeDescription());
  mom_L_ME_CCLabel = 
    VarLabel::create("mom_L_ME_CC",  CCVariable<Vector>::getTypeDescription());
  eng_L_ME_CCLabel = 
   VarLabel::create("eng_L_ME_CC",   CCVariable<double>::getTypeDescription());
  mass_advLabel = 
    VarLabel::create("mass_adv",    CCVariable<double>::getTypeDescription());
  mom_advLabel = 
    VarLabel::create("mom_adv",     CCVariable<Vector>::getTypeDescription());
  eng_advLabel = 
   VarLabel::create("eng_adv",      CCVariable<double>::getTypeDescription());
  sp_vol_advLabel = 
    VarLabel::create("sp_vol_adv",  CCVariable<double>::getTypeDescription());

  term2Label = 
    VarLabel::create("term2",        CCVariable<double>::getTypeDescription());
  term3Label = 
    VarLabel::create("term3",        CCVariable<double>::getTypeDescription());
  f_theta_CCLabel =
    VarLabel::create("f_theta",      CCVariable<double>::getTypeDescription());
  Tdot_CCLabel =
    VarLabel::create("Tdot",         CCVariable<double>::getTypeDescription());
  turb_viscosity_CCLabel =
    VarLabel::create("turb_viscosity_CC",CCVariable<double>::getTypeDescription());
  viscosityLabel =
    VarLabel::create("viscosity",    CCVariable<double>::getTypeDescription());
  thermalCondLabel =
    VarLabel::create("thermalCond",  CCVariable<double>::getTypeDescription());
  gammaLabel =
    VarLabel::create("gamma",        CCVariable<double>::getTypeDescription());
  specific_heatLabel =
    VarLabel::create("specific_heat",CCVariable<double>::getTypeDescription());
  temp_CC_XchangeLabel  = 
    VarLabel::create("temp_CC_Xchange",CCVariable<double>::getTypeDescription());
  vel_CC_XchangeLabel = 
   VarLabel::create("vel_CC_Xchange", CCVariable<Vector>::getTypeDescription());
  dTdt_CCLabel =
    VarLabel::create("dTdt_CC",       CCVariable<double>::getTypeDescription());
  dVdt_CCLabel =
    VarLabel::create("dVdt_CC",       CCVariable<Vector>::getTypeDescription());
 
  //__________________________________
  // Implicit Labels
  matrixLabel = 
    VarLabel::create("matrix",      CCVariable<Stencil7>::getTypeDescription());      
  rhsLabel = 
    VarLabel::create("rhs",         CCVariable<double>::getTypeDescription());        
  initialGuessLabel = 
    VarLabel::create("initialGuess",CCVariable<double>::getTypeDescription());     
  imp_delPLabel = 
    VarLabel::create("imp_delP",    CCVariable<double>::getTypeDescription());       
  sum_imp_delPLabel = 
    VarLabel::create("sum_imp_delP",CCVariable<double>::getTypeDescription());       
  betaLabel = 
    VarLabel::create("beta",        CCVariable<double>::getTypeDescription());
  sp_volX_FCLabel  = 
    VarLabel::create("sp_volX_FC",  SFCXVariable<double>::getTypeDescription() );
  sp_volY_FCLabel  = 
    VarLabel::create("sp_volY_FC",  SFCYVariable<double>::getTypeDescription() );
  sp_volZ_FCLabel  = 
    VarLabel::create("sp_volZ_FC",  SFCZVariable<double>::getTypeDescription() );
  vol_fracX_FCLabel  = 
    VarLabel::create("vol_fracX_FC",SFCXVariable<double>::getTypeDescription() );  
  vol_fracY_FCLabel  = 
    VarLabel::create("vol_fracY_FC",SFCYVariable<double>::getTypeDescription() );  
  vol_fracZ_FCLabel  = 
    VarLabel::create("vol_fracZ_FC",SFCZVariable<double>::getTypeDescription() );  
    
  //__________________________________
  // Face centered variables
  uvel_FCLabel       = 
    VarLabel::create("uvel_FC",   SFCXVariable<double>::getTypeDescription() );
  vvel_FCLabel       = 
    VarLabel::create("vvel_FC",   SFCYVariable<double>::getTypeDescription() );
  wvel_FCLabel       = 
    VarLabel::create("wvel_FC",   SFCZVariable<double>::getTypeDescription() );
  uvel_FCMELabel       = 
    VarLabel::create("uvel_FCME", SFCXVariable<double>::getTypeDescription() );
  vvel_FCMELabel       = 
    VarLabel::create("vvel_FCME", SFCYVariable<double>::getTypeDescription() );
  wvel_FCMELabel       = 
    VarLabel::create("wvel_FCME", SFCZVariable<double>::getTypeDescription() );
  pressX_FCLabel     = 
    VarLabel::create("pressX_FC", SFCXVariable<double>::getTypeDescription() );
  pressY_FCLabel     =
    VarLabel::create("pressY_FC", SFCYVariable<double>::getTypeDescription() );
  pressZ_FCLabel     =
    VarLabel::create("pressZ_FC", SFCZVariable<double>::getTypeDescription() );
  TempX_FCLabel     = 
    VarLabel::create("TempX_FC",  SFCXVariable<double>::getTypeDescription() );
  TempY_FCLabel     =
    VarLabel::create("TempY_FC",  SFCYVariable<double>::getTypeDescription() );
  TempZ_FCLabel     = 
    VarLabel::create("TempZ_FC",  SFCZVariable<double>::getTypeDescription() );
  grad_P_XFCLabel   =
    VarLabel::create("grad_P_XFC",SFCXVariable<double>::getTypeDescription() );
  grad_P_YFCLabel   =
    VarLabel::create("grad_P_YFC",SFCYVariable<double>::getTypeDescription() );
  grad_P_ZFCLabel   =
    VarLabel::create("grad_P_ZFC",SFCZVariable<double>::getTypeDescription() );
  grad_dp_XFCLabel   =
    VarLabel::create("grad_dp_XFC",SFCXVariable<double>::getTypeDescription() );
  grad_dp_YFCLabel   =
    VarLabel::create("grad_dp_YFC",SFCYVariable<double>::getTypeDescription() );
  grad_dp_ZFCLabel   =
    VarLabel::create("grad_dp_ZFC",SFCZVariable<double>::getTypeDescription() );
    
  // these should are for rate form and should be removed.  
  press_diffX_FCLabel = VarLabel::create("press_diffX_FC",
                                  SFCXVariable<double>::getTypeDescription() );
  press_diffY_FCLabel = VarLabel::create("press_diffY_FC",
                                  SFCYVariable<double>::getTypeDescription() );
  press_diffZ_FCLabel = VarLabel::create("press_diffZ_FC",
                                  SFCZVariable<double>::getTypeDescription() );  
  //__________________________________  
  // Misc labels
  machLabel     =
    VarLabel::create("mach",       CCVariable<double>::getTypeDescription() ); 
  scratchLabel     =
    VarLabel::create("scratch",    CCVariable<double>::getTypeDescription() );
  scratchVecLabel     =
    VarLabel::create("scratchVec", CCVariable<Vector>::getTypeDescription() );
  scratch_FCXLabel   =
    VarLabel::create("scratch_FCX",SFCXVariable<double>::getTypeDescription());
  scratch_FCYLabel   =
    VarLabel::create("scratch_FCY",SFCYVariable<double>::getTypeDescription());
  scratch_FCZLabel   =
    VarLabel::create("scratch_FCZ",SFCZVariable<double>::getTypeDescription());
  IveBeenHereLabel     =
    VarLabel::create("IveBeenHere",CCVariable<int>::getTypeDescription() );
     
 //__________________________________
 //Reduction labels (The names must be identical to those in MPMLabel.cc)
  KineticEnergyLabel = 
    VarLabel::create( "KineticEnergy", sum_vartype::getTypeDescription() );
  TotalMassLabel = 
    VarLabel::create( "TotalMass",     sum_vartype::getTypeDescription() );  
  TotalIntEngLabel = 
    VarLabel::create( "TotalIntEng",   sum_vartype::getTypeDescription() );
  eng_exch_errorLabel = 
    VarLabel::create( "eng_exch_error",sum_vartype::getTypeDescription() );
  max_RHSLabel = 
    VarLabel::create( "max_RHS",       max_vartype::getTypeDescription() );

  maxMach_xminusLabel = 
    VarLabel::create( "maxMach_xminus",   max_vartype::getTypeDescription() );
  maxMach_xplusLabel = 
    VarLabel::create( "maxMach_xplus",    max_vartype::getTypeDescription() );
  maxMach_yminusLabel =  
    VarLabel::create( "maxMach_yminus",   max_vartype::getTypeDescription() );
  maxMach_yplusLabel = 
    VarLabel::create( "maxMach_yplus",    max_vartype::getTypeDescription() );
  maxMach_zminusLabel = 
    VarLabel::create( "maxMach_zminus",   max_vartype::getTypeDescription() );  
  maxMach_zplusLabel = 
    VarLabel::create( "maxMach_zplus",    max_vartype::getTypeDescription() ); 

  CenterOfMassVelocityLabel = 
    VarLabel::create( "CenterOfMassVelocity",
                                      sumvec_vartype::getTypeDescription() );
  mom_exch_errorLabel = 
    VarLabel::create( "mom_exch_error",
                                      sumvec_vartype::getTypeDescription() );
  //__________________________________
  // Model variables
  modelMass_srcLabel =
    VarLabel::create( "modelMass_src",
		      CCVariable<double>::getTypeDescription());
  modelMom_srcLabel =
    VarLabel::create( "modelMom_src",
		      CCVariable<Vector>::getTypeDescription());
  modelEng_srcLabel =
    VarLabel::create( "modelEng_src",
		      CCVariable<double>::getTypeDescription());
  modelVol_srcLabel =
    VarLabel::create( "modelVol_src",
		      CCVariable<double>::getTypeDescription());
  //__________________________________
  // AMR variables
   AMR_SyncTaskgraphLabel = 
    VarLabel::create("AMR_SyncTaskgraph",CCVariable<int>::getTypeDescription()); 
  
  // magnitude of the gradient of q_CC
  mag_grad_rho_CCLabel = 
    VarLabel::create("mag_grad_rho_CC",     CCVariable<double>::getTypeDescription());
  mag_grad_temp_CCLabel = 
    VarLabel::create("mag_grad_temp_CC",    CCVariable<double>::getTypeDescription());  
  mag_div_vel_CCLabel = 
    VarLabel::create("mag_div_vel_CC",      CCVariable<double>::getTypeDescription());  
  mag_grad_vol_frac_CCLabel = 
    VarLabel::create("mag_grad_vol_frac_CC",CCVariable<double>::getTypeDescription());
  mag_grad_press_CCLabel = 
    VarLabel::create("mag_grad_press_CC",   CCVariable<double>::getTypeDescription());
    
  // refluxing variables  
  mass_X_FC_fluxLabel = 
    VarLabel::create("mass_X_FC_flux",  SFCXVariable<double>::getTypeDescription());
  mass_Y_FC_fluxLabel = 
    VarLabel::create("mass_Y_FC_flux",  SFCYVariable<double>::getTypeDescription());
  mass_Z_FC_fluxLabel = 
    VarLabel::create("mass_Z_FC_flux",  SFCZVariable<double>::getTypeDescription());
    
  mom_X_FC_fluxLabel = 
    VarLabel::create("mom_X_FC_flux",   SFCXVariable<Vector>::getTypeDescription()); 
  mom_Y_FC_fluxLabel = 
    VarLabel::create("mom_Y_FC_flux",   SFCYVariable<Vector>::getTypeDescription()); 
  mom_Z_FC_fluxLabel = 
    VarLabel::create("mom_Z_FC_flux",   SFCZVariable<Vector>::getTypeDescription()); 
    
  sp_vol_X_FC_fluxLabel = 
    VarLabel::create("sp_vol_X_FC_flux", SFCXVariable<double>::getTypeDescription()); 
  sp_vol_Y_FC_fluxLabel = 
    VarLabel::create("sp_vol_Y_FC_flux", SFCYVariable<double>::getTypeDescription()); 
  sp_vol_Z_FC_fluxLabel = 
    VarLabel::create("sp_vol_Z_FC_flux", SFCZVariable<double>::getTypeDescription()); 
    
  int_eng_X_FC_fluxLabel = 
    VarLabel::create("int_eng_X_FC_flux",SFCXVariable<double>::getTypeDescription()); 
  int_eng_Y_FC_fluxLabel = 
    VarLabel::create("int_eng_Y_FC_flux",SFCYVariable<double>::getTypeDescription()); 
  int_eng_Z_FC_fluxLabel = 
    VarLabel::create("int_eng_Z_FC_flux",SFCZVariable<double>::getTypeDescription()); 
    
  vol_frac_X_FC_fluxLabel = 
    VarLabel::create("vol_frac_X_FC_flux",SFCXVariable<double>::getTypeDescription()); 
  vol_frac_Y_FC_fluxLabel = 
    VarLabel::create("vol_frac_Y_FC_flux",SFCYVariable<double>::getTypeDescription()); 
  vol_frac_Z_FC_fluxLabel = 
    VarLabel::create("vol_frac_Z_FC_flux",SFCZVariable<double>::getTypeDescription());

  //__________________________________
  // Implicit AMR variable
  matrix_CFI_weightsLabel     =
    VarLabel::create("matrix_CFI_weights", CCVariable<double>::getTypeDescription() );
}

ICELabel::~ICELabel()
{
    // Cell centered variables
    VarLabel::destroy(delTLabel);
    VarLabel::destroy(press_CCLabel);
    VarLabel::destroy(TMV_CCLabel);
    VarLabel::destroy(press_equil_CCLabel);
    VarLabel::destroy(matl_press_CCLabel);
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
    VarLabel::destroy(press_force_CCLabel); 
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
    VarLabel::destroy(press_diffX_FCLabel);
    VarLabel::destroy(press_diffY_FCLabel);
    VarLabel::destroy(press_diffZ_FCLabel);
    // Misc labels
    VarLabel::destroy(IveBeenHereLabel);
    VarLabel::destroy(machLabel);
    VarLabel::destroy(scratchLabel);
    VarLabel::destroy(scratchVecLabel);
    VarLabel::destroy(scratch_FCXLabel);
    VarLabel::destroy(scratch_FCYLabel);
    VarLabel::destroy(scratch_FCZLabel);

    // Reduction Variables
    VarLabel::destroy(TotalMassLabel);
    VarLabel::destroy(KineticEnergyLabel);
    VarLabel::destroy(CenterOfMassVelocityLabel);
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

    VarLabel::destroy(doMechLabel);
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

    // Implicit AMR labels
    VarLabel::destroy(matrix_CFI_weightsLabel);
}
