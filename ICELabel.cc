#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

using namespace Uintah;

ICELabel::ICELabel()
{
  delTLabel = 0; // Placed in later, in problemSetup
  doMechLabel
    = VarLabel::create("doMech",    delt_vartype::getTypeDescription());
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
  spec_vol_source_CCLabel = 
   VarLabel::create("spVol_source_CC",CCVariable<double>::getTypeDescription());
  mom_L_CCLabel = 
    VarLabel::create("mom_L_CC",     CCVariable<Vector>::getTypeDescription());
  int_eng_L_CCLabel = 
    VarLabel::create("int_eng_L_CC", CCVariable<double>::getTypeDescription());
  spec_vol_L_CCLabel = 
    VarLabel::create("spec_vol_L_CC",CCVariable<double>::getTypeDescription());
  mass_L_CCLabel = 
    VarLabel::create("mass_L_CC",    CCVariable<double>::getTypeDescription());
  mom_L_ME_CCLabel = 
    VarLabel::create("mom_L_ME_CC",  CCVariable<Vector>::getTypeDescription());
  eng_L_ME_CCLabel = 
   VarLabel::create("eng_L_ME_CC",   CCVariable<double>::getTypeDescription());
  burnedMass_CCLabel =
    VarLabel::create("burnedMass",   CCVariable<double>::getTypeDescription());
  int_eng_comb_CCLabel =
    VarLabel::create("int_eng_comb", CCVariable<double>::getTypeDescription());
  mom_comb_CCLabel =
    VarLabel::create("mom_comb_CC",  CCVariable<Vector>::getTypeDescription());
  created_vol_CCLabel =
    VarLabel::create("created_vol",  CCVariable<double>::getTypeDescription());
  term2Label = 
    VarLabel::create("term2",        CCVariable<double>::getTypeDescription());
  term3Label = 
    VarLabel::create("term3",        CCVariable<double>::getTypeDescription());
  f_theta_CCLabel =
    VarLabel::create("f_theta",      CCVariable<double>::getTypeDescription());
  Tdot_CCLabel =
    VarLabel::create("Tdot",         CCVariable<double>::getTypeDescription());
 
  // Implicit Labels
  matrixLabel = 
    VarLabel::create("matrix",      CCVariable<Stencil7>::getTypeDescription());      
  rhsLabel = 
    VarLabel::create("rhs",         CCVariable<double>::getTypeDescription());        
  initialGuessLabel = 
    VarLabel::create("initialGuess",CCVariable<double>::getTypeDescription());     
  imp_delPLabel = 
    VarLabel::create("imp_delP",    CCVariable<double>::getTypeDescription());       
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
  press_diffX_FCLabel = VarLabel::create("press_diffX_FC",
                                  SFCXVariable<double>::getTypeDescription() );
  press_diffY_FCLabel = VarLabel::create("press_diffY_FC",
                                  SFCYVariable<double>::getTypeDescription() );
  press_diffZ_FCLabel = VarLabel::create("press_diffZ_FC",
                                  SFCZVariable<double>::getTypeDescription() );  
    
    // Misc labels
  scratchLabel     =
    VarLabel::create("scratch",  CCVariable<double>::getTypeDescription() );
  scratch_FCXLabel   =
    VarLabel::create("scratch_FCX",SFCXVariable<double>::getTypeDescription());
  scratch_FCYLabel   =
    VarLabel::create("scratch_FCY",SFCYVariable<double>::getTypeDescription());
  scratch_FCZLabel   =
    VarLabel::create("scratch_FCZ",SFCZVariable<double>::getTypeDescription());
  scratch_FCVectorLabel   =
    VarLabel::create("scratch_FCVector",
                                   SFCXVariable<Vector>::getTypeDescription());
  IveBeenHereLabel     =
    VarLabel::create("IveBeenHere",CCVariable<int>::getTypeDescription() );
     
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
  CenterOfMassVelocityLabel = 
    VarLabel::create( "CenterOfMassVelocity",
                                      sumvec_vartype::getTypeDescription() );
  mom_exch_errorLabel = 
    VarLabel::create( "mom_exch_error",
                                      sumvec_vartype::getTypeDescription() );
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
} 

ICELabel::~ICELabel()
{
    // Cell centered variables
    VarLabel::destroy(press_CCLabel);
    VarLabel::destroy(press_equil_CCLabel);
    VarLabel::destroy(matl_press_CCLabel);
    VarLabel::destroy(delP_DilatateLabel);
    VarLabel::destroy(delP_MassXLabel);
    VarLabel::destroy(rho_CCLabel);
    VarLabel::destroy(sum_rho_CCLabel);
    VarLabel::destroy(temp_CCLabel);
    VarLabel::destroy(vel_CCLabel);
    VarLabel::destroy(rho_micro_CCLabel);
    VarLabel::destroy(sp_vol_CCLabel);
    VarLabel::destroy(DLabel);
    VarLabel::destroy(burnedMass_CCLabel);
    VarLabel::destroy(int_eng_comb_CCLabel);
    VarLabel::destroy(created_vol_CCLabel);
    VarLabel::destroy(mom_comb_CCLabel);
    VarLabel::destroy(speedSound_CCLabel);
    VarLabel::destroy(vol_frac_CCLabel);
    VarLabel::destroy(press_force_CCLabel); 
    VarLabel::destroy(mom_source_CCLabel);
    VarLabel::destroy(int_eng_source_CCLabel);
    VarLabel::destroy(spec_vol_source_CCLabel);
    VarLabel::destroy(mom_L_CCLabel);
    VarLabel::destroy(int_eng_L_CCLabel);
    VarLabel::destroy(spec_vol_L_CCLabel);
    VarLabel::destroy(mass_L_CCLabel);
    VarLabel::destroy(mom_L_ME_CCLabel);
    VarLabel::destroy(eng_L_ME_CCLabel);
    VarLabel::destroy(term2Label);
    VarLabel::destroy(term3Label);
    VarLabel::destroy(f_theta_CCLabel);
    VarLabel::destroy(Tdot_CCLabel);
    
    // Implicit Labels
    VarLabel::destroy(matrixLabel);
    VarLabel::destroy(rhsLabel); 
    VarLabel::destroy(initialGuessLabel);
    VarLabel::destroy(betaLabel);
    VarLabel::destroy(imp_delPLabel);  
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
    VarLabel::destroy(press_diffX_FCLabel);
    VarLabel::destroy(press_diffY_FCLabel);
    VarLabel::destroy(press_diffZ_FCLabel);
    // Misc labels
    VarLabel::destroy(IveBeenHereLabel);
    VarLabel::destroy(scratchLabel);
    VarLabel::destroy(scratch_FCVectorLabel);
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
    VarLabel::destroy(doMechLabel);

    // Model variables
    VarLabel::destroy(modelMass_srcLabel);
    VarLabel::destroy(modelMom_srcLabel);
    VarLabel::destroy(modelEng_srcLabel);
    VarLabel::destroy(modelVol_srcLabel);
}
