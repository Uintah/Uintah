#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

using namespace Uintah;

ICELabel::ICELabel()
{
  delTLabel
    = VarLabel::create("delT",      delt_vartype::getTypeDescription());
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
  rho_CCLabel       = 
    VarLabel::create("rho_CC",       CCVariable<double>::getTypeDescription());
  rho_CC_top_cycleLabel       = 
    VarLabel::create("rho_top_cycle",CCVariable<double>::getTypeDescription());
  temp_CCLabel      = 
    VarLabel::create("temp_CC",      CCVariable<double>::getTypeDescription());
  vel_CCLabel       = 
    VarLabel::create("vel_CC",       CCVariable<Vector>::getTypeDescription());
  cv_CCLabel        = 
    VarLabel::create("cv_CC",        CCVariable<double>::getTypeDescription());
  rho_micro_CCLabel = 
    VarLabel::create("rho_micro_CC", CCVariable<double>::getTypeDescription());
  sp_vol_CCLabel =
    VarLabel::create("sp_vol_CC",    CCVariable<double>::getTypeDescription());

  mass_CCLabel =
    VarLabel::create("mass_CC",      CCVariable<double>::getTypeDescription());
  speedSound_CCLabel =
    VarLabel::create("speedSound_CC",CCVariable<double>::getTypeDescription());
  div_velfc_CCLabel =
    VarLabel::create("div_velfc_CC", CCVariable<double>::getTypeDescription());
  vol_frac_CCLabel =
    VarLabel::create("vol_frac_CC",  CCVariable<double>::getTypeDescription());
  viscosity_CCLabel =
    VarLabel::create("viscosity_CC", CCVariable<double>::getTypeDescription());
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
  int_eng_L_ME_CCLabel = 
   VarLabel::create("int_eng_L_ME_CC",CCVariable<double>::getTypeDescription());
  q_CCLabel = 
    VarLabel::create("q_CC",         CCVariable<double>::getTypeDescription());
  q_advectedLabel = 
    VarLabel::create("q_advected",   CCVariable<double>::getTypeDescription());
  qV_CCLabel = 
    VarLabel::create("qV_CC",        CCVariable<Vector>::getTypeDescription());
  qV_advectedLabel = 
    VarLabel::create("qV_advected",  CCVariable<Vector>::getTypeDescription());
  burnedMass_CCLabel =
    VarLabel::create("burnedMass",   CCVariable<double>::getTypeDescription());
  releasedHeat_CCLabel =
    VarLabel::create("releasedHeat", CCVariable<double>::getTypeDescription());
  created_vol_CCLabel =
    VarLabel::create("created_vol",  CCVariable<double>::getTypeDescription());

  term1Label = 
    VarLabel::create("term1",        CCVariable<double>::getTypeDescription());
  term2Label = 
    VarLabel::create("term2",        CCVariable<double>::getTypeDescription());
  term3Label = 
    VarLabel::create("term3",        CCVariable<double>::getTypeDescription());

  f_theta_CCLabel =
    VarLabel::create("f_theta",      CCVariable<double>::getTypeDescription());
  Tdot_CCLabel =
    VarLabel::create("Tdot",         CCVariable<double>::getTypeDescription());
  SumThermExpLabel =
    VarLabel::create("SumThermExp",  CCVariable<double>::getTypeDescription());
  
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
  tau_X_FCLabel       =
    VarLabel::create("tau_X_FC",  SFCXVariable<Vector>::getTypeDescription() );
  tau_Y_FCLabel       =
    VarLabel::create("tau_Y_FC",  SFCYVariable<Vector>::getTypeDescription() );
  tau_Z_FCLabel       =
    VarLabel::create("tau_Z_FC",  SFCZVariable<Vector>::getTypeDescription() );
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
  CenterOfMassVelocityLabel = 
    VarLabel::create( "CenterOfMassVelocity",
                                      sumvec_vartype::getTypeDescription() );
  TotalMassLabel = 
    VarLabel::create( "TotalMass",     sum_vartype::getTypeDescription() );  
  TotalIntEngLabel = 
    VarLabel::create( "TotalIntEng",   sum_vartype::getTypeDescription() );  

       
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
    VarLabel::destroy(rho_CC_top_cycleLabel);
    VarLabel::destroy(temp_CCLabel);
    VarLabel::destroy(vel_CCLabel);
    VarLabel::destroy(cv_CCLabel);
    VarLabel::destroy(rho_micro_CCLabel);
    VarLabel::destroy(sp_vol_CCLabel);
    VarLabel::destroy(mass_CCLabel);
    VarLabel::destroy(burnedMass_CCLabel);
    VarLabel::destroy(releasedHeat_CCLabel);
    VarLabel::destroy(created_vol_CCLabel);
    VarLabel::destroy(speedSound_CCLabel);
    VarLabel::destroy(div_velfc_CCLabel);
    VarLabel::destroy(vol_frac_CCLabel);
    VarLabel::destroy(viscosity_CCLabel);
    VarLabel::destroy(mom_source_CCLabel);
    VarLabel::destroy(int_eng_source_CCLabel);
    VarLabel::destroy(spec_vol_source_CCLabel);
    VarLabel::destroy(mom_L_CCLabel);
    VarLabel::destroy(int_eng_L_CCLabel);
    VarLabel::destroy(spec_vol_L_CCLabel);
    VarLabel::destroy(mass_L_CCLabel);
    VarLabel::destroy(mom_L_ME_CCLabel);
    VarLabel::destroy(int_eng_L_ME_CCLabel);
    VarLabel::destroy(q_CCLabel);
    VarLabel::destroy(q_advectedLabel);
    VarLabel::destroy(qV_CCLabel);
    VarLabel::destroy(qV_advectedLabel);
    VarLabel::destroy(term1Label);
    VarLabel::destroy(term2Label);
    VarLabel::destroy(term3Label);
    VarLabel::destroy(f_theta_CCLabel);
    VarLabel::destroy(Tdot_CCLabel);
    VarLabel::destroy(SumThermExpLabel);

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
    VarLabel::destroy(tau_X_FCLabel);
    VarLabel::destroy(tau_Y_FCLabel);
    VarLabel::destroy(tau_Z_FCLabel);
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
    VarLabel::destroy(delTLabel);
    VarLabel::destroy(TotalMassLabel);
    VarLabel::destroy(KineticEnergyLabel);
    VarLabel::destroy(CenterOfMassVelocityLabel);
    VarLabel::destroy(TotalIntEngLabel);    
    VarLabel::destroy(doMechLabel);
}

