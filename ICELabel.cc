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
  sp_vol_equilLabel =
    VarLabel::create("sp_vol_equil", CCVariable<double>::getTypeDescription());

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
  mom_L_CCLabel = 
    VarLabel::create("mom_L_CC",     CCVariable<Vector>::getTypeDescription());
  int_eng_L_CCLabel = 
    VarLabel::create("int_eng_L_CC", CCVariable<double>::getTypeDescription());
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
    delete press_CCLabel;
    delete press_equil_CCLabel;
    delete delP_DilatateLabel;
    delete delP_MassXLabel;
    delete rho_CCLabel;
    delete rho_CC_top_cycleLabel;
    delete temp_CCLabel;
    delete vel_CCLabel;
    delete cv_CCLabel;
    delete rho_micro_CCLabel;
    delete sp_vol_CCLabel;
    delete sp_vol_equilLabel;
    delete mass_CCLabel;
    delete burnedMass_CCLabel;
    delete releasedHeat_CCLabel;
    delete created_vol_CCLabel;
    delete speedSound_CCLabel;
    delete div_velfc_CCLabel;
    delete vol_frac_CCLabel;
    delete viscosity_CCLabel;
    delete mom_source_CCLabel;
    delete int_eng_source_CCLabel;
    delete mom_L_CCLabel;
    delete int_eng_L_CCLabel;
    delete mass_L_CCLabel;
    delete mom_L_ME_CCLabel;
    delete int_eng_L_ME_CCLabel;
    delete q_CCLabel;
    delete q_advectedLabel;
    delete term1Label;
    delete term2Label;
    delete term3Label;

    // Face centered variables
    delete uvel_FCLabel;
    delete vvel_FCLabel;
    delete wvel_FCLabel;
    delete uvel_FCMELabel;
    delete vvel_FCMELabel;
    delete wvel_FCMELabel;
    delete pressX_FCLabel;
    delete pressY_FCLabel;
    delete pressZ_FCLabel;
    delete tau_X_FCLabel;
    delete tau_Y_FCLabel;
    delete tau_Z_FCLabel;
    // Misc labels
    delete IveBeenHereLabel;
    delete scratchLabel;
    delete scratch_FCVectorLabel;
    
    // Reduction Variables
    delete delTLabel;
    delete TotalMassLabel;
    delete KineticEnergyLabel;
    delete CenterOfMassVelocityLabel;
    delete TotalIntEngLabel;    
    delete doMechLabel;
}

