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
    = scinew VarLabel("delT",      delt_vartype::getTypeDescription());
  doMechLabel
    = scinew VarLabel("doMech",    delt_vartype::getTypeDescription());
  press_CCLabel     =
    scinew VarLabel("press_CC",    CCVariable<double>::getTypeDescription());
  press_equil_CCLabel   =
    scinew VarLabel("press_equil_CC",CCVariable<double>::getTypeDescription());
  delPress_CCLabel  =
    scinew VarLabel("delPress_CC",  CCVariable<double>::getTypeDescription());
  rho_CCLabel       = 
    scinew VarLabel("rho_CC",       CCVariable<double>::getTypeDescription());
  rho_CC_top_cycleLabel       = 
    scinew VarLabel("rho_top_cycle",CCVariable<double>::getTypeDescription());
  temp_CCLabel      = 
    scinew VarLabel("temp_CC",      CCVariable<double>::getTypeDescription());
  vel_CCLabel       = 
    scinew VarLabel("vel_CC",       CCVariable<Vector>::getTypeDescription());
  cv_CCLabel        = 
    scinew VarLabel("cv_CC",        CCVariable<double>::getTypeDescription());
  rho_micro_CCLabel = 
    scinew VarLabel("rho_micro_CC", CCVariable<double>::getTypeDescription());
  sp_vol_CCLabel =
    scinew VarLabel("sp_vol_CC",    CCVariable<double>::getTypeDescription());
  sp_vol_equilLabel =
    scinew VarLabel("sp_vol_equil", CCVariable<double>::getTypeDescription());

  mass_CCLabel =
    scinew VarLabel("mass_CC",      CCVariable<double>::getTypeDescription());
  speedSound_CCLabel =
    scinew VarLabel("speedSound_CC",CCVariable<double>::getTypeDescription());
  div_velfc_CCLabel =
    scinew VarLabel("div_velfc_CC", CCVariable<double>::getTypeDescription());
  vol_frac_CCLabel =
    scinew VarLabel("vol_frac_CC",  CCVariable<double>::getTypeDescription());
  viscosity_CCLabel =
    scinew VarLabel("viscosity_CC", CCVariable<double>::getTypeDescription());
  mom_source_CCLabel = 
    scinew VarLabel("mom_source_CC",CCVariable<Vector>::getTypeDescription());
  int_eng_source_CCLabel = 
    scinew VarLabel("intE_source_CC",CCVariable<double>::getTypeDescription());
  mom_L_CCLabel = 
    scinew VarLabel("mom_L_CC",     CCVariable<Vector>::getTypeDescription());
  int_eng_L_CCLabel = 
    scinew VarLabel("int_eng_L_CC", CCVariable<double>::getTypeDescription());
  mass_L_CCLabel = 
    scinew VarLabel("mass_L_CC",    CCVariable<double>::getTypeDescription());
  mom_L_ME_CCLabel = 
    scinew VarLabel("mom_L_ME_CC",  CCVariable<Vector>::getTypeDescription());
  int_eng_L_ME_CCLabel = 
    scinew VarLabel("int_eng_L_ME_CC",CCVariable<double>::getTypeDescription());
  q_CCLabel = 
    scinew VarLabel("q_CC",         CCVariable<double>::getTypeDescription());
  q_advectedLabel = 
    scinew VarLabel("q_advected",   CCVariable<double>::getTypeDescription());
  qV_CCLabel = 
    scinew VarLabel("qV_CC",        CCVariable<Vector>::getTypeDescription());
  qV_advectedLabel = 
    scinew VarLabel("qV_advected",  CCVariable<Vector>::getTypeDescription());
  burnedMass_CCLabel =
    scinew VarLabel("burnedMass",   CCVariable<double>::getTypeDescription());
  releasedHeat_CCLabel =
    scinew VarLabel("releasedHeat", CCVariable<double>::getTypeDescription());
  created_vol_CCLabel =
    scinew VarLabel("created_vol",  CCVariable<double>::getTypeDescription());

  term1Label = 
    scinew VarLabel("term1",        CCVariable<double>::getTypeDescription());
  term2Label = 
    scinew VarLabel("term2",        CCVariable<double>::getTypeDescription());
  term3Label = 
    scinew VarLabel("term3",        CCVariable<double>::getTypeDescription());
  
  // Face centered variables
  uvel_FCLabel       = 
    scinew VarLabel("uvel_FC",   SFCXVariable<double>::getTypeDescription() );
  vvel_FCLabel       = 
    scinew VarLabel("vvel_FC",   SFCYVariable<double>::getTypeDescription() );
  wvel_FCLabel       = 
    scinew VarLabel("wvel_FC",   SFCZVariable<double>::getTypeDescription() );
  uvel_FCMELabel       = 
    scinew VarLabel("uvel_FCME", SFCXVariable<double>::getTypeDescription() );
  vvel_FCMELabel       = 
    scinew VarLabel("vvel_FCME", SFCYVariable<double>::getTypeDescription() );
  wvel_FCMELabel       = 
    scinew VarLabel("wvel_FCME", SFCZVariable<double>::getTypeDescription() );
  pressX_FCLabel     = 
    scinew VarLabel("pressX_FC", SFCXVariable<double>::getTypeDescription() );
  pressY_FCLabel     =
    scinew VarLabel("pressY_FC", SFCYVariable<double>::getTypeDescription() );
  pressZ_FCLabel     =
    scinew VarLabel("pressZ_FC", SFCZVariable<double>::getTypeDescription() );
  tau_X_FCLabel       =
    scinew VarLabel("tau_X_FC",  SFCXVariable<Vector>::getTypeDescription() );
  tau_Y_FCLabel       =
    scinew VarLabel("tau_Y_FC",  SFCYVariable<Vector>::getTypeDescription() );
  tau_Z_FCLabel       =
    scinew VarLabel("tau_Z_FC",  SFCZVariable<Vector>::getTypeDescription() );
    
    // Misc labels
    scratchLabel     =
     scinew VarLabel("scratch",  CCVariable<double>::getTypeDescription() );
    scratch_FCXLabel   =
     scinew VarLabel("scratch_FCX",SFCXVariable<double>::getTypeDescription());

    scratch_FCYLabel   =
     scinew VarLabel("scratch_FCY",SFCYVariable<double>::getTypeDescription());

    scratch_FCZLabel   =
     scinew VarLabel("scratch_FCZ",SFCZVariable<double>::getTypeDescription());

    scratch_FCVectorLabel   =
     scinew VarLabel("scratch_FCVector",
		     SFCXVariable<Vector>::getTypeDescription());
    IveBeenHereLabel     =
     scinew VarLabel("IveBeenHere",CCVariable<int>::getTypeDescription() );
     
 //Reduction labels (The names must be identical to those in MPMLabel.cc)
  KineticEnergyLabel = 
    scinew VarLabel( "KineticEnergy", sum_vartype::getTypeDescription() );
  CenterOfMassVelocityLabel = 
    scinew VarLabel( "CenterOfMassVelocity",
                                      sumvec_vartype::getTypeDescription() );
  TotalMassLabel = 
    scinew VarLabel( "TotalMass",     sum_vartype::getTypeDescription() );  
  TotalIntEngLabel = 
    scinew VarLabel( "TotalIntEng",   sum_vartype::getTypeDescription() );  

       
} 

ICELabel::~ICELabel()
{
    // Cell centered variables
    delete press_CCLabel;
    delete press_equil_CCLabel;
    delete delPress_CCLabel;
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

