#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

ICELabel::ICELabel()
{

    delTLabel
	 = scinew VarLabel("delT",      delt_vartype::getTypeDescription() );

    press_CCLabel     =
     scinew VarLabel("press_CC",    CCVariable<double>::getTypeDescription() );
    pressdP_CCLabel   =
     scinew VarLabel("pressdP_CC",  CCVariable<double>::getTypeDescription() );
    delPress_CCLabel  =
     scinew VarLabel("delPress_CC", CCVariable<double>::getTypeDescription() );

    rho_CCLabel       = 
     scinew VarLabel("rho_CC",    CCVariable<double>::getTypeDescription() );

    temp_CCLabel      = 
     scinew VarLabel("temp_CC",   CCVariable<double>::getTypeDescription() );

    uvel_CCLabel       = 
     scinew VarLabel("uvel_CC",    CCVariable<double>::getTypeDescription() );
    vvel_CCLabel       = 
     scinew VarLabel("vvel_CC",    CCVariable<double>::getTypeDescription() );
    wvel_CCLabel       = 
     scinew VarLabel("wvel_CC",    CCVariable<double>::getTypeDescription() );

    cv_CCLabel        = 
     scinew VarLabel("cv_CC",         CCVariable<double>::getTypeDescription());
    rho_micro_CCLabel = 
     scinew VarLabel("rho_micro_CC",  CCVariable<double>::getTypeDescription());
    rho_micro_equil_CCLabel = 
      scinew VarLabel("rho_micro_equil_CC",  CCVariable<double>::getTypeDescription());

    speedSound_CCLabel = 
     scinew VarLabel("speedSound_CC", CCVariable<double>::getTypeDescription());
    speedSound_equiv_CCLabel = 
     scinew VarLabel("speedSound_equiv_CC", CCVariable<double>::getTypeDescription());

    div_velfc_CCLabel = 
     scinew VarLabel("div_velfc_CC",  CCVariable<double>::getTypeDescription());
    vol_frac_CCLabel = 
     scinew VarLabel("vol_frac_CC",   CCVariable<double>::getTypeDescription());

    viscosity_CCLabel = 
     scinew VarLabel("viscosity_CC",  CCVariable<double>::getTypeDescription());
    xmom_source_CCLabel = 
     scinew VarLabel("xmom_source_CC",CCVariable<double>::getTypeDescription());
    ymom_source_CCLabel = 
     scinew VarLabel("ymom_source_CC",CCVariable<double>::getTypeDescription());
    zmom_source_CCLabel = 
     scinew VarLabel("zmom_source_CC",CCVariable<double>::getTypeDescription());
    int_eng_source_CCLabel = 
     scinew VarLabel("intE_source_CC",CCVariable<double>::getTypeDescription());
    xmom_L_CCLabel = 
     scinew VarLabel("xmom_L_CC",CCVariable<double>::getTypeDescription());
    ymom_L_CCLabel = 
     scinew VarLabel("ymom_L_CC",CCVariable<double>::getTypeDescription());
    zmom_L_CCLabel = 
     scinew VarLabel("zmom_L_CC",CCVariable<double>::getTypeDescription());
    int_eng_L_CCLabel = 
     scinew VarLabel("intE_L_CC",CCVariable<double>::getTypeDescription());
    mass_L_CCLabel = 
     scinew VarLabel("mass_L_CC",CCVariable<double>::getTypeDescription());
    rho_L_CCLabel = 
     scinew VarLabel("rho_L_CC",CCVariable<double>::getTypeDescription());
    xmom_L_ME_CCLabel = 
     scinew VarLabel("xmom_L_ME_CC",CCVariable<double>::getTypeDescription());
    ymom_L_ME_CCLabel = 
     scinew VarLabel("ymom_L_ME_CC",CCVariable<double>::getTypeDescription());
    zmom_L_ME_CCLabel = 
     scinew VarLabel("zmom_L_ME_CC",CCVariable<double>::getTypeDescription());
    int_eng_L_ME_CCLabel = 
     scinew VarLabel("intE_L_ME_CC",CCVariable<double>::getTypeDescription());
    q_CCLabel = 
     scinew VarLabel("q_CC",CCVariable<double>::getTypeDescription());
    q_advectedLabel = 
     scinew VarLabel("q_advected",CCVariable<double>::getTypeDescription());
    term1Label = 
     scinew VarLabel("term1",CCVariable<double>::getTypeDescription());
    term2Label = 
     scinew VarLabel("term2",CCVariable<double>::getTypeDescription());
    term3Label = 
     scinew VarLabel("term3",CCVariable<double>::getTypeDescription());

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
     scinew VarLabel("pressX_FC",  SFCXVariable<double>::getTypeDescription() );
    pressY_FCLabel     = 
     scinew VarLabel("pressY_FC",  SFCYVariable<double>::getTypeDescription() );
    pressZ_FCLabel     = 
     scinew VarLabel("pressZ_FC",  SFCZVariable<double>::getTypeDescription() );
    tau_X_FCLabel       = 
     scinew VarLabel("tau_X_FC",   SFCXVariable<Vector>::getTypeDescription() );
    tau_Y_FCLabel       = 
     scinew VarLabel("tau_Y_FC",   SFCYVariable<Vector>::getTypeDescription() );
    tau_Z_FCLabel       = 
     scinew VarLabel("tau_Z_FC",   SFCZVariable<Vector>::getTypeDescription() );
} 

ICELabel::~ICELabel()
{
    // Cell centered variables
    delete  press_CCLabel;
    delete  pressdP_CCLabel;
    delete  delPress_CCLabel;
    delete  rho_CCLabel;
    delete temp_CCLabel;
    delete uvel_CCLabel;
    delete vvel_CCLabel;
    delete wvel_CCLabel;
    delete speedSound_CCLabel;
    delete speedSound_equiv_CCLabel;
    delete cv_CCLabel;
    delete rho_micro_CCLabel;
    delete div_velfc_CCLabel;
    delete vol_frac_CCLabel;
    delete viscosity_CCLabel;
    delete xmom_source_CCLabel;
    delete ymom_source_CCLabel;
    delete zmom_source_CCLabel;
    delete int_eng_source_CCLabel;
    delete xmom_L_CCLabel;
    delete ymom_L_CCLabel;
    delete zmom_L_CCLabel;
    delete int_eng_L_CCLabel;
    delete mass_L_CCLabel;
    delete rho_L_CCLabel;
    delete xmom_L_ME_CCLabel;
    delete ymom_L_ME_CCLabel;
    delete zmom_L_ME_CCLabel;
    delete int_eng_L_ME_CCLabel;
    delete q_CCLabel;
    delete term1Label;
    delete term2Label;
    delete term3Label;

    // Face centered variables
    delete uvel_FCLabel;
    delete vvel_FCLabel;
    delete wvel_FCLabel;
    delete pressX_FCLabel;
    delete pressY_FCLabel;
    delete pressZ_FCLabel;
    delete tau_X_FCLabel;
    delete tau_Y_FCLabel;
    delete tau_Z_FCLabel;

    delete delTLabel;
}
