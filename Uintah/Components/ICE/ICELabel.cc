#include <Uintah/Components/ICE/ICELabel.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/PerPatch.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah;
using namespace Uintah::ICESpace;

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
// $Log$
// Revision 1.16  2000/12/18 23:25:55  jas
// 2d ice works for simple advection.
//
// Revision 1.15  2000/12/05 15:45:30  jas
// Now using SFC{X,Y,Z} data types.  Fixed some small bugs and things appear
// to be working up to the middle of step 2.
//
// Revision 1.14  2000/11/28 03:50:28  jas
// Added {X,Y,Z}FCVariables.  Things still don't work yet!
//
// Revision 1.13  2000/11/14 04:02:11  jas
// Added getExtraCellIterator and things now appear to be working up to
// face centered velocity calculations.
//
// Revision 1.12  2000/10/25 23:12:17  guilkey
// Fixed step2, reorganized 6and7 just a little bit.
//
// Revision 1.11  2000/10/24 23:07:21  guilkey
// Added code for steps6and7.
//
// Revision 1.10  2000/10/19 02:44:52  guilkey
// Added code for step5b.
//
// Revision 1.9  2000/10/18 21:02:17  guilkey
// Added code for steps 4 and 5.
//
// Revision 1.8  2000/10/17 04:13:25  jas
// Implement hydrostatic pressure adjustment as part of step 1b.  Still need
// to implement update bcs.
//
// Revision 1.7  2000/10/16 19:10:35  guilkey
// Combined step1e with step2 and eliminated step1e.
//
// Revision 1.6  2000/10/16 18:32:40  guilkey
// Implemented "step1e" of the ICE algorithm.
//
// Revision 1.5  2000/10/13 00:01:11  guilkey
// More work on ICE
//
// Revision 1.4  2000/10/09 22:37:01  jas
// Cleaned up labels and added more computes and requires for EOS.
//
// Revision 1.3  2000/10/06 03:47:26  jas
// Added computes for the initialization so that step 1 works.  Added a couple
// of CC labels for step 1. Can now go thru multiple timesteps doing work
// only in step 1.
//
// Revision 1.2  2000/10/04 20:17:52  jas
// Change namespace ICE to ICESpace.
//
// Revision 1.1  2000/10/04 19:26:14  guilkey
// Initial commit of some classes to help mainline ICE.
//
