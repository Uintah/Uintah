#include <Uintah/Components/ICE/ICELabel.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/FCVariable.h>
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
    speedSound_CCLabel = 
     scinew VarLabel("speedSound_CC", CCVariable<double>::getTypeDescription());
    div_velfc_CCLabel = 
     scinew VarLabel("div_velfc_CC",  CCVariable<double>::getTypeDescription());
    vol_frac_CCLabel = 
     scinew VarLabel("vol_frac_CC",   CCVariable<double>::getTypeDescription());

  // Face centered variables
    uvel_FCLabel       = 
     scinew VarLabel("uvel_FC",   FCVariable<double>::getTypeDescription() );
    vvel_FCLabel       = 
     scinew VarLabel("vvel_FC",   FCVariable<double>::getTypeDescription() );
    wvel_FCLabel       = 
     scinew VarLabel("wvel_FC",   FCVariable<double>::getTypeDescription() );
    uvel_FCMELabel       = 
     scinew VarLabel("uvel_FCME", FCVariable<double>::getTypeDescription() );
    vvel_FCMELabel       = 
     scinew VarLabel("vvel_FCME", FCVariable<double>::getTypeDescription() );
    wvel_FCMELabel       = 
     scinew VarLabel("wvel_FCME", FCVariable<double>::getTypeDescription() );
    press_FCLabel     = 
     scinew VarLabel("press_FC",  FCVariable<double>::getTypeDescription() );
    tau_FCLabel       = 
     scinew VarLabel("tau_FC",    FCVariable<Vector>::getTypeDescription() );
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
    delete cv_CCLabel;
    delete rho_micro_CCLabel;
    delete div_velfc_CCLabel;
    delete vol_frac_CCLabel;
    delete speedSound_CCLabel;

    // Face centered variables
    delete uvel_FCLabel;
    delete vvel_FCLabel;
    delete wvel_FCLabel;
    delete press_FCLabel;
    delete tau_FCLabel;

    delete delTLabel;
}
// $Log$
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
