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
     scinew VarLabel("press_CC",  CCVariable<double>::getTypeDescription() );

    rho_CCLabel       = 
     scinew VarLabel("rho_CC",    CCVariable<double>::getTypeDescription() );

    temp_CCLabel      = 
     scinew VarLabel("temp_CC",   CCVariable<double>::getTypeDescription() );

    vel_CCLabel       = 
     scinew VarLabel("vel_CC",    CCVariable<Vector>::getTypeDescription() );

    cv_CCLabel        = 
     scinew VarLabel("cv_CC",     CCVariable<double>::getTypeDescription() );
    rho_micro_CCLabel = 
     scinew VarLabel("rho_micro_CC",CCVariable<double>::getTypeDescription());
    speedSound_CCLabel = 
     scinew VarLabel("speedSound_CC",CCVariable<double>::getTypeDescription());
    div_velfc_CCLabel = 
     scinew VarLabel("div_velfc_CC", CCVariable<double>::getTypeDescription() );
    vol_frac_CCLabel = 
     scinew VarLabel("vol_frac_CC", CCVariable<double>::getTypeDescription() );

  // Face centered variables
    vel_FCLabel       = 
     scinew VarLabel("vel_FC",    FCVariable<Vector>::getTypeDescription() );
    press_FCLabel     = 
     scinew VarLabel("press_FC",  FCVariable<double>::getTypeDescription() );
    tau_FCLabel       = 
     scinew VarLabel("tau_FC",    FCVariable<Vector>::getTypeDescription() );
} 

ICELabel::~ICELabel()
{
    // Cell centered variables
    delete  press_CCLabel;
    delete  rho_CCLabel;
    delete temp_CCLabel;
    delete vel_CCLabel;
    delete cv_CCLabel;
    delete rho_micro_CCLabel;
    delete div_velfc_CCLabel;
    delete vol_frac_CCLabel;
    delete speedSound_CCLabel;

    // Face centered variables
    delete vel_FCLabel;
    delete press_FCLabel;
    delete tau_FCLabel;

    delete delTLabel;
}
// $Log$
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
