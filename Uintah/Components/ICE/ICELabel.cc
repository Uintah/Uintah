#include <Uintah/Components/ICE/ICELabel.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/PerPatch.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah;
using namespace Uintah::ICE;

ICELabel::ICELabel()
{

    delTLabel
	 = scinew VarLabel("delT",      delt_vartype::getTypeDescription() );

    press_CCLabel     =
     scinew VarLabel("press_CC",  CCVariable<double>::getTypeDescription() );
    press_CCLabel_0   =
     scinew VarLabel("press_CC_0",  CCVariable<double>::getTypeDescription() );
    press_CCLabel_1   =
     scinew VarLabel("press_CC_1",CCVariable<double>::getTypeDescription() );
    press_CCLabel_2   =
     scinew VarLabel("press_CC_2",CCVariable<double>::getTypeDescription() );
    press_CCLabel_3   = 
     scinew VarLabel("press_CC_3",CCVariable<double>::getTypeDescription() );
    press_CCLabel_4   = 
     scinew VarLabel("press_CC_4",CCVariable<double>::getTypeDescription() );
    press_CCLabel_5   = 
     scinew VarLabel("press_CC_5",CCVariable<double>::getTypeDescription() );
    press_CCLabel_6_7 = 
     scinew VarLabel("press_CC_6_7",CCVariable<double>::getTypeDescription() );

    rho_CCLabel       = 
     scinew VarLabel("rho_CC",    CCVariable<double>::getTypeDescription() );
    rho_CCLabel_0     = 
     scinew VarLabel("rho_CC_0",  CCVariable<double>::getTypeDescription() );
    rho_CCLabel_1     = 
     scinew VarLabel("rho_CC_1",  CCVariable<double>::getTypeDescription() );
    rho_CCLabel_2     = 
     scinew VarLabel("rho_CC_2",  CCVariable<double>::getTypeDescription() );
    rho_CCLabel_3     = 
     scinew VarLabel("rho_CC_3",  CCVariable<double>::getTypeDescription() );
    rho_CCLabel_4     = 
     scinew VarLabel("rho_CC_4",  CCVariable<double>::getTypeDescription() );
    rho_CCLabel_5     = 
     scinew VarLabel("rho_CC_5",  CCVariable<double>::getTypeDescription() );
    rho_CCLabel_6_7   = 
     scinew VarLabel("rho_CC_6_7",  CCVariable<double>::getTypeDescription() );

    temp_CCLabel      = 
     scinew VarLabel("temp_CC",   CCVariable<double>::getTypeDescription() );
    temp_CCLabel_0    = 
     scinew VarLabel("temp_CC_0", CCVariable<double>::getTypeDescription() );
    temp_CCLabel_1    = 
     scinew VarLabel("temp_CC_1", CCVariable<double>::getTypeDescription() );
    temp_CCLabel_2    = 
     scinew VarLabel("temp_CC_2", CCVariable<double>::getTypeDescription() );
    temp_CCLabel_3    = 
     scinew VarLabel("temp_CC_3", CCVariable<double>::getTypeDescription() );
    temp_CCLabel_4    = 
     scinew VarLabel("temp_CC_4", CCVariable<double>::getTypeDescription() );
    temp_CCLabel_5    = 
     scinew VarLabel("temp_CC_5", CCVariable<double>::getTypeDescription() );
    temp_CCLabel_6_7  = 
     scinew VarLabel("temp_CC_6_7", CCVariable<double>::getTypeDescription() );

    vel_CCLabel       = 
     scinew VarLabel("vel_CC",    CCVariable<Vector>::getTypeDescription() );
    vel_CCLabel_0     = 
     scinew VarLabel("vel_CC_0",  CCVariable<Vector>::getTypeDescription() );
    vel_CCLabel_1     = 
     scinew VarLabel("vel_CC_1",  CCVariable<Vector>::getTypeDescription() );
    vel_CCLabel_2     = 
     scinew VarLabel("vel_CC_2",  CCVariable<Vector>::getTypeDescription() );
    vel_CCLabel_3     = 
     scinew VarLabel("vel_CC_3",  CCVariable<Vector>::getTypeDescription() );
    vel_CCLabel_4     = 
     scinew VarLabel("vel_CC_4",  CCVariable<Vector>::getTypeDescription() );
    vel_CCLabel_5     = 
     scinew VarLabel("vel_CC_5",  CCVariable<Vector>::getTypeDescription() );
    vel_CCLabel_6_7   = 
     scinew VarLabel("vel_CC_6_7",  CCVariable<Vector>::getTypeDescription() );

    cv_CCLabel        = 
     scinew VarLabel("cv_CC",     CCVariable<double>::getTypeDescription() );
    div_velfc_CCLabel = 
     scinew VarLabel("div_velfc_CC", CCVariable<double>::getTypeDescription() );

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
    delete VarLabel* press_CCLabel;
    delete VarLabel* press_CCLabel_0;
    delete VarLabel* press_CCLabel_1;
    delete VarLabel* press_CCLabel_2;
    delete VarLabel* press_CCLabel_3;
    delete VarLabel* press_CCLabel_4;
    delete VarLabel* press_CCLabel_5;
    delete VarLabel* press_CCLabel_6_7;

    delete VarLabel* rho_CCLabel;
    delete VarLabel* rho_CCLabel_0;
    delete VarLabel* rho_CCLabel_1;
    delete VarLabel* rho_CCLabel_2;
    delete VarLabel* rho_CCLabel_3;
    delete VarLabel* rho_CCLabel_4;
    delete VarLabel* rho_CCLabel_5;
    delete VarLabel* rho_CCLabel_6_7;

    delete VarLabel* temp_CCLabel;
    delete VarLabel* temp_CCLabel_0;
    delete VarLabel* temp_CCLabel_1;
    delete VarLabel* temp_CCLabel_2;
    delete VarLabel* temp_CCLabel_3;
    delete VarLabel* temp_CCLabel_4;
    delete VarLabel* temp_CCLabel_5;
    delete VarLabel* temp_CCLabel_6_7;

    delete VarLabel* vel_CCLabel;
    delete VarLabel* vel_CCLabel_0;
    delete VarLabel* vel_CCLabel_1;
    delete VarLabel* vel_CCLabel_2;
    delete VarLabel* vel_CCLabel_3;
    delete VarLabel* vel_CCLabel_4;
    delete VarLabel* vel_CCLabel_5;
    delete VarLabel* vel_CCLabel_6_7;

    delete VarLabel* cv_CCLabel;
    delete VarLabel* div_velfc_CCLabel;

    // Face centered variables
    delete VarLabel* vel_FCLabel;
    delete VarLabel* press_FCLabel;
    delete VarLabel* tau_FCLabel;

    delete delTLabel;
}
// $Log$
// Revision 1.1  2000/10/04 19:26:14  guilkey
// Initial commit of some classes to help mainline ICE.
//
