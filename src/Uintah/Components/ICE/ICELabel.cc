#include <Uintah/Components/ICE/ICELabel.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/PerPatch.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/Vector.h>

using namespace Uintah;
using namespace Uintah::ICESpace;
using SCICore::Geometry::Vector;

ICELabel::ICELabel()
{
  delTLabel
    = scinew VarLabel("delT",      delt_vartype::getTypeDescription() );
  press_CCLabel     =
    scinew VarLabel("press_CC",    CCVariable<double>::getTypeDescription() );
  press_equil_CCLabel   =
    scinew VarLabel("press_equil_CC",  CCVariable<double>::getTypeDescription() );
  delPress_CCLabel  =
    scinew VarLabel("delPress_CC", CCVariable<double>::getTypeDescription() );
  rho_CCLabel       = 
    scinew VarLabel("rho_CC",    CCVariable<double>::getTypeDescription() );
  temp_CCLabel      = 
    scinew VarLabel("temp_CC",   CCVariable<double>::getTypeDescription() );
  vel_CCLabel       = 
    scinew VarLabel("vel_CC",    CCVariable<Vector>::getTypeDescription() );
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
  viscosity_CCLabel = 
    scinew VarLabel("viscosity_CC",  CCVariable<double>::getTypeDescription());
  mom_source_CCLabel = 
    scinew VarLabel("mom_source_CC",CCVariable<Vector>::getTypeDescription());
  int_eng_source_CCLabel = 
    scinew VarLabel("intE_source_CC",CCVariable<double>::getTypeDescription());
  mom_L_CCLabel = 
    scinew VarLabel("mom_L_CC",CCVariable<Vector>::getTypeDescription());
  int_eng_L_CCLabel = 
    scinew VarLabel("intE_L_CC",CCVariable<double>::getTypeDescription());
  mass_L_CCLabel = 
    scinew VarLabel("mass_L_CC",CCVariable<double>::getTypeDescription());
  rho_L_CCLabel = 
    scinew VarLabel("rho_L_CC",CCVariable<double>::getTypeDescription());
  mom_L_ME_CCLabel = 
    scinew VarLabel("mom_L_ME_CC",CCVariable<Vector>::getTypeDescription());
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
    delete press_CCLabel;
    delete press_equil_CCLabel;
    delete delPress_CCLabel;
    delete rho_CCLabel;
    delete temp_CCLabel;
    delete vel_CCLabel;
    delete cv_CCLabel;
    delete rho_micro_CCLabel;
    delete speedSound_CCLabel;
    delete div_velfc_CCLabel;
    delete vol_frac_CCLabel;
    delete viscosity_CCLabel;
    delete mom_source_CCLabel;
    delete int_eng_source_CCLabel;
    delete mom_L_CCLabel;
    delete int_eng_L_CCLabel;
    delete mass_L_CCLabel;
    delete rho_L_CCLabel;
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

    delete delTLabel;
}
// $Log$
// Revision 1.26  2001/01/20 00:39:03  jas
// Remove some memory leaks.
//
// Revision 1.25  2001/01/13 01:42:10  harman
// - eliminated speedSound_equiv_CCLabel
// - eliminated rho_micro_equil_CClabel
//
// Revision 1.24  2001/01/10 00:47:18  harman
// ICE.cc ICELabel.h ICELabel.cc ICE.h
// -changed how vol_frac is computed in actuallyStep1b
// -changed step4b requires to press_CC
// -Fixed small error in OFS calculation
// -Added instrumentation
// -Added
//     ICE::printVector
//     ICE::printData_FC
// -Added bullet proofing at the bottom of equilibration pressure function
// -changed pressdP_CCLabel to press_CCLabel in all steps downstream of explicit delPress
// -Added new label press_equil_CCLabel.  This represents the pressure coming out
// of the equilibration pressure calculation. This label replaced press_CCLabel between
// the equilibration press calc and explicit delPress calc.
//
// Revision 1.23  2001/01/08 22:01:55  jas
// Removed #if 0  #endif pairs surrounding unused code related to momentum
// variables that are now combined into CCVariables<Vector>.  This includes
// mom_source, mom_L and mom_L_ME.
//
// Revision 1.22  2001/01/08 20:40:51  jas
// Replace {x,y,z}mom_L_ME with a single CCVariable<Vector> mom_L_ME.
//
// Revision 1.21  2001/01/08 18:30:37  jas
// Replace {x,y,z}mom_L with a single CCVariable<Vector> mom_L.
//
// Revision 1.20  2001/01/05 20:01:30  jas
// Replaced {x,y,z}mom_source with a single mom_source that is a
// CCVariable<Vector>.  Fixed printData so it can handle CCVariable<Vector>.
//
// Revision 1.19  2001/01/05 17:55:29  jas
// Remove uvel_CC, vvel_CC and wvel_CC from labels.
//
// Revision 1.18  2001/01/05 16:34:10  jas
// Changed over uvel_CC, vvel_CC, wvel_CC to a CCVariable<Vector> in all steps
// where CC velocities are used.
//
// Revision 1.17  2001/01/03 00:51:53  harman
// - added cflux, OFC, IFC, q_in_CF, q_out_CF
// - Advection operator now in 3D, not fully tested
// - A little house cleaning on #includes
// - Removed *_old from step6&7 except cv_old
//
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
