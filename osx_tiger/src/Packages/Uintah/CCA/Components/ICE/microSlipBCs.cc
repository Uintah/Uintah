#include <Packages/Uintah/CCA/Components/ICE/microSlipBCs.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Math/MiscMath.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Math/MiscMath.h>
#include <typeinfo>

#define d_SMALL_NUM 1e-100

using namespace Uintah;
namespace Uintah {
//__________________________________
//  To turn on couts
//  setenv SCI_DEBUG "SLIP_DOING_COUT:+, SLIP_DBG_COUT:+"
static DebugStream cout_doing("SLIP_DOING_COUT", false);
static DebugStream cout_dbg("SLIP_DBG_COUT", false);

/* ______________________________________________________________________
 Function~  read_MicroSlip_BC_inputs--   
 Purpose~   -returns (true) if microSlip BC is specified on any face,
            -reads input parameters 
 ______________________________________________________________________  */
bool read_MicroSlip_BC_inputs(const ProblemSpecP& prob_spec,
                              Slip_variable_basket* svb)
{
  //__________________________________
  // search the BoundaryConditions problem spec
  // determine if Slip/creep bcs are specified
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
 
  bool usingSlip = false;
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
    map<string,string> face;
    face_ps->getAttributes(face);
    bool is_a_MicroSlip_face = false;
    
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
      

      if ((bc_type["var"] == "slip" || bc_type["var"] == "creep") 
          && !is_a_MicroSlip_face) {
        usingSlip = true;
        is_a_MicroSlip_face = true;
      }
    }
  }
  //__________________________________
  //  read in variables from microSlip section 
  if(usingSlip ){
    ProblemSpecP slip = bc_ps->findBlock("microSlip");
    if (!slip) {
      string warn="ERROR:\n Inputs:Boundary Conditions: Cannot find Slip block";
      throw ProblemSetupException(warn);
    }

    slip->require("alpha_momentum",   svb->alpha_momentum);
    slip->require("alpha_temperature",svb->alpha_temperature);
  }
  
  if (usingSlip) {
    cout << "\n WARNING:  Slip boundary conditions are "
         << " NOT set during the problem initialization \n " 
         << " THESE BOUNDARY CONDITIONS ONLY WORK FOR 1 MATL ICE PROBLEMS \n"
         << " (The material index has been hard coded in preprocess_MicroSlip_BCs)\n" <<endl;
  }
  return usingSlip;
}

/* ______________________________________________________________________ 
 Function~  addRequires_MicroSlip--   
 Purpose~   requires for all the tasks depends on which task you're in
 ______________________________________________________________________  */
void addRequires_MicroSlip(Task* t, 
                           const string& where,
                           ICELabel* lb,
                           const MaterialSubset* ice_matls,
                           Slip_variable_basket* var_basket)
{
  cout_doing<< "Doing addRequires_microSlip: \t\t" <<t->getName()
            << " " << where << endl;
  
  Ghost::GhostType  gn  = Ghost::None;
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  MaterialSubset* press_matl = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();

#if 0 
  if(where == "velFC_Exchange"){
    t->requires(Task::OldDW, lb->rho_CCLabel,   ice_matls, gn,0);        
    t->requires(Task::OldDW, lb->vel_CCLabel,   ice_matls, gn,0);
    t->requires(Task::OldDW, lb->temp_CCLabel,  ice_matls, gn,0); 
    t->requires(Task::NewDW, lb->viscosityLabel,ice_matls, gn,0);
    t->requires(Task::NewDW, lb->press_CCLabel, press_matl,oims,gn, 0);      
  }
  if(where == "imp_velFC_Exchange"){
    t->requires(Task::ParentOldDW, lb->rho_CCLabel,   ice_matls, gn,0);        
    t->requires(Task::ParentOldDW, lb->vel_CCLabel,   ice_matls, gn,0);
    t->requires(Task::ParentNewDW, lb->viscosityLabel,ice_matls, gn,0);
    t->requires(Task::ParentNewDW, lb->press_CCLabel, press_matl,oims,gn, 0);
  }
#endif
  
  if(where == "CC_Exchange"){
    t->requires(Task::NewDW, lb->rho_CCLabel,       ice_matls, gn);
    t->requires(Task::NewDW, lb->gammaLabel,        ice_matls, gn); 
    t->requires(Task::NewDW, lb->viscosityLabel,    ice_matls, gn);
    t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,oims,gn, 0);
    t->computes(lb->vel_CC_XchangeLabel);
    t->computes(lb->temp_CC_XchangeLabel);
  }
  if(where == "Advection"){
    t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,oims,gn, 0);
    t->requires(Task::NewDW, lb->gammaLabel,        ice_matls, gn); 
    t->requires(Task::NewDW, lb->viscosityLabel,    ice_matls, gn);
    // requires(Task::NewDW, lb->vel_CCLabel,       ice_matls, gn); 
    // requires(Task::NewDW, lb->rho_CCLabel,       ice_matls, gn); 
  }
}
/*__________________________________________________________________
 Function~ meanFreePath-
 Purpose~  compute the mean free path along an entire boundary face.
____________________________________________________________________*/
void meanFreePath(DataWarehouse* new_dw,
                  const Patch* patch,
                  SimulationStateP& sharedState,
                  Slip_vars* sv)                              
{
  cout_doing << "meanFreePath" << endl;
   new_dw->allocateTemporary(sv->lamda,patch);
   sv->lamda.initialize(-9e99);
  
  double R = 1.0;    // gas constant Need to do something with it
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    
    if (is_MicroSlip_face(patch,face, sharedState) ) {
      // hit the cells in once cell from the face direction
      IntVector offset = patch->faceDirection(face);
      
      for(CellIterator iter=patch->getFaceCellIterator(face, "plusEdgeCells"); 
          !iter.done();iter++) {
        IntVector c = *iter - offset;
        double A = sqrt(0.636620 * R * sv->Temp_CC[c]);
        sv->lamda[c] = sv->viscosity[c]/(sv->rho_CC[c] * A);
     
      }  // faceCelliterator 
    }  // is microSlip face
  }  // loop over faces
}
/*______________________________________________________________________ 
 Function~  preprocess_MicroSlip_BCs-- 
 Purpose~   get data from dw
______________________________________________________________________ */
void  preprocess_MicroSlip_BCs(DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               ICELabel* lb,
                               const Patch* patch,
                               const string& where,
                               const int /*indx*/,
                               SimulationStateP& sharedState,
                               bool& setMicroSlipBcs,
                               Slip_vars* sv,
                               Slip_variable_basket* var_basket)
{
  
  Ghost::GhostType  gn  = Ghost::None;
/*`==========TESTING==========*/
  int indx = 0;                 // ICE MATL IS HARD CODED TO 0
/*===========TESTING==========`*/
  setMicroSlipBcs = false;
  
  //__________________________________
  //    FC exchange
  if(where == "velFC_Exchange"){
#if 0
    setMicroSlipBcs = true;
    old_dw->get(sv->rho_CC,     lb->rho_CCLabel,        indx,patch,gn,0);
    old_dw->get(sv->vel_CC,     lb->vel_CCLabel,        indx,patch,gn,0);
    old_dw->get(sv->Temp_CC,    lb->temp_CCLabel,       indx,patch,gn,0);
    new_dw->get(sv->viscosity,  lb->viscosityLabel,     indx,patch,gn,0);
    new_dw->get(sv->press_CC,   lb->press_CCLabel,      0,   patch,gn,0);
#endif
  }
  //__________________________________
  //    cc_ Exchange
  if(where == "CC_Exchange"){
    setMicroSlipBcs = true;
    new_dw->get(sv->rho_CC,     lb->rho_CCLabel,         indx,patch,gn,0);
    new_dw->get(sv->vel_CC,     lb->vel_CC_XchangeLabel, indx,patch,gn,0);
    new_dw->get(sv->Temp_CC,    lb->temp_CC_XchangeLabel,indx,patch,gn,0);
    new_dw->get(sv->viscosity,  lb->viscosityLabel,      indx,patch,gn,0);
    new_dw->get(sv->gamma,      lb->gammaLabel,          indx,patch,gn,0);
    new_dw->get(sv->press_CC,   lb->press_CCLabel,       0,   patch,gn,0);
  }
  //__________________________________
  //    Advection
  if(where == "Advection"){
    setMicroSlipBcs = true;
    new_dw->get(sv->rho_CC,    lb->rho_CCLabel,        indx,patch,gn,0); 
    new_dw->get(sv->vel_CC,    lb->vel_CCLabel,        indx,patch,gn,0);
    new_dw->get(sv->Temp_CC,   lb->temp_CCLabel,       indx,patch,gn,0);
    new_dw->get(sv->viscosity, lb->viscosityLabel,     indx,patch,gn,0); 
    new_dw->get(sv->gamma,     lb->gammaLabel,         indx,patch,gn,0); 
    new_dw->get(sv->press_CC,  lb->press_CCLabel,      0,   patch,gn,0); 
  }
  //__________________________________
  //  compute the mean free path
  if(setMicroSlipBcs) {
    cout_doing << "preprocess_microSlip_BCs on patch "<<patch->getID()<< endl;
    sv->alpha_momentum    = var_basket->alpha_momentum;
    sv->alpha_temperature = var_basket->alpha_temperature;
    meanFreePath(new_dw, patch, sharedState, sv);
  }
}
/* ______________________________________________________________________ 
 Function~  is_MicroSlip_face--   
 Purpose~   returns true if this face on this patch is using MicroSlip bcs
 ______________________________________________________________________  */
bool is_MicroSlip_face(const Patch* patch,
                       Patch::FaceType face,
                       SimulationStateP& sharedState)
{
  bool is_MicroSlip_face = false;
  int numMatls = sharedState->getNumICEMatls();

  for (int m = 0; m < numMatls; m++ ) {
    ICEMaterial* ice_matl = sharedState->getICEMaterial(m);
    int indx= ice_matl->getDWIndex();
    bool slip_temperature = patch->haveBC(face,indx,"slip","Temperature");
    bool slip_velocity    = patch->haveBC(face,indx,"slip","Velocity");
    bool creep_velocity   = patch->haveBC(face,indx,"creep","Velocity");

    if (slip_temperature || slip_velocity || creep_velocity) {
      is_MicroSlip_face = true; 
    }
  }
  return is_MicroSlip_face;
}

/*_________________________________________________________________
 Function~ set_MicroSlipVelocity_BC--
 Purpose~  Set velocity boundary conditions
 Reference:   Jennifer please fill this in.
___________________________________________________________________*/
void set_MicroSlipVelocity_BC(const Patch* patch,
                              const Patch::FaceType face,
                              CCVariable<Vector>& vel_CC,
                              const string& var_desc,
                              const vector<IntVector> bound,
                              const string& bc_kind,
                              const Vector wall_vel,
                              Slip_vars* sv)                     

{
  if (var_desc == "Velocity" && (bc_kind == "slip" || bc_kind == "creep")) {
  
    cout_doing << "Setting FaceVel_MicroSlip on face " << face 
               << " wall Velocity " << wall_vel << endl;
    
    // bulletproofing
    if (!sv){
      throw InternalError("set_MicroSlipTemperature_BC: MicroSlip_vars = null");
    }

    // define shortcuts
    CCVariable<double>& lamda  = sv->lamda;
    constCCVariable<double> viscosity = sv->viscosity;
    constCCVariable<double> press_CC  = sv->press_CC;
    constCCVariable<double> rho_CC    = sv->rho_CC;
    
    IntVector offset = patch->faceDirection(face);
    IntVector axes = patch->faceAxes(face);
    int P_dir = axes[0];  // principal direction
    int dir1  = axes[1];  // Jennifer double check what these indicies
    int dir2  = axes[2];  // are for the different faces
    
    Vector DX = patch->dCell();
    double dx = DX[P_dir];
    double alpha_momentum = sv->alpha_momentum;


    cout_dbg << "____________________velocity";
    //__________________________________
    //   SLIP 
    if(bc_kind == "slip") {
      cout_dbg << " SLIP"<< endl;
      
      vector<IntVector>::const_iterator iter;
      for (iter=bound.begin(); iter != bound.end(); iter++) {
        IntVector c = *iter;
        IntVector in = c - offset;
        // normal direction velocity
        vel_CC[c] = wall_vel; 

        // transverse velocities    // Jennifer-- put equations here
        //double gradient_1 = 0;        
        //double gradient_2 = 0;
        //vel_CC[c][dir1] = ???????;
        //vel_CC[c][dir2] = ???????;
      }
    }
    //__________________________________
    //   CREEP 
    if(bc_kind == "creep") {
      cout_dbg << " CREEP"<< endl;
      
      vector<IntVector>::const_iterator iter;
      for (iter=bound.begin(); iter != bound.end(); iter++) {
        IntVector c = *iter;
        IntVector in = c - offset;
        // normal direction velocity
        vel_CC[c] = wall_vel; 

        // transverse velocities    // Jennifer-- put equations here
        //double gradient_1 = 0;        
        //double gradient_2 = 0;  // Tangential gradients MAY be a problem 
                                // with mult-patch problem.
        //vel_CC[c][dir1] = ???????;
        //vel_CC[c][dir2] = ???????;
      }
    }
  } 
}

/*_________________________________________________________________
 Function~ set_MicroSlipTemperature_BC--
 Purpose~  Compute temperature in boundary cells on faces
___________________________________________________________________*/
void set_MicroSlipTemperature_BC(const Patch* patch,
                                 const Patch::FaceType face,
                                 CCVariable<double>& temp_CC,
                                 const string& var_desc,
                                 const vector<IntVector> bound,
                                 const string& bc_kind,
                                 const double wall_temp,
                                 Slip_vars* sv)  
{
  if (var_desc == "Temperature" && bc_kind == "slip") {
    cout_doing << "Setting FaceTemp_MicroSlip on face " <<face
               << " wall Temperature " << wall_temp << endl; 

    // bulletproofing
    if (!sv){
      throw InternalError("set_MicroSlipTemperature_BC: Slip_vars = null");
    } 
    // shortcuts  
    constCCVariable<double>& gamma     = sv->gamma;
    constCCVariable<double>& viscosity = sv->viscosity;
    CCVariable<double>& lamda          = sv->lamda;
    constCCVariable<double>& Temp_CC  = sv->Temp_CC;

    IntVector axes = patch->faceAxes(face);
    int P_dir = axes[0];  // principal direction
    Vector DX = patch->dCell();
    double dx = DX[P_dir];
    double alpha_temperature = sv->alpha_temperature;
    double gas_constant = 1.0;   // Need to do something here
    IntVector offset = patch->faceDirection(face);
    cout_dbg << "\n____________________Temp"<< endl;

    //__________________________________
    //    S I D E     
    double A =( 2.0 - alpha_temperature) /alpha_temperature;
    
    vector<IntVector>::const_iterator iter;
    for (iter=bound.begin(); iter != bound.end(); iter++) {
      IntVector c = *iter;
      IntVector in = c - offset;
      
      double B = 2.0*(gamma[in] -1) / (gamma[in] + 1);
      double C = gas_constant/viscosity[in] * lamda[in];
      double grad = (Temp_CC[in] - Temp_CC[in-offset])/dx;
      // temp_CC[c] = A * B * C * grad + wall_temp???;  
      temp_CC[c] = Temp_CC[in];  // Jennifer-- put equations here
    }
  }
} 

  
}  // using namespace Uintah
