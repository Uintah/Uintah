#include <Packages/Uintah/CCA/Components/ICE/LODI2.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <Core/Math/MiscMath.h>
#include <typeinfo>

#define d_SMALL_NUM 1e-100

using namespace Uintah;
namespace Uintah {
//__________________________________
//  To turn on couts
//  setenv SCI_DEBUG "LODI_DOING_COUT:+, LODI_DBG_COUT:+"
static DebugStream cout_doing("LODI_DOING_COUT", false);
static DebugStream cout_dbg("LODI_DBG_COUT", false);

/* ______________________________________________________________________
 Function~  read_LODI_BC_inputs--   
 Purpose~   returns if we are using LODI BC on any face,
            reads in any lodi parameters 
            sets which boundaries are lodi
 ______________________________________________________________________  */
bool read_LODI_BC_inputs(const ProblemSpecP& prob_spec,
                         Lodi_variable_basket* vb)
{
  //__________________________________
  // search the BoundaryConditions problem spec
  // determine if LODI bcs are specified
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
 
  bool usingLODI = false;
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
    map<string,string> face;
    face_ps->getAttributes(face);
    bool is_a_Lodi_face = false;
    
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
      

      if (bc_type["var"] == "LODI" && !is_a_Lodi_face) {
        usingLODI = true;
        is_a_Lodi_face = true;
        // remember which faces are LODI
        if (face["side"] ==  "x-")
          vb->LodiFaces.push_back(Patch::xminus);
        if (face["side"] == "x+")
          vb->LodiFaces.push_back(Patch::xplus);
        if (face["side"] == "y-")
          vb->LodiFaces.push_back(Patch::yminus);
        if (face["side"] == "y+")
          vb->LodiFaces.push_back(Patch::yplus);
        if (face["side"] == "z-")
          vb->LodiFaces.push_back(Patch::zminus);
        if (face["side"] == "z+")
          vb->LodiFaces.push_back(Patch::zplus);
      }
    }
  }
  //__________________________________
  //  read in variables
  if(usingLODI ){
    ProblemSpecP lodi = bc_ps->findBlock("LODI");
    if (!lodi) {
      string warn="ERROR:\n Inputs:Boundary Conditions: Cannot find LODI block";
      throw ProblemSetupException(warn);
    }

    lodi->require("press_infinity",vb->press_infinity);
    lodi->getWithDefault("sigma",  vb->sigma, 0.27);
  }
  
  if (usingLODI) {
    cout << "\n WARNING:  LODI boundary conditions are "
         << " NOT set during the problem initialization \n " 
         << " THESE BOUNDARY CONDITIONS ONLY WORK FOR 1 MATL ICE PROBLEMS \n"
         << " (The material index has been hard coded in preprocess_Lodi_BCs)\n" <<endl;
  }
  return usingLODI;
}

/* ______________________________________________________________________ 
 Function~  addRequires_Lodi--   
 Purpose~   requires for all the tasks depends on which task you're in
 ______________________________________________________________________  */
void addRequires_Lodi(Task* t, 
                      const string& where,
                      ICELabel* lb,
                      const MaterialSubset* ice_matls,
                      Lodi_variable_basket* var_basket)
{
  cout_doing<< "Doing addRequires_Lodi: \t\t" <<t->getName()
            << " " << where << endl;
  
  Ghost::GhostType  gn  = Ghost::None;
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  MaterialSubset* press_matl = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();

  Task::WhichDW whichDW = Task::NewDW;
  bool setLODI_bcs = false;
  
  if(where == "EqPress"){
    setLODI_bcs = true;
    t->requires(Task::OldDW, lb->vel_CCLabel,       ice_matls, gn);
    // requires(Task::NewDW, lb->press_CCLabel,     press_matl,oims,gn, 0);
    // requires(Task::NewDW, lb->rho_CCLabel,       ice_matls, gn); 
    // requires(Task::NewDW, lb->speedSound_CCLabel,ice_matls, gn); 
  }
  if(where == "update_press_CC"){
    setLODI_bcs = true;
    t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,oims,gn, 0);
    t->requires(Task::OldDW, lb->vel_CCLabel,       ice_matls, gn);
    // requires(Task::NewDW, lb->rho_CCLabel,       ice_matls, gn); 
    // requires(Task::NewDW, lb->speedSound_CCLabel,ice_matls, gn);
  }
  if(where == "implicitPressureSolve"){
    setLODI_bcs=true;
  }
   
  if(where == "imp_update_press_CC"){
    setLODI_bcs = true;
    whichDW  = Task::ParentNewDW;
    t->requires(Task::ParentOldDW, lb->vel_CCLabel,        ice_matls, gn);
    t->requires(Task::ParentNewDW, lb->speedSound_CCLabel, ice_matls, gn);
    t->requires(Task::ParentNewDW, lb->rho_CCLabel,        ice_matls, gn);
    t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,oims,gn, 0);
  }
  if(where == "CC_Exchange"){
    setLODI_bcs = true;
    t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,oims,gn, 0);
    t->requires(Task::NewDW, lb->rho_CCLabel,       ice_matls, gn);      
    t->requires(Task::NewDW, lb->speedSound_CCLabel,ice_matls, gn);
    
    t->computes(lb->vel_CC_XchangeLabel);
    t->computes(lb->temp_CC_XchangeLabel);
  }
  if(where == "Advection"){
    setLODI_bcs = true;
    t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,oims,gn, 0);
    t->requires(Task::NewDW, lb->gammaLabel,        ice_matls, gn); 
    // requires(Task::NewDW, lb->vel_CCLabel,       ice_matls, gn); 
    // requires(Task::NewDW, lb->rho_CCLabel,       ice_matls, gn); 
    // requires(Task::NewDW, lb->speedSound_CCLabel,ice_matls, gn);
  }
  //__________________________________
  //   All tasks Lodi faces require(maxMach_<face>)
  if(setLODI_bcs){
    vector<Patch::FaceType>::iterator f ;
    for( f = var_basket->LodiFaces.begin();
         f !=var_basket->LodiFaces.end(); ++f) {
      VarLabel* V_Label = getMaxMach_face_VarLabel(*f);
      t->requires(whichDW,V_Label, ice_matls);
    }
  }  
}
/*______________________________________________________________________ 
 Function~  preprocess_Lodi_BCs-- 
 Purpose~   get data from dw, compute Li
______________________________________________________________________ */
void  preprocess_Lodi_BCs(DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          ICELabel* lb,
                          const Patch* patch,
                          const string& where,
                          const int /*indx*/,
                          SimulationStateP& sharedState,
                          bool& setLodiBcs,
                          Lodi_vars* lv,
                          Lodi_variable_basket* var_basket)
{
  cout_doing << "preprocess_Lodi_BCs on patch "<<patch->getID()<< endl;
  Ghost::GhostType  gn  = Ghost::None;
/*`==========TESTING==========*/
  int indx = 0;                 // ICE MATL IS HARD CODED TO 0
/*===========TESTING==========`*/
  //__________________________________
  //    Equilibration pressure
  if(where == "EqPress"){
    setLodiBcs = true; 
    old_dw->get(lv->vel_CC,    lb->vel_CCLabel,        indx,patch,gn,0);
    new_dw->get(lv->press_CC,  lb->press_equil_CCLabel, 0,   patch,gn,0);
    new_dw->get(lv->rho_CC,    lb->rho_CCLabel,        indx,patch,gn,0);
    new_dw->get(lv->speedSound,lb->speedSound_CCLabel, indx,patch,gn,0);
  }
  if(where == "EqPressMPMICE"){
    setLodiBcs = false;
  }
  //__________________________________
  //    FC exchange
  if(where == "velFC_Exchange"){
    setLodiBcs = false;
    // require(maxMach_face_varlabel);
  }
  //__________________________________
  //    update pressure
  if(where == "update_press_CC"){ 
    setLodiBcs = true;
    old_dw->get(lv->vel_CC,     lb->vel_CCLabel,        indx,patch,gn,0);
    new_dw->get(lv->press_CC,   lb->press_CCLabel,      0,   patch,gn,0);  
    new_dw->get(lv->rho_CC,     lb->rho_CCLabel,        indx,patch,gn,0);
    new_dw->get(lv->speedSound, lb->speedSound_CCLabel, indx,patch,gn,0); 
  }
  //__________________________________
  //    cc_ Exchange
  if(where == "CC_Exchange"){
    setLodiBcs = true;
    new_dw->get(lv->vel_CC,     lb->vel_CC_XchangeLabel,indx,patch,gn,0);
    new_dw->get(lv->press_CC,   lb->press_CCLabel,      0,   patch,gn,0);  
    new_dw->get(lv->rho_CC,     lb->rho_CCLabel,        indx,patch,gn,0);
    new_dw->get(lv->gamma,      lb->gammaLabel,         indx,patch,gn,0);
    new_dw->get(lv->speedSound, lb->speedSound_CCLabel, indx,patch,gn,0);
  }
  //__________________________________
  //    Advection
  if(where == "Advection"){
    setLodiBcs = true;
    new_dw->get(lv->rho_CC,    lb->rho_CCLabel,        indx,patch,gn,0); 
    new_dw->get(lv->vel_CC,    lb->vel_CCLabel,        indx,patch,gn,0);
    new_dw->get(lv->speedSound,lb->speedSound_CCLabel, indx,patch,gn,0); 
    new_dw->get(lv->gamma,     lb->gammaLabel,         indx,patch,gn,0); 
    new_dw->get(lv->press_CC,  lb->press_CCLabel,      0,   patch,gn,0); 
  }
  
  //__________________________________
  //compute Li at boundary cells
  if(setLodiBcs){
    for (int i = 0; i <= 5; i++){ 
      new_dw->allocateTemporary(lv->Li[i], patch);
      lv->Li[i].initialize(Vector(-9e30,-9e30,-9e30));
    }

    computeLi(lv->Li, lv->rho_CC,  lv->press_CC, lv->vel_CC, lv->speedSound, 
              patch, new_dw, sharedState, var_basket, false);
  }
}

/* ______________________________________________________________________ 
 Function~  getMaxMach_face_VarLabel--   
 Purpose~   returns varLabel for maxMach_<xminus.....zplus>
 ______________________________________________________________________  */
VarLabel* getMaxMach_face_VarLabel( Patch::FaceType face)
{
  string labelName = "maxMach_" + Patch::getFaceName(face);
  VarLabel* V_Label = VarLabel::find(labelName); 
  if (V_Label == NULL){
    throw InternalError("Label " + labelName+ " doesn't exist");
  }
  return V_Label;
}

/* ______________________________________________________________________ 
 Function~  Lodi_maxMach_patchSubset--   
 Purpose~   The reduction variables maxMach_<xminus, xplus....>
            need a patchSubset for each face.
 ______________________________________________________________________  */
void Lodi_maxMach_patchSubset(const LevelP& level,
                               SimulationStateP& sharedState,
                               vector<PatchSubset*> & maxMach_patchSubset)
{
  cout_doing << "Lodi_maxMach_patchSubset "<< endl;
  //__________________________________
  // Iterate over all patches on this levels
  vector<const Patch*> p[Patch::numFaces];
  for(Level::const_patchIterator iter = level->patchesBegin();
                                 iter != level->patchesEnd(); iter++){
    const Patch* patch = *iter;
    
    //_________________________________
    // Iterate over just the boundary faces
    vector<Patch::FaceType>::const_iterator itr;
    for (itr  = patch->getBoundaryFaces()->begin(); 
         itr != patch->getBoundaryFaces()->end(); ++itr){
      Patch::FaceType face = *itr;
      //__________________________________
      //  if Lodi face then keep track of the patch
      if (is_LODI_face(patch,face, sharedState) ) {
        p[face].push_back(patch);
      }
    }
  }
  //__________________________________
  // now put each patch into a patchSubsets
  for (int f = 0; f<Patch::numFaces; f++) {
    PatchSubset* subset = scinew PatchSubset(p[f].begin(), p[f].end());
    subset->addReference();
    maxMach_patchSubset[f]=subset;
  } 
}

/* ______________________________________________________________________ 
 Function~  is_LODI_face--   
 Purpose~   returns true if this face on this patch is using LODI bcs
 ______________________________________________________________________  */
bool is_LODI_face(const Patch* patch,
                  Patch::FaceType face,
                  SimulationStateP& sharedState)
{
  bool is_lodi_face = false;
  int numMatls = sharedState->getNumICEMatls();

  for (int m = 0; m < numMatls; m++ ) {
    ICEMaterial* ice_matl = sharedState->getICEMaterial(m);
    int indx= ice_matl->getDWIndex();
    bool lodi_pressure =    patch->haveBC(face,indx,"LODI","Pressure");
    bool lodi_density  =    patch->haveBC(face,indx,"LODI","Density");
    bool lodi_temperature = patch->haveBC(face,indx,"LODI","Temperature");
    bool lodi_velocity =    patch->haveBC(face,indx,"LODI","Velocity");

    if (lodi_pressure || lodi_density || lodi_temperature || lodi_velocity) {
      is_lodi_face = true; 
    }
  }
  return is_lodi_face;
}


/*__________________________________________________________________
 Function~ Sutherland_Vector_Components-
 Purpose:  Returns an IntVector filled with the vector components 
           from Sutherland and Kennedy's table 4 and 5
____________________________________________________________________*/
inline IntVector Sutherland_Vector_Components(const Patch::FaceType face) 
{
  IntVector dir(0,0,0);
  if (face == Patch::xminus || face == Patch::xplus ) {
    dir = IntVector(0,1,2);
  }
  if (face == Patch::yminus || face == Patch::yplus ) {
    dir = IntVector(1,0,2);
  }
  if (face == Patch::zminus || face == Patch::zplus ) {
    dir = IntVector(2,0,1);
  }
  return dir;
}
/*__________________________________________________________________
 Function~ oneSidedDifference_offsets-
 Purpose: utility function that computes offsets for 
          one-sided differencing
____________________________________________________________________*/
void oneSidedDifference_offsets(const Patch::FaceType face,
                                 const int P_dir,
                                 IntVector R_offset,
                                 IntVector L_offset) 
{
  R_offset= IntVector(0,0,0);
  L_offset= IntVector(0,0,0);     //  find the one sided derivative offsets

  if (face == Patch::xminus || face == Patch::yminus || face == Patch::zminus){
    R_offset[P_dir] += 1; 
  }
  if (face == Patch::xplus || face == Patch::yplus || face == Patch::zplus){
    L_offset[P_dir] -= 1;
  }
}
/*__________________________________________________________________
 Function~ characteristic_source_terms-
 Note:     Ignore all body force sources except those normal
           to the boundary face.
____________________________________________________________________*/
inline void characteristic_source_terms(const IntVector dir,
                                        const int P_dir,
                                        const Vector grav,
                                        const double rho_CC,
                                        const double speedSound,
                                        vector<double>& s)
{
  Vector s_mom = Vector(0,0,0);
  s_mom[P_dir] = grav[P_dir];
  double s_press = 0.0;
  //__________________________________
  //  compute sources, Appendix:Table 4
  // x_dir:  dir = (0,1,2) 
  // y_dir:  dir = (1,0,2)  
  // z_dir:  dir = (2,0,1)
  s[1] = 0.5 * (s_press - rho_CC * speedSound * s_mom[dir[0]] );
  s[2] = -s_press/(speedSound * speedSound);
  s[3] = s_mom[dir[1]];
  s[4] = s_mom[dir[2]];
  s[5] = 0.5 * (s_press + rho_CC * speedSound * s_mom[dir[0]] );
}


/*__________________________________________________________________
 Function~ debugging_Di-- this goes through the same logic as
           Di and dumps out data
____________________________________________________________________*/
void debugging_Li(const IntVector c,
                  const vector<double>& s,
                  const IntVector dir,
                  const Patch::FaceType face,
                  const double speedSound,
                  const Vector vel_CC,
                  double L1,
                  double L2, 
                  double L3,
                  double L4,
                  double L5 )
{
  int n_dir = dir[0];
  double normalVel = vel_CC[n_dir];
  double Mach = fabs(vel_CC.length()/speedSound);
   cout.setf(ios::scientific,ios::floatfield);
   cout.precision(10);  
  //__________________________________
  //right or left boundary
  // inflow or outflow
  double rightFace =1;
  double leftFace = 0;
  if (face == Patch::xminus || face == Patch::yminus || face == Patch::zminus){
    leftFace  = 1;
    rightFace = 0;
  }
  
  string flowDir = "outFlow";
  if (leftFace && normalVel >= 0 || rightFace && normalVel <= 0){
    flowDir = "inFlow";
  }
  cout << " \n ----------------- " << c << endl;
  cout << " default values " << endl;
  string s_A  = " A = rho * speedSound * dVel_dx[n_dir]; ";
  string s_L1 = " 0.5 * (normalVel - speedSound) * (dp_dx - A) ";
  string s_L2 = " normalVel * (drho_dx - dp_dx/speedSoundsqr) ";
  string s_L3 = " normalVel * dVel_dx[dir[1]];  ";
  string s_L4 = " normalVel * dVel_dx[dir[2]];  ";
  string s_L5 = " 0.5 * (normalVel + speedSound) * (dp_dx + A); \n";
  
  cout << "s[1] = "<< s[1] << "  0.5 * (s_press - rho_CC * speedSound * s_mom[dir[0]] ) "<< endl;
  cout << "s[2] = "<< s[2] << "  -s_press/(speedSound * speedSound);                     " << endl;
  cout << "s[3] = "<< s[3] << "  s_mom[dir[1]];                                          " << endl;
  cout << "s[4] = "<< s[4] << "  s_mom[dir[2]];                                          " << endl;
  cout << "s[5] = "<< s[5] << "  0.5 * (s_press + rho_CC * speedSound * s_mom[dir[0]] )  " << endl;
  cout << "\n"<< endl;
  //__________________________________
  //Subsonic non-reflective inflow
  if (flowDir == "inFlow" && Mach < 1.0){
    cout << " SUBSONIC INFLOW:  rightFace " <<rightFace << " leftFace " << leftFace<< endl;
    cout << "L1 = leftFace * L1 + rightFace * s[1]  " << L1 << endl;
    cout << "L2 = s[2];                             " << L2 << endl;
    cout << "L3 = s[3];                             " << L3 << endl;
    cout << "L4 = s[4];                             " << L4 << endl;
    cout << "L5 = rightFace * L5 + leftFace * s[5]; " << L5 << endl;
  }
  //__________________________________
  // Subsonic non-reflective outflow
  if (flowDir == "outFlow" && Mach < 1.0){
    cout << " SUBSONIC OUTFLOW:  rightFace " <<rightFace << " leftFace " << leftFace << endl;
    cout << "L1   "<< L1 
         << "  rightFace *(0.5 * K * (press - p_infinity)/domainLength[n_dir] + s[1]) + leftFace  * L1 " << endl;
    cout << "L2   " << L2 <<" " << s_L2 << endl;
    cout << "L3   " << L3 <<" " << s_L3 << endl;
    cout << "L4   " << L4 <<" " << s_L4 << endl;
    cout << "L5   " << L5
         << "  leftFace  *(0.5 * K * (press - p_infinity)/domainLength[n_dir] + s[5]) rightFace * L5 " << endl;
  }
  
  //__________________________________
  //Supersonic non-reflective inflow
  // see Thompson II pg 453
  if (flowDir == "inFlow" && Mach > 1.0){
    cout << " SUPER SONIC inflow " << endl;
    cout << " L1 = s[1] " << L1 << endl;
    cout << " L2 = s[2] " << L2 << endl;
    cout << " L3 = s[3] " << L3 << endl;
    cout << " L4 = s[4] " << L4 << endl;
    cout << " L5 = s[5] " << L5 << endl;
  }
  //__________________________________
  // Supersonic non-reflective outflow
  //if (flowDir == "outFlow" && Mach > 1.0){
  //}   do nothing see Thompson II pg 453
}

/*__________________________________________________________________
 Function~ Li-- do the actual work of computing di
 Reference:  "Improved Boundary conditions for viscous, reacting compressible
              flows", James C. Sutherland, Chistopher A. Kenndey
              Journal of Computational Physics, 191, 2003, pp. 502-524
____________________________________________________________________*/
inline void Li(StaticArray<CCVariable<Vector> >& L,
               const IntVector dir,
               const IntVector& c,
               const Patch::FaceType face,
               const Vector domainLength,
               const Lodi_variable_basket* user_inputs,
               const double maxMach,
               const vector<double>& s,
               const double press,
               const double speedSound,
               const double rho,
               const Vector vel_CC,
               const double drho_dx,
               const double dp_dx,
               const Vector dVel_dx)
{ 
  //__________________________________
  // compute Li terms
  //  see table 5 of Sutherland and Kennedy 
  // x_dir:  dir = (0,1,2) 
  // y_dir:  dir = (1,0,2)  
  // z_dir:  dir = (2,0,1)
  
  int n_dir = dir[0];
  double normalVel = vel_CC[n_dir];
  double Mach = fabs(vel_CC.length()/speedSound);
  
  double speedSoundsqr = speedSound * speedSound;
  
  //__________________________________
  // default Li terms
  double A = rho * speedSound * dVel_dx[n_dir];
  double L1 = 0.5 * (normalVel - speedSound) * (dp_dx - A);
  double L2 = normalVel * (drho_dx - dp_dx/speedSoundsqr);
  double L3 = normalVel * dVel_dx[dir[1]];  // u dv/dx
  double L4 = normalVel * dVel_dx[dir[2]];  // u dw/dx
  double L5 = 0.5 * (normalVel + speedSound) * (dp_dx + A);
  
  #if 0
  cout << c << " default L1 " << L1 << " L5 " << L5
     << " dVel_dx " << dVel_dx[n_dir]
     << " dp_dx " << dp_dx << endl;  
  #endif
  //__________________________________
  //  user_inputs
  double p_infinity = user_inputs->press_infinity;
  double sigma      = user_inputs->sigma;
  
  //____________________________________________________________
  //  Modify the Li terms based on whether or not the normal
  //  component of the velocity if flowing out of the domain.
  //  Equation 8 & 9 of Sutherland
  double K =  sigma * speedSound *(1.0 - maxMach*maxMach);
  
  //__________________________________
  //right or left boundary
  // inflow or outflow
  double rightFace =1;
  double leftFace = 0;
  if (face == Patch::xminus || face == Patch::yminus || face == Patch::zminus){
    leftFace  = 1;
    rightFace = 0;
  }
  
  string flowDir = "outFlow";
  if (leftFace && normalVel >= 0 || rightFace && normalVel <= 0){
    flowDir = "inFlow";
  }
  
  //__________________________________
  //Subsonic non-reflective inflow
  if (flowDir == "inFlow" && Mach < 1.0){
    L1 = leftFace * L1 + rightFace * s[1];
    L2 = s[2];
    L3 = s[3];
    L4 = s[4];
    L5 = rightFace * L5 + leftFace * s[5]; 
  }
  //__________________________________
  // Subsonic non-reflective outflow
  if (flowDir == "outFlow" && Mach < 1.0){
  
    L1 = rightFace *(0.5 * K * (press - p_infinity)/domainLength[n_dir] + s[1])
       + leftFace  * L1;
              
    L5 = leftFace  *(0.5 * K * (press - p_infinity)/domainLength[n_dir] + s[5])
       + rightFace * L5;
  }
  
  //__________________________________
  //Supersonic non-reflective inflow
  // see Thompson II pg 453
  if (flowDir == "inFlow" && Mach > 1.0){
    L1 = s[1];
    L2 = s[2];
    L3 = s[3];
    L4 = s[4];
    L5 = s[5]; 
  }
  //__________________________________
  // Supersonic non-reflective outflow
  //if (flowDir == "outFlow" && Mach > 1.0){
  //}   do nothing see Thompson II pg 453
  
  //__________________________________
  //  compute Di terms in the normal direction based on the 
  // modified Ls  See Sutherland Table 7
  L[1][c][n_dir] = L1;
  L[2][c][n_dir] = L2;
  L[3][c][n_dir] = L3;
  L[4][c][n_dir] = L4;
  L[5][c][n_dir] = L5;

  
#if 0
   //__________________________________
   //  debugging 
   vector<IntVector> dbgCells;
   dbgCells.push_back(IntVector(100,0,0));
   dbgCells.push_back(IntVector(99,0,0));
   dbgCells.push_back(IntVector(0,0,0));
   dbgCells.push_back(IntVector(-1,0,0));
           
   for (int i = 0; i<(int) dbgCells.size(); i++) {
     if (c == dbgCells[i]) {
      debugging_Li(c, s, dir, face, speedSound, vel_CC, L1, L2, L3, L4, L5 );
      cout << " press " << press << " p_infinity " << p_infinity
           << " gradient " << (press - p_infinity)/domainLength[n_dir] << endl;
     }  // if(dbgCells)
   }  // dbgCells loop
 #endif
} 

/*__________________________________________________________________
 Function~ computeLi--
 Purpose~  compute Li's at one cell inward using upwind first-order 
           differenceing scheme
____________________________________________________________________*/
void computeLi(StaticArray<CCVariable<Vector> >& L,
               const CCVariable<double>& rho,              
               const CCVariable<double>& press,                   
               const CCVariable<Vector>& vel,                  
               const CCVariable<double>& speedSound,              
               const Patch* patch,
               DataWarehouse* new_dw,
               SimulationStateP& sharedState,
               const Lodi_variable_basket* user_inputs,
               const bool recursion)                              
{
  cout_doing << "LODI computeLi "<< endl;
  Vector dx = patch->dCell();
  
  // Characteristic Length of the overall domain
  Vector domainLength;
  const Level* level = patch->getLevel();
  GridP grid = level->getGrid();
  grid->getLength(domainLength, "minusExtraCells");
  
  Vector grav = sharedState->getGravity();

  for (int i = 1; i<= 5; i++ ) {           // don't initialize inside main loop
    L[i].initialize(Vector(0.0,0.0,0.0));  // you'll overwrite previously compute LI
  }
  
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    
    if (is_LODI_face(patch,face, sharedState) ) {
      cout_dbg << " computing LI on face " << face 
               << " patch " << patch->getID()<<endl;
      //_____________________________________
      // S I D E S
      IntVector axes = patch->faceAxes(face);
      int P_dir = axes[0]; // find the principal dir
      double delta = dx[P_dir];

      IntVector R_offset(0,0,0);
      IntVector L_offset(0,0,0);     //  find the one sided derivative offsets

      if (face == Patch::xminus || face == Patch::yminus || face == Patch::zminus){
        R_offset[P_dir] += 1; 
      }
      if (face == Patch::xplus || face == Patch::yplus || face == Patch::zplus){
        L_offset[P_dir] -= 1;
      }

      IntVector dir = Sutherland_Vector_Components(face);
      
      // get the maxMach for that face
      DataWarehouse* pNewDW;
      if(recursion) {
        pNewDW  = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      } else {
        pNewDW  = new_dw;
      }
    
      VarLabel* V_Label = getMaxMach_face_VarLabel(face);
      max_vartype maxMach;
      pNewDW->get(maxMach,   V_Label);
      
      IntVector offset = patch->faceDirection(face);
      
      //__________________________________
      //  compute Li one cell in from
      // the face
      for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
          !iter.done();iter++) {
        IntVector c = *iter - offset;
        IntVector r = c + R_offset;
        IntVector l = c + L_offset;

        double drho_dx = (rho[r]   - rho[l])/delta; 
        double dp_dx   = (press[r] - press[l])/delta;
        Vector dVel_dx = (vel[r]   - vel[l])/delta;

        vector<double> s(6);
        characteristic_source_terms(dir, P_dir, grav, rho[c], speedSound[c], s);

        Li(L, dir, c, face, domainLength, user_inputs, maxMach, s, press[c],
           speedSound[c], rho[c], vel[c], drho_dx, dp_dx, dVel_dx); 

      }  // faceCelliterator 
    }  // is Lodi face
  }  // loop over faces
}// end of function
   
  
/*_________________________________________________________________
 Function~ getBoundaryEdges--
 Purpose~  returns a list of edges where boundary conditions
           should be applied.
___________________________________________________________________*/  
void getBoundaryEdges(const Patch* patch,
                      const Patch::FaceType face,
                      vector<Patch::FaceType>& face0)
{
  IntVector patchNeighborLow  = patch->neighborsLow();
  IntVector patchNeighborHigh = patch->neighborsHigh();
  
  //__________________________________
  // Looking down on the face, examine 
  // each edge (clockwise).  If there
  // are no neighboring patches then it's a valid edge
  IntVector axes = patch->faceAxes(face);
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];   
  IntVector minus(Patch::xminus, Patch::yminus, Patch::zminus);
  IntVector plus(Patch::xplus,   Patch::yplus,  Patch::zplus);
  
  if ( patchNeighborLow[dir1] == 1 ){
   face0.push_back(Patch::FaceType(minus[dir1]));
  }  
  if ( patchNeighborHigh[dir1] == 1 ) {
   face0.push_back(Patch::FaceType(plus[dir1]));
  }
  if ( patchNeighborLow[dir2] == 1 ){
   face0.push_back(Patch::FaceType(minus[dir2]));
  }  
  if ( patchNeighborHigh[dir2] == 1 ) {
   face0.push_back(Patch::FaceType(plus[dir2]));
  }
}

/*_________________________________________________________________
 Function~ remainingVectorComponent--
 Purpose~  returns the remaining vector component.
___________________________________________________________________*/  
int remainingVectorComponent(int dir1, int dir2)
{ 
  int dir3 = -9;
  if ((dir1 == 0 && dir2 == 1) || (dir1 == 1 && dir2 == 0) ){  // x, y
    dir3 = 2; // z
  }
  if ((dir1 == 0 && dir2 == 2) || (dir1 == 2 && dir2 == 0) ){  // x, z
    dir3 = 1; // y
  }
  if ((dir1 == 1 && dir2 == 2) || (dir1 == 2 && dir2 == 1) ){   // y, z
    dir3 = 0; //x
  }
  return dir3;
}
/*_________________________________________________________________
 Function~ FaceDensity_LODI--
 Purpose~  Compute density in boundary cells on any face
___________________________________________________________________*/
void FaceDensity_LODI(const Patch* patch,
                const Patch::FaceType face,
                CCVariable<double>& rho_CC,
                Lodi_vars* lv,
                const Vector& DX)
{
  cout_doing << "Setting FaceDensity_LODI on face " << face<<endl;
  // bulletproofing
  if (!lv){
    throw InternalError("FaceDensityLODI: Lodi_vars = null");
  }  
  
  StaticArray<CCVariable<Vector> >& L = lv->Li;
  constCCVariable<double>& speedSound = lv->speedSound;
  constCCVariable<Vector>& vel_CC     = lv->vel_CC;  
  
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  
  IntVector offset = patch->faceDirection(face);
  double plus_minus_one = (double) offset[P_dir];
  double dx = DX[P_dir];
  
  cout_dbg << "\n____________________density"<< endl;
  //__________________________________
  //    S I D E
  for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
                                                      !iter.done();iter++) {
    IntVector c = *iter;
    IntVector in = c - offset;
    
    double term1 = L[2][in][P_dir]/(vel_CC[in][P_dir] + d_SMALL_NUM);
    double term2 = L[5][in][P_dir]/(vel_CC[in][P_dir] + speedSound[in]);
    double term3 = L[1][in][P_dir]/(vel_CC[in][P_dir] - speedSound[in]);
    double drho_dx = term1 + (term2 + term3)/(speedSound[in] * speedSound[in]); 
    rho_CC[c] = rho_CC[in] + plus_minus_one * dx * drho_dx;
    
    cout_dbg << " c " << c << " in " << in << " rho_CC[c] "<< rho_CC[c] 
             << " drho_dx " << drho_dx << " rho_CC[in] " << rho_CC[in]<<endl;
  }
  
  
  //__________________________________
  //    E D G E S  -- on boundaryFaces only
  vector<Patch::FaceType> b_faces;
  getBoundaryEdges(patch,face,b_faces);
  
  vector<Patch::FaceType>::const_iterator iter;  
  for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
    Patch::FaceType face0 = *iter;
    
    IntVector offset = IntVector(0,0,0)  - patch->faceDirection(face) 
                                         - patch->faceDirection(face0);
    CellIterator iterLimits =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");
                
    for(CellIterator iter = iterLimits;!iter.done();iter++){ 
      IntVector c = *iter;      
      rho_CC[c] = rho_CC[c + offset];
    }
  }

  //__________________________________
  // C O R N E R S    
  const vector<IntVector> corner = patch->getCornerCells(face);
  vector<IntVector>::const_iterator itr;
  
  for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
    IntVector c = *itr;
    rho_CC[c] =  1.7899909957225715000;
  }      
}

/*_________________________________________________________________
 Function~ FaceVel_LODI--
 Purpose~  Compute velocity in boundary cells on face
___________________________________________________________________*/
void FaceVel_LODI(const Patch* patch,
                 Patch::FaceType face,
                 CCVariable<Vector>& vel_CC,                 
                 Lodi_vars* lv,
                 const Vector& DX,
                 SimulationStateP& sharedState)                     

{
  cout_doing << "Setting FaceVel_LODI on face " << face << endl;
  // bulletproofing
  if (!lv){
    throw InternalError("FaceVelLODI: Lodi_vars = null");
  }
     
  // shortcuts       
  StaticArray<CCVariable<Vector> >& L = lv->Li;      
  constCCVariable<double>& rho_CC     = lv->rho_CC;
  constCCVariable<double>& speedSound = lv->speedSound;

  IntVector dir= Sutherland_Vector_Components(face);                 
  int P_dir = dir[0];  // principal direction
  int dir1  = dir[1];  // transverse
  int dir2  = dir[2];  // transvers
  
  IntVector offset = patch->faceDirection(face);
  double plus_minus_one = (double) offset[P_dir];
  double dx = DX[P_dir];


  cout_dbg << "____________________velocity"<< endl;
  //__________________________________
  //    S I D E 
  for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
                                                      !iter.done();iter++) {
    IntVector c = *iter;
    IntVector in = c - offset;
    // normal direction velocity
    double term1 = L[5][in][P_dir]/(vel_CC[in][P_dir] + speedSound[in]);
    double term2 = L[1][in][P_dir]/(vel_CC[in][P_dir] - speedSound[in]);
    double dvel_norm_dx = (1.0/(rho_CC[in] * speedSound[in]) )* (term1 - term2);
    vel_CC[c][P_dir] = vel_CC[in][P_dir] + plus_minus_one * dx * dvel_norm_dx; 
    
    // transverse velocities
    double dvel_dir1_dx = L[3][in][P_dir]/(vel_CC[in][P_dir] + d_SMALL_NUM);
    double dvel_dir2_dx = L[4][in][P_dir]/(vel_CC[in][P_dir] + d_SMALL_NUM);
    
    vel_CC[c][dir1] = vel_CC[in][dir1] + plus_minus_one * dx * dvel_dir1_dx;
    vel_CC[c][dir2] = vel_CC[in][dir2] + plus_minus_one * dx * dvel_dir2_dx;
   
    //__________________________________
    //  debugging
    Vector vel, dvel;
    vel[P_dir] = vel_CC[c][P_dir];
    vel[dir1]  = vel_CC[c][dir1];
    vel[dir2]  = vel_CC[c][dir2];
    
    dvel[P_dir] = dvel_norm_dx;
    dvel[dir1]  = dvel_dir1_dx;
    dvel[dir2]  = dvel_dir2_dx;
    
    cout_dbg << " c " << c << " in " << in << " vel " << vel << endl;
    cout_dbg << " dvel_ " << dvel << endl;
  
  }
#if 1
  //__________________________________
  //    E D G E S  -- on boundaryFaces only
  vector<Patch::FaceType> b_faces;
  getBoundaryEdges(patch,face,b_faces);
  
  vector<Patch::FaceType>::const_iterator iter;  
  for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
    Patch::FaceType face0 = *iter;
   
    IntVector offset = IntVector(0,0,0)  - patch->faceDirection(face) 
                                         - patch->faceDirection(face0);
    CellIterator iterLimits =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");
                      
    for(CellIterator iter = iterLimits;!iter.done();iter++){ 
      IntVector c = *iter;
      vel_CC[c] = vel_CC[c + offset];
    }
  }  
  //________________________________________________________
  // C O R N E R S
  const vector<IntVector> corner = patch->getCornerCells(face);
  vector<IntVector>::const_iterator itr;
  
  for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
    IntVector c = *itr;
    vel_CC[c] = Vector(0,0,0);
  }
#endif
} //end of the function FaceVelLODI() 

/*_________________________________________________________________
 Function~ FaceTemp_LODI--
 Purpose~  Compute temperature in boundary cells on faces
___________________________________________________________________*/
void FaceTemp_LODI(const Patch* patch,
             const Patch::FaceType face,
             CCVariable<double>& temp_CC,
             Lodi_vars* lv, 
             const Vector& DX,
             SimulationStateP& sharedState)
{
  cout_doing << "Setting FaceTemp_LODI on face " <<face<< endl; 
  
  // bulletproofing
  if (!lv){
    throw InternalError("FaceTempLODI: Lodi_vars = null");
  } 
  // shortcuts  
  StaticArray<CCVariable<Vector> >& L = lv->Li;
  constCCVariable<double>& speedSound= lv->speedSound;
  constCCVariable<double>& gamma     = lv->gamma;
  constCCVariable<double>& rho_CC    = lv->rho_CC;
  constCCVariable<Vector>& vel_CC   = lv->vel_CC;
              
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  double dx = DX[P_dir];
  
  IntVector offset = patch->faceDirection(face);
  double plus_minus_one = (double) offset[P_dir];
  cout_dbg << "\n____________________Temp"<< endl;
  
  //__________________________________
  //    S I D E     
  for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
                                                      !iter.done();iter++) {
    IntVector c = *iter;
    IntVector in = c - offset;
    double term1 = temp_CC[in]/(rho_CC[in] * speedSound[in] * speedSound[in]);
    double term2 = L[2][in][P_dir]/(vel_CC[in][P_dir] + d_SMALL_NUM);
    double term3 = ( gamma[in] - 1.0);
    double term4 = L[5][in][P_dir]/(vel_CC[in][P_dir] + speedSound[in]);
    double term5 = L[1][in][P_dir]/(vel_CC[in][P_dir] - speedSound[in]);
    double dtemp_dx = term1 * (-term2 + term3*(term4 + term5) );
    temp_CC[c] = temp_CC[in] + plus_minus_one * dx * dtemp_dx;
   
    cout_dbg << " c " << c << " in " << in << " temp_CC[c] "<< temp_CC[c]
             << " temp_CC[in] " << temp_CC[in] << endl;
     
    cout_dbg << " term1 " << term1
             << " term2 " << term2
             << " term3 " << term3
             << " term4 " << term4
             << " term5 " << term5
             << " dtemp_dx " << dtemp_dx << endl;
  }

  //__________________________________
  //    E D G E S  -- on boundaryFaces only
  vector<Patch::FaceType> b_faces;
  getBoundaryEdges(patch,face,b_faces);
  
  vector<Patch::FaceType>::const_iterator iter;  
  for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
    Patch::FaceType face0 = *iter;
    IntVector offset = IntVector(0,0,0)  - patch->faceDirection(face) 
                                         - patch->faceDirection(face0); 
     
    CellIterator iterLimits =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");
             
    for(CellIterator iter = iterLimits;!iter.done();iter++){ 
      IntVector c = *iter;
      temp_CC[c] = temp_CC[c+offset]; 
    }
  }  
 
  //________________________________________________________
  // C O R N E R S    
  const vector<IntVector> corner = patch->getCornerCells(face);
  vector<IntVector>::const_iterator itr;
  
  for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
    IntVector c = *itr;
    temp_CC[c] = 300;
  }

} //end of function FaceTempLODI()  


/* ______________________________________________________________________ 
 Function~  FacePress_LODI--
 Purpose~   Back out the pressure See Table 6 of Sutherland and Kennedy
______________________________________________________________________  */
void FacePress_LODI(const Patch* patch,
                    CCVariable<double>& press_CC,
                    StaticArray<CCVariable<double> >& rho_micro,
                    SimulationStateP& sharedState, 
                    Patch::FaceType face,
                    Lodi_vars* lv)
{
  cout_doing << " I am in FacePress_LODI on face " <<face<< endl;
  // bulletproofing
  if (!lv){
    throw InternalError("FacePress_LODI: Lodi_vars = null");
  }

  StaticArray<CCVariable<Vector> >& L = lv->Li;
 
  Vector DX =patch->dCell();
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  
  IntVector offset = patch->faceDirection(face);
  double plus_minus_one = (double) offset[P_dir];
  double dx = DX[P_dir];
 
  cout_dbg << "\n____________________press"<< endl; 
  
  //__________________________________ 
  for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
                                                    !iter.done();iter++) {

    IntVector c = *iter;
    IntVector in = c - offset;  
    
    double term1 = L[5][in][P_dir]/(lv->vel_CC[in][P_dir] + lv->speedSound[in]);
    double term2 = L[1][in][P_dir]/(lv->vel_CC[in][P_dir] - lv->speedSound[in]);
    double dpress_dx = term1 + term2;

    press_CC[c] = press_CC[in] + plus_minus_one * dx * dpress_dx; 
        
    cout_dbg << " c " << c << " in " << in << " press_CC[c] "<< press_CC[c] 
             << " dpress_dx " << dpress_dx << " press_CC[in] " << press_CC[in]<<endl;

  }
  //__________________________________
  //    E D G E S  -- on boundaryFaces only
  vector<Patch::FaceType> b_faces;
  getBoundaryEdges(patch,face,b_faces);
  
  vector<Patch::FaceType>::const_iterator iter;  
  for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
    Patch::FaceType face0 = *iter;
    IntVector offset = IntVector(0,0,0)  - patch->faceDirection(face) 
                                         - patch->faceDirection(face0); 
     
    CellIterator iterLimits =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");
             
    for(CellIterator iter = iterLimits;!iter.done();iter++){ 
      IntVector c = *iter;
      press_CC[c] = press_CC[c+offset]; 
    }
  }  
 
  //________________________________________________________
  // C O R N E R S    
  const vector<IntVector> corner = patch->getCornerCells(face);
  vector<IntVector>::const_iterator itr;
  
  for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
    IntVector c = *itr;
    press_CC[c] = 101325;
  }
} 
   
}  // using namespace Uintah
