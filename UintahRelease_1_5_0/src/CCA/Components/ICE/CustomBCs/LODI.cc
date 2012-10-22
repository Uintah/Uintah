/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/ICE/CustomBCs/LODI.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/EOS/EquationOfState.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <Core/Math/MiscMath.h>
#include <typeinfo>

using namespace Uintah;
namespace Uintah {
//__________________________________
//  To turn on couts
//  setenv SCI_DEBUG "LODI_DOING_COUT:+, LODI_DBG_COUT:+"
static DebugStream cout_doing("LODI_DOING_COUT", false);
static DebugStream cout_dbg("LODI_DBG_COUT", false);

/* ---------------------------------------------------------------------
 Function~  read_LODI_BC_inputs--   
 Purpose~   returns if we are using LODI BC on any face,
            reads in any lodi parameters 
            sets which boundaries are lodi
 ---------------------------------------------------------------------  */
bool read_LODI_BC_inputs(const ProblemSpecP& prob_spec,
                         Lodi_variable_basket* userInputs)
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
    
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
      
      if (bc_type["var"] == "LODI") {
       usingLODI = true;
      }
    }
  }
  //__________________________________
  //  read in variables
  if(usingLODI ){
    ProblemSpecP lodi = bc_ps->findBlock("LODI");
    if (!lodi) {
      string warn="ERROR:\n Inputs:Boundary Conditions: Cannot find LODI block";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }

    lodi->require("press_infinity",userInputs->press_infinity);
    lodi->getWithDefault("sigma",  userInputs->sigma, 0.27);
  }
  
  if (usingLODI) {
    cout << "\n WARNING:  LODI boundary conditions are "
         << " NOT set during the problem initialization \n " << endl;
  }
  return usingLODI;
}
/* --------------------------------------------------------------------- 
 Function~  getMaxMach_face_VarLabel--   
 Purpose~   returns varLabel for maxMach_<xminus.....zplus>
 ---------------------------------------------------------------------  */
VarLabel* getMaxMach_face_VarLabel( Patch::FaceType face)
{
  string labelName = "maxMach_" + Patch::getFaceName(face);
  VarLabel* V_Label = VarLabel::find(labelName); 
  if (V_Label == NULL){
    throw InternalError("Label " + labelName+ " doesn't exist", __FILE__, __LINE__);
  }
  return V_Label;
}

/* --------------------------------------------------------------------- 
 Function~  Lodi_maxMach_patchSubset--   
 Purpose~   The reduction variables maxMach_<xminus, xplus....>
            need a patchSubset for each face.
 ---------------------------------------------------------------------  */
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
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for (itr  = bf.begin(); itr != bf.end(); ++itr){
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

/* --------------------------------------------------------------------- 
 Function~  is_LODI_face--   
 Purpose~   returns true if this face on this patch is using LODI bcs
 ---------------------------------------------------------------------  */
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

/* --------------------------------------------------------------------- 
 Function~  lodi_getVars_pressBC-- 
 Purpose~   Get all the data required for setting the pressure BC
 ---------------------------------------------------------------------  */
void lodi_getVars_pressBC( const Patch* patch,
                           Lodi_vars_pressBC* lodi_vars,
                           ICELabel* lb,
                           SimulationStateP sharedState,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  cout_doing << "lodi_getVars_pressBC on patch "<<patch->getID()<< endl;
  int numMatls = sharedState->getNumMatls();
  StaticArray<constCCVariable<double> > Temp_CC(numMatls);
  StaticArray<constCCVariable<double> > f_theta_CC(numMatls);
  StaticArray<constCCVariable<double> > gamma(numMatls);
  StaticArray<constCCVariable<double> > cv(numMatls);
  Ghost::GhostType  gn = Ghost::None;
  
  for(int m = 0; m < numMatls; m++) {
    Material* matl = sharedState->getMaterial( m );
    int indx = matl->getDWIndex();
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

    if(ice_matl){                // I C E
      old_dw->get(Temp_CC[m],     lb->temp_CCLabel,       indx,patch,gn,0);
      new_dw->get(f_theta_CC[m],  lb->f_theta_CCLabel,    indx,patch,gn,0);
      new_dw->get(gamma[m],       lb->gammaLabel,         indx,patch,gn,0);
      new_dw->get(cv[m],          lb->specific_heatLabel, indx,patch,gn,0);
    }
    if(mpm_matl){                // M P M
      new_dw->get(Temp_CC[m],     lb->temp_CCLabel,    indx,patch,gn,0);
      new_dw->get(f_theta_CC[m],  lb->f_theta_CCLabel, indx,patch,gn,0);
    }
    lodi_vars->f_theta[m] = f_theta_CC[m];
    lodi_vars->Temp_CC[m] = Temp_CC[m];
    lodi_vars->gamma[m]   = gamma[m];
    lodi_vars->cv[m]      = cv[m];
  }
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
void debugging_Di(const IntVector c,
                  StaticArray<CCVariable<Vector> >& d,
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
  
  //__________________________________
  //  compute Di terms in the normal direction based on the 
  // modified Ls  See Sutherland Table 7
    cout << "\nd[1][c]["<<n_dir<<"] = L2 + (L1 + L5)/speedSoundsqr    " << d[1][c][n_dir] << endl;
    cout << "d[2][c]["  <<n_dir<<"] = (L5 + L1);                     " << d[2][c][n_dir] << endl;
 
  if (n_dir == 0) {   // X-normal
    cout << "d[3][c][0] = (L5 - L1)/(rho * speedSound)    " << d[3][c][n_dir] << endl;
    cout << "d[4][c][0] = L3                              " << d[4][c][n_dir] << endl;
    cout << "d[5][c][0] = L4                              " << d[5][c][n_dir] << endl;
  }
  if (n_dir == 1) {   // Y-normal
    cout << "d[3][c][1] = L3                              " << d[3][c][n_dir] << endl;
    cout << "d[4][c][1] = (L5 - L1)/(rho * speedSound)    " << d[4][c][n_dir] << endl;
    cout << "d[5][c][1] = L4;                             " << d[5][c][n_dir] << endl;
  }
  if (n_dir == 2) {   // Z-normal
    cout << "d[3][c][3] = L3;                             " << d[3][c][n_dir] << endl;
    cout << "d[4][c][3] = L4;                             " << d[4][c][n_dir] << endl;
    cout << "d[5][c][3] = (L5 - L1)/(rho * speedSound);   " << d[5][c][n_dir] << endl;
  }
}

/*__________________________________________________________________
 Function~ Di-- do the actual work of computing di
 Reference:  "Improved Boundary conditions for viscous, reacting compressible
              flows", James C. Sutherland, Chistopher A. Kenndey
              Journal of Computational Physics, 191, 2003, pp. 502-524
____________________________________________________________________*/
inline void Di(StaticArray<CCVariable<Vector> >& d,
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
  d[1][c][n_dir] = L2 + (L1 + L5)/speedSoundsqr;
  d[2][c][n_dir] = (L5 + L1);
 
  if (n_dir == 0) {   // X-normal
    d[3][c][n_dir] = (L5 - L1)/(rho * speedSound);
    d[4][c][n_dir] = L3;
    d[5][c][n_dir] = L4;
  }
  if (n_dir == 1) {   // Y-normal
    d[3][c][n_dir] = L3;
    d[4][c][n_dir] = (L5 - L1)/(rho * speedSound);
    d[5][c][n_dir] = L4;
  }
  if (n_dir == 2) {   // Z-normal
    d[3][c][n_dir] = L3;
    d[4][c][n_dir] = L4;
    d[5][c][n_dir] = (L5 - L1)/(rho * speedSound);
  }
  
#if 0
   //__________________________________
   //  debugging
   vector<IntVector> dbgCells;
   dbgCells.push_back(IntVector(0,50,0));
           
   for (int i = 0; i<(int) dbgCells.size(); i++) {
     if (c == dbgCells[i]) {
      debugging_Di(c, d, s, dir, face, speedSound, vel_CC, L1, L2, L3, L4, L5 );
     }  // if(dbgCells)
   }  // dbgCells loop
 #endif
} 

/*__________________________________________________________________
 Function~ computeDi--
 Purpose~  compute Di's at the boundary cells using upwind first-order 
           differenceing scheme
____________________________________________________________________*/
void computeDi(StaticArray<CCVariable<Vector> >& d,
               constCCVariable<double>& rho,              
               const CCVariable<double>& press,                   
               constCCVariable<Vector>& vel,                  
               constCCVariable<double>& speedSound,              
               const Patch* patch,
               DataWarehouse* new_dw,
               SimulationStateP& sharedState,
               const Lodi_variable_basket* user_inputs)                              
{
  cout_doing << "LODI computeDi "<< endl;
  Vector dx = patch->dCell();
  
  // Characteristic Length of the overall domain
  Vector domainLength;
  const Level* level = patch->getLevel();
  GridP grid = level->getGrid();
  grid->getLength(domainLength, "minusExtraCells");
  
  Vector grav = sharedState->getGravity();

  for (int i = 1; i<= 5; i++ ) {           // don't initialize inside main loop
    d[i].initialize(Vector(0.0,0.0,0.0));  // you'll overwrite previously compute di
  }
  
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for (iter  = bf.begin(); iter != bf.end(); ++iter){
    Patch::FaceType face = *iter;
    
    if (is_LODI_face(patch,face, sharedState) ) {
      cout_dbg << " computing DI on face " << face 
               << " patch " << patch->getID()<<endl;
      //_____________________________________
      // S I D E S
      IntVector axes = patch->getFaceAxes(face);
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
      VarLabel* V_Label = getMaxMach_face_VarLabel(face);
      max_vartype maxMach;
      new_dw->get(maxMach,   V_Label);
      
      //__________________________________
      Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
      for(CellIterator iter=patch->getFaceIterator(face, PEC); 
          !iter.done();iter++) {
        IntVector c = *iter;
        IntVector r = c + R_offset;
        IntVector l = c + L_offset;

        double drho_dx = (rho[r]   - rho[l])/delta; 
        double dp_dx   = (press[r] - press[l])/delta;
        Vector dVel_dx = (vel[r]   - vel[l])/delta;

        vector<double> s(6);
        characteristic_source_terms(dir, P_dir, grav, rho[c], speedSound[c], s);

        Di(d, dir, c, face, domainLength, user_inputs, maxMach, s, press[c],
           speedSound[c], rho[c], vel[c], drho_dx, dp_dx, dVel_dx); 

      }  // faceCelliterator 
    }  // is Lodi face
  }  // loop over faces
}// end of function


/*__________________________________________________________________
 Function~ computeNu--                    L   O   D   I
 Purpose~  compute dissipation coefficients 
__________________________________________________________________*/ 
void computeNu(CCVariable<Vector>& nu,
               const CCVariable<double>& p, 
               const Patch* patch,
               SimulationStateP& sharedState)
{
  cout_doing << "LODI computeNu "<< endl;
  double d_SMALL_NUM = 1.0e-100;
    
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  for (iter  = bf.begin(); iter != bf.end(); ++iter){
    Patch::FaceType face = *iter;
    
    if (is_LODI_face(patch, face, sharedState) ) {
      cout_dbg << " computing Nu on face " << face 
               << " patch " << patch->getID()<<endl;   
              
      vector<int> otherDir(2);
      IntVector axes = patch->getFaceAxes(face);
      int P_dir   = axes[0]; // principal direction
      otherDir[0] = axes[1]; // other vector directions
      otherDir[1] = axes[2];  
      
      //__________________________________
      //  At patch boundaries you need to extend
      // the computational footprint by one cell in ghostCells
      CellIterator hiLo = patch->getFaceCellIterator(face, "minusEdgeCells");
      CellIterator iterPlusGhost = patch->addGhostCell_Iter(hiLo,1);
                        
      for(CellIterator iter=iterPlusGhost; !iter.done();iter++) {
        IntVector c = *iter;

        for ( int i = 0; i < 2 ; i++ ) {  // set both orthogonal components
          int dir = otherDir[i];
          IntVector r = c;
          IntVector l = c;
          r[dir] += 1;  // tweak the r and l cell indices
          l[dir] -= 1; 
                        // 2nd order cell centered difference 
          nu[c][dir] = fabs(p[r] - 2.0 * p[c] + p[l])/
                        (fabs(p[r] - p[c]) + fabs(p[c] - p[l])  + d_SMALL_NUM);
        }
      }
      //__________________________________
      //    E D G E S  -- on boundaryFaces only
      // use cell centered AND one sided differencing
      vector<Patch::FaceType> b_faces;
      getBoundaryEdges(patch,face,b_faces);

      vector<Patch::FaceType>::const_iterator iter;  
      for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
        Patch::FaceType face0 = *iter;
        //__________________________________
        //  Find the Vector components Edir1 and Edir2
        //  for this particular edge
        IntVector faceDir = patch->faceDirection(face0);
        IntVector axes = patch->getFaceAxes(face0);
        int Edir1 = axes[0];
        int Edir2 = remainingVectorComponent(P_dir, Edir1);
        
        //-----------  THIS IS GROSS-------
        // Find an edge iterator that 
        // a) doesn't hit the corner cells and
        // b) extends one cell into the next patch over
        CellIterator edgeIter =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");

        IntVector patchNeighborLow  = patch->noNeighborsLow();
        IntVector patchNeighborHigh = patch->noNeighborsHigh();
        
        IntVector lo = edgeIter.begin();
        IntVector hi = edgeIter.end();
        lo[Edir2] -= abs(1 - patchNeighborLow[Edir2]);  // increase footprint
        hi[Edir2] += abs(1 - patchNeighborHigh[Edir2]); // by 1
        CellIterator iterLimits(lo,hi);
        //__________________________________
        
        for(CellIterator iter = iterLimits;!iter.done();iter++){ 

          IntVector c = *iter;
          IntVector r  = c;
          IntVector rr = c;
          r[Edir1]  -= faceDir[Edir1];      // tweak the r and l cell indices
          rr[Edir1] -= 2 * faceDir[Edir1];  // One sided differencing
          nu[c][Edir1] = fabs(p[c] - 2.0 * p[r] + p[rr])/
                        (fabs(p[c] - p[r]) + fabs(p[r] - p[rr])  + d_SMALL_NUM);

          IntVector r2 = c;
          IntVector l2 = c;
          r2[Edir2] += 1;  // tweak the r and l cell indices
          l2[Edir2] -= 1;  // cell centered differencing
          nu[c][Edir2] = fabs(p[r2] - 2.0 * p[c] + p[l2])/
                        (fabs(p[r2] - p[c]) + fabs(p[c] - p[l2])  + d_SMALL_NUM);
        }
      }
      //________________________________________________________
      // C O R N E R S   
  /*`==========TESTING==========*/
  // Need a clever way to figure out the r and rr indicies
  //  for the two different directions

      vector<IntVector> corner;
      patch->getCornerCells(corner,face);
      vector<IntVector>::const_iterator itr;
      for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
        IntVector c = *itr;
        nu[c] = Vector(0,0,0);
        //cout << " I'm working on cell " << c << endl;
      } 
  /*==========TESTING==========`*/
    }  // on the LODI bc face
  }
}
/* --------------------------------------------------------------------- 
 Function~  lodi_bc_preprocess-- 
 Purpose~   Take care: getting data from dw, allocate temporary vars,
            initialize vars, compute some vars, compute di and nu for LODI bcs
 ---------------------------------------------------------------------  */
void  lodi_bc_preprocess( const Patch* patch,
                          Lodi_vars* lv,
                          ICELabel* lb,
                          const int indx,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          SimulationStateP& sharedState)
{
  cout_doing << "lodi_bc_preprocess on patch "<<patch->getID()<< endl;
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;
  constCCVariable<double> press_old;
  constCCVariable<double> vol_frac_old;
  
  CCVariable<double>& E   = lv->E;  // shortcuts to Lodi_vars struct
  CCVariable<Vector>& vel_CC = lv->vel_CC;
  CCVariable<double>& rho_CC = lv->rho_CC;
  CCVariable<double>& press_tmp = lv->press_tmp; 
  CCVariable<Vector>& nu = lv->nu;
  constCCVariable<double>& cv         = lv->cv;
  constCCVariable<double>& gamma      = lv->gamma;
  constCCVariable<double>& rho_old    = lv->rho_old;
  constCCVariable<double>& temp_old   = lv->temp_old;
  constCCVariable<double>& speedSound = lv->speedSound;
  constCCVariable<Vector>& vel_old    = lv->vel_old;
  StaticArray<CCVariable<Vector> >& di = lv->di;
  
  //__________________________________
  //   get the data LODI needs from old dw
  new_dw->get(gamma,        lb->gammaLabel,       indx,patch,gn ,0);
  old_dw->get(temp_old,     lb->temp_CCLabel,     indx,patch,gac,1);
  old_dw->get(rho_old,      lb->rho_CCLabel,      indx,patch,gac,1);
  old_dw->get(vel_old,      lb->vel_CCLabel,      indx,patch,gac,1);
  old_dw->get(press_old,    lb->press_CCLabel,    0,   patch,gac,2);
  old_dw->get(vol_frac_old, lb->vol_frac_CCLabel, indx,patch,gac,2);

  new_dw->allocateTemporary(press_tmp, patch, gac, 2);
  new_dw->allocateTemporary(nu,        patch, gac, 1);
  new_dw->allocateTemporary(E,         patch, gac, 1);
  new_dw->allocateTemporary(rho_CC,    patch, gac, 1);  
  new_dw->allocateTemporary(vel_CC,    patch, gac, 1);

  for (int i = 0; i <= 5; i++){ 
    new_dw->allocateTemporary(di[i], patch, gac, 1);
  }                             
   
  //__________________________________
  // only work on those faces that have lodi bcs
  // and are on the edge of the computational domain
  vector<Patch::FaceType>::const_iterator iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for (iter  = bf.begin(); iter != bf.end(); ++iter){
    Patch::FaceType face = *iter;
    
    if (is_LODI_face(patch,face, sharedState) ) {
      //__________________________________
      // Create an iterator that iterates over the face
      // + 2 cells inward.  We don't need to hit every
      // cell on the patch.  At patch boundaries you need to extend
      // the footprint by one/two cells into the next patch
      
      Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
      CellIterator iter=patch->getFaceIterator(face, PEC);
      IntVector lo = iter.begin();
      IntVector hi = iter.end();
    
      int P_dir = patch->getFaceAxes(face)[0];  //principal dir.
      if(face==Patch::xminus || face==Patch::yminus || face==Patch::zminus){
        hi[P_dir] += 2;
      }
      if(face==Patch::xplus || face==Patch::yplus || face==Patch::zplus){
        lo[P_dir] -= 2;
      }
      CellIterator iterLimits(lo,hi);
      CellIterator iterPlusGhost1 = patch->addGhostCell_Iter(iterLimits,1);
      CellIterator iterPlusGhost2 = patch->addGhostCell_Iter(iterLimits,2);

      //  plus one layer of ghostcells
      Vector nanV(-9e1000, -9e1000, -9e1000);
      for(CellIterator iter = iterPlusGhost1; !iter.done(); iter++) {  
        IntVector c = *iter;
        nu[c] = nanV;
        E[c] = rho_old[c] * (cv[c] * temp_old[c]  + 0.5 * vel_old[c].length2() ); 
      }
      //  plut two layers of ghostcells
      for(CellIterator iter = iterPlusGhost2; !iter.done(); iter++) {  
        IntVector c = *iter;
        press_tmp[c] = vol_frac_old[c] * press_old[c];
      }
    }  //lodi face
  }  // boundary face

  //compute dissipation coefficients
  computeNu(nu, press_tmp, patch, sharedState);

  //compute Di at boundary cells
  computeDi(di, rho_old,  press_tmp, vel_old, 
            speedSound, patch, new_dw, sharedState, lv->var_basket);
}  
 
/*________________________________________________________
 Function~ computeConvection--
 Purpose~  Compute the convection term in conservation law
_________________________________________________________*/
inline double computeConvection(
              const Vector& nuFrt, const Vector& nuMid, const Vector& nuLast,    
              const double& qFrt,  const double& qMid,  const double& qLast,
              const double& q_R,   const double& q_C,   const double& q_L,
              const double& deltaT,    const Vector& dx,
              const int& dir) 
{
   //__________________________________
   // Artifical dissipation term
/*`==========TESTING==========*/
#if 0
    double k_const = 0.4;
    doubl eeplus  = 0.5 * k_const * dx[dir] * (nuFrt[dir]   + nuMid[dir])/deltaT;
    double eminus = 0.5 * k_const * dx[dir] * (nuLast[dir]  + nuMid[dir])/deltaT;
    double dissipation = (eplus * qFrt - (eplus + eminus) * qMid 
              +  eminus * qLast)/dx[dir]; 
#endif
    double dissipation  = 0;
/*==========TESTING==========`*/
 double conv;
 
  // central differenceing
  conv = 0.5 * (q_R - q_L)/dx[dir] - dissipation;
   
#if 0      
   // upwind differenceing  (we'll make this fast after we know that it works)
   if (q_C > 0) {
    conv = (q_R - q_C)/dx[dir] - dissipation;
   }else{
    conv = (q_L - q_C)/dx[dir] - dissipation;
   }         
#endif
   return conv; 
} 
/*_________________________________________________________________
 Function~ getBoundaryEdges--
 Purpose~  returns a list of edges where boundary conditions
           should be applied.
___________________________________________________________________*/  
void getBoundaryEdges(const Patch* patch,
                      const Patch::FaceType face,
                      vector<Patch::FaceType>& face0)
{
  IntVector patchNeighborLow  = patch->noNeighborsLow();
  IntVector patchNeighborHigh = patch->noNeighborsHigh();
  
  //__________________________________
  // Looking down on the face, examine 
  // each edge (clockwise).  If there
  // are no neighboring patches then it's a valid edge
  IntVector axes = patch->getFaceAxes(face);
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
                const Vector& dx)
{
  cout_doing << "Setting FaceDensity_LODI on face " << face<<endl;
  // bulletproofing
  if (!lv){
    throw InternalError("FaceDensityLODI: Lodi_vars = null", __FILE__, __LINE__);
  }  
  
  // shortcuts
  StaticArray<CCVariable<Vector> >& d = lv->di;
  const CCVariable<Vector>& nu      = lv->nu;
  const CCVariable<double>& rho_old = lv->rho_old;
  const CCVariable<Vector>& vel_old = lv->vel_old;
  const double delT = lv->delT;
  
  IntVector axes = patch->getFaceAxes(face);
  int P_dir = axes[0];  // principal direction
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];
  
  IntVector offset = IntVector(1,1,1) - Abs(patch->faceDirection(face));
  //__________________________________
  //    S I D E   M I N U S   E D G E S
  Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
  for(CellIterator iter=patch->getFaceIterator(face, MEC); 
                                                      !iter.done();iter++) {
    IntVector c = *iter;
    
    IntVector r1 = c;
    IntVector l1 = c;
    r1[dir1] += offset[dir1];  // tweak the r and l cell indices
    l1[dir1] -= offset[dir1];
    double q_R = rho_old[r1] * vel_old[r1][dir1];
    double q_C = rho_old[c]  * vel_old[c][dir1];
    double q_L = rho_old[l1] * vel_old[l1][dir1];
    
    double conv_dir1 = computeConvection(nu[r1], nu[c], nu[l1], 
                                         rho_old[r1], rho_old[c], rho_old[l1], 
                                         q_R, q_C, q_L, 
                                         delT, dx, dir1);
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];
    
    q_R = rho_old[r2] * vel_old[r2][dir2];
    q_C = rho_old[c]  * vel_old[c][dir2];
    q_L = rho_old[l2] * vel_old[l2][dir2];
    double conv_dir2 = computeConvection(nu[r2], nu[c], nu[l2],
                                         rho_old[r2], rho_old[c], rho_old[l2], 
                                         q_R, q_C, q_L, 
                                         delT, dx, dir2);
                                         
    double rho_src = -delT * (d[1][c][P_dir] + conv_dir1 + conv_dir2);

    rho_CC[c] = rho_old[c] + rho_src;
#if 0
    //__________________________________
    //  debugging
    vector<IntVector> dbgCells;
    dbgCells.push_back(IntVector(0,50,0));

    for (int i = 0; i<(int) dbgCells.size(); i++) {
      if (c == dbgCells[i]) {
        cout.setf(ios::scientific,ios::floatfield);
        cout.precision(10);
        cout << " \n c " << c << "--------------------------  F A C E " << face << " P_dir " << P_dir << endl;
        cout << c <<" P_dir " << P_dir << " dir1 " << dir1 << "dir2 " << dir2 << endl;
        cout << " rho_old                 " << rho_old[c] << endl;
        cout << " rho_src                 " << rho_src << endl;
        cout << " conv_dir1               " << conv_dir1 << endl;
        cout << " conv_dir                " << conv_dir2 << endl;
        cout << " BN_convect              " << d[1][c][P_dir] <<endl;
        cout << " rho_CC                  " << rho_CC[c] << endl;
      }
    }  //  dbgCells loop
#endif
  }
 
  //__________________________________
  //    E D G E S  -- on boundaryFaces only
  vector<Patch::FaceType> b_faces;
  getBoundaryEdges(patch,face,b_faces);
  
  vector<Patch::FaceType>::const_iterator iter;  
  for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
    Patch::FaceType face0 = *iter;
    //__________________________________
    //  Find the offset for the r and l cells
    //  and the Vector components Edir1 and Edir2
    //  for this particular edge
    IntVector offset = IntVector(1,1,1)  - Abs(patch->faceDirection(face)) 
                                         - Abs(patch->faceDirection(face0));
           
    IntVector axes = patch->getFaceAxes(face0);
    int Edir1 = axes[0];
    int Edir2 = remainingVectorComponent(P_dir, Edir1);
  
    CellIterator iterLimits =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");
                
    for(CellIterator iter = iterLimits;!iter.done();iter++){ 
      IntVector c = *iter;  
      IntVector r = c + offset;  
      IntVector l = c - offset;
      double q_R = rho_old[r] * vel_old[r][Edir2];
      double q_C = rho_old[c] * vel_old[c][Edir2];
      double q_L = rho_old[l] * vel_old[l][Edir2];
      double conv = computeConvection(nu[r], nu[c], nu[l],
                                      rho_old[r], rho_old[c], rho_old[l], 
                                      q_R, q_C, q_L, 
                                      delT, dx, Edir2); 
                                                                
      rho_CC[c] = rho_old[c] - delT * (d[1][c][P_dir] + d[1][c][Edir1] + conv);
    }
  }

  //__________________________________
  // C O R N E R S    
  vector<IntVector> corner;
  patch->getCornerCells(conrner,face);
  vector<IntVector>::const_iterator itr;
  
  for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
    IntVector c = *itr;
    rho_CC[c] = rho_old[c] 
              - delT * (d[1][c].x() + d[1][c].y() + d[1][c].z());
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
                 const Vector& dx,
                 SimulationStateP& sharedState)                     

{
  cout_doing << "Setting FaceVel_LODI on face " << face << endl;
  // bulletproofing
  if (!lv){
    throw InternalError("FaceVelLODI: Lodi_vars = null", __FILE__, __LINE__);
  }
 
/*`==========TESTING==========*/
// hardCode the cells you want to intergate;
vector<IntVector> dbgCells;
#if 1
   dbgCells.push_back(IntVector(0,50,0));
#endif
double time = sharedState->getElapsedTime();
/*===========TESTING==========`*/
     
  // shortcuts       
  StaticArray<CCVariable<Vector> >& d = lv->di;      
  constCCVariable<double>& rho_old = lv->rho_old;
  constCCVariable<Vector>& vel_old = lv->vel_old; 
  CCVariable<double>& rho_new = lv->rho_CC;
  
  CCVariable<double>& p  = lv->press_tmp;          
  double delT = lv->delT;
                 
  IntVector axes = patch->getFaceAxes(face);
  int P_dir = axes[0];  // principal direction
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];
  
  Vector gravity = sharedState->getGravity();
  IntVector offset = IntVector(1,1,1) - Abs(patch->faceDirection(face));
  //__________________________________
  //    S I D E   M I N U S   E D G E S 
  Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
  for(CellIterator iter=patch->getFaceIterator(face, MEC); 
                                                      !iter.done();iter++) {
    IntVector c = *iter;
    //__________________________________
    // Transverse convection terms
    IntVector r1 = c;
    IntVector l1 = c;
    r1[dir1] += offset[dir1];  // tweak the r and l cell indices
    l1[dir1] -= offset[dir1]; 
       
    Vector convect1(0,0,0);
    for(int dir = 0; dir <3; dir ++ ) {              
      double q_R = rho_old[r1] * vel_old[r1][dir] * vel_old[r1][dir1];
      double q_C = rho_old[c]  * vel_old[c][dir]  * vel_old[c][dir1];
      double q_L = rho_old[l1] * vel_old[l1][dir] * vel_old[l1][dir1];
      Vector placeHolder;

      convect1[dir] = computeConvection(placeHolder, placeHolder, placeHolder, 
                                        rho_old[r1], rho_old[c], rho_old[l1], 
                                        q_R, q_C, q_L, 
                                        delT, dx, dir1);
    }
    
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];  
    
    Vector convect2(0,0,0);
    for(int dir = 0; dir <3; dir ++ ) {            
      double q_R = rho_old[r2] * vel_old[r2][dir] * vel_old[r2][dir2];
      double q_C = rho_old[c]  * vel_old[c][dir]  * vel_old[c][dir2];
      double q_L = rho_old[l2] * vel_old[l2][dir] * vel_old[l2][dir2];
      Vector placeHolder;
      
      convect2[dir] = computeConvection(placeHolder, placeHolder, placeHolder, 
                                        rho_old[r2], rho_old[c], rho_old[l2], 
                                        q_R, q_C, q_L, 
                                        delT, dx, dir2); 
     }       
    //__________________________________
    // Pressure gradient terms
    Vector pressGradient;  
    pressGradient[P_dir] = 0.0;   // This is accounted for in Li terms
    pressGradient[dir1]  = 0.5 * (p[r1] - p[l1])/dx[dir1];  // centered difference  
    pressGradient[dir2]  = 0.5 * (p[r2] - p[l2])/dx[dir2];  //  ---//---- 
    
    //__________________________________
    // Boundary normal convection Terms
    // See Sutherland Table 8
    Vector BN_convect;
    BN_convect.x( vel_old[c].x() * d[1][c][P_dir] + rho_old[c] *d[3][c][P_dir] );
    BN_convect.y( vel_old[c].y() * d[1][c][P_dir] + rho_old[c] *d[4][c][P_dir] );
    BN_convect.z( vel_old[c].z() * d[1][c][P_dir] + rho_old[c] *d[5][c][P_dir] );
     
    //__________________________________
    //  Equations 9.9-9.10
    Vector momOld, momChange;
    momOld    = rho_old[c] * vel_old[c];
    
    Vector bodyForce = rho_old[c] * gravity;
                         
    momChange =  - delT * ( BN_convect + convect1 + convect2 + pressGradient )
                 + delT *bodyForce;
         
    vel_CC[c] = (momOld + momChange)/rho_new[c];

#if 0
    //__________________________________
    //  debugging
    for (int i = 0; i<(int) dbgCells.size(); i++) {
      if (c == dbgCells[i]) {
        cout.setf(ios::scientific,ios::floatfield);
        cout.precision(10);
        cout << " \n c " << c << "--------------------------  F A C E " << face << " P_dir " << P_dir << endl;
        cout << c <<" P_dir " << P_dir << " dir1 " << dir1 << "dir2 " << dir2 << endl;
        cout << " rho_old[c] * vel_old[c] " << momOld << endl;
        cout << " convect1                " << convect1 << endl;
        cout << " convect2                " << convect2 << endl;
        cout << " BN_convect              " << BN_convect <<endl;
        cout << " pressGradient           " << pressGradient << endl;
        cout << " rho_old * gravity       " << bodyForce << endl;
        cout << " vel                     " << vel_CC[c] << "\n"<<endl;
        cout << " convect1: rho_old, vel_old["<<dir1<<"], vel_old["<<dir1<<"], dx["<<dir1<<"]" << endl;
        cout << " convect2: rho_old, vel_old["<<dir1<<"], vel_old["<<dir2<<"], dx["<<dir2<<"]" << endl;
        cout << " vel_old[c].y() * d[1][c][P_dir] " <<  vel_old[c].y() * d[1][c][P_dir]
             << " - rho_old[c] *d[4][c][P_dir] "    << rho_old[c] *d[4][c][P_dir] << "  "<< BN_convect.y() << endl;
      }
    }  //  dbgCells loop
#endif
  }

  //__________________________________
  //    E D G E S  -- on boundaryFaces only
  vector<Patch::FaceType> b_faces;
  getBoundaryEdges(patch,face,b_faces);
  
  vector<Patch::FaceType>::const_iterator iter;  
  for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
    Patch::FaceType face0 = *iter;
    
    //__________________________________
    //  Find the offset for the r and l cells
    //  and the Vector components Edir1 and Edir2
    //  for this particular edge
    IntVector offset = IntVector(1,1,1)  - Abs(patch->faceDirection(face)) 
                                         - Abs(patch->faceDirection(face0));
           
    IntVector axes = patch->getFaceAxes(face0);
    int Edir1 = axes[0];
    int Edir2 = remainingVectorComponent(P_dir, Edir1);
     
    CellIterator iterLimits =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");
                      
    for(CellIterator iter = iterLimits;!iter.done();iter++){ 
      IntVector c = *iter;

      //__________________________________
      // convective terms
      IntVector r2 = c;
      IntVector l2 = c;
      r2[Edir2] += offset[Edir2];  // tweak the r and l cell indices
      l2[Edir2] -= offset[Edir2]; 

      Vector convect1(0,0,0);
      for(int dir = 0; dir <3; dir ++ ) {
        double q_R = rho_old[r2] * vel_old[r2][dir] * vel_old[r2][Edir2];
        double q_C = rho_old[c]  * vel_old[c][dir]  * vel_old[c][Edir2];
        double q_L = rho_old[l2] * vel_old[l2][dir] * vel_old[l2][Edir2];
        Vector placeHolder;
        
        convect1[dir] = computeConvection(placeHolder, placeHolder, placeHolder, 
                                          rho_old[r2], rho_old[c], rho_old[l2], 
                                          q_R, q_C, q_L, 
                                          delT, dx, Edir2);
      }
      //__________________________________
      // Pressure gradient terms
      Vector pressGradient; 
      pressGradient[P_dir] = 0.0;   // accounted for in Li terms
      pressGradient[Edir1] = 0.0;
      pressGradient[Edir2] = 0.5 * (p[r2] - p[l2])/dx[Edir2];

      //__________________________________
      // Equation 9.9 - 9.10
      Vector mom; // momentum
      
      Vector BN_convect;    // boundary normal convection
      BN_convect.x( vel_old[c].x() * (d[1][c][P_dir] + d[1][c][Edir1])
                  + rho_old[c]     * (d[3][c][P_dir] + d[3][c][Edir1]) );
                  
      BN_convect.y( vel_old[c].y() * (d[1][c][P_dir] + d[1][c][Edir1])
                  + rho_old[c]     * (d[4][c][P_dir] + d[4][c][Edir1]) );
                  
      BN_convect.z( vel_old[c].z() * (d[1][c][P_dir] + d[1][c][Edir1])
                  + rho_old[c]     * (d[5][c][P_dir] + d[5][c][Edir1]) );
                  
      Vector bodyForce = rho_old[c] * gravity;
      
      mom  = rho_old[c] * vel_old[c]                                
           - delT * ( BN_convect + convect1 + pressGradient )
           + delT * bodyForce;                     
              
      vel_CC[c] = mom/rho_new[c];

#if 0
      //__________________________________
      //  debugging
      for (int i = 0; i<(int) dbgCells.size(); i++) {
        if (c == dbgCells[i] ) {
          cout.setf(ios::scientific,ios::floatfield);
          cout.precision(10);

          cout << " -------------------------- E D G E " << endl;
          cout << c <<" P_dir " << P_dir << " Edir1 " << Edir1 << " Edir2 " << Edir2 << endl;
          cout << " rho_old[c] * vel_old[c] " << rho_old[c] * vel_old[c] << endl;
          cout << " BN_convect              " << BN_convect << endl;
          cout << " convect1                " << convect1 << endl;
          cout << " pressGradient           " << pressGradient << endl;
          cout << " rho_old gravity         " << bodyForce << endl;
          cout << " mom                     " << mom << endl;
          cout << " vel                     " << vel_CC[c] << "\n"<<endl;

          for (int i = 1; i<= 5; i++ ) {
            cout << " d[" << i << "]:\t"<< d[i][c]<< endl;
          }
          cout << " -------------------------- " << endl;
        }
      }
#endif
    }
  }  
  //________________________________________________________
  // C O R N E R S
  vector<IntVector> corner;
  patch->getCornerCells(corner,face);
  vector<IntVector>::const_iterator itr;
  
  for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
    IntVector c = *itr;
    Vector term1, term2;
   
    term1.x((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_old[c].x());
    term1.y((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_old[c].y());
    term1.z((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_old[c].z());
    
    term2.x((d[3][c].x() + d[3][c].y() + d[3][c].z()) * rho_old[c]);
    term2.y((d[4][c].x() + d[4][c].y() + d[4][c].z()) * rho_old[c]);
    term2.z((d[5][c].x() + d[5][c].y() + d[5][c].z()) * rho_old[c]);
              
    Vector bodyForce = rho_old[c] * gravity;
                                        
    Vector mom = rho_old[c] * vel_old[c] - delT * (term1 + term2)
                                         + delT * bodyForce;

    vel_CC[c] = mom/ rho_new[c];

      //__________________________________
      //  debugging
#if 0
  cout << " corner " << c << endl;
      for (int i = 0; i<(int) dbgCells.size(); i++) {
        if (c == dbgCells[i]) {
         cout << "-------------------------- C O R N E R " << c << endl;
         cout << " rho_old[c] * vel_old[c] " << rho_old[c] * vel_old[c] << endl;
         cout << " term1 =                 " << term1 << endl;
         cout << " term2 =                 " << term2 << endl;
         cout << " rho_old gravity         " << bodyForce << endl;
 
         cout << " mom/delT                " <<  mom/delT<< endl;
         cout << " vel_CC[c]               " << vel_CC[c] << endl;
         for (int i = 1; i<= 5; i++ ) {
           cout << " d[" << i << "]:\t"<< d[i][c]<< endl;
         }
       }
    }
#endif 
    
  }
} //end of the function FaceVelLODI() 

/*_________________________________________________________________
 Function~ FaceTemp_LODI--
 Purpose~  Compute temperature in boundary cells on faces
           Solves equation 9.8 of Reference.
___________________________________________________________________*/
void FaceTemp_LODI(const Patch* patch,
             const Patch::FaceType face,
             CCVariable<double>& temp_CC,
             Lodi_vars* lv, 
             const Vector& dx,
             SimulationStateP& sharedState)
{
  cout_doing << "Setting FaceTemp_LODI on face " <<face<< endl; 
  
  // bulletproofing
  if (!lv){
    throw InternalError("FaceTempLODI: Lodi_vars = null", __FILE__, __LINE__);
  } 
  // shortcuts  
  StaticArray<CCVariable<Vector> >& d = lv->di;
  const CCVariable<double>& E         = lv->E;
  const CCVariable<double>& cv        = lv->cv;
  const CCVariable<double>& gamma     = lv->gamma;
  const CCVariable<double>& rho_new   = lv->rho_CC;
  const CCVariable<double>& rho_old   = lv->rho_old;
  const CCVariable<double>& Temp_old  = lv->temp_old;
  const CCVariable<double>& press_tmp = lv->press_tmp;
  const CCVariable<Vector>& vel_old   = lv->vel_old;
  const CCVariable<Vector>& vel_new   = lv->vel_CC;
  const CCVariable<Vector>& nu  = lv->nu;
  const double delT  = lv->delT;
              
  IntVector axes = patch->getFaceAxes(face);
  int P_dir = axes[0];  // principal direction
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];
  
  Vector gravity = sharedState->getGravity();
  IntVector offset = IntVector(1,1,1) - Abs(patch->faceDirection(face));
  //__________________________________
  //    S I D E   M I N U S   E D G E S  
  Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
  for(CellIterator iter=patch->getFaceIterator(face, MEC); 
                                                      !iter.done();iter++) {
    IntVector c = *iter;
    
    IntVector r1 = c;
    IntVector l1 = c;
    r1[dir1] += offset[dir1];  // tweak the r and l cell indices
    l1[dir1] -= offset[dir1];
    double q_R = vel_old[r1][dir1] * (E[r1] + press_tmp[r1]);
    double q_C = vel_old[c][dir1]  * (E[c]  + press_tmp[c]);
    double q_L = vel_old[l1][dir1] * (E[l1] + press_tmp[l1]);
    
    double conv_dir1 = computeConvection(nu[r1], nu[c], nu[l1], 
                                          E[r1], E[c], E[l1], 
                                          q_R, q_C, q_L, 
                                          delT, dx, dir1);
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];
    
    q_R = vel_old[r2][dir2] * (E[r2] + press_tmp[r2]);
    q_C = vel_old[c][dir2]  * (E[c]  + press_tmp[c]);
    q_L = vel_old[l2][dir2] * (E[l2] + press_tmp[l2]);
    double conv_dir2 = computeConvection(nu[r2], nu[c], nu[l2],
                                         E[r2], E[c], E[l2], 
                                         q_R, q_C, q_L, 
                                         delT, dx, dir2);

    double vel_old_sqr = vel_old[c].length2();
    double vel_new_sqr = vel_new[c].length2();
    double term1 = 0.5 * d[1][c][P_dir] * vel_old_sqr;
    
    double term2 = d[2][c][P_dir]/(gamma[c] - 1.0);
    double term3 = rho_old[c] * ( vel_old[c].x() * d[3][c][P_dir] + 
                                  vel_old[c].y() * d[4][c][P_dir] +
                                  vel_old[c].z() * d[5][c][P_dir] );
    double BN_convect = term1 + term2 + term3;
    //__________________________________
    //  See Thompson II, pg 451, eq 56
    double gravityTerm = 0.0;
    for(int dir = 0; dir <3; dir ++ ) { 
      gravityTerm += rho_old[c] * vel_old[c][dir] * gravity[dir]; 
    }
                   
    double E_src = -delT * (BN_convect + conv_dir1 + conv_dir2
                           - gravityTerm);            

    temp_CC[c] = (E[c] + E_src)/(rho_new[c]*cv[c]) - 0.5 * vel_new_sqr/cv[c];
    
    
#if 0
    //__________________________________
    //  debugging
    vector<IntVector> dbgCells;
    dbgCells.push_back(IntVector(0,50,0));

    for (int i = 0; i<(int) dbgCells.size(); i++) {
      if (c == dbgCells[i]) {
        cout.setf(ios::scientific,ios::floatfield);
        cout.precision(10);
        cout << " \n c " << c << "--------------------------  F A C E " << face << " P_dir " << P_dir << endl;
        cout << c <<" P_dir " << P_dir << " dir1 " << dir1 << "dir2 " << dir2 << endl;
        cout << " Temp_old                " << Temp_old[c] << endl;
        cout << " E_old[c]                " << E[c] << endl;
        cout << " E_src                   " << E_src << endl;
        cout << " conv_dir1               " << conv_dir1 << endl;
        cout << " conv_dir                " << conv_dir2 << endl;
        cout << " BN_convect              " << BN_convect <<endl;
        cout << " gravityTerm             " << gravityTerm << endl;
        cout << " rho_old                 " << rho_old[c] << " \t rho_new " << rho_new[c] << endl;
        cout << " Temp_CC                 " << temp_CC[c] << endl;
      }
    }  //  dbgCells loop
#endif
  }
  
  //__________________________________
  //    E D G E S  -- on boundaryFaces only
  vector<Patch::FaceType> b_faces;
  getBoundaryEdges(patch,face,b_faces);
  
  vector<Patch::FaceType>::const_iterator iter;  
  for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
    Patch::FaceType face0 = *iter;
    //__________________________________
    //  Find the offset for the r and l cells
    //  and the Vector components Edir1 and Edir2
    //  for this particular edge
    
    IntVector edge = Abs(patch->faceDirection(face)) 
                   + Abs(patch->faceDirection(face0));
    IntVector offset = IntVector(1,1,1) - edge;
           
    IntVector axes = patch->getFaceAxes(face0);
    int Edir1 = axes[0];
    int Edir2 = remainingVectorComponent(P_dir, Edir1);
     
    CellIterator iterLimits =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");
                  
    for(CellIterator iter = iterLimits;!iter.done();iter++){ 
      IntVector c = *iter;  
      IntVector r = c + offset;  
      IntVector l = c - offset;

      double q_R = vel_old[r][Edir2] * (E[r] + press_tmp[r]);
      double q_C = vel_old[c][Edir2] * (E[c] + press_tmp[c]);
      double q_L = vel_old[l][Edir2] * (E[l] + press_tmp[l]);
    
      double conv = computeConvection(nu[r], nu[c], nu[l],
                                      E[r], E[c], E[l],               
                                      q_R, q_C, q_L, 
                                      delT, dx, Edir2); 
                                      
      double vel_old_sqr = vel_old[c].length2();

      double term1 = 0.5 * (d[1][c][P_dir] + d[1][c][Edir1]) * vel_old_sqr;

      double term2 = (d[2][c][P_dir] + d[2][c][Edir1])/(gamma[c] - 1.0);
      
      double term3 =
           rho_old[c] * (vel_old[c].x() * (d[3][c][P_dir] + d[3][c][Edir1])  
                      +  vel_old[c].y() * (d[4][c][P_dir] + d[4][c][Edir1])  
                      +  vel_old[c].z() * (d[5][c][P_dir] + d[5][c][Edir1]) );
     
      double BN_convect = term1 + term2 + term3;
      //__________________________________
      //  See Thompson II, pg 451, eq 56
      double gravityTerm = 0.0;
      for(int dir = 0; dir <3; dir ++ ) { 
        gravityTerm += rho_old[c] * vel_old[c][dir] * gravity[dir]; 
      }
      
      double E_src = - delT * ( BN_convect + conv - gravityTerm);
      double vel_new_sqr = vel_new[c].length2();

      temp_CC[c] = (E[c] + E_src)/(rho_new[c] *cv[c]) - 0.5 * vel_new_sqr/cv[c];
    }
  }  
 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> corner;
  patch->getCornerCells(corner,face);
  vector<IntVector>::const_iterator itr;
  
  for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
    IntVector c = *itr;
    double vel_old_sqr = vel_old[c].length2();
    double vel_new_sqr = vel_new[c].length2();
    double term1, term2, term3;
    
    term1 = 0.5 * (d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_old_sqr;
    term2 =       (d[2][c].x() + d[2][c].y() + d[2][c].z())/(gamma[c] - 1.0);
    
    term3 =
        rho_old[c] * vel_old[c].x() * (d[3][c].x() + d[3][c].y() + d[3][c].z())  
      + rho_old[c] * vel_old[c].y() * (d[4][c].x() + d[4][c].y() + d[4][c].z())  
      + rho_old[c] * vel_old[c].z() * (d[5][c].x() + d[5][c].y() + d[5][c].z()); 

    //__________________________________
    //  See Thompson II, pg 451, eq 56
    double gravityTerm = 0.0;
    for(int dir = 0; dir <3; dir ++ ) { 
      gravityTerm += rho_old[c] * vel_old[c][dir] * gravity[dir]; 
    }

    double E_new = E[c] - delT * ( term1 + term2 + term3 - gravityTerm);

    temp_CC[c] = E_new/(rho_new[c] * cv[c]) - 0.5 * vel_new_sqr/cv[c];
  }
} //end of function FaceTempLODI()  


/* --------------------------------------------------------------------- 
 Function~  FacePress_LODI--
 Purpose~   Back out the pressure from f_theta and P_EOS
---------------------------------------------------------------------  */
void FacePress_LODI(const Patch* patch,
                    CCVariable<double>& press_CC,
                    StaticArray<CCVariable<double> >& rho_micro,
                    SimulationStateP& sharedState, 
                    Patch::FaceType face,
                    Lodi_vars_pressBC* lv)
{
  cout_doing << " I am in FacePress_LODI on face " <<face<< endl;
  // bulletproofing
  if (!lv){
    throw InternalError("FacePress_LODI: Lodi_vars_pressBC = null", __FILE__, __LINE__);
  }

  int numMatls = sharedState->getNumMatls();
  StaticArray<double> press_eos(numMatls);
  StaticArray<constCCVariable<double> >& gamma   = lv->gamma;
  StaticArray<constCCVariable<double> >& cv      = lv->cv;
  StaticArray<constCCVariable<double> >& Temp_CC = lv->Temp_CC;
  StaticArray<constCCVariable<double> >& f_theta = lv->f_theta;

  //__________________________________  
  double press_ref= sharedState->getRefPress();    
     
  //__________________________________ 
  Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
  for(CellIterator iter=patch->getFaceIterator(face, PEC); 
    !iter.done();iter++) {
    IntVector c = *iter;

    press_CC[c] = 0.0;
    for (int m = 0; m < numMatls; m++) {
      Material* matl = sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      double tmp;

      if(ice_matl){                // I C E
        ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                       cv[m][c],Temp_CC[m][c],
                                       press_eos[m],tmp,tmp);        
      } 
#ifndef __APPLE__
      if(mpm_matl){                //  M P M
        mpm_matl->getConstitutiveModel()->
          computePressEOSCM(rho_micro[m][c],press_eos[m], press_ref,
                            tmp, tmp,mpm_matl);
      }
#else
      // This needs to be rethought, due to circular dependencies...
      cerr << "Temporarily commented out by Steve\n";
#endif
      press_CC[c] += f_theta[m][c]*press_eos[m];
//     cout << "press_CC" << c << press_CC[c] << endl;           
    }  // for ALLMatls...
  }
} 
  
/*______________________________________________________________________
          S H E M A T I C   D I A G R A M S

                          Looking down x(plus/minus) faces
                             yplus face  Edir0 = 0 (x)
                                         Edir1 = 1 (y)
                                         Edir2 = 2 (z)
                        _____________________
                        |   |   |   |   |   |
                        | + | + | + | + | + |
                        |---+-----------+---|
                        |   |           |   |
  z(plus/minus)  face   | + |           | + |   z(plus/minus)  face
                        |---+           +---|
                        |   |           |   |    
                        | + |           | + |  
                        |---+           +---|
                        |   |           |   |        
                        | + |           | + |        
                        |---+-----------+---|
                        |   |   |   |   |   |
                        | + | + | + | + | + |     
                        -------------------- 
                              yminus face  Edir0 = 0 (x)
                                           Edir1 = 1 (y)
                                           Edir2 = 2 (z)
______________________________________________________________________*/   
}  // using namespace Uintah
