/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#include <CCA/Components/ICE/CustomBCs/microSlipBCs.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Math/MiscMath.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Math/MiscMath.h>
#include <typeinfo>

#define d_SMALL_NUM 1e-100

using namespace std;
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
                              slip_globalVars* svb)
{
  //__________________________________
  // search the BoundaryConditions problem spec
  // determine if Slip/creep bcs are specified
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");

  bool usingSlip = false;

  for( ProblemSpecP face_ps = bc_ps->findBlock( "Face" ); face_ps != nullptr; face_ps = face_ps->findNextBlock( "Face" ) ) {

    map<string,string> face;
    face_ps->getAttributes(face);
    bool is_a_MicroSlip_face = false;

    for( ProblemSpecP bc_iter = face_ps->findBlock( "BCType" ); bc_iter != nullptr; bc_iter = bc_iter->findNextBlock( "BCType" ) ) {
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);

      if ((bc_type["var"] == "slip" || bc_type["var"] == "creep")  && !is_a_MicroSlip_face) {
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
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }
    slip->require("alpha_momentum",    svb->alpha_momentum);
    slip->require("alpha_temperature", svb->alpha_temperature);
    slip->require("SlipModel",         svb->SlipModel);
    slip->require("CreepFlow",         svb->CreepFlow);
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
void addRequires_MicroSlip(Task         * t,
                           const string & where,
                           ICELabel     * lb,
                           const MaterialSubset * ice_matls,
                           slip_globalVars      * var_basket)
{
  cout_doing<< "Doing addRequires_microSlip: \t\t" <<t->getName()
            << " " << where << endl;

  Ghost::GhostType  gn  = Ghost::None;
#if 0
  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  MaterialSubset* press_matl = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();


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
    t->requires(Task::NewDW, lb->rho_CCLabel,        ice_matls, gn);
    t->requires(Task::NewDW, lb->gammaLabel,         ice_matls, gn);
    t->requires(Task::NewDW, lb->specific_heatLabel, ice_matls, gn);
    t->requires(Task::NewDW, lb->thermalCondLabel,   ice_matls, gn);
    t->requires(Task::NewDW, lb->viscosityLabel,     ice_matls, gn);

    t->computes(lb->vel_CC_XchangeLabel);
    t->computes(lb->temp_CC_XchangeLabel);
  }
  if(where == "Advection"){
    t->requires(Task::NewDW, lb->gammaLabel,         ice_matls, gn);
    t->requires(Task::NewDW, lb->specific_heatLabel, ice_matls, gn);
    t->requires(Task::NewDW, lb->thermalCondLabel,   ice_matls, gn);
    t->requires(Task::NewDW, lb->viscosityLabel,     ice_matls, gn);
    // requires(Task::NewDW, lb->vel_CCLabel,        ice_matls, gn);
    // requires(Task::NewDW, lb->rho_CCLabel,        ice_matls, gn);
  }
}
/*__________________________________________________________________
 Function~ meanFreePath-
 Purpose~  compute the mean free path along an entire boundary face.
____________________________________________________________________*/
void meanFreePath(DataWarehouse     * new_dw,
                  const Patch       * patch,
                  SimulationStateP  & sharedState,
                  slip_localVars    * sv)
{
  cout_doing << "meanFreePath" << endl;

  // for readability
  CCVariable<double>&     lamda          = sv->lamda;
  constCCVariable<double>& gamma         = sv->gamma;
  constCCVariable<double>& specific_heat = sv->specific_heat;
  constCCVariable<double>& viscosity     = sv->viscosity;
  constCCVariable<double>& rho_CC        = sv->rho_CC;
  constCCVariable<double>& temp_CC       = sv->temp_CC;
  new_dw->allocateTemporary(sv->lamda,patch);

  sv->lamda.initialize(-9e99);

  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;

    if ( is_MicroSlip_face(patch,face, sharedState) ) {

      // hit the cells in one cell from the face direction
      IntVector offset    = patch->faceDirection(face);
      IntVector axes      = patch->getFaceAxes(face);
      IntVector patchLoEC = patch->getExtraCellLowIndex();
      IntVector patchHiEC = patch->getExtraCellHighIndex();
      Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

      // Investigate using patch::InteriorFaceCells instead of ExtraPlusEdgeCells  -Todd

      for(CellIterator cIter=patch->getFaceIterator(face, PEC); !cIter.done(); cIter++) {

        IntVector c = *cIter - offset;
        IntVector b = c;

        //  find the interior
        for(int i = 0; i < 3; i++) {            // Sultan:  double check what this is doing
          int dir = axes[i];                    // this could use some software engineering....

          if( c[dir] == patchLoEC[dir] ) {
            b[dir] += 1;
          }
          if( c[dir] == (patchHiEC[dir]-1) ) {
            b[dir] -= 1;
          }
        }

        double R = specific_heat[b] * (gamma[b]-1);
        lamda[c] = viscosity[b]/(rho_CC[b] * sqrt( 2 * R * temp_CC[b]/M_PI) );
      }  // faceCelliterator
    }  // is microSlip face
  }  // loop over faces
}
/*______________________________________________________________________
 Function~  preprocess_MicroSlip_BCs--
 Purpose~   Retrieve variables from the dw
______________________________________________________________________ */
void  preprocess_MicroSlip_BCs(DataWarehouse    * old_dw,
                               DataWarehouse    * new_dw,
                               ICELabel         * lb,
                               const Patch      * patch,
                               const string     & where,
                               const int /*indx*/,
                               SimulationStateP & sharedState,
                               bool             & setMicroSlipBcs,
                               slip_localVars   * lv,
                               slip_globalVars  * gv)
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
    old_dw->get(lv->rho_CC,     lb->rho_CCLabel,        indx,patch,gn,0);
    old_dw->get(lv->vel_CC,     lb->vel_CCLabel,        indx,patch,gn,0);
    old_dw->get(lv->temp_CC,    lb->temp_CCLabel,       indx,patch,gn,0);
    new_dw->get(lv->viscosity,  lb->viscosityLabel,     indx,patch,gn,0);
    new_dw->get(lv->press_CC,   lb->press_CCLabel,      0,   patch,gn,0);
#endif
  }
  //__________________________________
  //    cc_ Exchange
  if(where == "CC_Exchange"){
    setMicroSlipBcs = true;
    new_dw->get(lv->rho_CC,        lb->rho_CCLabel,          indx,patch,gn,0);
    new_dw->get(lv->vel_CC,        lb->vel_CC_XchangeLabel,  indx,patch,gn,0);
    new_dw->get(lv->temp_CC,       lb->temp_CC_XchangeLabel, indx,patch,gn,0);
    new_dw->get(lv->gamma,         lb->gammaLabel,           indx,patch,gn,0);
    new_dw->get(lv->specific_heat, lb->specific_heatLabel,   indx,patch,gn,0);
    new_dw->get(lv->thermalCond,   lb->thermalCondLabel,     indx,patch,gn,0);
    new_dw->get(lv->viscosity,     lb->viscosityLabel,       indx,patch,gn,0);
  }
  //__________________________________
  //    Advection
  if(where == "Advection"){
    setMicroSlipBcs = true;
    new_dw->get(lv->rho_CC,        lb->rho_CCLabel,        indx,patch,gn,0);
    new_dw->get(lv->vel_CC,        lb->vel_CCLabel,        indx,patch,gn,0);
    new_dw->get(lv->temp_CC,       lb->temp_CCLabel,       indx,patch,gn,0);
    new_dw->get(lv->gamma,         lb->gammaLabel,         indx,patch,gn,0);
    new_dw->get(lv->specific_heat, lb->specific_heatLabel, indx,patch,gn,0);
    new_dw->get(lv->thermalCond,   lb->thermalCondLabel,   indx,patch,gn,0);
    new_dw->get(lv->viscosity,     lb->viscosityLabel,     indx,patch,gn,0);
  }
  //__________________________________
  //  compute the mean free path
  if(setMicroSlipBcs) {
    cout_doing << "preprocess_microSlip_BCs on patch "<<patch->getID()<< endl;
    lv->alpha_momentum    = gv->alpha_momentum;
    lv->alpha_temperature = gv->alpha_temperature;
    lv->SlipModel         = gv->SlipModel;
    lv->CreepFlow         = gv->CreepFlow;
    meanFreePath(new_dw, patch, sharedState, lv);
  }
}
/* ______________________________________________________________________
 Function~  is_MicroSlip_face--
 Purpose~   returns true if this face on this patch is using MicroSlip bcs
 ______________________________________________________________________  */
bool is_MicroSlip_face(const Patch      * patch,
                       Patch::FaceType    face,
                       SimulationStateP & sharedState)
{
  bool is_MicroSlip_face = false;
  int numMatls = sharedState->getNumICEMatls();

  for (int m = 0; m < numMatls; m++ ) {
    ICEMaterial* ice_matl = sharedState->getICEMaterial(m);
    int indx              = ice_matl->getDWIndex();
    bool slip_temperature = patch->haveBC( face, indx, "slip", "Temperature");
    bool slip_velocity    = patch->haveBC( face, indx, "slip",  "Velocity");
    bool creep_velocity   = patch->haveBC( face, indx, "creep", "Velocity");

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
int set_MicroSlipVelocity_BC(const Patch          * patch,
                             const Patch::FaceType  face,
                             CCVariable<Vector>   & vel_CC,
                             const string         & var_desc,
                             Iterator             & bound_ptr,
                             const string         & bc_kind,
                             const Vector           wall_vel,
                             slip_localVars       * lv)

{
  int nCells = 0;
  if (var_desc == "Velocity" && (bc_kind == "slip" || bc_kind == "creep")) {

    cout_doing << "Setting FaceVel_MicroSlip on face " << face
               << " wall Velocity " << wall_vel << endl;

    // bulletproofing
    if (!lv){
      throw InternalError("set_MicroSlipTemperature_BC: Microslip_localVars = null", __FILE__, __LINE__);
    }


    // define shortcuts
    CCVariable<double>&     lamda     = lv->lamda;
    constCCVariable<double> viscosity = lv->viscosity;
    constCCVariable<double> press_CC  = lv->press_CC;
    constCCVariable<double> rho_CC    = lv->rho_CC;
    constCCVariable<double> temp_CC   = lv->temp_CC;

    double alpha_momentum = lv->alpha_momentum;
    std::string SlipModel = lv->SlipModel;
    bool        CreepFlow = lv->CreepFlow;

    IntVector patchLoEC = patch->getExtraCellLowIndex();
    IntVector patchHiEC = patch->getExtraCellHighIndex();
    IntVector offset    = patch->faceDirection(face);
    IntVector axes      = patch->getFaceAxes(face);

    int pDir  = axes[0];                       // principal direction
    Vector DX = patch->dCell();

    cout_dbg << "____________________velocity";

    //__________________________________
    //   SLIP
    if(bc_kind == "slip") {
      cout_dbg << " SLIP"<< endl;

      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
        IntVector c = *bound_ptr;

        IntVector b = c;

        for (int i = 0; i < 3; i++){
          int dir = axes[i];

          if (c[dir] == patchLoEC[dir]) {
            b[dir] += 1;
          }
          if (c[dir] == (patchHiEC[dir]-1)) {
            b[dir] -= 1;
          }
        }

        Vector V1 = vel_CC[b];
        Vector V2 = vel_CC[b-offset];
        Vector V3 = vel_CC[b-offset-offset];
        Vector V4 = vel_CC[b-offset-offset-offset];

        double velgrad1[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        double velgrad2[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        double tempgrad[3]    = {0,0,0};

        //__________________________________
        // normal velocity gradients
        for(int j = 0;j < 3;j++){  //normal derivatives
          double dx = DX[pDir];       // dx in normal dir
          velgrad1[pDir][j] = ( -5*V1[j] + 8*V2[j] - 3*V3[j])/(2*dx);
          velgrad2[pDir][j] = (  3*V1[j] - 8*V2[j] + 7*V3[j] - 2*V4[j])/( dx * dx );
        }

        //__________________________________
        //transverse derivatives
        for(int i=1;i<3;i++){
          int dir = axes[i];
          double dx = DX[dir];

          if( (patchHiEC[dir] - patchLoEC[dir]) >4 ){
            IntVector a = b;

            if( b[dir] == (patchLoEC[dir] + 1) ) {
              a[dir] += 1;
            }
            if( b[dir] == (patchHiEC[dir] - 2) ) {
              a[dir] -= 1;
            }

            IntVector R = a;
            IntVector L = a;
            R[dir] += 1;
            L[dir] -= 1;

            tempgrad[dir] = ( temp_CC[R] - temp_CC[L] )/( 2*dx );

            for(int j = 0;j < 3;j++){
              velgrad1[dir][j] = ( vel_CC[R][j] -  vel_CC[L][j])/( 2*dx );
              velgrad2[dir][j] = ( vel_CC[R][j] - 2*vel_CC[a][j] + vel_CC[L][j] )/( dx * dx );
            }
          }  // hi - lo > 4 loop
        }  // i loop

        //__________________________________
        //  update the velocity
        double Bv1 = lamda[b] * (2 - alpha_momentum)/alpha_momentum;
        vel_CC[c] = wall_vel;

        for(int i = 1; i < 3; i++){
          int dir = axes[i];
          vel_CC[c][dir] += Bv1 * ( velgrad1[pDir][dir] + velgrad1[dir][pDir] );

          // add creep contribution
          if( CreepFlow ){
            vel_CC[c][dir] += 0.75 * viscosity[b] * tempgrad[dir]/( rho_CC[b] * temp_CC[b] );
          }

          if(SlipModel == "Deissler"){
            double Bv2 = (9.0/8.0) * lamda[b] * lamda[b];
            vel_CC[c][dir] -= Bv2 * velgrad2[pDir][dir];
            vel_CC[c][dir] -= 0.5 * Bv2 * ( velgrad2[axes[1]][dir] + velgrad2[axes[2]][dir] );
          }

          if(SlipModel == "Karniadakis-Beskok"){
            double Bv2 = -0.5 * Bv1 * lamda[b];
            vel_CC[c][dir] -= Bv2 * velgrad2[pDir][dir];
          }
        }
      }
      nCells +=bound_ptr.size();;
    }
  }
  return nCells;
}

/*_________________________________________________________________
 Function~ set_MicroSlipTemperature_BC--
 Purpose~  Compute temperature in boundary cells on faces
___________________________________________________________________*/
int  set_MicroSlipTemperature_BC(const Patch            * patch,
                                 const Patch::FaceType    face,
                                 CCVariable<double>     & temp_CC,
                                 Iterator               & bound_ptr,
                                 const string           & bc_kind,
                                 const double             wall_temp,
                                 slip_localVars         * lv)
{
  int nCells = 0;
  if (bc_kind == "slip") {
    cout_doing << "Setting FaceTemp_MicroSlip on face " <<face
               << " wall Temperature " << wall_temp << endl;

    // bulletproofing
    if (!lv){
      throw InternalError("set_MicroSlipTemperature_BC: slip_localVars = null", __FILE__, __LINE__);
    }

    constCCVariable<double>& gamma         = lv->gamma;
    constCCVariable<double>& specific_heat = lv->specific_heat;
    constCCVariable<double>& thermalCond   = lv->thermalCond;
    constCCVariable<double>& viscosity     = lv->viscosity;
    CCVariable<double>& lamda              = lv->lamda;

    double alpha_temperature = lv->alpha_temperature;
    string SlipModel         = lv->SlipModel;

    IntVector axes      = patch->getFaceAxes(face);
    IntVector offset    = patch->faceDirection(face);
    IntVector patchLoEC = patch->getExtraCellLowIndex();        // Aimie double check this -Todd
    IntVector patchHiEC = patch->getExtraCellHighIndex();

    Vector DX = patch->dCell(); //axes[0], normal direction
    int pDir  = axes[0];

    cout_dbg << "\n____________________Temp"<< endl;

    //__________________________________
    //
    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
      IntVector c  = *bound_ptr;
      IntVector in = c - offset;

      IntVector b = c;

      //__________________________________
      // compute the gradients
      double tempgrad1[3] = {0,0,0};                // consider using a Uintah::Vector.
      double tempgrad2[3] = {0,0,0};

      for (int i = 0; i < 3; i++){
        int dir = axes[i];

        if (c[dir] == patchLoEC[dir]) {
          b[dir] += 1;
        }
        if (c[dir] == (patchHiEC[dir]-1)) {
          b[dir] -= 1;
        }
      }

      double T1 = temp_CC[b];
      double T2 = temp_CC[b-offset];
      double T3 = temp_CC[b-offset-offset];
      double T4 = temp_CC[b-offset-offset-offset];

      tempgrad1[pDir] = ( -5*T1 + 8*T2 - 3*T3 )/( 2*DX[pDir] ); //normal derivatives
      tempgrad2[pDir] = (  3*T1 - 8*T2 + 7*T3 - 2*T4 )/( DX[pDir]*DX[pDir] );

      //__________________________________
      // compute the transverse gradients
      for(int i = 1; i < 3; i++){
        int dir   = axes[i];
        double dx = DX[dir];

        if( (patchHiEC[dir] - patchLoEC[dir] ) > 4){
          IntVector a = b;

          if( b[dir] == (patchLoEC[dir]+1) ) {
            a[dir] += 1;
          }
          if( b[dir] == (patchHiEC[dir]-2) ) {
            a[dir] -= 1;
          }

          IntVector R = a;
          IntVector L = a;

          R[dir] += 1;
          L[dir] -= 1;

          tempgrad1[dir] = ( temp_CC[R] - temp_CC[L] )/( 2*dx );
          tempgrad2[dir] = ( temp_CC[R] - 2*temp_CC[a] + temp_CC[L] )/( dx*dx );
        }
      }
      double Bt1 = lamda[b] * ( 2 - alpha_temperature)/alpha_temperature;
      Bt1        = Bt1 * 2 * thermalCond[b]/( (gamma[b] + 1) * specific_heat[b] * viscosity[b]);

      temp_CC[c] = wall_temp + Bt1 * tempgrad1[pDir];

      if(SlipModel == "Deissler"){
        double Bt2 = lamda[b] * lamda[b] * (9.0/128.0) * (177.0 * gamma[b] - 145.0)/(gamma[b] + 1.0);
        temp_CC[c] -= Bt2 * ( tempgrad2[pDir] + 0.5*( tempgrad2[axes[1]] + tempgrad2[axes[2]] ) );
      }
      if(SlipModel == "Karniadakis-Beskok"){
        double Bt2 = -(Bt1/2) * lamda[b];
        temp_CC[c] -= Bt2 * tempgrad2[pDir];
      }
    }
    nCells = bound_ptr.size();
  }
  return nCells;
}


}  // using namespace Uintah
