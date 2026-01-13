/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#include <CCA/Components/ICE/CustomBCs/temporalBCs.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Math/MiscMath.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Math/MiscMath.h>
#include <typeinfo>
#include <Core/Util/DebugStream.h>

using namespace std;

namespace Uintah {
//__________________________________
//  To turn on couts

static Uintah::DebugStream cout_doing("ICE_BC_CC", false);

/* ______________________________________________________________________
 Function~  read_temporal_BC_inputs--
 Purpose~   -returns (true) if the sine BC is specified on any face,
            -reads input parameters thar are need by the setBC routines
 ______________________________________________________________________  */
bool readInputs_temporal_BCs(const ProblemSpecP    & prob_spec,
                             temporal_globalVars * gv)
{
  //__________________________________
  // search the BoundaryConditions problem spec
  // determine if temporalBCs
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");

  bool usingTemporalBC = false;

  for( ProblemSpecP face_ps = bc_ps->findBlock( "Face" ); face_ps != nullptr; face_ps=face_ps->findNextBlock( "Face" ) ) {

    for( ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != nullptr; bc_iter = bc_iter->findNextBlock( "BCType" ) ) {
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);

      if (bc_type["var"] == "TemporalBC") {
        usingTemporalBC = true;
      }
    }
  }

  //__________________________________
  //  read in variables need by the boundary
  //  conditions and put them in the variable basket
  if(usingTemporalBC ){
    ProblemSpecP sine = bc_ps->findBlock("TEMPORAL_BC");
    if (!sine) {
      string warn="ERROR:\n Inputs:Boundary Conditions: Cannot find TEMPORAL_BC block";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }
  }
  return usingTemporalBC;
}

/* ______________________________________________________________________
 Function~  addRequires_temporalBCs--
 Purpose~   requires
 ______________________________________________________________________  */
void addRequires_Temporal( Task          * t,
                           const string  & where,
                           ICELabel      * lb,
                           const MaterialSubset * /*ice_matls*/)
{
  cout_doing<< "Doing addRequires_temporalBCs: \t\t" <<t->getName()
            << " " << where << endl;


#if 0
  Ghost::GhostType  gn  = Ghost::None;
  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  MaterialSubset* press_matl = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();

  if(where == "CC_Exchange"){

  }
  if(where == "Advection"){
  }
#endif
}

/*______________________________________________________________________
 Function~  preprocess_temporal_BCs--
 Purpose~   get data from the datawarehouse
______________________________________________________________________ */
void  preprocess_temporal_BCs(DataWarehouse * new_dw,
                              DataWarehouse * /*old_dw*/,
                              ICELabel      * lb,
                              const int /*indx*/,
                              const Patch   * patch,
                              const string  & where,
                              bool          & setTemporal_BCS,
                              temporal_localVars* lv)
{
//  Ghost::GhostType  gn  = Ghost::None;
  setTemporal_BCS = false;
  lv->where = where;
  //__________________________________
  //    Equilibrium pressure
  if(where == "EqPress"){
    setTemporal_BCS = true;
    lv->delT = 0.0;  // Don't include delt at this point in the timestep
  }
  //__________________________________
  //    Explicit and semi-implicit update pressure
  if(where == "update_press_CC"){
    setTemporal_BCS = true;
  }
  if(where == "implicitPressureSolve"){
    setTemporal_BCS = true;
  }

  if(where == "imp_update_press_CC"){
    setTemporal_BCS = true;
  }
  //__________________________________
  //    cc_ Exchange
  if(where == "CC_Exchange"){
    setTemporal_BCS = true;
  }
  //__________________________________
  //    Advection
  if(where == "Advection"){
    setTemporal_BCS = true;
  }
}
/*_________________________________________________________________
 Function~ setVelocity_temporalBC--
 Purpose~  Set velocity boundary conditions
___________________________________________________________________*/
int  setVelocity_temporalBC(  const Patch         * patch,
                              const Patch::FaceType face,
                              CCVariable<Vector>  & vel_CC,
                              const string        & var_desc,
                              Iterator            & bound_ptr,
                              const string        & bc_kind,
                              MaterialManagerP    & materialManager,
                              temporal_globalVars * gv,
                              temporal_localVars  * lv)

{
  int nCells = 0;
  if (var_desc == "Velocity" && bc_kind == "TemporalBC") {
    cout_doing << "    Vel_CC (temporalBC) \t\t" <<patch->getFaceName(face)<< endl;

    // bulletproofing
    if (!gv || !lv){
      throw InternalError("setVelocity_temporalBC", __FILE__, __LINE__);
    }

    // double t  = lv->simTime + lv->delT;
    // only alter the velocity in the direction that the reference_velocity
    // is non-zero.
    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
      IntVector c = *bound_ptr;
      vel_CC[c].x(0 );
      vel_CC[c].y(0 );
      vel_CC[c].z(0 );
    }
    nCells += bound_ptr.size();
  }
  return nCells;
}

/*_________________________________________________________________
 Function~ setTemp_temporalBC--
 Purpose~  Set temperature boundary conditions
___________________________________________________________________*/
int setTemp_temporalBC(const Patch         * patch,
                       const Patch::FaceType face,
                       CCVariable<double>  & temp_CC,
                       Iterator            & bound_ptr,
                       const string        & bc_kind,
                       temporal_globalVars * gv,
                       temporal_localVars  * lv)
{
  int nCells = 0;
  if (bc_kind == "TemporalBC") {
    cout_doing << "    Temp_CC (temporalBC) \t\t" <<patch->getFaceName(face)<< endl;

    // bulletproofing
    if (!gv || !lv){
      throw InternalError("setTemp_temporalBC", __FILE__, __LINE__);
    }

    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
      IntVector c = *bound_ptr;
    }
    nCells += bound_ptr.size();
  }
  return nCells;
}

/*_________________________________________________________________
 Function~ setPress_temporalBC--
 Purpose~  Set press boundary conditions
___________________________________________________________________*/
int setPress_temporalBC(const Patch         * patch,
                        const Patch::FaceType face,
                        CCVariable<double>  & press_CC,
                        Iterator            & bound_ptr,
                        const string        & bc_kind,
                        MaterialManagerP    & materialManager,
                        temporal_globalVars * gv,
                        temporal_localVars  * lv)
{
  cout_doing << "    press_CC (temporalBC) \t\t" <<patch->getFaceName(face)<< endl;

  // bulletproofing
  if (!gv || !lv){
    throw InternalError("setPress_temporalBC: lvars = null", __FILE__, __LINE__);
  }

  int nCells = 0;
  // double t   = lv->simTime + lv->delT;

  for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
    IntVector c = *bound_ptr;
  }
  nCells += bound_ptr.size();
  return nCells;
}

}  // using namespace Uintah
