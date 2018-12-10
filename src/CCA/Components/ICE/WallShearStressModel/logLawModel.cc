/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/ICE/WallShearStressModel/logLawModel.h>

#include <Core/Grid/MaterialManager.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Patch.h>
#include <cmath>
#include <iomanip>

using namespace Uintah;
using namespace std;
static DebugStream dbg("ICE_DOING_COUT", false);
#define SMALL_NUM 1e-100

//______________________________________________________________________
//  Reference:  R. Stoll, F. Porte-Agel, "Effect of Roughness on Surface Boundary
//  Conditions, for Large-Eddy Simulation", Boundary Layer Meteorology (2006),
//  118: 169-187
//______________________________________________________________________

logLawModel::logLawModel(ProblemSpecP& ps, MaterialManagerP& materialManager)
  : WallShearStress(ps, materialManager)
{
  d_materialManager = materialManager;
    
  string face;
  ps->require("domainFace", face);
  
  if ( face == "x-" ) {
    d_face = Patch::xminus;
  }
  if ( face == "x+" ) {
    d_face = Patch::xplus;
  }
  if ( face == "y-" ) {
    d_face = Patch::yminus;
  }
  if ( face == "y+" ) {
    d_face = Patch::yplus;
  }
  if ( face == "z-" ) {
    d_face = Patch::zminus;
  }
  if ( face == "z+" ) {
    d_face = Patch::zplus;
  }

  
  d_vonKarman = 0.4;                // default value
  ps->get( "vonKarmanConstant", d_vonKarman );

  ps->require( "roughnessConstant", d_roughnessConstant );
  
  string roughnessInputFile = "none";
  ps->get( "roughnessInputFile", roughnessInputFile);
  
  if( d_roughnessConstant != -9 && roughnessInputFile != "none") {
    ostringstream warn;
    warn << "\nERROR ICE::WallShearStressModel:logLawModel\n"
         << "    You cannot specify both a constant roughness ("<< d_roughnessConstant<< ")"
         << " and read in the roughness from a file (" << roughnessInputFile << ").";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  d_roughnessLabel  = VarLabel::create("logLaw_roughness", CCVariable<double>::getTypeDescription());
  
}

logLawModel::~logLawModel()
{
   VarLabel::destroy(d_roughnessLabel);
}

//______________________________________________________________________
//
void logLawModel::sched_Initialize(SchedulerP& sched,
                                   const LevelP& level,        
                                   const MaterialSet* matls)   
{
  printSchedule(level,dbg,"logLawModel::schedInitialize");
  
  Task* t = scinew Task("logLawModel::Initialize",
                  this, &logLawModel::Initialize);

  t->computes(d_roughnessLabel);
  sched->addTask(t, level->eachPatch(), d_materialManager->allMaterials( "ICE" ));
}
//______________________________________________________________________
//
void logLawModel::Initialize(const ProcessorGroup*, 
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse*, 
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"logLawModel::Initialize");
    
    for (int m = 0; m < matls->size(); m++ ) {
      int indx = matls->get(m);
      
      CCVariable<double> roughness;
      new_dw->allocateAndPut(roughness, d_roughnessLabel, indx,patch);
      roughness.initialize( d_roughnessConstant );            // This needs to be changed
      
      new_dw->put( roughness, d_roughnessLabel, indx, patch );
    }
  }
}

//______________________________________________________________________
//  Schedule variables that are needed by this model
void logLawModel::sched_AddComputeRequires(Task* task, 
                                           const MaterialSubset* matls)
{
 // printSchedule(level,dbg,"logLawModel::schedcomputeWallShearStresses");
  task->requires(Task::OldDW, d_roughnessLabel, matls, Ghost::None, 0);
  task->computes(d_roughnessLabel);
}


//______________________________________________________________________
//  Wrapper around the calls for the individual components   
void logLawModel::computeWallShearStresses( DataWarehouse* old_dw,
                                            DataWarehouse* new_dw,
                                            const Patch* patch,
                                            const int indx,
                                            constCCVariable<double>& vol_frac_CC,  
                                            constCCVariable<Vector>& vel_CC,      
                                            const CCVariable<double>& viscosity,    
                                            constCCVariable<double>& /* rho_CC */,    
                                            SFCXVariable<Vector>& tau_X_FC,
                                            SFCYVariable<Vector>& tau_Y_FC,
                                            SFCZVariable<Vector>& tau_Z_FC )
{
  if( d_face == Patch::xminus || d_face == Patch::xplus ){
    wallShearStresses< SFCXVariable<Vector> >( old_dw, new_dw, patch, indx, vol_frac_CC, vel_CC, tau_X_FC);
  }
  
  if( d_face == Patch::yminus || d_face == Patch::yplus ){  
    wallShearStresses< SFCYVariable<Vector> >( old_dw, new_dw, patch, indx, vol_frac_CC, vel_CC, tau_Y_FC);
  }
  
  if( d_face == Patch::zminus || d_face == Patch::zplus ){  
    wallShearStresses< SFCZVariable<Vector> >( old_dw, new_dw, patch, indx, vol_frac_CC, vel_CC, tau_Z_FC);
  }                              
}

//______________________________________________________________________
//
template<class T>
void logLawModel::wallShearStresses(DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    const Patch* patch,
                                    const int indx,
                                    constCCVariable<double>& vol_frac_CC,
                                    constCCVariable<Vector>& vel_CC,
                                    T& Tau_FC)
{
  // transfer variable forward
  constCCVariable<double> roughnessOld;
  CCVariable<double>      roughness;
  
  old_dw->get(        roughnessOld,  d_roughnessLabel, indx, patch, Ghost::None, 0);
  new_dw->allocateAndPut( roughness, d_roughnessLabel, indx, patch );
  roughness.copyData( roughnessOld );
    
  //__________________________________
  // 
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);  
  for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
    Patch::FaceType face = *itr;
    
    if( face == d_face ){
      
      // Determine the streamwise (dir1) and transverse directions (dir2)
      IntVector axes = patch->getFaceAxes(face);
      int p_dir = axes[0];
      int dir1  = axes[1];
      int dir2  = axes[2];
  
      // The height (z_CC) above the domain wall
      double z_CC  = patch->dCell()[p_dir]/2.0;
      
      double denominator = log( z_CC/d_roughnessConstant );       // this assumes constant roughness
      
      IntVector oneCell(0,0,0);
      if( d_face == Patch::xplus || d_face == Patch::yplus || d_face == Patch::zplus){
        oneCell= patch->faceDirection(face);
      }
      
      
/*`==========TESTING==========*/
#if 0
      cout << " logLawModel " << patch->getFaceName(face) << " iterator: "
           << patch->getFaceIterator(face, Patch::SFCVars) << endl;

      cout << "    oneCell: " << oneCell << " dir1: " << dir1 << " dir2 " << dir2 << " p_dir " << p_dir 
           << " z_CC: " << z_CC << endl; 
#endif
/*===========TESTING==========`*/       

      //__________________________________
      //
      for(CellIterator iter = patch->getFaceIterator(face, Patch::SFCVars); !iter.done(); iter++){
        IntVector c = *iter;
        IntVector adj  = c - oneCell;                           // for x+, y+, z+ faces use the velocity from the adjacent cell
        
        double vel1 = vel_CC[adj][dir1];                        // transverse velocity components
        double vel2 = vel_CC[adj][dir2];
        
        double u_tilde = ( pow(vel1, 2)  + pow(vel2, 2) );
        u_tilde = sqrt( u_tilde ) + SMALL_NUM;                   // avoid division by 0
          
        // eq (5)
        double tau_s = pow( ( u_tilde * d_vonKarman )/denominator, 2);
        
        // eq (6)
        Vector tau_tmp(0,0,0);
        tau_tmp[dir1] = tau_s * vol_frac_CC[adj] *( vel1/u_tilde );
        tau_tmp[dir2] = tau_s * vol_frac_CC[adj] *( vel2/u_tilde );
        
/*`==========TESTING==========*/
#if 0
        cout << " c " << c << " adj " << adj << setw(8) <<" u_tilde: " << u_tilde 
             << setw(8) << " tau_s: " << tau_s << " Tau_tmp " << tau_tmp << setw(8) << "vel1: " << vel1 << " vel2 " << vel2 << endl; 
#endif
/*===========TESTING==========`*/
             
        Tau_FC[c] = tau_tmp;
      }
    }  // face
  }  // face iterator
}
