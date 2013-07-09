/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
#include <CCA/Components/ICE/BoundaryCond.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Patch.h>
#include <cmath>
using namespace Uintah;
static DebugStream cout_doing("ICE_DOING_COUT", false);

//______________________________________________________________________
//  Reference:  R. Stoll, F. Porte-Agel, "Effect of Roughness on Surface Boundary
//  Conditions, for Large-Eddy Simulation", Boundary Layer Meteorology (2006),
//  118: 169-187
//______________________________________________________________________

logLawModel::logLawModel(ProblemSpecP& ps, SimulationStateP& sharedState)
  : WallShearStress(ps, sharedState)
{
  string face;
  ps->require("face", face);
  
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
  if ( face == "z-" ) {
    d_face = Patch::zplus;
  }
  
  d_vonKarman = 0.4;                // default value
  ps->get( "vonKarmanConstant", d_vonKarman );

  ps->require( "roughnessConstant", d_roughnessConstant );
  
  string roughnessInputFile = "none";
  ps->get( "roughnessInputFile", roughnessInputFile);
  
  if( d_roughnessConstant != -9 && roughnessInputFile != "none") {
    ostringstream warn;
    warn << "ERROR ICE::WallShearStressModel:logLaw\n"
         << "    You cannot specify both a constant roughness ("<< d_roughnessConstant<< ")"
         << "     and read in the roughness from an input file (" << roughnessInputFile << ").";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
}

logLawModel::~logLawModel()
{
}

//______________________________________________________________________
//
void logLawModel::scheduleInitialize(SchedulerP& sched,
                                     const LevelP& level)
{
}


//______________________________________________________________________
//
template<class T>
void logLawModel::wallShearStresses(DataWarehouse* new_dw,
                                    const Patch* patch,
                                    const CCVariable<Vector>& vel_CC,
                                    T& Tau_FC)
{
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
      
      IntVector oneCell = patch->faceDirection(face);

      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
      for(CellIterator iter = patch->getFaceIterator(face, MEC); !iter.done(); iter++){
        IntVector c = *iter;
        IntVector adj  = c - oneCell;
        
        double vel1 = vel_CC[c][dir1];
        double vel2 = vel_CC[c][dir2];
        
        double u_tilde = ( pow( vel1, 2)  + pow(vel2, 2) );
        u_tilde = sqrt(u_tilde);
        
        // eq (5)
        double tau_s = -pow( (u_tilde * d_vonKarman )/denominator, 2);
        
        // eq (6)
        Vector Tau_tmp(0,0,0);
        Tau_tmp[dir1] = tau_s * ( vel1/u_tilde );
        Tau_tmp[dir2] = tau_s * ( vel2/u_tilde );
        Tau_FC[c] = Tau_tmp;
      }
    }  // face
  }  // face iterator
}
