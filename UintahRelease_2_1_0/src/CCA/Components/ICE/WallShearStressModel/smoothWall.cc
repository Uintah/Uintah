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

#include <CCA/Components/ICE/WallShearStressModel/smoothWall.h>
#include <CCA/Components/ICE/BoundaryCond.h>

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
//  Reference 1:  R. Stoll, F. Porte-Agel, "Effect of Roughness on Surface Boundary
//  Conditions, for Large-Eddy Simulation", Boundary Layer Meteorology (2006),
//  118: 169-187
//  Reference 2: Pope 2000
//  Reference 3: Handout on evernote - urbanmodelpaper
//______________________________________________________________________

smoothwall::smoothwall(ProblemSpecP& ps, SimulationStateP& sharedState)
  : WallShearStress(ps, sharedState)
{
  d_sharedState = sharedState;

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

  // default values
  d_invVonKarman      = 1.0/0.4;  // default value
  d_B_const           = 5.2;      // B constant
  d_convergence_uTau  = 0.1;      // convergence criteria for uTau (%)
  d_max_iter          = 100;      // Maximum iterations
  d_uTau_guess        = 1.0;

#if 0             // do we need these??
  ps->get( "friction_vel",        d_uTau_guess);
#endif
}

//__________________________________
//  Destructor
smoothwall::~smoothwall()
{
}

//______________________________________________________________________
//
void smoothwall::sched_Initialize(SchedulerP& sched,
                                   const LevelP& level,
                                   const MaterialSet* matls)
{
}
//______________________________________________________________________
//
void smoothwall::Initialize(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse*,
                             DataWarehouse* new_dw)
{
}

//______________________________________________________________________
//  Schedule variables that are needed by this model
void smoothwall::sched_AddComputeRequires(Task* task,
                                           const MaterialSubset* matls)
{
}

//______________________________________________________________________
//  Wrapper around the calls for the individual components
void smoothwall::computeWallShearStresses( DataWarehouse* old_dw,
                                           DataWarehouse* new_dw,
                                           const Patch* patch,
                                           const int indx,
                                           constCCVariable<double>& vol_frac_CC,
                                           constCCVariable<Vector>& vel_CC,
                                           const CCVariable<double>& viscosity,
                                           constCCVariable<double>& rho_CC,
                                           SFCXVariable<Vector>& tau_X_FC,
                                           SFCYVariable<Vector>& tau_Y_FC,
                                           SFCZVariable<Vector>& tau_Z_FC )
{
  if( d_face == Patch::xminus || d_face == Patch::xplus ){
    wallShearStresses< SFCXVariable<Vector> >( old_dw, new_dw, patch, indx, viscosity, vol_frac_CC, rho_CC, vel_CC, tau_X_FC);
  }

  if( d_face == Patch::yminus || d_face == Patch::yplus ){
    wallShearStresses< SFCYVariable<Vector> >( old_dw, new_dw, patch, indx, viscosity, vol_frac_CC, rho_CC, vel_CC, tau_Y_FC);
  }

  if( d_face == Patch::zminus || d_face == Patch::zplus ){
    wallShearStresses< SFCZVariable<Vector> >( old_dw, new_dw, patch, indx, viscosity, vol_frac_CC, rho_CC, vel_CC, tau_Z_FC);
  }
}

//______________________________________________________________________
//
template<class T>
void smoothwall::wallShearStresses(DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    const Patch* patch,
                                    const int indx,
                                    const CCVariable<double>& viscosity,
                                    constCCVariable<double>& vol_frac_CC,
                                    constCCVariable<double>& rho_CC,
                                    constCCVariable<Vector>& vel_CC,
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
      IntVector oneCell(0,0,0);
      if( d_face == Patch::xplus || d_face == Patch::yplus || d_face == Patch::zplus){
        oneCell= patch->faceDirection(face);
      }
/*`==========TESTING==========*/
#if 0
     cout << " smoothWallModel " << patch->getFaceName(face) << " iterator: "
          << patch->getFaceIterator(face, Patch::SFCVars) << endl;
 
     cout << "    oneCell: " << oneCell << " streamwiseDir: "<< d_streamwiseDir <<" dir1: " << dir1 << " dir2 " << dir2 << " p_dir " << p_dir
          << " z_CC: " << z_CC << endl;
#endif
/*===========TESTING==========`*/
      double uTau        = d_uTau_guess;                      // We need to talk about this
      //__________________________________
      //
      for(CellIterator iter = patch->getFaceIterator(face, Patch::SFCVars); !iter.done(); iter++){
        IntVector c = *iter;
        IntVector adj  = c - oneCell;                           // for x+, y+, z+ faces use the velocity from the adjacent cell
        
        if (vol_frac_CC[c] < 0.5 ) {                           
          //cout << "  smoothWallModel: skipping cell: " << c << endl;
          continue;
        }

        double vel1 = vel_CC[adj][dir1];                        // transverse velocity components
        double vel2 = vel_CC[adj][dir2];
        double u_tilde = ( pow(vel1, 2)  + pow(vel2, 2) );
	 u_tilde = sqrt( u_tilde ) + SMALL_NUM;                   // avoid division by 0
        double u_streamwise = u_tilde;

        double rel_error   = DBL_MAX;
        double k_viscosity = viscosity[c] / rho_CC[c];          // for readability and optimization
        unsigned int count = 0;

        // Solve implicitly for uTau using Newton's Method, Eq 5 - handout
        while( count < d_max_iter && rel_error > d_convergence_uTau) {
          count ++;
          double uTau_old = uTau;

          double fx  = u_streamwise - uTau * (d_invVonKarman * log( z_CC * uTau/k_viscosity)) + d_B_const;
          
          double dfx = -uTau * ( d_invVonKarman * 1/uTau) - (d_invVonKarman * log( z_CC * uTau/k_viscosity));

          double uTau_new = uTau_old - fx / dfx;
          rel_error = fabs( (uTau_new - uTau_old)/uTau_new )*100;
          uTau = uTau_new;
/*`==========TESTING==========*/
          #if 0
            cout << setw(15) << " uTau_old: " << uTau_old << " d_invVonKarman: " << d_invVonKarman << " u_streamwise: " << u_streamwise << " k_viscosity: " << k_viscosity << " dfx: " << dfx << " fx: " << fx << " change: " << fx/dfx << " rel_error: " << rel_error << endl;
          #endif

/*===========TESTING==========`*/ 
        }
        // end of solve for uTau

        // bulletproofing
        if (count == d_max_iter || std::isnan(uTau) ) {
          ostringstream warn;
          warn << "ERROR: ICE:smoothwall::wallShearStresses cell (" << c << ") exeeded the max iterations, or a NAN was detected"
               << " when computing uTau.\n"
               << "     u_streamwise: " << u_streamwise << "\n"
               << "     k_viscosity:  " << k_viscosity << "\n"
               << "     viscosity:    " << viscosity[c] << "\n"
               << "     rho_CC:       " << rho_CC[c];
          throw InternalError(warn.str(), __FILE__, __LINE__);
        }

        // Eq 6 - handout
        double tau_s   = rho_CC[c] * pow(uTau, 2);

        Vector tau_tmp(0,0,0);
        tau_tmp[dir1] = tau_s * vol_frac_CC[adj] *( vel1/u_tilde );
        tau_tmp[dir2] = tau_s * vol_frac_CC[adj] *( vel2/u_tilde );

/*`==========TESTING==========*/
#if 0
        cout << " c " << c << " adj " << adj << setw(8) <<" u_tilde: " << u_tilde
             << setw(8) << " tau_s: " << tau_s << " Tau_tmp " << tau_tmp << setw(8) << " u_streamwise: " << u_streamwise
             << " count: " << count << " uTau:  " << uTau << " rho_CC[c]: " << rho_CC[c] << endl;
#endif
/*===========TESTING==========`*/

        Tau_FC[c] = tau_tmp;
      }
    }  // face
  }  // face iterator
}
