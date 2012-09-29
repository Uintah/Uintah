/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/********************************************************************************
    Crack.cc
    PART SIX: MOVE CRACK POINTS 

    Created by Yajun Guo in 2002-2005.
********************************************************************************/

#include "Crack.h"
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <vector>
#include <iostream>
#include <fstream>

// AIX is defining hz to something else
#ifdef _AIX
    #ifdef hz
        #undef hz
    #endif
#endif

using namespace Uintah;
using namespace std;

using std::vector;
using std::string;


void
Crack::addComputesAndRequiresCrackPointSubset(Task* /*t*/,
                                              const PatchSet* /*patches*/,
                                              const MaterialSet* /*matls*/) const
{
  // Currently do nothing
}

void
Crack::CrackPointSubset(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* /*matls*/,
                        DataWarehouse* /*old_dw*/,
                        DataWarehouse* /*new_dw*/)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      cnset[m][pid].clear();
      
      // Collect crack nodes in each patch
      for(int i=0; i<(int)cx[m].size(); i++) {
        if(patch->containsPointInExtraCells(cx[m][i])) {
          cnset[m][pid].push_back(i);
        }
      } 
      
      MPI_Barrier(mpi_crack_comm);

      // Broadcast cnset to all the ranks
      for(int i=0; i<patch_size; i++) {
        int num; // number of crack nodes in patch i
        if(i==pid) num=cnset[m][i].size();
        MPI_Bcast(&num,1,MPI_INT,i,mpi_crack_comm);
        if(pid!=i) cnset[m][i].resize(num);
        MPI_Bcast(&cnset[m][i][0],num,MPI_INT,i,mpi_crack_comm);
      }
      
    } // End of loop over matls
  }  
}

void
Crack::addComputesAndRequiresMoveCracks(Task* t,
                                        const PatchSet* /*patches*/,
                                        const MaterialSet* /*matls*/) const
{
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType  gac = Ghost::AroundCells;
  int NGC=2*NGN;
  t->requires(Task::NewDW, lb->gMassLabel,         gac, NGC);
  t->requires(Task::NewDW, lb->gNumPatlsLabel,     gac, NGC);
  t->requires(Task::NewDW, lb->gVelocityStarLabel, gac, NGC);
  t->requires(Task::NewDW, lb->GMassLabel,         gac, NGC);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,     gac, NGC);
  t->requires(Task::NewDW, lb->GVelocityStarLabel, gac, NGC);

  t->requires(Task::OldDW,lb->pSizeLabel, Ghost::None);
  t->requires(Task::OldDW,lb->pDeformationMeasureLabel, Ghost::None);
}

void
Crack::MoveCracks(const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset* /*matls*/,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);
    MPI_Datatype MPI_POINT=fun_getTypeDescription((Point*)0)->getMPIType();


    Vector dx = patch->dCell();
    double dx_min=Min(dx.x(),dx.y(),dx.z());

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),getLevel(patches) );

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){ 
      if((int)ce[m].size()==0) // for materials with no cracks
        continue;

      
      // Task 1: Move crack nodes (cx)
     
      // Get the necessary information
      MPMMaterial* mpm_matl=d_sharedState->getMPMMaterial(m);
      int dwi=mpm_matl->getDWIndex();

      ParticleSubset* pset=old_dw->getParticleSubset(dwi,patch);
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> deformationGradient;
      old_dw->get(psize,lb->pSizeLabel,pset);
      old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

      Ghost::GhostType  gac = Ghost::AroundCells;
      constNCVariable<double> gmass,Gmass;
      constNCVariable<int>    gnum,Gnum;
      constNCVariable<Vector> gvelocity_star, Gvelocity_star;
      int NGC=2*NGN;
      new_dw->get(gmass,         lb->gMassLabel,        dwi,patch,gac,NGC);
      new_dw->get(gnum,          lb->gNumPatlsLabel,    dwi,patch,gac,NGC);
      new_dw->get(gvelocity_star,lb->gVelocityStarLabel,dwi,patch,gac,NGC);
      new_dw->get(Gmass,         lb->GMassLabel,        dwi,patch,gac,NGC);
      new_dw->get(Gnum,          lb->GNumPatlsLabel,    dwi,patch,gac,NGC);
      new_dw->get(Gvelocity_star,lb->GVelocityStarLabel,dwi,patch,gac,NGC);

      for(int i=0; i<patch_size; i++) { 
        int numNodes=cnset[m][i].size();
        if(numNodes>0) {
          Point* cptmp=new Point[numNodes];
          if(pid==i) { // Rank i updates the nodes in patch i
            for(int j=0; j<numNodes; j++) {
              int idx=cnset[m][i][j];
              Point pt=cx[m][idx];

              double mg,mG;
              Vector vg,vG;
              Vector vcm = Vector(0.0,0.0,0.0);

              interpolator->findCellAndWeights(pt, ni, S, psize[idx],deformationGradient[idx]);

              // Calculate center-of-velocity (vcm)
              // Sum of shape functions from nodes with particle(s) around them
              // This part is necessary for crack nodes outside the material
              double sumS=0.0;
              for(int k =0; k < n8or27; k++) {
                Point pi=patch->nodePosition(ni[k]);
                if(PhysicalGlobalGridContainsPoint(dx_min,pi) &&  //ni[k] in real grid
                     (gnum[ni[k]]+Gnum[ni[k]]!=0)) {
                  sumS += S[k];
                }
              }
              if(sumS>1.e-6) {
                for(int k = 0; k < n8or27; k++) {
                  Point pi=patch->nodePosition(ni[k]);
                  if(PhysicalGlobalGridContainsPoint(dx_min,pi) &&
                             (gnum[ni[k]]+Gnum[ni[k]]!=0)) {
                    mg = gmass[ni[k]];
                    mG = Gmass[ni[k]];
                    vg = gvelocity_star[ni[k]];
                    vG = Gvelocity_star[ni[k]];
                    vcm += (mg*vg+mG*vG)/(mg+mG)*S[k]/sumS;
                  }
                }
              }

              // Update the position
              cptmp[j]=pt+vcm*delT;

              // Apply symmetric BCs for the new position
              ApplySymmetricBCsToCrackPoints(dx,pt,cptmp[j]);

              // Check if the node is still inside the grid
              if(!PhysicalGlobalGridContainsPoint(dx_min,cptmp[j])) {
                cout << "Error: cx[" << m << "," << idx << "]=" << cx[m][idx]
                     << " has moved to " << cptmp[j] << ", outside the global grid."
                     << " Center-of-mass velocity, vcm=" << vcm 
                     << ". Program terminated." << endl;
                exit(1);
              }   
              
            } // End of loop over numNodes
          } // End if(pid==i)

          // Broadcast the updated position to all ranks
          MPI_Bcast(cptmp,numNodes,MPI_POINT,i,mpi_crack_comm);

          // Update cx
          for(int j=0; j<numNodes; j++) {
            int idx=cnset[m][i][j];
            cx[m][idx]=cptmp[j];
          }

          delete [] cptmp;

        } // End of if(numNodes>0)
      } // End of loop over patch_size

      MPI_Barrier(mpi_crack_comm);
      

      // Task 2: Update crack extent
     
      cmin[m]=Point(9.e16,9.e16,9.e16);
      cmax[m]=Point(-9.e16,-9.e16,-9.e16);
      for(int i=0; i<(int)cx[m].size(); i++) {
        cmin[m]=Min(cmin[m],cx[m][i]);
        cmax[m]=Max(cmax[m],cx[m][i]);
      } 

    } // End of loop over matls
    delete interpolator;
  }
}

// Find if a point is inside the real global grid
short Crack::PhysicalGlobalGridContainsPoint(const double& dx,const Point& pt)
{
  // Return true if pt is inside the real global grid or
  // around it (within 1% of cell-size)

  double px=pt.x(),  py=pt.y(),  pz=pt.z();
  double lx=GLP.x(), ly=GLP.y(), lz=GLP.z();
  double hx=GHP.x(), hy=GHP.y(), hz=GHP.z();

  return ((px>lx || fabs(px-lx)/dx<0.01) && (px<hx || fabs(px-hx)/dx<0.01) &&
          (py>ly || fabs(py-ly)/dx<0.01) && (py<hy || fabs(py-hy)/dx<0.01) &&
          (pz>lz || fabs(pz-lz)/dx<0.01) && (pz<hz || fabs(pz-hz)/dx<0.01));
}

// Apply symmetric boundary condition to crack points
void
Crack::ApplySymmetricBCsToCrackPoints(const Vector& cs,
                                      const Point& old_pt,Point& new_pt)
{
  // cs -- cell size
  for(Patch::FaceType face = Patch::startFace;
       face<=Patch::endFace; face=Patch::nextFace(face)) {
    if(GridBCType[face]=="symmetry") {
      if( face==Patch::xminus && fabs(old_pt.x()-GLP.x())/cs.x()<1.e-2 )
        new_pt(0)=GLP.x(); // On symmetric face x-
      if( face==Patch::xplus  && fabs(old_pt.x()-GHP.x())/cs.x()<1.e-2 )
        new_pt(0)=GHP.x(); // On symmetric face x+
      if( face==Patch::yminus && fabs(old_pt.y()-GLP.y())/cs.y()<1.e-2 )
        new_pt(1)=GLP.y(); // On symmetric face y-
      if( face==Patch::yplus  && fabs(old_pt.y()-GHP.y())/cs.y()<1.e-2 )
        new_pt(1)=GHP.y(); // On symmetric face y+
      if( face==Patch::zminus && fabs(old_pt.z()-GLP.z())/cs.z()<1.e-2 )
        new_pt(2)=GLP.z(); // On symmetric face z-
      if( face==Patch::zplus  && fabs(old_pt.z()-GHP.z())/cs.z()<1.e-2 )
        new_pt(2)=GHP.z(); // On symmetric face z+
    }
  }
}

