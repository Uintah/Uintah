/********************************************************************************
    Crack.cc
    PART SIX: MOVE CRACK POINTS 

    Created by Yajun Guo in 2002-2004.
********************************************************************************/

#include "Crack.h"
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

using std::vector;
using std::string;


void Crack::addComputesAndRequiresCrackPointSubset(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  // Currently do nothing
}

void Crack::CrackPointSubset(const ProcessorGroup*,
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
        if(patch->containsPoint(cx[m][i])) {
          cnset[m][pid].push_back(i);
        }
      } // End of loop over nodes
      MPI_Barrier(mpi_crack_comm);

      // Broadcast cnset to all ranks
      for(int i=0; i<patch_size; i++) {
        int num; // number of crack nodes in patch i
        if(i==pid) num=cnset[m][i].size();
        MPI_Bcast(&num,1,MPI_INT,i,mpi_crack_comm);
        if(pid!=i) cnset[m][i].resize(num);
        MPI_Bcast(&cnset[m][i][0],num,MPI_INT,i,mpi_crack_comm);
      }
    } // End of loop over matls
  }  // End of loop over pathces
}

void Crack::addComputesAndRequiresMoveCracks(Task* t,
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
}

void Crack::MoveCracks(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());

    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);
    MPI_Datatype MPI_POINT=fun_getTypeDescription((Point*)0)->getMPIType();


    Vector dx = patch->dCell();
    double dx_min=Min(dx.x(),dx.y(),dx.z());
    //double vc=dx.x()*dx.y()*dx.z();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),getLevel(patches) );

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){ // loop over matls
      if((int)ce[m].size()==0) // for materials with no cracks
        continue;

      /* Task 1: Move crack points (cx)
      */
      //Get the necessary information
      MPMMaterial* mpm_matl=d_sharedState->getMPMMaterial(m);
      int dwi=mpm_matl->getDWIndex();

      // Mass of a grad cell 
      //double rho_init=mpm_matl->getInitialDensity();
      //double mc=rho_init*vc;

      ParticleSubset* pset=old_dw->getParticleSubset(dwi,patch);
      constParticleVariable<Vector> psize;
      old_dw->get(psize,lb->pSizeLabel,pset);

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

      for(int i=0; i<patch_size; i++) { // Loop over all patches
        int numNodes=cnset[m][i].size();
        if(numNodes>0) {
          Point* cptmp=new Point[numNodes];
          if(pid==i) { // Processor i updates the position of nodes in patch i
            for(int j=0; j<numNodes; j++) {
              int idx=cnset[m][i][j];
              Point pt=cx[m][idx];

              double mg,mG;
              Vector vg,vG;
              Vector vcm = Vector(0.0,0.0,0.0);

              // Get element nodes and shape functions
	      interpolator->findCellAndWeights(pt, ni, S, psize[idx]);

              // Calculate center-of-velocity (vcm)
              // Sum of shape functions from nodes with particle(s) around them
              // This part is necessary for pt located outside the body
              double sumS=0.0;
              for(int k =0; k < n8or27; k++) {
                Point pi=patch->nodePosition(ni[k]);
                if(PhysicalGlobalGridContainsPoint(dx_min,pi) &&  //ni[k] in real grid
                     (gnum[ni[k]]+Gnum[ni[k]]!=0)) {
                  //if(gmass[ni[k]]+Gmass[ni[k]]>mc/20.)
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
                    //if(mg+mG>mc/100.) {
                      vcm += (mg*vg+mG*vG)/(mg+mG)*S[k]/sumS;
                    //}
                  }
                }
              }

              // Update the position
              cptmp[j]=pt+vcm*delT;

              // Apply symmetric BCs for the new position
              ApplySymmetricBCsToCrackPoints(dx,pt,cptmp[j]);
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

      // Detect if crack points outside the global grid
      for(int i=0; i<(int)cx[m].size();i++) {
        if(!PhysicalGlobalGridContainsPoint(dx_min,cx[m][i])) {
          cout << "cx[" << m << "," << i << "]=" << cx[m][i]
               << " outside the global grid."
               << " Program terminated." << endl;
          exit(1);
        }
      }

      MPI_Barrier(mpi_crack_comm);

      /* Task 2: Update crack extent
      */
      cmin[m]=Point(9.e16,9.e16,9.e16);
      cmax[m]=Point(-9.e16,-9.e16,-9.e16);
      for(int i=0; i<(int)cx[m].size(); i++) {
        cmin[m]=Min(cmin[m],cx[m][i]);
        cmax[m]=Max(cmax[m],cx[m][i]);
      } // End of loop over crack points

    } // End of loop over matls
    delete interpolator;
  }
}

