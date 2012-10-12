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
    PART SEVEN: UPDATE CRACK FRONT AND CALCULATE CRACK-FRONT NORMALS

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
#include <cstring>
#include <sys/stat.h>

using namespace Uintah;
using namespace std;

using std::vector;
using std::string;


void Crack::addComputesAndRequiresCrackFrontNodeSubset(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  // Currently do nothing
}

void Crack::CrackFrontNodeSubset(const ProcessorGroup*,
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
      if(d_calFractParameters || d_doCrackPropagation) {
        // cfnset - subset of crack-front nodes in each patch
        cfnset[m][pid].clear();
        for(int j=0; j<(int)cfSegNodes[m].size(); j++) {
          int node=cfSegNodes[m][j];
          if(patch->containsPointInExtraCells(cx[m][node]))
            cfnset[m][pid].push_back(j);
        }
        MPI_Barrier(mpi_crack_comm);

        // Broadcast cfnset to all the ranks
        for(int i=0; i<patch_size; i++) {
          int num; // number of crack-front nodes in patch i
          if(i==pid) num=cfnset[m][i].size();
          MPI_Bcast(&num,1,MPI_INT,i,mpi_crack_comm);
          cfnset[m][i].resize(num);
          MPI_Bcast(&cfnset[m][i][0],num,MPI_INT,i,mpi_crack_comm);
        }
        MPI_Barrier(mpi_crack_comm);

        // cfsset - subset of crack-front segment center in each patch
        cfsset[m][pid].clear();
        int ncfSegs=(int)cfSegNodes[m].size()/2;
        for(int j=0; j<ncfSegs; j++) {
          int n1=cfSegNodes[m][2*j]; 
          int n2=cfSegNodes[m][2*j+1];
          Point cent=cx[m][n1]+(cx[m][n2]-cx[m][n1])/2.;
          if(patch->containsPointInExtraCells(cent)) {
            cfsset[m][pid].push_back(j); 
          } 
        }

        MPI_Barrier(mpi_crack_comm);
        
        // Broadcast cfsset to all the ranks
        for(int i=0; i<patch_size; i++) {
          int num; // number of crack-front segments in patch i
          if(i==pid) num=cfsset[m][i].size();
          MPI_Bcast(&num,1,MPI_INT,i,mpi_crack_comm);
          cfsset[m][i].resize(num);
          MPI_Bcast(&cfsset[m][i][0],num,MPI_INT,i,mpi_crack_comm);
        }
      } 
    } // End of loop over matls
  } 
}

void Crack::addComputesAndRequiresRecollectCrackFrontSegments(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  int NGC=2*NGN;
  t->requires(Task::NewDW, lb->gMassLabel, gac, NGC);
  t->requires(Task::NewDW, lb->GMassLabel, gac, NGC);
  t->requires(Task::OldDW,lb->pSizeLabel, Ghost::None);
  t->requires(Task::OldDW,lb->pDeformationMeasureLabel, Ghost::None);
}

void Crack::RecollectCrackFrontSegments(const ProcessorGroup*,
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

    Vector dx = patch->dCell();

    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);

      // Cell mass of the material
      double d_cell_mass=mpm_matl->getInitialDensity()*dx.x()*dx.y()*dx.z();
                   
      // Get nodal mass information
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      Ghost::GhostType  gac = Ghost::AroundCells;
      constNCVariable<double> gmass,Gmass;
      int NGC=2*NGN;
      new_dw->get(gmass, lb->gMassLabel, dwi, patch, gac, NGC);
      new_dw->get(Gmass, lb->GMassLabel, dwi, patch, gac, NGC);

      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> deformationGradient;
      old_dw->get(psize, lb->pSizeLabel, pset);
      old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

      if(d_doCrackPropagation) {

              
        // Task 1: Detect if crack-front nodes are inside the material
        
        vector<short> cfSegNodesInMat;
        cfSegNodesInMat.resize(cfSegNodes[m].size());
        for(int i=0; i<(int)cfSegNodes[m].size();i++) {
          cfSegNodesInMat[i]=YES;
        }

        for(int i=0; i<patch_size; i++) {
          int num=cfnset[m][i].size();
          short* inMat=new short[num];

          if(pid==i) { // Rank i does it
            for(int j=0; j<num; j++) {
              int idx=cfnset[m][i][j];
              int node=cfSegNodes[m][idx];
              Point pt=cx[m][node];
              inMat[j]=YES;

              interpolator->findCellAndWeights(pt, ni, S, psize[j],deformationGradient[j]);

              for(int k = 0; k < n8or27; k++) {
                double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                if(totalMass<d_cell_mass/32.) {
                  inMat[j]=NO;
                  break;
                }
              }
            } 
          } 

          MPI_Bcast(&inMat[0],num,MPI_SHORT,i,mpi_crack_comm);

          for(int j=0; j<num; j++) {
            int idx=cfnset[m][i][j];
            cfSegNodesInMat[idx]=inMat[j];
          }
          delete [] inMat;
        } // End of loop over patches

        MPI_Barrier(mpi_crack_comm);

        
        // Task 2: Detect if the centers of crack-front segments
        //         are inside the material
       
        vector<short> cfSegCenterInMat;
        cfSegCenterInMat.resize(cfSegNodes[m].size()/2);
        for(int i=0; i<(int)cfSegNodes[m].size()/2;i++) {
          cfSegCenterInMat[i]=YES;
        }

        for(int i=0; i<patch_size; i++) {
          int num=cfsset[m][i].size();
          short* inMat=new short[num];

          if(pid==i) { // Rank i does it
            for(int j=0; j<num; j++) {
              int idx=cfsset[m][i][j];
              int node1=cfSegNodes[m][2*idx];
              int node2=cfSegNodes[m][2*idx+1];
              Point cent=cx[m][node1]+(cx[m][node2]-cx[m][node1])/2.;
              inMat[j]=YES;

              interpolator->findCellAndWeights(cent, ni, S, psize[j],deformationGradient[j]);

              for(int k = 0; k < n8or27; k++) {
                double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                if(totalMass<d_cell_mass/32.) {
                  inMat[j]=NO;
                  break;
                }
              }
            } 
          } 

          MPI_Bcast(&inMat[0],num,MPI_SHORT,i,mpi_crack_comm);

          for(int j=0; j<num; j++) {
            int idx=cfsset[m][i][j];
            cfSegCenterInMat[idx]=inMat[j];
          }
          delete [] inMat;
        } // End of loop over patches

        MPI_Barrier(mpi_crack_comm);

        
        // Task 3: Recollect crack-front segments, discarding the 
        //         dead ones. A segment is regarded dead if both
        //         ends of it are outside the material 
       
        // Store crack-front parameters in temporary arraies
        int old_size=(int)cfSegNodes[m].size();
        int*  copyData = scinew int[old_size];

        for(int i=0; i<old_size; i++) copyData[i]=cfSegNodes[m][i];
        cfSegNodes[m].clear();

        // Collect the active crack-front segments
        for(int i=0; i<old_size/2; i++) { 
          int nd1=copyData[2*i];
          int nd2=copyData[2*i+1];

          short thisSegActive=NO;
          if(cfSegNodesInMat[2*i] || cfSegNodesInMat[2*i+1] || cfSegCenterInMat[i])
            thisSegActive=YES;
          
          if(thisSegActive) { 
            cfSegNodes[m].push_back(nd1);
            cfSegNodes[m].push_back(nd2);
          }
        } 
        delete [] copyData;

        if(cfSegNodes[m].size()>0) { // New crack front is still in material
          // Seek the start crack point (sIdx), re-arrange crack-front nodes 
          int node0=cfSegNodes[m][0];
          int segs[2];
          FindSegsFromNode(m,node0,segs);

          if(segs[R]>=0) {
            int numNodes=(int)cfSegNodes[m].size();
            int sIdx=0;
            for(int i=0; i<numNodes; i++) {
              int segsT[2];
              FindSegsFromNode(m,cfSegNodes[m][i],segsT);
              if(segsT[R]<0) {sIdx=i; break;}
            }

            int* copyData =new int[numNodes];
            int rIdx=2*segs[R]+1;
            for(int i=0; i<numNodes; i++) {
              int oldIdx=sIdx+i;
              if(oldIdx>rIdx) oldIdx-=(rIdx+1);
              copyData[i]=cfSegNodes[m][oldIdx];
            }

            for(int i=0; i<numNodes; i++) {
              cfSegNodes[m][i]=copyData[i];
            }      

            delete [] copyData;
          }
       
          
          // Task 4: Get previous index, and minimum & maximum indexes 
          //         for crack-front nodes
         
          FindCrackFrontNodeIndexes(m);
    
         
          // Task 5: Calculate outer normals, tangential normals and binormals
          //         of crack plane at crack-front nodes  
                  
          if(smoothCrackFront) { 
            short smoothSuccessfully=SmoothCrackFrontAndCalculateNormals(m);
            if(!smoothSuccessfully) CalculateCrackFrontNormals(m);
          }
          else { // Calculate crack-front normals directly
            CalculateCrackFrontNormals(m);
          }
        } // End if(cfSegNodes[m].size()>0)
        
        else { // Crack has penetrated the material
          // If all crack-front segments dead, the material is broken.
          if(ce[m].size()>0) { // for the material with crack(s) initially              
            if(pid==0) cout << "!!! Material " << m << " is broken." << endl;
          }  
        }
        
      } // End of if(d_doCrackPropagation!="false")

      // Save crack elements, crack nodes and crack-front nodes
      // for crack geometry visualization
      if(saveCrackGeometry) {
        int curTimeStep=d_sharedState->getCurrentTopLevelTimeStep();
        if(pid==0) OutputCrackGeometry(m,curTimeStep);
      }

    } // End of loop over matls
    delete interpolator;
  } 
}

// Output cracks for visualization
void Crack::OutputCrackGeometry(const int& m, const int& timestep)
{
  if(ce[m].size()>0) { // for the materials with cracks
    bool timeToDump = dataArchiver->isOutputTimestep();
    if(timeToDump) {
      // Create output files in format:
      // ce.matXXX.timestepYYYYY (crack elems)
      // cx.matXXX.timestepYYYYY (crack points)
      // cf.matXXX.timestepYYYYY (crack front nodes)
      // Those files are stored in .uda.XXX/tXXXXX/crackData/

      char timestepbuf[10],matbuf[10];
      sprintf(timestepbuf,"%d",timestep);
      sprintf(matbuf,"%d",m);

      // Create output directories
      char crackDir[200]="";
      strcat(crackDir,udaDir.c_str());
      strcat(crackDir,"/t");
      if(timestep<10) strcat(crackDir,"0000");
      else if(timestep<100) strcat(crackDir,"000");
      else if(timestep<1000) strcat(crackDir,"00");
      else if(timestep<10000) strcat(crackDir,"0");
      strcat(crackDir,timestepbuf);
      strcat(crackDir,"/crackData");

      MKDIR(crackDir,0777);

      // Specify output file names
      char ceFileName[200]="";
      strcat(ceFileName,crackDir);
      strcat(ceFileName,"/ce.mat");
      if(m<10) strcat(ceFileName,"00");
      else if(m<100) strcat(ceFileName,"0");
      strcat(ceFileName,matbuf);

      char cxFileName[200]="";
      strcat(cxFileName,crackDir);
      strcat(cxFileName,"/cx.mat");
      if(m<10) strcat(cxFileName,"00");
      else if(m<100) strcat(cxFileName,"0");
      strcat(cxFileName,matbuf);

      char cfFileName[200]="";
      strcat(cfFileName,crackDir);
      strcat(cfFileName,"/cf.mat");
      if(m<10) strcat(cfFileName,"00");
      else if(m<100) strcat(cfFileName,"0");
      strcat(cfFileName,matbuf);

      ofstream outputCE(ceFileName, ios::out);
      ofstream outputCX(cxFileName, ios::out);
      ofstream outputCF(cfFileName, ios::out);

      if(!outputCE || !outputCX || !outputCF) {
        cout << "Error: failure to open files for storing crack geometry" << endl;
        exit(1);
      }

      // Output crack elems
      for(int i=0; i<(int)ce[m].size(); i++) {
        outputCE << ce[m][i].x() << " " << ce[m][i].y() << " "
                 << ce[m][i].z() << endl;
      }

      // Output crack nodes
      for(int i=0; i<(int)cx[m].size(); i++) {
        outputCX << cx[m][i].x() << " " << cx[m][i].y() << " "
                 << cx[m][i].z() << endl;
      }

      // Output crack-front nodes
      for(int i=0; i<(int)cfSegNodes[m].size()/2; i++) {
        outputCF << cfSegNodes[m][2*i] << " "
                 << cfSegNodes[m][2*i+1] << endl;
      }
    }
  } // End if(ce[m].size()>0)
}

