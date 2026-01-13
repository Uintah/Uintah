
//______________________________________________________________________
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
#include <CCA/Components/MPM/MPMGranular.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Core/Math/Matrix3.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Components/MPM/Core/MPMBoundCond.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DebugStream.h>
#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("MPM", true);
static DebugStream cout_dbg("MPMGranular", true);


//______________________________________________________________________
//_________________________________________________________________

MPMGranular::MPMGranular(MaterialManagerP& ss, MPMFlags* flags)
{
  d_lb = scinew MPMLabel();

  d_flags = flags;

  if(d_flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }

  d_materialManager = ss;
}

MPMGranular::~MPMGranular()
{
  delete d_lb;
}
//______________________________________________________________________

void MPMGranular::MPMGranularProblemSetup(const ProblemSpecP& prob_spec, 
                                                 MPMFlags* flags)
{
  
}

//_____________________________________________________________________
//
void MPMGranular::scheduleGranularMPM(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
    if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
        getLevel(patches)->getGrid()->numLevels()))
        return;
  
    printSchedule(patches, cout_doing, "MPM::scheduleGranularMPM");


  Task* t = scinew Task("MPM::GranularMPM", this, &MPMGranular::GranularMPM);
    
   //  MaterialSubset* zeroth_matl = scinew MaterialSubset();
    // zeroth_matl->add(0);
     //zeroth_matl->addReference();

    // Add required modifications for GranularMPM
    t->modifiesVar(d_lb->pParticleIDLabel_preReloc);
    t->modifiesVar(d_lb->pXLabel_preReloc);
    t->modifiesVar(d_lb->pVolumeLabel_preReloc);
    t->modifiesVar(d_lb->pVelocityLabel_preReloc);
    t->modifiesVar(d_lb->pMassLabel_preReloc);
    t->modifiesVar(d_lb->pSizeLabel_preReloc);
    t->modifiesVar(d_lb->pDispLabel_preReloc);
    t->modifiesVar(d_lb->pStressLabel_preReloc);
    t->modifiesVar(d_lb->pdTdtLabel);

    if (d_flags->d_with_color) {
        t->modifiesVar(d_lb->pColorLabel_preReloc);
    }
    if (d_flags->d_useLoadCurves) {
        t->modifiesVar(d_lb->pLoadCurveIDLabel_preReloc);
    }

    // Handle Scalar Diffusion variables if needed
    if (d_flags->d_doScalarDiffusion) {
        t->modifiesVar(d_lb->diffusion->pConcentration_preReloc);
        t->modifiesVar(d_lb->diffusion->pConcPrevious_preReloc);
        t->modifiesVar(d_lb->diffusion->pGradConcentration_preReloc);
        t->modifiesVar(d_lb->diffusion->pExternalScalarFlux_preReloc);
        t->modifiesVar(d_lb->diffusion->pArea_preReloc);
        t->modifiesVar(d_lb->diffusion->pDiffusivity_preReloc);
    }
    t->modifiesVar(d_lb->pLocalizedMPMLabel_preReloc);
    t->modifiesVar(d_lb->pExtForceLabel_preReloc);
    t->modifiesVar(d_lb->pTemperatureLabel_preReloc);
    t->modifiesVar(d_lb->pTemperatureGradientLabel_preReloc);
    t->modifiesVar(d_lb->pTempPreviousLabel_preReloc);
    t->modifiesVar(d_lb->pDeformationMeasureLabel_preReloc);

    if (d_flags->d_computeScaleFactor) {
        t->modifiesVar(d_lb->pScaleFactorLabel_preReloc);
    }
    t->modifiesVar(d_lb->pVelGradLabel_preReloc);
        
    
//  t->requires(Task::OldDW, d_lb->pCellNAPIDLabel, zeroth_matl, Ghost::None);
 // t->computes(             d_lb->pCellNAPIDLabel, zeroth_matl);

  unsigned int numMatls =  d_materialManager->getNumMatls( "MPM" );
  for(unsigned int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*)  d_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->modifyComputesAndRequires(t, mpm_matl, patches);
  }
	

  sched->addTask(t, patches,  d_materialManager->allMaterials( "MPM" ));
 
}
//______________________________________________________________________
//
void MPMGranular::GranularMPM(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)                         
{
	
    simTime_vartype simTimeVar;
    old_dw->get(simTimeVar, d_lb->simulationTimeLabel);
    double time = simTimeVar;
    delt_vartype delT;
    old_dw->get(delT, d_lb->delTLabel, getLevel(patches) );
	int totalNumPartls = 0;
	for (int p = 0; p < patches->size(); ++p) {
		const Patch* patch = patches->get(p);	
		printTask(patches, patch, cout_doing, "Doing MPM::GranularMPM");
		Vector dx = patch->dCell();
		unsigned int numMatls =  d_materialManager->getNumMatls("MPM");
		vector<int> AssignedMatls(numMatls, -1); 
		vector<int> numPartsPerMatIdx(numMatls, 0);  
		int numMatTypeIdx = 0;                      
		int maxMaterialTypes = numMatls;
		vector<int> assignedMatTypeIdx(maxMaterialTypes, -1);  
		for (unsigned int m = 0; m < numMatls; m++) {
			MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial("MPM", m);
			int dwi = mpm_matl->getDWIndex(); 
			int dwi_type = mpm_matl->getMatTypeIdx(); 
			if (assignedMatTypeIdx[dwi_type] == -1) {
				assignedMatTypeIdx[dwi_type] = 0;  
				numMatTypeIdx++; 
			} 
			ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
			numPartsPerMatIdx[dwi] = pset->addParticles(0);  
			AssignedMatls[dwi] = (numPartsPerMatIdx[dwi] == 0) ? 0 : 1;  
			if(time >= mpm_matl->getActivationTime() &&  time <= mpm_matl->getDeactivationTime()){
				if( numPartsPerMatIdx[dwi] != 0){
					mpm_matl->setIsActive(true);
					cout_dbg<<"dwi : "<<dwi
						<<", time: "<<time
						<< ", numPartsPerMatIdx[" << dwi << "] = " << numPartsPerMatIdx[dwi] 
						<<", is_active: "<<mpm_matl->getIsActive()
						<< endl;
						totalNumPartls += numPartsPerMatIdx[dwi];  
					cout_dbg <<"totalNumPartls = " << totalNumPartls<< endl;
				}else {
					mpm_matl->setIsActive(false);   
									}
			}else {
				mpm_matl->setIsActive(false);   
								}						
		}
		if (cout_dbg.active()){
		 cout_dbg << "totalNumPartls = " << totalNumPartls << ", numMatls = " << numMatls<<", numMatTypeIdx = "<<numMatTypeIdx << endl;
		} 


		vector<vector<int>> availableMatIndices_Can(numMatTypeIdx, vector<int>(numMatls, -1)); 
		vector<vector<int>> availableMatIndices_nonCan(numMatTypeIdx, vector<int>(numMatls, -1));   
		vector<int> availableMatIndices_Can_size(numMatTypeIdx, 0); 
		vector<int> availableMatIndices_nonCan_size(numMatTypeIdx, 0); 
		
		vector<int> MatlIdx(totalNumPartls);
		vector<int> PartIdx(totalNumPartls);
		vector<double> px1(totalNumPartls), px2(totalNumPartls), px3(totalNumPartls);
		vector<double> DI1(totalNumPartls), DI2(totalNumPartls), DI3(totalNumPartls);
		int PrtCtr = 0; 
		for (unsigned int m = 0; m < numMatls; m++) {
			MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial("MPM", m);
			int dwi = mpm_matl->getDWIndex(); 
			ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
			ParticleVariable<Point> px;
			ParticleVariable<Matrix3> pSize;
			ParticleVariable<Matrix3> pF;
			new_dw->getModifiable(px, d_lb->pXLabel_preReloc, pset);
			new_dw->getModifiable(pSize, d_lb->pSizeLabel_preReloc, pset);
			new_dw->getModifiable(pF, d_lb->pDeformationMeasureLabel_preReloc, pset);
			if( mpm_matl->getIsActive()){ 
				cout<< dwi<<" activation is "<<mpm_matl->getIsActive()<<endl;
				for (auto iter = pset->begin(); iter != pset->end(); ++iter) {
					particleIndex idx = *iter;        
					MatlIdx[PrtCtr] = dwi;
					PartIdx[PrtCtr] = idx;
					px1[PrtCtr] = px[idx].x();
					px2[PrtCtr] = px[idx].y();
					px3[PrtCtr] = px[idx].z();
					double alpha = mpm_matl->getInitialDensity() / (pF[idx].Determinant() * mpm_matl->getCriticalDensity());
					
					if (d_flags->d_ndim == 3) {
						DI1[PrtCtr] = cbrt(alpha) * dx.x() * pSize[idx](0, 0);
						DI2[PrtCtr] = cbrt(alpha) * dx.y() *pSize[idx](1, 1);
						DI3[PrtCtr] = cbrt(alpha) * dx.z() *pSize[idx](2, 2); 
					
					} else if (d_flags->d_ndim == 2) {
						DI1[PrtCtr] = sqrt(alpha) * dx.x() *pSize[idx](0, 0); 
						DI2[PrtCtr] = sqrt(alpha) * dx.y() *pSize[idx](1, 1);
						DI3[PrtCtr] = sqrt(alpha) * dx.z() * pSize[idx](2, 2);
											
					} else if (d_flags->d_ndim == 1) {
						DI1[PrtCtr] = alpha * dx.x() * pSize[idx](0, 0);
						DI2[PrtCtr] = dx.y() * pSize[idx](1, 1);
						DI3[PrtCtr] = dx.z() * pSize[idx](2, 2);
					
					} 	
					++PrtCtr;
				}//particels	
			}
			if (time <= mpm_matl->getDeactivationTime()){//active materials
				int dwi_type = mpm_matl->getMatTypeIdx(); 
				if (AssignedMatls[dwi] == 0) {    
					if (!mpm_matl->getAppliedContactModel()) {  
						availableMatIndices_nonCan[dwi_type][availableMatIndices_nonCan_size[dwi_type]] = dwi;
						cout_dbg << "availableMatIndices_nonCan["<<dwi_type<<"]["<<availableMatIndices_nonCan_size[dwi_type]<<"] : " << dwi << endl; 
						availableMatIndices_nonCan_size[dwi_type]++;
					} else if (mpm_matl->getAppliedContactModel()) { 
						availableMatIndices_Can[dwi_type][availableMatIndices_Can_size[dwi_type]] = dwi;
						cout_dbg << "availableMatIndices_Can["<<dwi_type<<"]["<<availableMatIndices_Can_size[dwi_type]<<"] : " << dwi << endl; 
						availableMatIndices_Can_size[dwi_type]++;
					} 
				} 	
			}			
		}
	     
		vector<vector<int>> Interaction(totalNumPartls, vector<int>(totalNumPartls, -1));
		vector<int> InteractionSize(totalNumPartls, 0);
		vector<vector<int>> AssignedInteraction(totalNumPartls, vector<int>(totalNumPartls, 0));
		for (int i = 0; i < totalNumPartls; ++i) {
			double ximin = px1[i] - 0.5 * DI1[i];
			double ximax = px1[i] + 0.5 * DI1[i];        
			double yimin = px2[i] - 0.5 * DI2[i];
			double yimax = px2[i] + 0.5 * DI2[i];    
			double zimin = px3[i] - 0.5 * DI3[i];
			double zimax = px3[i] + 0.5 * DI3[i];            
			for (int j = 0; j < totalNumPartls; ++j) {
				if (AssignedInteraction[i][j] == 0  && j != i) { 
					double xjmin = px1[j] - 0.5 * DI1[j];
					double xjmax = px1[j] + 0.5 * DI1[j];
					double yjmin = px2[j] - 0.5 * DI2[j];
					double yjmax = px2[j] + 0.5 * DI2[j];
					double zjmin = px3[j] - 0.5 * DI3[j];
					double zjmax = px3[j] + 0.5 * DI3[j];
					if ((ximax >= xjmin) && (ximin <= xjmax) && 
						(yimax >= yjmin) && (yimin <= yjmax) && 
						(zimax >= zjmin) && (zimin <= zjmax)){
						Interaction[i][InteractionSize[i]] = j;
						InteractionSize[i] += 1;    
						AssignedInteraction[i][j] = 1;
					}
					else if ((ximax >= xjmin) && (ximin <= xjmax) && 
						(yimax >= yjmin) && (yimin <= yjmax) && 
						(zjmax >= zimin) && (zjmin <= zimax)) {
						Interaction[i][InteractionSize[i]] = j;
						InteractionSize[i] += 1;    
						AssignedInteraction[i][j] = 1;
					}
					else if ((ximax >= xjmin) && (ximin <= xjmax) && 
						(yjmax >= yimin) && (yjmin <= yimax) && 
						(zimax >= zjmin) && (zimin <= zjmax)) {
						Interaction[i][InteractionSize[i]] = j;
						InteractionSize[i] += 1;    
						AssignedInteraction[i][j] = 1;
					}
					else if ((ximax >= xjmin) && (ximin <= xjmax) && 
						(yjmax >= yimin) && (yjmin <= yimax) && 
						(zjmax >= zimin) && (zjmin <= zimax)) {
						Interaction[i][InteractionSize[i]] = j;
						InteractionSize[i] += 1;    
						AssignedInteraction[i][j] = 1;
					}
					else if ((xjmax >= ximin) && (xjmin <= ximax) && 
						(yimax >= yjmin) && (yimin <= yjmax) && 
						(zimax >= zjmin) && (zimin <= zjmax)) {
						Interaction[i][InteractionSize[i]] = j;
						InteractionSize[i] += 1;    
						AssignedInteraction[i][j] = 1;
					}
					else if ((xjmax >= ximin) && (xjmin <= ximax) && 
						(yimax >= yjmin) && (yimin <= yjmax) && 
						(zjmax >= zimin) && (zjmin <= zimax)) {
						Interaction[i][InteractionSize[i]] = j;
						InteractionSize[i] += 1;    
						AssignedInteraction[i][j] = 1;	
					}
					else if ((xjmax >= ximin) && (xjmin <= ximax) && 
						(yjmax >= yimin) && (yjmin <= yimax) && 
						(zimax >= zjmin) && (zimin <= zjmax)) {
						Interaction[i][InteractionSize[i]] = j;
						InteractionSize[i] += 1;    
						AssignedInteraction[i][j] = 1;
					}
					else if ((xjmax >= ximin) && (xjmin <= ximax) && 
						(yjmax >= yimin) && (yjmin <= yimax) && 
						(zjmax >= zimin) && (zjmin <= zimax)) {
						Interaction[i][InteractionSize[i]] = j;
						InteractionSize[i] += 1;    
						AssignedInteraction[i][j] = 1;
					}	
				}        
			}             
		}
		//if (cout_dbg.active()){
			// for (int i = 0; i < totalNumPartls; ++i) {
		//	cout<<"InteractionSize["<<i<<"] = "<<InteractionSize[i]<<endl;
		//	for (int j = 0; j < InteractionSize[i]; ++j) {
		//		cout<<" Interaction["<<i<<"]["<<j<<"] = "<<Interaction[i][j]<< ", MatlIdx["<<Interaction[i][j]<<"] = " <<MatlIdx[Interaction[i][j]]<<endl;
		//	}
		//}	
			
		//}	
		
      
		vector<vector<int>> Grids(totalNumPartls, vector<int>(totalNumPartls, -1));
		vector<int> GridsSize(totalNumPartls, 0);
		vector<int> Assigned(totalNumPartls, -1); 
		vector<int> UnAssigned(totalNumPartls);  	
		int GridCount = 0;
		int UnAssignCount = totalNumPartls;
		for (int i = 0; i < totalNumPartls; i++) {
			UnAssigned[i] = i;                  
		}
		while (UnAssignCount > 0) {
			GridCount++;	 
			int ini_part = UnAssigned[0];
			if (find(Grids[GridCount-1].begin(), Grids[GridCount-1].end(), ini_part) == Grids[GridCount-1].end()) { 
				Grids[GridCount-1][GridsSize[GridCount-1]] = ini_part;
				GridsSize[GridCount-1]++;
			}
			for (int i = 0; i < InteractionSize[ini_part]; i++) {
				int neighborPart_i = Interaction[ini_part][i];
				Grids[GridCount-1][GridsSize[GridCount-1]] = neighborPart_i;
				GridsSize[GridCount-1]++;		
			}
			Assigned[ini_part] = 1;	
			int counter = 0;
			int stopCounter = 0;
			while (counter < 1) {
				for (int i = 0; i < GridsSize[GridCount-1]; i++) {
					int part_1 = Grids[GridCount-1][i];
					if ( Assigned[part_1] == -1) {
						if (find(Grids[GridCount-1].begin(), Grids[GridCount-1].end(), part_1) == Grids[GridCount-1].end()) { 
							Grids[GridCount-1][GridsSize[GridCount-1]] = part_1;
							GridsSize[GridCount-1]++;
						}
						for (int j = 0; j < InteractionSize[part_1]; j++) {
							int part_2 = Interaction[part_1][j];
							if (find(Grids[GridCount-1].begin(), Grids[GridCount-1].end(), part_2) == Grids[GridCount-1].end()) { 
								Grids[GridCount-1][GridsSize[GridCount-1]] = part_2;
								GridsSize[GridCount-1]++;
							}
						}
						Assigned[part_1] = 1;
					}
				}
				if (stopCounter == GridsSize[GridCount-1]) {
					counter = 1;
				} else {
					stopCounter = GridsSize[GridCount-1];
				}
			} 
			UnAssignCount = 0;
			for (int i = 0; i < totalNumPartls; i++) {
				UnAssigned[i] = -1;
				if ( Assigned[i] == -1) { 
					UnAssigned[UnAssignCount] = i;
					UnAssignCount++;
				}
			}
		} 
		
		//if (cout_dbg.active()){ cout_dbg << "GridCount: " << GridCount << endl;}
		vector<vector<vector<int>>>gridMatTypeMatIdslist(GridCount, vector<vector<int>>(numMatTypeIdx, vector<int>(numMatls, -1))); 
		vector<vector<int>>gridMatTypeMIDsSize(GridCount, vector<int>(numMatTypeIdx, 0)); 
		vector<vector<map<int, vector<int>>>> gridMatTypeMIdPartList(GridCount, vector<map<int, vector<int>>>(numMatTypeIdx)); 
		vector<int>gridMatTypeSize(GridCount, 0); 
		vector<vector<int>>gridMatTypeList(GridCount, vector<int>(numMatTypeIdx, -1)); 
		vector<vector<int>>MatIdsMax(GridCount, vector<int>(numMatTypeIdx, -1)); 
		vector<vector<int>>MatIdsTest(GridCount, vector<int>(numMatTypeIdx, -1)); 
		vector<vector<int>>gridMatTypePartSize(GridCount, vector<int>(numMatTypeIdx, 0)); 
		vector<vector<int>>with_contact(GridCount, vector<int>(numMatTypeIdx, -1)); 
		for (int i = 0; i < GridCount; ++i) { 
			vector<int> assignedMatTypeIdx(numMatTypeIdx, -1); 
			vector<vector<int>> assignedMatIdx(numMatTypeIdx, vector<int>(numMatls, -1)); 
			for (int j = 0; j < GridsSize[i]; ++j) { 
				int partID = PartIdx[Grids[i][j]]; 
				int materialID = MatlIdx[Grids[i][j]]; 
				MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial("MPM", materialID);
				int dwi_type = mpm_matl->getMatTypeIdx(); 
				if (dwi_type < 0 || dwi_type >= numMatTypeIdx) { 
				//	cout << "Error: dwi_type out of bounds: " << dwi_type << endl;
					continue; 
				}
				if (assignedMatTypeIdx[dwi_type] == -1) { 
					assignedMatTypeIdx[dwi_type] = 0;
					gridMatTypeList[i][gridMatTypeSize[i]] = dwi_type;  
					gridMatTypeSize[i]++; 
				}
				if (assignedMatIdx[dwi_type][materialID] == -1) { 
					assignedMatIdx[dwi_type][materialID] = 0;
					gridMatTypeMatIdslist[i][dwi_type][gridMatTypeMIDsSize[i][dwi_type]] = materialID;
				//	cout<<"gridMatTypeMatIdslist["<<i<<"]["<<dwi_type<<"]["<<gridMatTypeMIDsSize[i][dwi_type]<<"] = "<<gridMatTypeMatIdslist[i][dwi_type][gridMatTypeMIDsSize[i][dwi_type]]<<endl;
					gridMatTypeMIDsSize[i][dwi_type]++; 
				}
				gridMatTypePartSize[i][dwi_type]++;
				gridMatTypeMIdPartList[i][dwi_type][materialID].push_back(partID); 
				int MIDmax = MatIdsMax[i][dwi_type]; 
				vector<int>& particleCount_max = gridMatTypeMIdPartList[i][dwi_type][MIDmax]; 
				vector<int>& particleCount = gridMatTypeMIdPartList[i][dwi_type][materialID]; 
				if (particleCount.size() > particleCount_max.size()) { 
					MatIdsMax[i][dwi_type] = materialID; 
				}
			}
			cout << "gridMatTypeSize[" << i << "] = " << gridMatTypeSize[i] << endl;
			if (gridMatTypeSize[i] == 1) {	//one material type
				int j = 0;  
				int dwi_type = gridMatTypeList[i][j]; 
				cout << "dwi_type = " << dwi_type
					<< ", gridMatTypeMIDsSize[" << i << "][" << dwi_type << "] = " << gridMatTypeMIDsSize[i][dwi_type] 
					<< ", MatIdsMax[" << i << "][" << dwi_type << "] = " << MatIdsMax[i][dwi_type] << endl;
				if (gridMatTypeMIDsSize[i][dwi_type] == 1) {  // only one material Id
					MatIdsTest[i][dwi_type] = MatIdsMax[i][dwi_type];
					cout << "MatIdsTest[" << i << "][" << dwi_type << "] = " << MatIdsTest[i][dwi_type] << endl;
					MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial("MPM", MatIdsTest[i][dwi_type]);
					if (mpm_matl->getAppliedContactModel()) { 
						with_contact[i][j] = mpm_matl->DoDWIndexCorrection() ? 0 : 1;  //DoDWIndexCorrection: 0 means that <allow_DWIndex_correction>false</allow_DWIndex_correction> so it will be in contact and with_contact[i][j] is 1 
						cout << "with_contact[" << i << "][" << j << "] = " << with_contact[i][j]
							<< ", DoDWIndexCorrection: " << mpm_matl->DoDWIndexCorrection()
							<< ", getAppliedContactModel: " << mpm_matl->getAppliedContactModel() << endl; 
					} else if (!mpm_matl->getAppliedContactModel()) {
						with_contact[i][j] = 0; 
						cout << "with_contact[" << i << "][" << j << "] = " << with_contact[i][j]
							<< ", DoDWIndexCorrection: " << mpm_matl->DoDWIndexCorrection()
							<< ", getAppliedContactModel: " << mpm_matl->getAppliedContactModel() << endl;
					}
				} else if (gridMatTypeMIDsSize[i][dwi_type] > 1) { // more than one material Id
					with_contact[i][j] = 1;  // self contact
					for (int k = 0; k < gridMatTypeMIDsSize[i][dwi_type]; k++) { 
						MatIdsTest[i][dwi_type] = gridMatTypeMatIdslist[i][dwi_type][k];
						if (MatIdsTest[i][dwi_type] != MatIdsMax[i][dwi_type]) {
							break;
						}
					}
					cout <<"MatIdsTest[" << i << "][" << dwi_type << "] = " << MatIdsTest[i][dwi_type] << endl;	
					cout <<"with_contact[" << i << "][" << j << "] = " << with_contact[i][j] << endl;
				}	
			} else if (gridMatTypeSize[i] > 1) { // more that one materila type
				for (int j = 0; j < gridMatTypeSize[i]; j++) {
					int dwi_type = gridMatTypeList[i][j]; 
					cout << "dwi_type = " << dwi_type
						<< ", gridMatTypeMIDsSize[" << i << "][" << dwi_type << "] = " << gridMatTypeMIDsSize[i][dwi_type] 
						<< ", MatIdsMax[" << i << "][" << dwi_type << "] = " << MatIdsMax[i][dwi_type] << endl;
					if (gridMatTypeMIDsSize[i][dwi_type] == 1) { // in each type one materila ID
						MatIdsTest[i][dwi_type] = MatIdsMax[i][dwi_type];
						cout << "MatIdsTest[" << i << "][" << dwi_type << "] = " << MatIdsTest[i][dwi_type] << endl;
						with_contact[i][j] = 1; 
						cout << "with_contact[" << i << "][" << j << "] = " << with_contact[i][j] << endl;	
					} else if (gridMatTypeMIDsSize[i][dwi_type] > 1) { // in each several materila iDs
						with_contact[i][j] = 1; 
						for (int k = 0; k < gridMatTypeMIDsSize[i][dwi_type]; k++) { 
							MatIdsTest[i][dwi_type] = gridMatTypeMatIdslist[i][dwi_type][k];
							if (MatIdsTest[i][dwi_type] != MatIdsMax[i][dwi_type]) {
								break;
							}
						}
						cout << "MatIdsTest[" << i << "][" << dwi_type << "] = " << MatIdsTest[i][dwi_type] << endl;	
						cout << "with_contact[" << i << "][" << j << "] = " << with_contact[i][j] << endl;		
					}
				}
			}
		}
		
		vector<vector<int>> dwi_dest_for_grid(GridCount, vector<int>(numMatls * GridCount, -1));
		vector<vector<int>> AssignedGrids(GridCount, vector<int>(numMatTypeIdx * GridCount,  -1));
		vector<vector<int>> used_dwi_dests_Can(numMatTypeIdx); // Empty inner vectors
        vector<vector<int>> used_dwi_dests_nonCan(numMatTypeIdx); // Empty inner vectors
		// Assign materials to grids
		for (int i = 0; i < GridCount; ++i) {
			cout_dbg << "Grid [" << i << "] contains " << gridMatTypeSize[i] << " dwi_type" << endl;
			for (int j = 0; j < gridMatTypeSize[i]; j++) {
				int dwi_type = gridMatTypeList[i][j];
				
				int mId_dest = MatIdsMax[i][dwi_type];
				MPMMaterial* mpm_matl_dest = (MPMMaterial*) d_materialManager->getMaterial("MPM", mId_dest);
				int dwi_dest = mpm_matl_dest->getDWIndex();
				cout_dbg << "dwi_type = " << dwi_type << ", dwi_dest = " << dwi_dest << endl;
				
				int mId_cur = MatIdsTest[i][dwi_type];
				MPMMaterial* mpm_matl_cur = (MPMMaterial*) d_materialManager->getMaterial("MPM", mId_cur);
				int dwi_cur = mpm_matl_cur->getDWIndex();
				cout_dbg << "dwi_cur = " << dwi_cur << endl;
				
				int gridSize = gridMatTypePartSize[i][dwi_type];
				vector<int>& destParticleIDs = gridMatTypeMIdPartList[i][dwi_type][mId_dest];
				int numDestParticles = destParticleIDs.size();
				cout_dbg << "numDestParticles = " << numDestParticles << ", gridSize = " << gridSize << endl;
				
				bool isContact = (with_contact[i][j] == 1);
				cout_dbg << "isContact : " << isContact << ", with_contact[" << i << "][" << j << "] = " << with_contact[i][j]<< "dwi_dest contact: "<<mpm_matl_dest->getAppliedContactModel() << endl;
				
				bool assigned = false;
				
				auto& used_dwi_dests = isContact ? used_dwi_dests_Can[dwi_type] : used_dwi_dests_nonCan[dwi_type];
				cout << "Size of used_dwi_dests[" << dwi_type << "]: " << used_dwi_dests.size() << endl;
	
				auto& availableMatIndices = isContact ? availableMatIndices_Can : availableMatIndices_nonCan;
                auto& availableMatIndicesSize = isContact ? availableMatIndices_Can_size : availableMatIndices_nonCan_size;

                vector<int>& selectedMatIndices = availableMatIndices[dwi_type];  // Get correct material vector
                int availableSize = availableMatIndicesSize[dwi_type];  // Get correct size
                cout << "Available indices count for dwi_type " << dwi_type << " is " << availableSize << endl;
			  
				 
				if (mpm_matl_dest->getAppliedContactModel() == isContact && gridSize == numDestParticles && dwi_cur == dwi_dest) {
					cout_dbg << "isContact : MId : " << mpm_matl_dest->getAppliedContactModel() << ", and  " << isContact << endl;
					int dwi_candidate = dwi_dest;
					cout_dbg << "dwi_candidate " << dwi_candidate << ", and  " << dwi_dest << endl;
					// Check if the candidate is already in the list
					if(std::find(used_dwi_dests.begin(), used_dwi_dests.end(), dwi_candidate) == used_dwi_dests.end()) {
						dwi_dest_for_grid[i][j] = dwi_candidate;
						AssignedGrids[i][j] = 1;  //no need to apply materil correction
						assigned = true;
						cout << "1_Grid " << i << " Assigned GridMatlIdx = " << dwi_dest_for_grid[i][j]
							 << ", AssignedGrids[" << i << "][" << j << "] = " << AssignedGrids[i][j] << endl;
						used_dwi_dests.push_back(dwi_candidate);
                        cout << "Size of used_dwi_dests[" << dwi_type << "]: " << used_dwi_dests.size() << endl;	
					}
				}
				if (!assigned) {
					for (int k = 0; k < availableSize; ++k) {
						int dwi_candidate = selectedMatIndices[k]; 
						cout_dbg << "dwi_candidate " << dwi_candidate << ", and, k:  " << k<< ", availableSize: "<< availableSize<<endl;
						if (find(used_dwi_dests.begin(), used_dwi_dests.end(), dwi_candidate) == used_dwi_dests.end()) {
							dwi_dest_for_grid[i][j] = dwi_candidate;
							AssignedGrids[i][j] = 0; // need to apply materil correction
							assigned = true;
							cout << "2_Grid " << i << " Assigned GridMatlIdx = " << dwi_dest_for_grid[i][j]
								 << ", AssignedGrids[" << i << "][" << j << "] = " << AssignedGrids[i][j] << endl;
						   used_dwi_dests.push_back(dwi_candidate);
						   cout << "Size of used_dwi_dests[" << dwi_type << "]: " << used_dwi_dests.size() << endl;
							break;		
						}
					}
				}	
				if (!assigned) {
					cout_dbg << "**ERROR** grid "<< i<<" with dwi_type = " << dwi_type <<" is not assigned"<< endl;
					throw ProblemSetupException("ERROR**:", __FILE__, __LINE__);
				}
			}
		}

		vector<vector<int>> RemovedParticle(numMatls, vector<int>(totalNumPartls, 0));
		for (int i = 0; i < GridCount; ++i) {
			for (int k = 0; k < gridMatTypeSize[i]; k++) { 
				int dwi_type_dest = gridMatTypeList[i][k]; 
				MPMMaterial* mpm_matl_dest = (MPMMaterial*) d_materialManager->getMaterial("MPM", dwi_dest_for_grid[i][k]); 
				int dwi_dest = mpm_matl_dest->getDWIndex();
				cout << " dwi_dest_for_grid[" << i << "][" << k << "] = " << dwi_dest_for_grid[i][k] << endl;
				if (AssignedGrids[i][k] == 0) { 
					cout << "Material correction of Grid " << i << ", dwi_dest = " << dwi_dest << endl;
					vector<vector<int>> AssignedCopyInitialCMData(numMatls, vector<int>(numMatls, 0));
					for (int j = 0; j < GridsSize[i]; j++) {
						MPMMaterial* mpm_matl_ref = (MPMMaterial*) d_materialManager->getMaterial("MPM", MatlIdx[Grids[i][j]]);
						int dwi_ref = mpm_matl_ref->getDWIndex();
						int dwi_type_ref = mpm_matl_ref->getMatTypeIdx(); 
						if (dwi_type_dest == dwi_type_ref) { // this only works for the same type of materiasl such as lood soil or dense soil or soil and wall
							//cout_dbg<<"dwi_ref = "<<dwi_ref<<", dwi_type_ref = "<<dwi_type_ref<<endl;
							ParticleSubset* pset_ref = old_dw->getParticleSubset(dwi_ref, patch);
							
							//getting the origin material and Creating necessary vectors
							ParticleVariable<Point> px1;
							ParticleVariable<Matrix3> pF1, pSize1, pstress1, pvelgrad1, pscalefac1;
							ParticleVariable<long64> pids1;
							ParticleVariable<double> pvolume1, pmass1, ptemp1, ptempP1, pcolor1, pdTdt1;// pJThermal1, p_q1, ptemperature1;
							ParticleVariable<Vector> pvelocity1, pextforce1, pdisp1, ptempgrad1;
							ParticleVariable<int> ploc1;// pLocalized1;
							ParticleVariable<IntVector> pLoadCID1;			
							// JBH -- Scalar diffusion variables
							ParticleVariable<double> pConc1, pConcPrev1, pD1, pESFlux1;
							ParticleVariable<Vector> pGradConc1, pArea1;
							// Getting modifiable labels from new_dwh
							new_dw->getModifiable(px1, d_lb->pXLabel_preReloc, pset_ref);
							new_dw->getModifiable(pids1, d_lb->pParticleIDLabel_preReloc, pset_ref);
							new_dw->getModifiable(pmass1, d_lb->pMassLabel_preReloc, pset_ref);
							new_dw->getModifiable(pSize1, d_lb->pSizeLabel_preReloc, pset_ref);
							new_dw->getModifiable(pdisp1, d_lb->pDispLabel_preReloc, pset_ref);
							new_dw->getModifiable(pstress1, d_lb->pStressLabel_preReloc, pset_ref);
							new_dw->getModifiable(pvolume1, d_lb->pVolumeLabel_preReloc, pset_ref);
							new_dw->getModifiable(pvelocity1, d_lb->pVelocityLabel_preReloc, pset_ref);
							new_dw->getModifiable(pextforce1, d_lb->pExtForceLabel_preReloc, pset_ref);
							new_dw->getModifiable(ptemp1, d_lb->pTemperatureLabel_preReloc, pset_ref);
							new_dw->getModifiable(ptempgrad1, d_lb->pTemperatureGradientLabel_preReloc, pset_ref);
							new_dw->getModifiable(ptempP1, d_lb->pTempPreviousLabel_preReloc, pset_ref);
							new_dw->getModifiable(pdTdt1, d_lb->pdTdtLabel, pset_ref); // Added missing label ??
							new_dw->getModifiable(ploc1, d_lb->pLocalizedMPMLabel_preReloc, pset_ref);
							new_dw->getModifiable(pvelgrad1, d_lb->pVelGradLabel_preReloc, pset_ref);
							new_dw->getModifiable(pF1, d_lb->pDeformationMeasureLabel_preReloc, pset_ref);
							
							if (d_flags->d_computeScaleFactor) {
								new_dw->getModifiable(pscalefac1, d_lb->pScaleFactorLabel_preReloc, pset_ref);
							}
							if (d_flags->d_with_color) {
								new_dw->getModifiable(pcolor1, d_lb->pColorLabel_preReloc, pset_ref);
							}
							// JBH -- Scalar diffusion variables
							if (d_flags->d_doScalarDiffusion) {
								new_dw->getModifiable(pConc1,     d_lb->diffusion->pConcentration_preReloc, pset_ref);
								new_dw->getModifiable(pConcPrev1, d_lb->diffusion->pConcPrevious_preReloc, pset_ref);
								new_dw->getModifiable(pGradConc1, d_lb->diffusion->pGradConcentration_preReloc, pset_ref);
								new_dw->getModifiable(pESFlux1,   d_lb->diffusion->pExternalScalarFlux_preReloc, pset_ref);
								new_dw->getModifiable(pArea1,     d_lb->diffusion->pArea_preReloc, pset_ref);
								new_dw->getModifiable(pD1,        d_lb->diffusion->pDiffusivity_preReloc, pset_ref);
							}
							if (d_flags->d_useLoadCurves) {
								new_dw->getModifiable(pLoadCID1, d_lb->pLoadCurveIDLabel_preReloc, pset_ref);
							}
							
                         
							//___________________________
							//
							ParticleSubset* pset_dest = old_dw->getParticleSubset(dwi_dest, patch);
							ParticleVariable<Point> px2;
							ParticleVariable<Matrix3> pF2, pSize2, pstress2, pvelgrad2, pscalefac2;
							ParticleVariable<long64> pids2;
							ParticleVariable<double> pvolume2, pmass2, ptemp2, ptempP2, pcolor2, pdTdt2;//, pJThermal2, p_q2, ptemperature2;
							ParticleVariable<Vector> pvelocity2, pextforce2, pdisp2, ptempgrad2;
							ParticleVariable<int> ploc2;//, pLocalized2;
							ParticleVariable<IntVector> pLoadCID2;
							// JBH -- Scalar diffusion variables
							ParticleVariable<double> pConc2, pConcPrev2, pD2, pESFlux2;
							ParticleVariable<Vector> pGradConc2, pArea2;
							// Getting modifiable labels from pset_dest
							new_dw->getModifiable(px2, d_lb->pXLabel_preReloc, pset_dest);
							new_dw->getModifiable(pids2, d_lb->pParticleIDLabel_preReloc, pset_dest);
							new_dw->getModifiable(pmass2, d_lb->pMassLabel_preReloc, pset_dest);
							new_dw->getModifiable(pSize2, d_lb->pSizeLabel_preReloc, pset_dest);
							new_dw->getModifiable(pdisp2, d_lb->pDispLabel_preReloc, pset_dest);
							new_dw->getModifiable(pstress2, d_lb->pStressLabel_preReloc, pset_dest);
							new_dw->getModifiable(pvolume2, d_lb->pVolumeLabel_preReloc, pset_dest);
							new_dw->getModifiable(pvelocity2, d_lb->pVelocityLabel_preReloc, pset_dest);
							new_dw->getModifiable(pextforce2, d_lb->pExtForceLabel_preReloc, pset_dest);
							new_dw->getModifiable(ptemp2, d_lb->pTemperatureLabel_preReloc, pset_dest);
							new_dw->getModifiable(ptempgrad2, d_lb->pTemperatureGradientLabel_preReloc, pset_dest);
							new_dw->getModifiable(ptempP2, d_lb->pTempPreviousLabel_preReloc, pset_dest);
							new_dw->getModifiable(pdTdt2, d_lb->pdTdtLabel, pset_dest); 
							new_dw->getModifiable(ploc2, d_lb->pLocalizedMPMLabel_preReloc, pset_dest);
							new_dw->getModifiable(pvelgrad2, d_lb->pVelGradLabel_preReloc, pset_dest);
							new_dw->getModifiable(pF2, d_lb->pDeformationMeasureLabel_preReloc, pset_dest);
							if (d_flags->d_computeScaleFactor) {
								new_dw->getModifiable(pscalefac2, d_lb->pScaleFactorLabel_preReloc, pset_dest);
							}
							if (d_flags->d_with_color) {
								new_dw->getModifiable(pcolor2, d_lb->pColorLabel_preReloc, pset_dest);
							}
							// JBH -- Scalar diffusion variables
							if (d_flags->d_doScalarDiffusion) {
								new_dw->getModifiable(pConc2, d_lb->diffusion->pConcentration_preReloc, pset_dest);
								new_dw->getModifiable(pConcPrev2, d_lb->diffusion->pConcPrevious_preReloc, pset_dest);
								new_dw->getModifiable(pGradConc2, d_lb->diffusion->pGradConcentration_preReloc, pset_dest);
								new_dw->getModifiable(pESFlux2, d_lb->diffusion->pExternalScalarFlux_preReloc, pset_dest);
								new_dw->getModifiable(pArea2, d_lb->diffusion->pArea_preReloc, pset_dest);
								new_dw->getModifiable(pD2, d_lb->diffusion->pDiffusivity_preReloc, pset_dest);
							}
							if (d_flags->d_useLoadCurves) {
								new_dw->getModifiable(pLoadCID2, d_lb->pLoadCurveIDLabel_preReloc, pset_dest);
							}
													
							//_________________________
							//
							const  int oldNumPar = pset_dest->addParticles(1);
							const  int newNumPar = pset_dest->addParticles(0);
							//cout_dbg << "oldNumPar = "<<oldNumPar<<", newNumPar = "<<newNumPar<<endl;
							//_________________________
							//
							ParticleVariable<Point> px_tmp;
							ParticleVariable<Matrix3> pF_tmp, pSize_tmp, pstress_tmp, pvelgrad_tmp, pscalefac_tmp;
							ParticleVariable<long64> pids_tmp;
							ParticleVariable<double> pvolume_tmp, pmass_tmp, ptemp_tmp, ptempP_tmp, pcolor_tmp, pdTdt_tmp;//, pJThermal_tmp, p_q_tmp, ptemperature_tmp;
							ParticleVariable<Vector> pvelocity_tmp, pextforce_tmp, pdisp_tmp, ptempgrad_tmp;
							ParticleVariable<int> ploc_tmp;//,pLocalized_tmp;
							ParticleVariable<IntVector> pLoadCID_tmp;							                       							
							// JBH -- Scalar diffusion variables
							ParticleVariable<double> pConc_tmp, pConcPrev_tmp, pD_tmp, pESFlux_tmp;
							ParticleVariable<Vector> pGradConc_tmp, pArea_tmp;
							// Allocattign Temporarylabels to new_dw
							new_dw->allocateTemporary(px_tmp,  pset_dest);
							new_dw->allocateTemporary(pids_tmp,  pset_dest);
							new_dw->allocateTemporary(pmass_tmp, pset_dest);
							new_dw->allocateTemporary(pSize_tmp,  pset_dest);
							new_dw->allocateTemporary(pdisp_tmp, pset_dest);
							new_dw->allocateTemporary(pstress_tmp, pset_dest);
							new_dw->allocateTemporary(pvolume_tmp,  pset_dest);
							new_dw->allocateTemporary(pvelocity_tmp,  pset_dest);
							new_dw->allocateTemporary(pextforce_tmp, pset_dest);
							new_dw->allocateTemporary(ptemp_tmp,  pset_dest);
							new_dw->allocateTemporary(ptempgrad_tmp, pset_dest);
							new_dw->allocateTemporary(ptempP_tmp, pset_dest);
							new_dw->allocateTemporary(pdTdt_tmp,  pset_dest); //Added missing label ????
							new_dw->allocateTemporary(ploc_tmp,  pset_dest);
							new_dw->allocateTemporary(pvelgrad_tmp,  pset_dest);
							new_dw->allocateTemporary(pF_tmp,  pset_dest);
							if (d_flags->d_computeScaleFactor) {
								new_dw->allocateTemporary(pscalefac_tmp,  pset_dest);
							}
							if (d_flags->d_with_color) {
								new_dw->allocateTemporary(pcolor_tmp, pset_dest);
							}
							// JBH -- Scalar diffusion variables
							if (d_flags->d_doScalarDiffusion) {
								new_dw->allocateTemporary(pConc_tmp,  pset_dest);
								new_dw->allocateTemporary(pConcPrev_tmp,  pset_dest);
								new_dw->allocateTemporary(pGradConc_tmp,  pset_dest);
								new_dw->allocateTemporary(pESFlux_tmp, pset_dest);
								new_dw->allocateTemporary(pArea_tmp,  pset_dest);
								new_dw->allocateTemporary(pD_tmp,  pset_dest);
							}
							if (d_flags->d_useLoadCurves) {
								new_dw->allocateTemporary(pLoadCID_tmp, pset_dest);
							}
						
							//_________________________________
							// Assigning tmp variables to the corresponding 2 variables
							for ( int pp = 0; pp < oldNumPar; ++pp) {
								px_tmp[pp] = px2[pp];
								pids_tmp[pp] = pids2[pp];
								pmass_tmp[pp] = pmass2[pp];
								pSize_tmp[pp] = pSize2[pp];
								pdisp_tmp[pp] = pdisp2[pp];
								pstress_tmp[pp] = pstress2[pp];
								pvolume_tmp[pp] = pvolume2[pp];
								pvelocity_tmp[pp] = pvelocity2[pp];
								pextforce_tmp[pp] = pextforce2[pp];
								ptemp_tmp[pp] = ptemp2[pp];
								ptempgrad_tmp[pp] = ptempgrad2[pp];
								ptempP_tmp[pp] = ptempP2[pp];
								pdTdt_tmp[pp] = pdTdt2[pp];
								ploc_tmp[pp] = ploc2[pp];
								pvelgrad_tmp[pp] = pvelgrad2[pp];
								pF_tmp[pp] = pF2[pp];
								if (d_flags->d_computeScaleFactor) {
									pscalefac_tmp[pp] = pscalefac2[pp];
								}
								if (d_flags->d_with_color) {
									pcolor_tmp[pp] = pcolor2[pp];
								}
								if (d_flags->d_doScalarDiffusion) {
									pConc_tmp[pp] = pConc2[pp];
									pConcPrev_tmp[pp] = pConcPrev2[pp];
									pGradConc_tmp[pp] = pGradConc2[pp];
									pESFlux_tmp[pp] = pESFlux2[pp];
									pArea_tmp[pp] = pArea2[pp];
									pD_tmp[pp] = pD2[pp];
								}
								if (d_flags->d_useLoadCurves) {
									pLoadCID_tmp[pp] = pLoadCID2[pp];
								}	 
							}	
							//___________________________		
							// Assigning tmp variables to the corresponding 1 variables
							int partID_ref = PartIdx[Grids[i][j]];
							px_tmp[newNumPar - 1] = px1[partID_ref];
							pids_tmp[newNumPar - 1] = pids1[partID_ref];
							pmass_tmp[newNumPar - 1] = pmass1[partID_ref];
							pSize_tmp[newNumPar - 1] = pSize1[partID_ref];
							pdisp_tmp[newNumPar - 1] = pdisp1[partID_ref];
							pstress_tmp[newNumPar - 1] = pstress1[partID_ref];
							pvolume_tmp[newNumPar - 1] = pvolume1[partID_ref];
							pvelocity_tmp[newNumPar - 1] = pvelocity1[partID_ref];
							pextforce_tmp[newNumPar - 1] = pextforce1[partID_ref];
							ptemp_tmp[newNumPar - 1] = ptemp1[partID_ref];
							ptempgrad_tmp[newNumPar - 1] = ptempgrad1[partID_ref];
							ptempP_tmp[newNumPar - 1] = ptempP1[partID_ref];
							pdTdt_tmp[newNumPar - 1] = pdTdt1[partID_ref];
							ploc_tmp[newNumPar - 1] = ploc1[partID_ref];
							pvelgrad_tmp[newNumPar - 1] = pvelgrad1[partID_ref];
							pF_tmp[newNumPar - 1] = pF1[partID_ref];				
							if (d_flags->d_computeScaleFactor) {
								pscalefac_tmp[newNumPar - 1] = pscalefac1[partID_ref];
							}
							if (d_flags->d_with_color) {
								pcolor_tmp[newNumPar - 1] = pcolor1[partID_ref];
							}
							if (d_flags->d_doScalarDiffusion) {
								pConc_tmp[newNumPar - 1] = pConc1[partID_ref];
								pConcPrev_tmp[newNumPar - 1] = pConcPrev1[partID_ref];
								pGradConc_tmp[newNumPar - 1] = pGradConc1[partID_ref];
								pESFlux_tmp[newNumPar - 1] = pESFlux1[partID_ref];
								pArea_tmp[newNumPar - 1] = pArea1[partID_ref];
								pD_tmp[newNumPar - 1] = pD1[partID_ref];
							}
							if (d_flags->d_useLoadCurves) {
								pLoadCID_tmp[newNumPar - 1] = pLoadCID1[partID_ref];
							}
							
							//_____________________________
							// copy model parameters from the mpm_matl_ref to mpm_matl_dest , this later can be improved to do for all particles at the same time
							 mpm_matl_dest->getConstitutiveModel()->CopyInitialCMData(patch, mpm_matl_ref, mpm_matl_dest, oldNumPar, newNumPar, partID_ref, old_dw, new_dw); // dwi_ref and dwi_dest follow the Constitutive Model
							//_____________________________ 							
							//Now that the data are copied we should sign the material point to be removed from the origin.
							RemovedParticle[dwi_ref][partID_ref] = { 1 };
							//if (cout_dbg.active())
						    //cout_dbg <<"change the material ID from "<<dwi_ref<<" to "<<dwi_dest<<", for paricle "<<partID_ref<<endl;	
						    	
						    //__________________________				  
							//putting back temporary data
							new_dw->put(px_tmp, d_lb->pXLabel_preReloc, true);
							new_dw->put(pids_tmp, d_lb->pParticleIDLabel_preReloc, true);
							new_dw->put(pmass_tmp, d_lb->pMassLabel_preReloc, true);
							new_dw->put(pSize_tmp, d_lb->pSizeLabel_preReloc, true);
							new_dw->put(pdisp_tmp, d_lb->pDispLabel_preReloc, true);
							new_dw->put(pstress_tmp, d_lb->pStressLabel_preReloc, true);
							new_dw->put(pvolume_tmp, d_lb->pVolumeLabel_preReloc, true);
							new_dw->put(pvelocity_tmp, d_lb->pVelocityLabel_preReloc, true);
							new_dw->put(pextforce_tmp, d_lb->pExtForceLabel_preReloc, true);
							new_dw->put(ptemp_tmp, d_lb->pTemperatureLabel_preReloc, true);
							new_dw->put(ptempgrad_tmp, d_lb->pTemperatureGradientLabel_preReloc, true);
							new_dw->put(ptempP_tmp, d_lb->pTempPreviousLabel_preReloc, true);
							new_dw->put(pdTdt_tmp, d_lb->pdTdtLabel, true); 
							new_dw->put(ploc_tmp, d_lb->pLocalizedMPMLabel_preReloc, true);
							new_dw->put(pvelgrad_tmp, d_lb->pVelGradLabel_preReloc, true);
							new_dw->put(pF_tmp, d_lb->pDeformationMeasureLabel_preReloc, true);
							if (d_flags->d_computeScaleFactor) {
								new_dw->put(pscalefac_tmp, d_lb->pScaleFactorLabel_preReloc, true);
							}
							if (d_flags->d_with_color) {
								new_dw->put(pcolor_tmp, d_lb->pColorLabel_preReloc, true);
							}
							if (d_flags->d_doScalarDiffusion) {
								new_dw->put(pConc_tmp, d_lb->diffusion->pConcentration_preReloc, true);
								new_dw->put(pConcPrev_tmp, d_lb->diffusion->pConcPrevious_preReloc, true);
								new_dw->put(pGradConc_tmp, d_lb->diffusion->pGradConcentration_preReloc, true);
								new_dw->put(pESFlux_tmp, d_lb->diffusion->pExternalScalarFlux_preReloc, true);
								new_dw->put(pArea_tmp, d_lb->diffusion->pArea_preReloc, true);
								new_dw->put(pD_tmp, d_lb->diffusion->pDiffusivity_preReloc, true);
							}
							if (d_flags->d_useLoadCurves) {
								new_dw->put(pLoadCID_tmp, d_lb->pLoadCurveIDLabel_preReloc, true);
							}
							
							 
							//_________________________						
							//
							if (!mpm_matl_dest->getIsActive()){
								mpm_matl_dest->setIsActive(true);
								cout<<"dwi_dest : "<<dwi_dest<<", at time: "<<time <<", is_active: "<<mpm_matl_dest->getIsActive()<<endl;
					        } 
					        //_________________________
					        //       
						}
					}	 	 		       	
				} 
			}
		}
	//_______________________		
    //removing the marked material points from origin material
    
    for (unsigned int m = 0; m <  d_materialManager->getNumMatls("MPM"); m++) {
		MPMMaterial* mpm_matl = 
	                        (MPMMaterial*) d_materialManager->getMaterial("MPM", m);
		int dwi = mpm_matl->getDWIndex();
		ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
		ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);        
		for (ParticleSubset::iterator iter = pset->begin();	iter != pset->end(); iter++) {
			particleIndex idx = *iter;
			if (RemovedParticle[dwi][idx] ==  1 ) {
				delset->addParticle(idx);
			}
		}
		new_dw->deleteParticles(delset);			
	}		   
  }  // loop over patches  
 }
//______________________________________________________________________
//

void MPMGranular::readInsertGranularParticlesFile(string filename)
{

 if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR Opening particle insertion file '"+filename+"'\n",
                                  __FILE__, __LINE__);
    }
    while(is) {
        double t1, MatId, color,transx,transy,transz,v_new_x,v_new_y,v_new_z;
        is >> t1 >>  MatId>>  color >> transx >> transy >> transz >> v_new_x >> v_new_y >> v_new_z;
        if(is) {
            d_IPTimes.push_back(t1);
            d_IPdwi.push_back(MatId); //material index
            d_IPColor.push_back(color);
            d_IPTranslate.push_back(Vector(transx,transy,transz));
            d_IPVelNew.push_back(Vector(v_new_x,v_new_y,v_new_z));
        }
    }
  }
}

void MPMGranular::scheduleInsertGranularParticles(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  if(d_flags->d_insertParticles){
    printSchedule(patches,cout_doing,"MPM::scheduleInsertGranularParticles");

    Task* t=scinew Task("MPM::insertGranularParticles",this,
                  &MPMGranular::insertGranularParticles);
              

    t->requiresVar(Task::OldDW, d_lb->simulationTimeLabel);
    t->requiresVar(Task::OldDW, d_lb->delTLabel );

    t->modifiesVar(d_lb->pXLabel_preReloc);
    t->modifiesVar(d_lb->pVelocityLabel_preReloc);
    t->requiresVar(Task::OldDW, d_lb->pColorLabel,  Ghost::None);

    sched->addTask(t, patches, matls);
  }
}

void MPMGranular::insertGranularParticles(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing MPM::insertGranularParticles");

    // Get the current simulation time
    simTime_vartype simTimeVar;
    old_dw->get(simTimeVar, d_lb->simulationTimeLabel);
    double time = simTimeVar;

    delt_vartype delT;
    old_dw->get(delT, d_lb->delTLabel, getLevel(patches) );

    // activate materials if it is their time
    unsigned int numMPMMatls=d_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                       (MPMMaterial*) d_materialManager->getMaterial("MPM", m);      
		int dwi = mpm_matl->getDWIndex(); 
		unsigned int numMPMMatls=d_materialManager->getNumMatls( "MPM" );
        vector<int> numPartsPerMatIdx(numMPMMatls, 0); 
		ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
	    numPartsPerMatIdx[dwi] = pset->addParticles(0);  
	   if(time >= mpm_matl->getActivationTime() &&  time <= mpm_matl->getDeactivationTime()){ 
			 if ( numPartsPerMatIdx[dwi] != 0){
				mpm_matl->setIsActive(true);
			} else{
				mpm_matl->setIsActive(false);
			}				
		 }	
    } //end matID
    
    
    int index = -999;
    for(int i = 0; i<(int) d_IPTimes.size(); i++){
      if(time+delT > d_IPTimes[i] && time <= d_IPTimes[i]){
        index = i;
        if(index>=0){
			MPMMaterial* mpm_matl_index = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  d_IPdwi[index] ); 
			int dwi_index = mpm_matl_index->getDWIndex();  
			unsigned int numMPMMatls=d_materialManager->getNumMatls( "MPM" );
			for(unsigned int m = 0; m < numMPMMatls; m++){
				MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
				int dwi = mpm_matl->getDWIndex(); 
				if (dwi == dwi_index){//HK instead of color
					ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
					// Get the arrays of particle values to be changed
					ParticleVariable<Point> px;
					ParticleVariable<Vector> pvelocity;
					constParticleVariable<double> pcolor;
					old_dw->get(pcolor,             d_lb->pColorLabel,              pset);
					new_dw->getModifiable(px,       d_lb->pXLabel_preReloc,         pset);
					new_dw->getModifiable(pvelocity,d_lb->pVelocityLabel_preReloc,  pset);
					// Loop over particles here
					for(ParticleSubset::iterator iter  = pset->begin();
											iter != pset->end();   iter++){
						particleIndex idx = *iter;
						if(pcolor[idx]==d_IPColor[index]){
							pvelocity[idx]=d_IPVelNew[index];
							px[idx] = px[idx] + d_IPTranslate[index];
						} // end if
					}   // end for idx
				}     // end for	dwi_index
			}	//end for m 
		}	   	   //end if index>0

      }         // end if time
    }           // end for
    
   
  }//end patch
}
//______________________________________________________________________
//

