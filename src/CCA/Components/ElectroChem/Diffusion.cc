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

#include <CCA/Components/ElectroChem/Diffusion.h>
#include <CCA/Components/ElectroChem/ECMaterial.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <iostream>

using namespace Uintah;

Diffusion::Diffusion(const ProcessorGroup* myworld,
                     const MaterialManagerP materialManager)
    : ApplicationCommon(myworld, materialManager) {

  d_delt       = 0.0;

  d_one_mat_set.add(0);
  d_one_mat_set.addReference();

  d_one_mat_subset.add(0);
  d_one_mat_subset.addReference();

  std::cout << "**** Constructor." << std::endl;
}
    
Diffusion::~Diffusion(){
}

void Diffusion::problemSetup(const ProblemSpecP& ps,
                             const ProblemSpecP& restart_ps,
                                   GridP&        grid){

  std::cout << "***** Begin Problem Setup. " << std::endl;
  ProblemSpecP root_ps = 0;
  if (restart_ps){
    root_ps = restart_ps;
  } else{
    root_ps = ps;
  }

  ProblemSpecP ec_ps = root_ps->findBlock("ElectroChem");

  ec_ps->require("delt", d_delt);

  ProblemSpecP mat_ps = root_ps->findBlockWithOutAttribute("MaterialProperties");
  if( !mat_ps ) {
      throw ProblemSetupException("ERROR: Cannot find the Material Properties block", __FILE__, __LINE__);
  }

  ProblemSpecP ec_mat_ps = mat_ps->findBlock("ElectroChem");
  if( !ec_mat_ps ) {
    throw ProblemSetupException("ERROR: Cannot find the ElectroChem Materials Properties block", __FILE__, __LINE__);
  }

  for(ProblemSpecP tmp_ps = ec_mat_ps->findBlock("material"); tmp_ps != nullptr;
                   tmp_ps = tmp_ps->findNextBlock("material") ) {
    ElectroChem::ECMaterial *mat =
                    scinew ElectroChem::ECMaterial(tmp_ps, m_materialManager);

    m_materialManager->registerMaterial( "ElectroChem", mat);
  }
  std::cout << "***** End Problem Setup. " << std::endl;
}

void Diffusion::scheduleInitialize(const LevelP&     level,
                                         SchedulerP& sched){

  Task* t = scinew Task("Diffusion::initialize", this,
                        &Diffusion::initialize);

  t->computes(d_eclabel.cc_concentration);
  t->computes(d_eclabel.cc_matid);
  sched->addTask(t, level->eachPatch(), &d_one_mat_set);
}

void Diffusion::initialize(const ProcessorGroup* pg,
                           const PatchSubset*    patches,
                           const MaterialSubset* matls,
                                 DataWarehouse*  old_dw,
                                 DataWarehouse*  new_dw){

  std::cout << "***** Begin Initialize. " << std::endl;
  int num_matls = m_materialManager->getNumMatls( "ElectroChem" );

  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    CCVariable<double> concentration;
    CCVariable<int>    matid;
    new_dw->allocateAndPut(concentration, d_eclabel.cc_concentration, 0, patch);
    new_dw->allocateAndPut(matid,         d_eclabel.cc_matid,         0, patch);
    concentration.initialize(0.0);
    matid.initialize(-1);

    for(int m = 0; m < num_matls; m++){
      ElectroChem::ECMaterial* ec_matl = (ElectroChem::ECMaterial* )
                               m_materialManager->getMaterial("ElectroChem", m);

      for(int obj=0; obj < ec_matl->GetNumObjects(); ++obj){
        GeometryPieceP piece = ec_matl->GetGeomPiece(obj);
        for(CellIterator iter = patch->getExtraCellIterator();
                         !iter.done(); ++iter){
          IntVector c = *iter;
          Point center = patch->cellPosition(c);

          if(piece->inside(center)){
            concentration[c] = ec_matl->GetConcentration(obj);
            matid[c] = m;
          }
        } // end for loop - CellIterator
      } // end for loop - obj
    } // end for loop - matls
  } // end for loop - patches
  std::cout << "***** End Initialize. " << std::endl;
}

void Diffusion::scheduleRestartInitialize(const LevelP&     level,
                                                SchedulerP& sched){
}

void Diffusion::scheduleComputeStableTimeStep(const LevelP&     level,
                                                    SchedulerP& sched){
  Task* task = scinew Task("Diffusion::computeStableTimeStep",this, 
                           &Diffusion::computeStableTimeStep);
  task->computes(getDelTLabel(),level.get_rep());
  sched->addTask(task, level->eachPatch(),
                 m_materialManager->allMaterials("ElectroChem"));
}

void Diffusion::computeStableTimeStep(const ProcessorGroup* pg,
                                      const PatchSubset*    patches,
                                      const MaterialSubset* matls,
                                            DataWarehouse*  old_dw,
                                            DataWarehouse*  new_dw){
  new_dw->put(delt_vartype(d_delt), getDelTLabel(),getLevel(patches));
}

void Diffusion::scheduleTimeAdvance(const LevelP&     level,
                                          SchedulerP& sched){
  const PatchSet* patches = level->eachPatch();

  scheduleComputeFlux(patches, sched);
  scheduleForwardEuler(patches, sched);
}

void Diffusion::scheduleComputeFlux(const PatchSet* patches, SchedulerP& sched){
  Task* t = scinew Task("Diffusion::computeFlux", this,
                        &Diffusion::computeFlux);

  t->requires(Task::OldDW, d_eclabel.cc_concentration, Ghost::AroundCells,  1);
  t->requires(Task::OldDW, d_eclabel.cc_matid,         Ghost::AroundCells,  1);

  t->computes(d_eclabel.fcx_flux);
  t->computes(d_eclabel.fcy_flux);
  t->computes(d_eclabel.fcz_flux);

  sched->addTask(t, patches, &d_one_mat_set);
}

void Diffusion::computeFlux(const ProcessorGroup* pg,
                            const PatchSubset*    patches,
                            const MaterialSubset* matls,
                                  DataWarehouse*  old_dw,
                                  DataWarehouse*  new_dw) {

  IntVector offsets[3];

  offsets[0].x(1.0); offsets[0].y(0.0); offsets[0].z(0.0);
  offsets[1].x(0.0); offsets[1].y(1.0); offsets[1].z(0.0);
  offsets[2].x(0.0); offsets[2].y(0.0); offsets[2].z(1.0);

  int num_matls = m_materialManager->getNumMatls( "ElectroChem" );

  double* diff_coeff = new double[num_matls];

  for(int m = 0; m < num_matls; ++m){
    ElectroChem::ECMaterial* ec_matl = (ElectroChem::ECMaterial* )
                               m_materialManager->getMaterial("ElectroChem", m);
    diff_coeff[m] = ec_matl->GetDiffusionCoeff();
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    IntVector lower_idx = patch->getCellLowIndex();
    Vector dx = patch->dCell();

    int lower_bnd[] {0,0,0};
    if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
      lower_bnd[0] = 1;
    }

    if(patch->getBCType(Patch::yminus) != Patch::Neighbor){
      lower_bnd[1] = 1;
    }

    if(patch->getBCType(Patch::zminus) != Patch::Neighbor){
      lower_bnd[2] = 1;
    }

    constCCVariable<double> conc;
    constCCVariable<int>    matid;

    SFCXVariable<double> fcx_flux;
    SFCYVariable<double> fcy_flux;
    SFCZVariable<double> fcz_flux;

    old_dw->get(conc,  d_eclabel.cc_concentration, 0, patch,
                Ghost::AroundCells, 1);
    old_dw->get(matid, d_eclabel.cc_matid,         0, patch,
                Ghost::AroundCells, 1);

    new_dw->allocateAndPut(fcx_flux, d_eclabel.fcx_flux, 0, patch);
    new_dw->allocateAndPut(fcy_flux, d_eclabel.fcy_flux, 0, patch);
    new_dw->allocateAndPut(fcz_flux, d_eclabel.fcz_flux, 0, patch);

    double flux;
    double dc;
    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      for(int i = 0; i < 3; ++i){
        if(c[i] == lower_idx[i]){
          if(lower_bnd[i]){
            flux = 0.0;
          }else{
            dc = .5*(diff_coeff[matid[c]] + diff_coeff[matid[c-offsets[i]]]);
            flux = -dc*(conc[c] - conc[c-offsets[i]])/dx[i];
          }
        }else{
          dc = .5*(diff_coeff[matid[c]] + diff_coeff[matid[c-offsets[i]]]);
          flux = -dc*(conc[c] - conc[c-offsets[i]])/dx[i];
        }
        if(i == 0){ fcx_flux[c] = flux; }
        else if(i == 1){ fcy_flux[c] = flux; }
        else if(i == 2){ fcz_flux[c] = flux; }
      }
    }

  } // end of for patch loop
  
  delete[] diff_coeff;
}


void Diffusion::scheduleForwardEuler(const PatchSet* patches, SchedulerP& sched){
  Task* t = scinew Task("Diffusion::forwardEuler", this,
                        &Diffusion::forwardEuler);

  t->requires(Task::OldDW, d_eclabel.cc_concentration, Ghost::AroundCells,  0);
  t->requires(Task::OldDW, d_eclabel.cc_matid,         Ghost::AroundCells,  0);
  t->requires(Task::NewDW, d_eclabel.fcx_flux,         Ghost::AroundFacesX, 0);
  t->requires(Task::NewDW, d_eclabel.fcy_flux,         Ghost::AroundFacesY, 0);
  t->requires(Task::NewDW, d_eclabel.fcz_flux,         Ghost::AroundFacesZ, 0);

  t->computes(d_eclabel.cc_concentration);
  t->computes(d_eclabel.cc_matid);
  sched->addTask(t, patches, &d_one_mat_set);
}


void Diffusion::forwardEuler(const ProcessorGroup* pg,
                             const PatchSubset*    patches,
                             const MaterialSubset* matls,
                                   DataWarehouse*  old_dw,
                                   DataWarehouse*  new_dw){
  IntVector xoffset(1,0,0);
  IntVector yoffset(0,1,0);
  IntVector zoffset(0,0,1);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();

    constCCVariable<double>   cc_old_conc;
    constCCVariable<int>      cc_old_matid;
    constSFCXVariable<double> fcx_flux;
    constSFCYVariable<double> fcy_flux;
    constSFCZVariable<double> fcz_flux;

    CCVariable<double> cc_conc;
    CCVariable<int>    cc_matid;

    old_dw->get(cc_old_conc,  d_eclabel.cc_concentration, 0, patch,
                Ghost::AroundCells, 0);
    old_dw->get(cc_old_matid, d_eclabel.cc_matid,         0, patch,
                Ghost::AroundCells, 0);

    new_dw->get(fcx_flux, d_eclabel.fcx_flux, 0, patch, Ghost::AroundFacesX, 0);
    new_dw->get(fcy_flux, d_eclabel.fcy_flux, 0, patch, Ghost::AroundFacesY, 0);
    new_dw->get(fcz_flux, d_eclabel.fcz_flux, 0, patch, Ghost::AroundFacesZ, 0);

    new_dw->allocateAndPut(cc_conc,  d_eclabel.cc_concentration, 0, patch);
    new_dw->allocateAndPut(cc_matid, d_eclabel.cc_matid,         0, patch);

    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      cc_conc[c]  = cc_old_conc[c]
                  - d_delt * ((fcx_flux[c + xoffset] - fcx_flux[c])/dx.x()
                           +  (fcy_flux[c + yoffset] - fcy_flux[c])/dx.y()
                           +  (fcz_flux[c + zoffset] - fcz_flux[c])/dx.z());

      cc_matid[c] = cc_old_matid[c];
    }
  } // end patch loop
}
