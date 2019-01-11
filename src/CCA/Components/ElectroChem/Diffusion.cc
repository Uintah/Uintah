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
#include <CCA/Components/ElectroChem/FluxModels.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <iostream>

using namespace Uintah;

Diffusion::Diffusion(const ProcessorGroup* myworld,
                     const MaterialManagerP materialManager)
    : ApplicationCommon(myworld, materialManager) {

  d_one_mat_set.add(0);
  d_one_mat_set.addReference();

  d_one_mat_subset.add(0);
  d_one_mat_subset.addReference();

  offsets[0].x(1.0); offsets[0].y(0.0); offsets[0].z(0.0);
  offsets[1].x(0.0); offsets[1].y(1.0); offsets[1].z(0.0);
  offsets[2].x(0.0); offsets[2].y(0.0); offsets[2].z(1.0);
}
    
Diffusion::~Diffusion(){
  d_one_mat_set.removeReference();
  d_one_mat_subset.removeReference();
}

void Diffusion::problemSetup(const ProblemSpecP& ps,
                             const ProblemSpecP& restart_ps,
                                   GridP&        grid){

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
}

void Diffusion::scheduleInitialize(const LevelP&     level,
                                         SchedulerP& sched){

  Task* t1 = scinew Task("Diffusion::initializeMaterialId", this,
                         &Diffusion::initializeMaterialId);

  Task* t2 = scinew Task("Diffusion::initializeFluxModel", this,
                         &Diffusion::initializeFluxModel);

  t1->computes(d_eclabel.cc_concentration);
  t1->computes(d_eclabel.cc_matid);

  t2->computes(d_eclabel.fcx_fluxmodel);
  t2->computes(d_eclabel.fcy_fluxmodel);
  t2->computes(d_eclabel.fcz_fluxmodel);

  t2->requires(Task::NewDW, d_eclabel.cc_matid, Ghost::AroundCells, 1);

  sched->addTask(t1, level->eachPatch(), &d_one_mat_set);
  sched->addTask(t2, level->eachPatch(), &d_one_mat_set);
}

void Diffusion::initializeMaterialId(const ProcessorGroup* pg,
                                     const PatchSubset*    patches,
                                     const MaterialSubset* matls,
                                           DataWarehouse*  old_dw,
                                           DataWarehouse*  new_dw){

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
}

void Diffusion::initializeFluxModel(const ProcessorGroup* pg,
                                    const PatchSubset*    patches,
                                    const MaterialSubset* matls,
                                          DataWarehouse*  old_dw,
                                           DataWarehouse*  new_dw){
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    IntVector low_idx  = patch->getCellLowIndex();
    IntVector high_idx = patch->getCellHighIndex();

    int low_bnd[] {0,0,0};
    int high_bnd[] {0,0,0};
    if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
      low_bnd[0] = 1;
    }

    if(patch->getBCType(Patch::xplus) != Patch::Neighbor){
      high_bnd[0] = 1;
    }

    if(patch->getBCType(Patch::yminus) != Patch::Neighbor){
      low_bnd[1] = 1;
    }

    if(patch->getBCType(Patch::yplus) != Patch::Neighbor){
      high_bnd[1] = 1;
    }

    if(patch->getBCType(Patch::zminus) != Patch::Neighbor){
      low_bnd[2] = 1;
    }

    if(patch->getBCType(Patch::zplus) != Patch::Neighbor){
      high_bnd[2] = 1;
    }

    constCCVariable<int> matid;

    SFCXVariable<int> fcx_fluxmodel;
    SFCYVariable<int> fcy_fluxmodel;
    SFCZVariable<int> fcz_fluxmodel;

    new_dw->get(matid, d_eclabel.cc_matid, 0, patch, Ghost::AroundCells, 1);

    new_dw->allocateAndPut(fcx_fluxmodel, d_eclabel.fcx_fluxmodel, 0, patch);
    new_dw->allocateAndPut(fcy_fluxmodel, d_eclabel.fcy_fluxmodel, 0, patch);
    new_dw->allocateAndPut(fcz_fluxmodel, d_eclabel.fcz_fluxmodel, 0, patch);

    FluxModels::FluxModel fmodel;

    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      for(int i = 0; i < 3; ++i){
        fmodel = FluxModels::PNP;
        if(c[i] == low_idx[i]){
          if(low_bnd[i]){
            fmodel = FluxModels::BC;
          }else{
            if(matid[c] != matid[c-offsets[i]]){ fmodel = FluxModels::MaterialInterface; }
            else{ fmodel = FluxModels::Basic; }
          }
        }else{
          if(matid[c] != matid[c-offsets[i]]){ fmodel = FluxModels::MaterialInterface; }
          else{ fmodel = FluxModels::PNP; }
        }
        switch (i) {
          case 0: fcx_fluxmodel[c] = fmodel; 
                  break;
          case 1: fcy_fluxmodel[c] = fmodel;
                  break;
          case 2: fcz_fluxmodel[c] = fmodel;
                  break;
        }

        if(c[i] == high_idx[i]-1 && high_bnd[i]){
          switch (i) {
            case 0: fcx_fluxmodel[c+offsets[i]] = FluxModels::BC;
                    break;
            case 1: fcy_fluxmodel[c+offsets[i]] = FluxModels::BC;
                    break;
            case 2: fcz_fluxmodel[c+offsets[i]] = FluxModels::BC;
                    break;
          }
        }
      } // end i for loop
    } // end CellIterator
  } // end of for patch loop
}

void Diffusion::scheduleRestartInitialize(const LevelP&     level,
                                                SchedulerP& sched){
}

void Diffusion::scheduleComputeStableTimeStep(const LevelP&     level,
                                                    SchedulerP& sched){
  Task* task = scinew Task("Diffusion::computeStableTimeStep",this, 
                           &Diffusion::computeStableTimeStep);
  task->computes(getDelTLabel(),level.get_rep());
  sched->addTask(task, level->eachPatch(), &d_one_mat_set);
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
  t->requires(Task::OldDW, d_eclabel.fcx_fluxmodel,    Ghost::AroundFacesX, 0);
  t->requires(Task::OldDW, d_eclabel.fcy_fluxmodel,    Ghost::AroundFacesY, 0);
  t->requires(Task::OldDW, d_eclabel.fcz_fluxmodel,    Ghost::AroundFacesZ, 0);

  t->computes(d_eclabel.fcx_flux);
  t->computes(d_eclabel.fcy_flux);
  t->computes(d_eclabel.fcz_flux);
  t->computes(d_eclabel.fcx_fluxmodel);
  t->computes(d_eclabel.fcy_fluxmodel);
  t->computes(d_eclabel.fcz_fluxmodel);

  sched->addTask(t, patches, &d_one_mat_set);
}

void Diffusion::computeFlux(const ProcessorGroup* pg,
                            const PatchSubset*    patches,
                            const MaterialSubset* matls,
                                  DataWarehouse*  old_dw,
                                  DataWarehouse*  new_dw) {

  int num_matls = m_materialManager->getNumMatls( "ElectroChem" );

  double* diff_coeff = new double[num_matls];

  for(int m = 0; m < num_matls; ++m){
    ElectroChem::ECMaterial* ec_matl = (ElectroChem::ECMaterial* )
                               m_materialManager->getMaterial("ElectroChem", m);
    diff_coeff[m] = ec_matl->GetDiffusionCoeff();
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    IntVector high_idx = patch->getCellHighIndex();
    Vector dx = patch->dCell();

    int high_bnd[] {0,0,0};

    if(patch->getBCType(Patch::xplus) != Patch::Neighbor){
      high_bnd[0] = 1;
    }

    if(patch->getBCType(Patch::yplus) != Patch::Neighbor){
      high_bnd[1] = 1;
    }

    if(patch->getBCType(Patch::zplus) != Patch::Neighbor){
      high_bnd[2] = 1;
    }

    constCCVariable<double> conc;
    constCCVariable<int>    matid;

    constSFCXVariable<int> fcx_fluxmodel_old;
    constSFCYVariable<int> fcy_fluxmodel_old;
    constSFCZVariable<int> fcz_fluxmodel_old;

    SFCXVariable<double> fcx_flux;
    SFCYVariable<double> fcy_flux;
    SFCZVariable<double> fcz_flux;

    SFCXVariable<int> fcx_fluxmodel;
    SFCYVariable<int> fcy_fluxmodel;
    SFCZVariable<int> fcz_fluxmodel;

    old_dw->get(conc,  d_eclabel.cc_concentration, 0, patch,
                Ghost::AroundCells, 1);
    old_dw->get(matid, d_eclabel.cc_matid,         0, patch,
                Ghost::AroundCells, 1);
    old_dw->get(fcx_fluxmodel_old, d_eclabel.fcx_fluxmodel, 0, patch,
                Ghost::AroundFacesX, 0);
    old_dw->get(fcy_fluxmodel_old, d_eclabel.fcy_fluxmodel, 0, patch,
                Ghost::AroundFacesY, 0);
    old_dw->get(fcz_fluxmodel_old, d_eclabel.fcz_fluxmodel, 0, patch,
                Ghost::AroundFacesZ, 0);

    new_dw->allocateAndPut(fcx_flux, d_eclabel.fcx_flux, 0, patch);
    new_dw->allocateAndPut(fcy_flux, d_eclabel.fcy_flux, 0, patch);
    new_dw->allocateAndPut(fcz_flux, d_eclabel.fcz_flux, 0, patch);

    new_dw->allocateAndPut(fcx_fluxmodel, d_eclabel.fcx_fluxmodel, 0, patch);
    new_dw->allocateAndPut(fcy_fluxmodel, d_eclabel.fcy_fluxmodel, 0, patch);
    new_dw->allocateAndPut(fcz_fluxmodel, d_eclabel.fcz_fluxmodel, 0, patch);

    double flux;
    double dc;

    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      for(int i = 0; i < 3; ++i){
        /**
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
      **/
        flux = 0;
        switch (i) {
          case 0: fcx_flux[c] = flux;
                  break;
          case 1: fcy_flux[c] = flux;
                  break;
          case 2: fcz_flux[c] = flux;
                  break;
        }

        if(c[i] == high_idx[i]-1 && high_bnd[i]){
          switch (i) {
            case 0: fcx_flux[c+offsets[0]] = flux;
                    break;
            case 1: fcy_flux[c+offsets[1]] = flux;
                    break;
            case 2: fcz_flux[c+offsets[2]] = flux;
                    break;
          }
        }
      }
      fcx_fluxmodel[c] = fcx_fluxmodel_old[c];
      fcy_fluxmodel[c] = fcy_fluxmodel_old[c];
      fcz_fluxmodel[c] = fcz_fluxmodel_old[c];

      if(c[0] == high_idx[0]-1 && high_bnd[0]){
        fcx_fluxmodel[c+offsets[0]] = fcx_fluxmodel_old[c+offsets[0]];
      }
      if(c[1] == high_idx[1]-1 && high_bnd[1]){
        fcy_fluxmodel[c+offsets[1]] = fcy_fluxmodel_old[c+offsets[1]];
      }
      if(c[2] == high_idx[2]-1 && high_bnd[2]){
        fcz_fluxmodel[c+offsets[2]] = fcz_fluxmodel_old[c+offsets[2]];
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
  t->requires(Task::NewDW, d_eclabel.fcx_flux,         Ghost::AroundFacesX, 1);
  t->requires(Task::NewDW, d_eclabel.fcy_flux,         Ghost::AroundFacesY, 1);
  t->requires(Task::NewDW, d_eclabel.fcz_flux,         Ghost::AroundFacesZ, 1);

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

    new_dw->get(fcx_flux, d_eclabel.fcx_flux, 0, patch, Ghost::AroundFacesX, 1);
    new_dw->get(fcy_flux, d_eclabel.fcy_flux, 0, patch, Ghost::AroundFacesY, 1);
    new_dw->get(fcz_flux, d_eclabel.fcz_flux, 0, patch, Ghost::AroundFacesZ, 1);

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
