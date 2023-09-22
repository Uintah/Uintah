/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#include <CCA/Components/Models/FluidsBased/BinaryProperties.h>
#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>

#include <CCA/Components/ICE/Core/ICELabel.h>
#include <CCA/Components/ICE/Core/ConservationTest.h>
#include <CCA/Components/ICE/Core/Diffusion.h>
#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>

#include <CCA/Ports/Output.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/InvalidValue.h>


using namespace Uintah;
using namespace std;

Dout m_cout_tasks(   "BinaryProperties_tasks", "Models:BinaryProperties", "Print task scheduling & execution", false );
//______________________________________________________________________
BinaryProperties::BinaryProperties(const ProcessorGroup* myworld,
                                   const MaterialManagerP & materialManager,
                                   const ProblemSpecP& params)
  : FluidsBasedModel(myworld, materialManager), d_params(params)
{
  m_modelComputesThermoTransportProps = true;

  d_matl_set = 0;
  Ilb  = scinew ICELabel();
  Slb = scinew BinaryPropertiesLabel();
}

//______________________________________________________________________
//
BinaryProperties::~BinaryProperties()
{
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  VarLabel::destroy( d_scalar->scalar_CCLabel );
  VarLabel::destroy( d_scalar->source_CCLabel );
  VarLabel::destroy( d_scalar->diffusionCoefLabel );
  VarLabel::destroy( Slb->sum_scalar_fLabel );
  delete Ilb;
  delete Slb;

  for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                iter != d_scalar->regions.end(); iter++){
    Region* region = *iter;
    delete region;

  }
}

//__________________________________
BinaryProperties::Region::Region(GeometryPieceP piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("scalar", initialScalar);
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void
BinaryProperties::problemSetup( GridP &, const bool isRestart )
{
  DOUTR( m_cout_tasks, " BinaryProperties::problemSetup " );

  d_matl = m_materialManager->parseAndLookupMaterial(d_params, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  //__________________________________
  // - create Label names
  // - Let ICE know that this model computes the
  //   thermoTransportProperties.
  // - register the scalar to be transported
  d_scalar = scinew Scalar();
  d_scalar->index = 0;
  d_scalar->name  = "f";

  const TypeDescription* td_CCdouble = CCVariable<double>::getTypeDescription();
  d_scalar->scalar_CCLabel =     VarLabel::create("scalar-f",       td_CCdouble);
  d_scalar->diffusionCoefLabel = VarLabel::create("scalar-diffCoef",td_CCdouble);
  d_scalar->source_CCLabel =
                                 VarLabel::create("scalar-f_src",  td_CCdouble);

  Slb->sum_scalar_fLabel      =  VarLabel::create("sum_scalar_f",
                                            sum_vartype::getTypeDescription());

  registerTransportedVariable(d_matl_set,
                              d_scalar->scalar_CCLabel,
                              d_scalar->source_CCLabel);

  //__________________________________
  // Read in the constants for the scalar
  ProblemSpecP child = d_params->findBlock("BinaryProperties")->findBlock("scalar");
  if (!child){
    throw ProblemSetupException("BinaryProperties: Couldn't find (BinaryProperties) or (scalar) tag", __FILE__, __LINE__);
  }

  child->getWithDefault("test_conservation", d_test_conservation, false);

  ProblemSpecP const_ps = child->findBlock("constants");
  if(!const_ps) {
    throw ProblemSetupException("BinaryProperties: Couldn't find constants tag", __FILE__, __LINE__);
  }

  const_ps->getWithDefault("rho_A",         d_rho_A,         -9);
  const_ps->getWithDefault("rho_B",         d_rho_B,        -9);
  const_ps->getWithDefault("cv_A",          d_cv_A,          -9);
  const_ps->getWithDefault("cv_B",          d_cv_B,         -9);
  const_ps->getWithDefault("R_A",           d_R_A,           -9);
  const_ps->getWithDefault("R_B",           d_R_B,          -9);
  const_ps->getWithDefault("thermalCond_A", d_thermalCond_A, 0);
  const_ps->getWithDefault("thermalCond_B", d_thermalCond_B, 0);
  const_ps->getWithDefault("dynamic_viscosity_A",   d_viscosity_A,   0);
  const_ps->getWithDefault("dynamic_viscosity_B",   d_viscosity_B,   0);
  const_ps->getWithDefault("diffusivity",           d_scalar->diff_coeff, -9);
  const_ps->getWithDefault("initialize_diffusion_knob",
                            d_scalar->initialize_diffusion_knob,   0);

  if( d_rho_A   == -9  || d_rho_B == -9 ||
      d_cv_A    == -9  || d_cv_B  == -9 ||
      d_R_A     == -9  || d_R_B   == -9  ) {
    ostringstream warn;
    warn << " ERROR BinaryProperties: Input variable(s) not specified \n"
         << "\n diffusivity      "<< d_scalar->diff_coeff
         << "\n rho_A            "<< d_rho_A
         << "\n rho_B            "<< d_rho_B
         << "\n R_B              "<< d_R_B
         << "\n R_A              "<< d_R_A<<endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in the geometry objects for the scalar
  if( !isRestart ){
   for ( ProblemSpecP geom_obj_ps = child->findBlock("geom_object"); geom_obj_ps != nullptr; geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
    vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);

    GeometryPieceP mainpiece;
    if( pieces.size() == 0 ){
      throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
    }
    else if(pieces.size() > 1){
      mainpiece = scinew UnionGeometryPiece(pieces);
    }
    else {
      mainpiece = pieces[0];
    }

    d_scalar->regions.push_back(scinew Region(mainpiece, geom_obj_ps));
   }
  }
  if(d_scalar->regions.size() == 0 && !isRestart) {
    throw ProblemSetupException("Variable: scalar-f does not have any initial value regions", __FILE__, __LINE__);
  }
}
//______________________________________________________________________
//      S C H E D U L E   I N I T I A L I Z E
void BinaryProperties::scheduleInitialize(SchedulerP   & sched,
                                          const LevelP & level)
{
  printSchedule( level, m_cout_tasks, " BinaryProperties::scheduleInitialize" );

  Task* t = scinew Task("BinaryProperties::initialize", this, &BinaryProperties::initialize);

  t->requires(Task::NewDW, Ilb->timeStepLabel );

  t->modifies( Ilb->sp_vol_CCLabel);
  t->modifies( Ilb->rho_micro_CCLabel);
  t->modifies( Ilb->rho_CCLabel);
  t->modifies( Ilb->specific_heatLabel);
  t->modifies( Ilb->gammaLabel);
  t->modifies( Ilb->thermalCondLabel);
  t->modifies( Ilb->viscosityLabel);

  t->computes(d_scalar->scalar_CCLabel);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
//       I N I T I A L I Z E
void BinaryProperties::initialize(const ProcessorGroup *,
                                  const PatchSubset    * patches,
                                  const MaterialSubset *,
                                  DataWarehouse        *,
                                  DataWarehouse        * new_dw)
{
  timeStep_vartype timeStep;
  new_dw->get(timeStep, VarLabel::find( timeStep_name) );

  bool isNotInitialTimeStep = (timeStep > 0);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask( patches, patch, m_cout_tasks, "BinaryProperties::initialize" );

    int indx = d_matl->getDWIndex();

    CCVariable<double>  f;
    CCVariable<double>  cv;
    CCVariable<double>  gamma;
    CCVariable<double>  thermalCond;
    CCVariable<double>  viscosity;
    CCVariable<double>  rho_CC;
    CCVariable<double>  sp_vol;
    CCVariable<double>  press;
    CCVariable<double>  rho_micro;

    constCCVariable<double> Temp;

    new_dw->allocateAndPut(f, d_scalar->scalar_CCLabel, indx, patch);
    new_dw->getModifiable( rho_CC,      Ilb->rho_CCLabel,       indx,patch );
    new_dw->getModifiable( sp_vol,      Ilb->sp_vol_CCLabel,    indx,patch );
    new_dw->getModifiable( rho_micro,   Ilb->rho_micro_CCLabel, indx,patch );
    new_dw->getModifiable( gamma,       Ilb->gammaLabel,        indx,patch );
    new_dw->getModifiable( cv,          Ilb->specific_heatLabel,indx,patch );
    new_dw->getModifiable( thermalCond, Ilb->thermalCondLabel,  indx,patch );
    new_dw->getModifiable( viscosity,   Ilb->viscosityLabel,    indx,patch );
    //__________________________________
    //  initialize the scalar field in a region
    f.initialize(0);

    for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                  iter != d_scalar->regions.end(); iter++){
      Region* region = *iter;
      for(CellIterator iter = patch->getExtraCellIterator();!iter.done(); iter++){
        IntVector c = *iter;
        Point p = patch->cellPosition(c);

        if(region->piece->inside(p)) {
          f[c] = region->initialScalar;
        }
      } // Over cells
    } // regions

    //__________________________________
    //  Smooth out initial distribution with some diffusion
    // WARNING::This may have problems on multiple patches
    //          look at the initial distribution where the patches meet
    CCVariable<double> FakeDiffusivity;
    new_dw->allocateTemporary(FakeDiffusivity, patch, Ghost::AroundCells, 1);
    FakeDiffusivity.initialize(1.0);    //  HARDWIRED
    double fakedelT = 1.0;

    for( int i =1 ; i < d_scalar->initialize_diffusion_knob; i++ ){
      bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
      CCVariable<double> placeHolder;
      scalarDiffusionOperator(new_dw, patch, use_vol_frac, f, placeHolder,
                              f, FakeDiffusivity, fakedelT);
    }

    setBC( f, "scalar-f", patch, m_materialManager, indx, new_dw, isNotInitialTimeStep );

    //__________________________________
    // compute thermo-transport-physical quantities
    // This MUST be identical to what's in the task modifyThermoTransport....
    for(CellIterator iter = patch->getExtraCellIterator();!iter.done(); iter++){
      IntVector c = *iter;
      double oneMinus_f = 1.0 - f[c];
      cv[c]          = f[c]*d_cv_B          + oneMinus_f*d_cv_A;
      viscosity[c]   = f[c]*d_viscosity_B   + oneMinus_f*d_viscosity_A;
      thermalCond[c] = f[c]*d_thermalCond_B + oneMinus_f*d_thermalCond_A;
      double R_mix   = f[c]*d_R_B           + oneMinus_f*d_R_A;
      gamma[c]       = R_mix/cv[c]  + 1.0;

      rho_CC[c]      = d_rho_B * d_rho_A /
                      ( f[c] * d_rho_A + oneMinus_f * d_rho_B);
      rho_micro[c]   = rho_CC[c];
      sp_vol[c]      = 1.0/rho_CC[c];
    } // Over cells
  }  // patches
}

/* _____________________________________________________________________
  Task:     modifyThermoTransportProperties--
  Purpose:  Simple method for changing thermoTransporProperties
 _____________________________________________________________________  */
void BinaryProperties::scheduleModifyThermoTransportProperties(SchedulerP  & sched,
                                                               const LevelP     & level,
                                                               const MaterialSet* /*ice_matls*/)
            {
  printSchedule( level, m_cout_tasks, " BinaryProperties::scheduleModifyThermoTransportProperties" );

  Task* t = scinew Task("BinaryProperties::modifyThermoTransportProperties",
                   this,&BinaryProperties::modifyThermoTransportProperties);

  t->requires( Task::OldDW, d_scalar->scalar_CCLabel, Ghost::None,0 );
  t->modifies( Ilb->specific_heatLabel );
  t->modifies( Ilb->gammaLabel );
  t->modifies( Ilb->thermalCondLabel );
  t->modifies( Ilb->viscosityLabel );
  t->computes( d_scalar->diffusionCoefLabel );
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
/* _____________________________________________________________________
// Purpose:  Compute the thermo and transport properties.  This gets
//           called at the top of the timestep.
 _____________________________________________________________________  */
void BinaryProperties::modifyThermoTransportProperties(const ProcessorGroup  *,
                                                       const PatchSubset     * patches,
                                                       const MaterialSubset  *,
                                                       DataWarehouse         * old_dw,
                                                       DataWarehouse         * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_cout_tasks, "BinaryProperties::modifyThermoTransportProperties" );

    int indx = d_matl->getDWIndex();
    CCVariable<double> diffusionCoeff;
    CCVariable<double> gamma;
    CCVariable<double> cv;
    CCVariable<double> thermalCond;
    CCVariable<double> viscosity;
    constCCVariable<double> f_old;

    new_dw->allocateAndPut(diffusionCoeff,
                           d_scalar->diffusionCoefLabel,indx, patch);

    new_dw->getModifiable(gamma,       Ilb->gammaLabel,        indx,patch);
    new_dw->getModifiable(cv,          Ilb->specific_heatLabel,indx,patch);
    new_dw->getModifiable(thermalCond, Ilb->thermalCondLabel,  indx,patch);
    new_dw->getModifiable(viscosity,   Ilb->viscosityLabel,    indx,patch);

    old_dw->get(f_old,  d_scalar->scalar_CCLabel,  indx, patch, Ghost::None,0);

    //__________________________________
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      double  f = f_old[c];
      double oneMinus_f = 1.0 - f;
      cv[c]          = f * d_cv_B          + oneMinus_f*d_cv_A;
      viscosity[c]   = f * d_viscosity_B   + oneMinus_f*d_viscosity_A;
      thermalCond[c] = f * d_thermalCond_B + oneMinus_f*d_thermalCond_A;
      double R_mix   = f * d_R_B           + oneMinus_f*d_R_A;
      gamma[c]       = R_mix/cv[c]  + 1.0;
    }
    diffusionCoeff.initialize(d_scalar->diff_coeff);
  }
}

//______________________________________________________________________
// Purpose:  Compute the specific heat at time.  This gets called immediately
//           after (f) is advected
//  TO DO:  FIGURE OUT A WAY TO ONLY COMPUTE CV ONCE
void BinaryProperties::computeSpecificHeat(CCVariable<double>& cv_new,
                                           const Patch    * patch,
                                           DataWarehouse  * new_dw,
                                           const int        indx)
{
  printTask(patch, m_cout_tasks, "BinaryProperties::computeSpecificHeat" );

  int test_indx = d_matl->getDWIndex();
  //__________________________________
  //  Compute cv for only one matl.
  if (test_indx == indx) {
    constCCVariable<double> f;
    new_dw->get(f,  d_scalar->scalar_CCLabel,  indx, patch, Ghost::None,0);

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      cv_new[c]  = f[c] * d_cv_B + (1.0 - f[c])*d_cv_A;
    }
  }
}

/* _____________________________________________________________________
  Task:     computeModelSources--
  Purpose:  Computes the src due to diffusion
 _____________________________________________________________________  */
void BinaryProperties::scheduleComputeModelSources( SchedulerP   & sched,
                                                    const LevelP & level)
{
  printSchedule( level, m_cout_tasks, " BinaryProperties::scheduleComputeModelSources" );

  Task* t = scinew Task("BinaryProperties::computeModelSources",
                   this,&BinaryProperties::computeModelSources);

  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires(Task::OldDW, Ilb->delTLabel, level.get_rep());
  t->requires(Task::NewDW, d_scalar->diffusionCoefLabel, gac,1);
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel,     gac,1);

  t->modifies(d_scalar->source_CCLabel);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//
void BinaryProperties::computeModelSources(const ProcessorGroup  *,
                                           const PatchSubset     * patches,
                                           const MaterialSubset  * /*matls*/,
                                           DataWarehouse         * old_dw,
                                           DataWarehouse         * new_dw)
{
  const Level* level = getLevel(patches);

  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel, level);

  Ghost::GhostType gac = Ghost::AroundCells;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch, m_cout_tasks, "BinaryProperties::computeModelSources" );

    constCCVariable<double> f_old;
    constCCVariable<double> diff_coeff;
    CCVariable<double> f_src;

    int indx = d_matl->getDWIndex();
    old_dw->get( f_old,      d_scalar->scalar_CCLabel,      indx, patch, gac,1);
    new_dw->get( diff_coeff, d_scalar->diffusionCoefLabel,  indx, patch, gac,1);

    new_dw->getModifiable(f_src, d_scalar->source_CCLabel,indx, patch);

    //__________________________________
    //  Tack on diffusion
    double diff_coeff_test = d_scalar->diff_coeff;
    if( diff_coeff_test != 0.0 ){

      bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
      CCVariable<double> placeHolder;

      scalarDiffusionOperator(new_dw, patch, use_vol_frac, f_old,
                              placeHolder,f_src, diff_coeff, delT);
    }
  }
}
/* _____________________________________________________________________
  Task:  testConservation
  Purpose:  compute sum_scalar_f
 _____________________________________________________________________  */
void BinaryProperties::scheduleTestConservation(SchedulerP    & sched,
                                                const PatchSet* patches)
{
  if(d_test_conservation){
    printSchedule( patches, m_cout_tasks, " BinaryProperties::scheduleTestConservation" );

    Task* t = scinew Task("BinaryProperties::testConservation",
                     this,&BinaryProperties::testConservation);

    Ghost::GhostType  gn = Ghost::None;
    t->requires(Task::OldDW, Ilb->delTLabel, getLevel(patches) );
    t->requires(Task::NewDW, d_scalar->scalar_CCLabel, gn,0);
    t->requires(Task::NewDW, Ilb->rho_CCLabel,          gn,0);
    t->requires(Task::NewDW, Ilb->uvel_FCMELabel,       gn,0);
    t->requires(Task::NewDW, Ilb->vvel_FCMELabel,       gn,0);
    t->requires(Task::NewDW, Ilb->wvel_FCMELabel,       gn,0);
    t->computes(Slb->sum_scalar_fLabel);

    sched->addTask(t, patches, d_matl_set);
  }
}

//______________________________________________________________________
void BinaryProperties::testConservation(const ProcessorGroup*,
                                        const PatchSubset   * patches,
                                        const MaterialSubset* /*matls*/,
                                        DataWarehouse       * old_dw,
                                        DataWarehouse       * new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel, level);
  Ghost::GhostType gn = Ghost::None;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch, m_cout_tasks, "BinaryProperties::testConservation" );

    //__________________________________
    //  conservation of f test
    constCCVariable<double> rho_CC, f;
    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;

    int indx = d_matl->getDWIndex();
    new_dw->get( f,       d_scalar->scalar_CCLabel, indx,patch,gn,0 );
    new_dw->get( rho_CC,  Ilb->rho_CCLabel,         indx,patch,gn,0 );
    new_dw->get( uvel_FC, Ilb->uvel_FCMELabel,      indx,patch,gn,0 );
    new_dw->get( vvel_FC, Ilb->vvel_FCMELabel,      indx,patch,gn,0 );
    new_dw->get( wvel_FC, Ilb->wvel_FCMELabel,      indx,patch,gn,0 );
    Vector dx = patch->dCell();
    double cellVol = dx.x()*dx.y()*dx.z();

    CCVariable<double> q_CC;
    new_dw->allocateTemporary(q_CC, patch);

    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      q_CC[c] = rho_CC[c] * cellVol * f[c];
    }

    double sum_mass_f;
    conservationTest<double>(patch, delT, q_CC, uvel_FC, vvel_FC, wvel_FC,
                             sum_mass_f);

    new_dw->put(sum_vartype(sum_mass_f), Slb->sum_scalar_fLabel);
  }
}


//__________________________________
void BinaryProperties::scheduleComputeStableTimeStep(SchedulerP&,
                                              const LevelP&)
{
  // None necessary...
}
//______________________________________________________________________
//
void BinaryProperties::scheduleErrorEstimate(const LevelP&,
                                      SchedulerP&)
{
  // Not implemented yet
}
