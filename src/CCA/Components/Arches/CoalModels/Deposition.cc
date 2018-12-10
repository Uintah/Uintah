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


#include <CCA/Components/Arches/CoalModels/Deposition.h>

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>

#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>

//===========================================================================

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:

DepositionBuilder::DepositionBuilder( const std::string         & modelName,
                                      const vector<std::string> & reqICLabelNames,
                                      const vector<std::string> & reqScalarLabelNames,
                                            ArchesLabel         * fieldLabels,
                                            MaterialManagerP    & materialManager,
                                            int                   qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, materialManager, qn )
{}

DepositionBuilder::~DepositionBuilder(){}

ModelBase*
DepositionBuilder::build(){
  return scinew Deposition( d_modelName, d_materialManager, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}

// End Builder
//---------------------------------------------------------------------------

Deposition::Deposition( std::string           modelName,
                        MaterialManagerP    & materialManager,
                        ArchesLabel         * fieldLabels,
                        vector<std::string>   icLabelNames,
                        vector<std::string>   scalarLabelNames,
                        int                   qn )
: ModelBase(modelName, materialManager, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );

  _is_weight = false;

}

Deposition::~Deposition()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------

void
Deposition::problemSetup(const ProblemSpecP& inputdb, int qn)
{

  ProblemSpecP db = inputdb;

  if ( db->findBlock("is_weight")){
    _is_weight = true;
  }

  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  if ( !_is_weight ){

    if ( db->findBlock("abscissa") ){
      db->findBlock("abscissa")->getAttribute( "label", _abscissa_name );
    } else {
      throw ProblemSetupException("Error: Must specify an abscissa label for this model.",__FILE__,__LINE__);
    }

    std::string a_name = ArchesCore::append_qn_env( _abscissa_name, d_quadNode );
    EqnBase& a_eqn = dqmomFactory.retrieve_scalar_eqn( a_name );
    _a_scale = a_eqn.getScalingConstant(d_quadNode);

  }

  std::string w_name = ArchesCore::append_qn_env( "w", d_quadNode );
  EqnBase& temp_eqn = dqmomFactory.retrieve_scalar_eqn(w_name);
  _w_scale = temp_eqn.getScalingConstant(d_quadNode);

  _w_label = VarLabel::find(w_name);

  if ( _w_label == 0 ){
    throw InvalidValue("Error:Weight not found: "+w_name, __FILE__, __LINE__);
  }

  // check to see if rate_deposition model is active
  bool missing_rate_depostion = true;
  const ProblemSpecP params_root = db->getRootNode();
  ProblemSpecP db_PM = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleModels");
  for( ProblemSpecP db_var = db_PM->findBlock("model"); db_var != nullptr; db_var = db_var->findNextBlock("model") ) {

    std::string type;
    std::string role_found;
    db_var->getAttribute("type", type);
    if ( type == "rate_deposition" ){
      missing_rate_depostion = false;
    }
  }
  if ( missing_rate_depostion ){
    throw InvalidValue("Error: Deposition model requires a rate_deposition model in ParticleModels.", __FILE__, __LINE__);
  }
  // create rate_deposition model base names
  std::string rate_dep_base_nameX = "RateDepositionX";
  std::string rate_dep_base_nameY = "RateDepositionY";
  std::string rate_dep_base_nameZ = "RateDepositionZ";
  std::string rate_dep_X = ArchesCore::append_env( rate_dep_base_nameX, d_quadNode );
  std::string rate_dep_Y = ArchesCore::append_env( rate_dep_base_nameY, d_quadNode );
  std::string rate_dep_Z = ArchesCore::append_env( rate_dep_base_nameZ, d_quadNode );
  _rate_depX_varlabel = VarLabel::find(rate_dep_X);
  _rate_depY_varlabel = VarLabel::find(rate_dep_Y);
  _rate_depZ_varlabel = VarLabel::find(rate_dep_Z);

  // Need a size IC:
  std::string length_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
  std::string length_name = ArchesCore::append_env( length_root, d_quadNode );
  _length_varlabel = VarLabel::find(length_name);

  // Need a density
  std::string density_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DENSITY);
  std::string density_name = ArchesCore::append_env( density_root, d_quadNode );
  _particle_density_varlabel = VarLabel::find(density_name);

  _pi = acos(-1.0);

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
Deposition::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "Deposition::initVars";
  Task* tsk = scinew Task(taskname, this, &Deposition::initVars);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
Deposition::initVars( const ProcessorGroup * pc,
                            const PatchSubset    * patches,
                            const MaterialSubset * matls,
                            DataWarehouse        * old_dw,
                            DataWarehouse        * new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> model;
    CCVariable<double> gas_source;

    new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
    model.initialize(0.0);
    new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
    gas_source.initialize(0.0);
  }
}



//---------------------------------------------------------------------------
// Method: Schedule the calculation of the model
//---------------------------------------------------------------------------
void
Deposition::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "Deposition::computeModel";
  Task* tsk = scinew Task(taskname, this, &Deposition::computeModel, timeSubStep );

  Ghost::GhostType gn = Ghost::None;
  Ghost::GhostType  gaf = Ghost::AroundFaces;

  if ( !_is_weight ){
    std::string abscissa_name = ArchesCore::append_env( _abscissa_name, d_quadNode );
    _abscissa_label = VarLabel::find(abscissa_name);
    if ( _abscissa_label == 0 )
      throw InvalidValue("Error: Abscissa not found: "+abscissa_name, __FILE__, __LINE__);
  }

  if (timeSubStep == 0) {
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    tsk->requires(Task::OldDW, _w_label, gn, 0);
    tsk->requires(Task::OldDW, _rate_depX_varlabel, gaf, 1);
    tsk->requires(Task::OldDW, _rate_depY_varlabel, gaf, 1);
    tsk->requires(Task::OldDW, _rate_depZ_varlabel, gaf, 1);
    tsk->requires(Task::OldDW, _length_varlabel, gn, 0 );
    tsk->requires(Task::OldDW, _particle_density_varlabel, gn, 0 );
    if ( !_is_weight )
      tsk->requires(Task::OldDW, _abscissa_label, gn, 0);
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);
    tsk->requires(Task::NewDW, _w_label, gn, 0);
    tsk->requires(Task::NewDW, _rate_depX_varlabel, gaf, 1);
    tsk->requires(Task::NewDW, _rate_depY_varlabel, gaf, 1);
    tsk->requires(Task::NewDW, _rate_depZ_varlabel, gaf, 1);
    tsk->requires(Task::NewDW, _length_varlabel, gn, 0 );
    tsk->requires(Task::NewDW, _particle_density_varlabel, gn, 0 );
    if ( !_is_weight )
      tsk->requires(Task::NewDW, _abscissa_label, gn, 0);
  }

  tsk->requires(Task::OldDW, VarLabel::find("volFraction"), gn, 0 );

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
Deposition::computeModel( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   const int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gaf = Ghost::AroundFaces;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    Vector DX = patch->dCell();
    double vol = DX.x()*DX.y()*DX.z();
    double area_x = DX.y()*DX.z();
    double area_y = DX.x()*DX.z();
    double area_z = DX.x()*DX.y();

    CCVariable<double> model;
    CCVariable<double> gas_source;
    DataWarehouse* which_dw;

    if ( timeSubStep == 0 ){
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
      gas_source.initialize(0.0);
      model.initialize(0.0);
      which_dw = old_dw;
    } else {
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch );
      new_dw->getModifiable( gas_source, d_gasLabel, matlIndex, patch );
      which_dw = new_dw;
    }

    constCCVariable<double> w;
    constCCVariable<double> a;
    constCCVariable<double> diam;
    constCCVariable<double> rhop;
    constCCVariable<double> vol_fraction;
    constSFCXVariable<double> rate_X;
    constSFCYVariable<double> rate_Y;
    constSFCZVariable<double> rate_Z;

    which_dw->get( w, _w_label, matlIndex, patch, gn, 0 );
    which_dw->get( diam, _length_varlabel, matlIndex, patch, gn, 0 );
    which_dw->get( rhop, _particle_density_varlabel, matlIndex, patch, gn, 0 );
    which_dw->get( rate_X, _rate_depX_varlabel, matlIndex, patch, gaf, 1 );
    which_dw->get( rate_Y, _rate_depY_varlabel, matlIndex, patch, gaf, 1 );
    which_dw->get( rate_Z, _rate_depZ_varlabel, matlIndex, patch, gaf, 1 );
    old_dw->get( vol_fraction, VarLabel::find("volFraction"), matlIndex, patch, gn, 0 );
    if ( !_is_weight ){
      which_dw->get( a, _abscissa_label, matlIndex, patch, gn, 0 );
    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      IntVector cpx = c + IntVector(1,0,0);
      IntVector cpy = c + IntVector(0,1,0);
      IntVector cpz = c + IntVector(0,0,1);
      if ( vol_fraction[c] > 0. ){
        // add all deposit flux per area
        double mass_out = 0.0;
        mass_out += abs(rate_X[c])*area_x;
        mass_out += abs(rate_Y[c])*area_y;
        mass_out += abs(rate_Z[c])*area_z;
        mass_out += abs(rate_X[cpx])*area_x;
        mass_out += abs(rate_Y[cpy])*area_y;
        mass_out += abs(rate_Z[cpz])*area_z;
        double vol_p = (_pi/6.0)*pow(diam[c],3.0);
        if ( _is_weight ){
          model[c] = - mass_out / (rhop[c] * vol_p * vol * _w_scale); // scaled #/(m^3s)
        } else {
          model[c] = - mass_out / (rhop[c] * vol_p * vol * _w_scale) * (a[c]/_a_scale); // scaled #/(m^3s)
        }

      } else {

        model[c] = 0.0;

      }

      gas_source[c] = 0.0;

    }
  }
}
