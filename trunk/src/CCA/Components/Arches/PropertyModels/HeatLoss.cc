#include <CCA/Components/Arches/PropertyModels/HeatLoss.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/Properties.h>

using namespace Uintah;
using namespace std;

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
HeatLoss::HeatLoss( std::string prop_name, MaterialManagerP& materialManager ) : PropertyModelBase( prop_name, materialManager )
{
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() );

  // Evaluated before or after table lookup:
  _before_table_lookup = true;

  _constant_heat_loss = false;

  _boundary_condition = scinew BoundaryCondition_new( materialManager->getMaterial( "Arches", 0)->getDWIndex() );

  _low_hl  = -1;
  _high_hl =  1;

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
HeatLoss::~HeatLoss( )
{
  delete _boundary_condition;

  if ( _constant_heat_loss ){
    VarLabel::destroy( _actual_hl_label );
  }

}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void
HeatLoss::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb;

  _boundary_condition->problemSetup( db, _prop_name );

  if ( db->findBlock("constant_heat_loss") ){

    _constant_heat_loss = true;

    std::string name = _prop_name + "_actual";
    _actual_hl_label = VarLabel::create( name, CCVariable<double>::getTypeDescription() );

  }

  db->getWithDefault( "sensible_enthalpy_label"  , _sen_h_label_name   , "sensibleenthalpy" );
  db->getWithDefault( "adiabatic_enthalpy_label"  , _adiab_h_label_name   , "adiabaticenthalpy" );

  db->require( "enthalpy_label", _enthalpy_label_name );

  db->getWithDefault( "use_Ha_lookup", _use_h_ad_lookup, false);

  _noisy_heat_loss = false;
  if ( db->findBlock( "noisy_hl_warning" ) ){
    _noisy_heat_loss = true;
  }

  _prop_type = "heat_loss";

  commonProblemSetup( inputdb );
}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void HeatLoss::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{

  std::string taskname = "HeatLoss::computeProp";
  Task* tsk = scinew Task( taskname, this, &HeatLoss::computeProp, time_substep );

  _enthalpy_label = 0;

  _enthalpy_label = VarLabel::find( _enthalpy_label_name );
  _vol_frac_label= VarLabel::find( "volFraction" );

  if ( _enthalpy_label == 0 ){
    throw InvalidValue( "Error: Could not find enthalpy label with name: "+_enthalpy_label_name, __FILE__, __LINE__);
  }
  if ( _vol_frac_label == 0 ){
    throw InvalidValue("Error: Cannot match volume fraction name with label.",__FILE__, __LINE__);
  }

  //mixture fractions:
  vector<string> ivs;
  ivs = _rxn_model->getAllIndepVars();

  for ( vector<string>::iterator iter = ivs.begin(); iter != ivs.end(); iter++ ){

    if ( *iter != _prop_name ){
      const VarLabel* label = VarLabel::find( *iter );

      if ( label == 0 ){
        throw InvalidValue( "Error: Could not find table IV label with name: "+*iter, __FILE__, __LINE__);
      }
      tsk->requires( Task::NewDW , label , Ghost::None , 0 );
    }
  }

  tsk->modifies( _prop_label );
  if ( time_substep == 0 ){

    if ( _constant_heat_loss ){
      tsk->computes( _actual_hl_label );
    }

    tsk->requires( Task::NewDW , _enthalpy_label , Ghost::None , 0 );
    tsk->requires( Task::OldDW,  _vol_frac_label , Ghost::None , 0 );

  } else {


    if ( _constant_heat_loss ){
      tsk->modifies( _actual_hl_label );
    }

    tsk->requires( Task::NewDW , _enthalpy_label , Ghost::None , 0 );
    tsk->requires( Task::NewDW,  _vol_frac_label , Ghost::None , 0 );

  }

  //inerts
  _inert_map = _rxn_model->getInertMap();
  for ( MixingRxnModel::InertMasterMap::iterator iter = _inert_map.begin(); iter != _inert_map.end(); iter++ ){
    const VarLabel* label = VarLabel::find( iter->first );
    tsk->requires( Task::NewDW, label, Ghost::None, 0 );
    _use_h_ad_lookup = true;
  }

  sched->addTask( tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ) );

}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void HeatLoss::computeProp(const ProcessorGroup* pc,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           int time_substep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CellIterator iter = patch->getCellIterator();

    bool oob_up = false;
    bool oob_dn = false;

    CCVariable<double> prop;

    constCCVariable<double> h;       //enthalpy
    constCCVariable<double> eps;     //volume fraction

    DataWarehouse* which_dw;
    new_dw->getModifiable( prop, _prop_label, matlIndex, patch );
    if ( time_substep == 0 ){
      which_dw = new_dw;
      prop.initialize(0.0);
      old_dw->get( eps, _vol_frac_label, matlIndex, patch, Ghost::None, 0 );
    } else {
      which_dw = new_dw;
      new_dw->get( eps, _vol_frac_label, matlIndex, patch, Ghost::None, 0 );
    }

    MixingRxnModel::StringToCCVar inerts;
    inerts.clear();
    for ( MixingRxnModel::InertMasterMap::iterator iter = _inert_map.begin(); iter != _inert_map.end(); iter++ ){

      constCCVariable<double> the_inert;
      const VarLabel* the_label = VarLabel::find( iter->first );
      which_dw->get( the_inert, the_label, matlIndex, patch, Ghost::None, 0 );

      MixingRxnModel::ConstVarContainer v;
      v.var = the_inert;
      inerts.insert( make_pair( iter->first, v) );

    }

    which_dw->get( h, _enthalpy_label , matlIndex , patch , Ghost::None , 0 );

    vector<string> ivs;
    ivs = _rxn_model->getAllIndepVars();
    std::map<std::string, constCCVariable<double> > mix_fracs;

    int index = 0;
    for ( vector<string>::iterator iter = ivs.begin(); iter != ivs.end(); iter++ ){

      if ( *iter != _prop_name ){
        const VarLabel* label = VarLabel::find( *iter );

        constCCVariable<double> f;
        new_dw->get( f, label, matlIndex, patch, Ghost::None, 0);
        mix_fracs.insert(std::make_pair( *iter, f ) );
        index++;
      }
    }

    for (iter.begin(); !iter.done(); iter++){

      IntVector c = *iter;

      if ( eps[c] > 0.0 ){

        vector<double> iv_values;
        for ( unsigned int ii = 0; ii < ivs.size(); ii++ ){

          if ( ivs[ii] != _prop_name ){

            std::map<std::string, constCCVariable<double> >::iterator iMF = mix_fracs.find(ivs[ii]);
            iv_values.push_back( iMF->second[c] );

          } else {

            iv_values.push_back(0.0);

          }
        }

        double small = 1e-16;
        double hl = 0.0;
        double h_sens = _rxn_model->getTableValue( iv_values, _sen_h_label_name, inerts, c );
        double h_ad_lookup = 0.0;

        if ( _use_h_ad_lookup ){

          h_ad_lookup = _rxn_model->getTableValue( iv_values, _adiab_h_label_name, inerts, c );
          hl = h_ad_lookup - h[c];

        } else {

          double h_adiab = _rxn_model->get_Ha( iv_values, 0.0 );
          hl = h_adiab - h[c];

        }

        hl /= ( h_sens + small );

        if ( hl < _low_hl ){
          hl     = _low_hl;
          oob_dn = true;
        }
        if ( hl > _high_hl ){
          hl     = _high_hl;
          oob_up = true;
        }

        prop[c] = hl;

      } else {

        prop[c] = 0.0;

      }

    }

    //Apply boundary conditions
    _boundary_condition->setScalarValueBC( 0, patch, prop, _prop_name );

    if ( _noisy_heat_loss ) {

      if ( oob_up || oob_dn ) {
        std::cout << "Patch with bounds: " << patch->getCellLowIndex() << " to " << patch->getCellHighIndex()  << std::endl;
        if ( oob_dn ){
          std::cout << "   --> lower heat loss exceeded. " << std::endl;
        }
        if ( oob_up ){
          std::cout << "   --> upper heat loss exceeded. " << std::endl;
        }
      }
    }

    if ( _constant_heat_loss ){
      //assuming this will be used for debugging and not production cases.
      CCVariable<double> actual_heat_loss;

      if ( time_substep == 0 ){

        new_dw->allocateAndPut( actual_heat_loss, _actual_hl_label, matlIndex, patch );

      } else {

        new_dw->getModifiable( actual_heat_loss, _actual_hl_label, matlIndex, patch );

      }

      actual_heat_loss.copyData( prop );
      prop.initialize( _const_init );

    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void HeatLoss::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "HeatLoss::initialize";

  Task* tsk = scinew Task(taskname, this, &HeatLoss::initialize);
  tsk->computes(_prop_label);

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void HeatLoss::initialize( const ProcessorGroup* pc,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> prop;

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch );

    PropertyModelBase::base_initialize( patch, prop ); // generic initialization functionality

    vector<Patch::FaceType> bf;
    vector<Patch::FaceType>::const_iterator bf_iter;
    patch->getBoundaryFaces(bf);

    //check the BCs + initialize BC
    _boundary_condition->checkBCs( patch, _prop_name, matlIndex );

    //Apply boundary conditions
    _boundary_condition->setScalarValueBC( 0, patch, prop, _prop_name );

  }
}

void HeatLoss::sched_restartInitialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "HeatLoss::restartInitialize";
  Task* tsk = scinew Task(taskname, this, &HeatLoss::restartInitialize);
  tsk->computes(_prop_label);

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));
}
void HeatLoss::restartInitialize( const ProcessorGroup * pc,
                                  const PatchSubset    * patches,
                                  const MaterialSubset * matls,
                                  DataWarehouse        * old_dw,
                                  DataWarehouse        * new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> prop;
    //new_dw->getModifiable( prop, _prop_label, matlIndex, patch );
    new_dw->allocateAndPut(prop, _prop_label, matlIndex, patch );

    //check the BCs + initialize BC
    _boundary_condition->checkBCs( patch, _prop_name, matlIndex );

    //Apply boundary conditions
    _boundary_condition->setScalarValueBC( 0, patch, prop, _prop_name );

  }
}
