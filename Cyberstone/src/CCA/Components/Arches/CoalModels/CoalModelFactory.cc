#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h> 
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <sstream>
#include <iostream>

//===========================================================================

using namespace std;
using namespace Uintah;

CoalModelFactory::CoalModelFactory()
{
  b_labelSet = false;
}

CoalModelFactory::~CoalModelFactory()
{

  // delete the builders
  for( BuildMap::iterator i = builders_.begin(); i != builders_.end(); ++i ){
    delete i->second;
  }

  // delete all constructed solvers
  for( ModelMap::iterator i = models_.begin(); i != models_.end(); ++i ){
    delete i->second;
  }

  for( DevolModelMap::iterator i = devolmodels_.begin(); i != devolmodels_.end(); ++i ){
    delete i->second;
  }

  for( CharOxiModelMap::iterator i = charoximodels_.begin(); i != charoximodels_.end(); ++i ){
    delete i->second;
  }

  for ( ModelMap::iterator i = birth_models_.begin(); i != birth_models_.end(); ++i ) {
    delete i->second; 
  }

  VarLabel::destroy(d_Length_ICLabel); 
  VarLabel::destroy(d_Length_GasLabel); 

  VarLabel::destroy(d_RawCoal_ICLabel); 
  VarLabel::destroy(d_RawCoal_GasLabel); 

  if( b_useParticleTemperature ) {
    VarLabel::destroy(d_ParticleTemperature_ICLabel); 
    VarLabel::destroy(d_ParticleTemperature_GasLabel); 
  } else {
    VarLabel::destroy(d_ParticleEnthalpy_ICLabel); 
    VarLabel::destroy(d_ParticleEnthalpy_GasLabel); 
  }
  
  if( b_useMoisture ) {
    VarLabel::destroy(d_Moisture_ICLabel);
    VarLabel::destroy(d_Moisture_GasLabel);
  }
}

//---------------------------------------------------------------------------
// Method: Return a reference to itself. 
//---------------------------------------------------------------------------
CoalModelFactory&
CoalModelFactory::self()
{
  static CoalModelFactory s;
  return s;
}

//---------------------------------------------------------------------------
// Method: Problem setup
//---------------------------------------------------------------------------
void CoalModelFactory::problemSetup(const ProblemSpecP& params)
{

  ProblemSpecP db = params; // Should be the <DQMOM> block
  ProblemSpecP params_root = db->getRootNode(); 

  ProblemSpecP dqmom_db = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM");
  std::string which_dqmom; 
  dqmom_db->getAttribute( "type", which_dqmom ); 
  if ( which_dqmom == "unweightedAbs" ){
    d_unweighted = true; 
   } else { 
    d_unweighted = false; 
   }

}

//---------------------------------------------------------------------------
// Method: Register a model  
//---------------------------------------------------------------------------
void
CoalModelFactory::register_model( const std::string name,
                                  ModelBuilder* builder )
{
  ASSERT( builder != nullptr );

  BuildMap::iterator iBuilder = builders_.find( name );
  if( iBuilder == builders_.end() ){
    iBuilder = builders_.insert( std::make_pair(name,builder) ).first;
  } else {
    string err_msg = "ERROR: Arches: CoalModelFactory: A duplicate ModelBuilder object named "+name+" was already loaded. This is forbidden.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }
  
  ModelBase* model = builder->build();

  model->setUnweightedAbscissas(d_unweighted);

  string modelType = model->getType();
  if( modelType == "Devolatilization" ) {

    DevolModelMap::iterator i = devolmodels_.find(name); 
    if ( i == devolmodels_.end() ){ 
      Devolatilization* devolmodel = dynamic_cast<Devolatilization*>(model);
      devolmodels_[name] = devolmodel;
    } else { 
      throw InvalidValue("Error: Devol model already loaded: "+name, __FILE__, __LINE__); 
    }

  } else if( modelType == "CharOxidation" ) {

    CharOxiModelMap::iterator i = charoximodels_.find(name); 

    if ( i == charoximodels_.end() ){ 
      CharOxidation* charoximodel = dynamic_cast<CharOxidation*>(model);
      charoximodels_[name] = charoximodel;
    } else { 
      throw InvalidValue("Error: Char model already loaded: "+name, __FILE__, __LINE__); 
    }

  } else { 

    if ( modelType == "birth" ){ 
      ModelMap::iterator i = birth_models_.find(name); 
      if ( i == birth_models_.end() ){ 
        birth_models_[name] = model; 
      } else { 
        throw InvalidValue("Error: Coal model already loaded: "+name, __FILE__, __LINE__); 
      }
    } else { 
      ModelMap::iterator i = models_.find(name); 
      if ( i == models_.end() ){
        models_[name] = model;
      } else { 
        throw InvalidValue("Error: Coal model already loaded: "+name, __FILE__, __LINE__); 
      }
    }

  }
}

//---------------------------------------------------------------------------
// Method: Retrieve a model from the map. 
//---------------------------------------------------------------------------
ModelBase&
CoalModelFactory::retrieve_model( const std::string name )
{
  const ModelMap::iterator imodel= models_.find( name );
  if( imodel != models_.end() ) {
    return *(imodel->second);
  } 

  const DevolModelMap::iterator idevol = devolmodels_.find( name ); 
  if ( idevol != devolmodels_.end() ){ 
    return *(idevol->second); 
  }

  const CharOxiModelMap::iterator ichar = charoximodels_.find( name ); 
  if ( ichar != charoximodels_.end() ){ 
    return *(ichar->second); 
  }

  const ModelMap::iterator ibirth=birth_models_.find(name); 
  if ( ibirth != birth_models_.end() ){ 
    return *(ibirth->second); 
  }

  string err_msg = "Error: In CoalModelFactory: No model registered for " + name + "\n";
  throw InvalidValue(err_msg,__FILE__,__LINE__);

}

//---------------------------------------------------------------------------
// Method: Schedule calculation of all models
//---------------------------------------------------------------------------
void
CoalModelFactory::sched_coalParticleCalculation( const LevelP& level, 
                                                 SchedulerP& sched, 
                                                 int timeSubStep )
{

  //Eval all birth models first
  for ( ModelMap::iterator iModel = birth_models_.begin(); iModel != birth_models_.end(); ++iModel ){ 
    iModel->second->sched_computeModel( level, sched, timeSubStep );
  }

  //Devol models second
  for( DevolModelMap::iterator iModel = devolmodels_.begin(); iModel != devolmodels_.end(); ++iModel ) {
    iModel->second->sched_computeModel( level, sched, timeSubStep );
  }

  //Char models
  for( CharOxiModelMap::iterator iModel = charoximodels_.begin(); iModel != charoximodels_.end(); ++iModel ) {
    iModel->second->sched_computeModel( level, sched, timeSubStep );
  }

  //everything else
  for( ModelMap::iterator iModel = models_.begin(); iModel != models_.end(); ++iModel ) {
    iModel->second->sched_computeModel( level, sched, timeSubStep );
  }

}

void 
CoalModelFactory::sched_init_all_models(const LevelP&  level, SchedulerP& sched ){ 

  //birth models 
  for ( ModelMap::iterator iModel = birth_models_.begin(); iModel != birth_models_.end(); iModel++){
    iModel->second->sched_initVars(level, sched); 
  }
  //Devol models
  for ( DevolModelMap::iterator iModel = devolmodels_.begin(); iModel != devolmodels_.end(); iModel++){
    iModel->second->sched_initVars(level, sched); 
  }
  //Char models
  for ( CharOxiModelMap::iterator iModel = charoximodels_.begin(); iModel != charoximodels_.end(); iModel++){
    iModel->second->sched_initVars(level, sched); 
  }
  //everything else
  for ( ModelMap::iterator iModel = models_.begin(); iModel != models_.end(); iModel++){
    iModel->second->sched_initVars(level, sched); 
  }
} 
