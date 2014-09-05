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

using namespace Uintah;

CoalModelFactory::CoalModelFactory()
{
  b_labelSet = false;
}

CoalModelFactory::~CoalModelFactory()
{
//  // delete the builders
  for( BuildMap::iterator i=builders_.begin(); i!=builders_.end(); ++i ){
      delete i->second;
    }

  // delete all constructed solvers
  for( ModelMap::iterator i=models_.begin(); i!=models_.end(); ++i ){
      delete i->second;
  }

//  for( DevolModelMap::iterator i=devolmodels_.begin(); i!=devolmodels_.end(); ++i ){
//      delete i->second;
//  }

//  for( CharOxiModelMap::iterator i=charoximodels_.begin(); i!=charoximodels_.end(); ++i ){
//      delete i->second;
//  }

//  for( HeatTransferModelMap::iterator i=heatmodels_.begin(); i!=heatmodels_.end(); ++i ){
//      delete i->second;
//  }

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
  if ( which_dqmom == "unweightedAbs" )
    d_unweighted = true; 
  else 
    d_unweighted = false; 

  // Grab coal properties from input file
  ProblemSpecP db_coalProperties = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("CoalProperties");
  if( db_coalProperties ) {
    db_coalProperties->require("C", yelem[0]);
    db_coalProperties->require("H", yelem[1]);
    db_coalProperties->require("N", yelem[2]);
    db_coalProperties->require("O", yelem[3]);
    db_coalProperties->require("S", yelem[4]);
  } else {
    // not a problem yet
    //string err_msg="Missing <Coal_Properties> section in input file!";
    //throw ProblemSetupException(err_msg,__FILE__,__LINE__);
  }

  ProblemSpecP db_coalParticleCalculation = db->findBlock("coalParticleCalculation");
  if( !db_coalParticleCalculation ) {
    b_coupled_physics = false;

  } else {
    
    // Coupled or separable physics calculations?
    string calculation_type;
    db_coalParticleCalculation->getAttribute("type",calculation_type);
  
    if( calculation_type == "separable" ) {
      b_coupled_physics = false;
      proc0cout << endl << "DQMOM coal particle calculation: using separable multiphysics calculation." << endl << endl;
    } else if( calculation_type == "coupled" ) {
      b_coupled_physics = true;
      proc0cout << endl << "DQMOM coal particle calculation: using coupled multiphysics calculation." << endl << endl;
    } else {
      string err_msg = "ERROR: Arches: CoalModelFactory: Unrecognized <coalParticleCalculation> type: " + calculation_type + ": should be 'coupled' or 'separable'.";
      throw ProblemSetupException(err_msg,__FILE__,__LINE__);
    }
  
    // Now grab specific internal coordinates (only if using coupled algorithm)
    b_useParticleTemperature = false;
    b_useParticleEnthalpy = false; 
    b_useMoisture = false;
    b_useAsh = false;
    if( b_coupled_physics ) {
  
      // Check for length internal coordinate (required)
      db_coalParticleCalculation->get("Length",s_LengthName);
      if( s_LengthName == "" ) {
        string err_msg = "ERROR: Arches: CoalModelFactory: You specified that you wanted to use the coupled multiphysics particle algorithm, but you didn't specify a length internal coordiante!\n";
        throw ProblemSetupException(err_msg,__FILE__,__LINE__);
      }
  
      // Check for raw coal internal coordiante (required)
      db_coalParticleCalculation->get("RawCoal",s_RawCoalName);
      if( s_RawCoalName == "" ) {
        string err_msg = "ERROR: Arches: CoalModelFactory: You specified that you wanted to use the coupled multiphysics particle algorithm, but you didn't specify a raw coal internal coordinate!\n";
        throw ProblemSetupException(err_msg,__FILE__,__LINE__);
      }
  
      // Check for char internal coordinate (required)
      db_coalParticleCalculation->get("Char",s_CharName);
      if( s_CharName == "" ) {
        string err_msg = "ERROR: Arches: CoalModelFactory: You specified that you wanted to use the coupled multiphysics particle algorithm, but you didn't specify a char internal coordinate!\n";
        throw ProblemSetupException(err_msg,__FILE__,__LINE__);
      }
  
      // Check for temperature or enthalpy internal coordinate (required)
      if( db_coalParticleCalculation->findBlock("ParticleTemperature") ) {
        b_useParticleTemperature = true;
      }
      if( db_coalParticleCalculation->findBlock("ParticleEnthalpy") ) {
        b_useParticleEnthalpy = true;
      }
      if( b_useParticleTemperature == b_useParticleEnthalpy ) {
        string err_msg = "ERROR: Arches: CoalModelFactory: You specified BOTH <ParticleEnthalpy> and <ParticleTemperature> in your <coalParticleCalculation> tags (or you didn't specify either). Please fix your input file.";
        throw ProblemSetupException(err_msg,__FILE__,__LINE__);
      }
      if( b_useParticleTemperature ) {
        db_coalParticleCalculation->get("ParticleTemperature",s_ParticleTemperatureName);
      } else if( b_useParticleEnthalpy ) {
        db_coalParticleCalculation->get("ParticleEnthalpy",s_ParticleEnthalpyName);
      }
  
      // Check for moisture internal coordinate (optional)
      if( db_coalParticleCalculation->findBlock("Moisture") ) {
        b_useMoisture = true;
        db_coalParticleCalculation->get("Moisture",s_MoistureName);
      }
  
      // Check for ash internal coordiante (optional)
      if( db_coalParticleCalculation->findBlock("Ash") ) {
        b_useAsh = true;
        db_coalParticleCalculation->get("Ash",s_AshName);
      }
  
      // Now create variable labels
      d_Length_ICLabel  = VarLabel::create( s_LengthName, CCVariable<double>::getTypeDescription() );
      d_Length_GasLabel = VarLabel::create( s_LengthName+"_gasSource", CCVariable<double>::getTypeDescription() );
  
      d_RawCoal_ICLabel  = VarLabel::create( s_RawCoalName, CCVariable<double>::getTypeDescription() );
      d_RawCoal_GasLabel = VarLabel::create( s_RawCoalName+"_gasSource", CCVariable<double>::getTypeDescription() );
  
      if( b_useParticleTemperature ) {
        d_ParticleTemperature_ICLabel  = VarLabel::create( s_ParticleTemperatureName, CCVariable<double>::getTypeDescription() ); 
        d_ParticleTemperature_GasLabel = VarLabel::create( s_ParticleTemperatureName+"_gasSource", CCVariable<double>::getTypeDescription() );
      } else {
        d_ParticleEnthalpy_ICLabel  = VarLabel::create( s_ParticleEnthalpyName, CCVariable<double>::getTypeDescription() );
        d_ParticleEnthalpy_GasLabel = VarLabel::create( s_ParticleEnthalpyName+"_gasSource", CCVariable<double>::getTypeDescription() );
      }
  
      if( b_useMoisture ) {
        d_Moisture_ICLabel  = VarLabel::create( s_MoistureName, CCVariable<double>::getTypeDescription() );
        d_Moisture_GasLabel = VarLabel::create( s_MoistureName+"_gasSource", CCVariable<double>::getTypeDescription() );
      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Register a model  
//---------------------------------------------------------------------------
void
CoalModelFactory::register_model( const std::string name,
                                  ModelBuilder* builder )
{
  ASSERT( builder != NULL );

  BuildMap::iterator iBuilder = builders_.find( name );
  if( iBuilder == builders_.end() ){
    iBuilder = builders_.insert( std::make_pair(name,builder) ).first;
  } else {
    string err_msg = "ERROR: Arches: CoalModelFactory: A duplicate ModelBuilder object named "+name+" was already loaded. This is forbidden.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }

  // why wait until retrieve_model to build the models?
  // (why have builders in the first place...?)
  const ModelMap::iterator iModel = models_.find( name );
  if( iModel != models_.end() ) {
    string err_msg = "ERROR: Arches: CoalModelFactory: A duplicate ModelBase object named "+name+" was already built. This is forbidden.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }
  
  ModelBase* model = builder->build();
  models_[name] = model;

  model->setUnweightedAbscissas(d_unweighted);

  string modelType = model->getType();
  if( modelType == "Devolatilization" ) {
    Devolatilization* devolmodel = dynamic_cast<Devolatilization*>(model);
    devolmodels_[name] = devolmodel;
  } else if( modelType == "CharOxidation" ) {
    CharOxidation* charoximodel = dynamic_cast<CharOxidation*>(model);
    charoximodels_[name] = charoximodel;
  } else if( modelType == "HeatTransfer" ) {
    HeatTransfer* heatmodel = dynamic_cast<HeatTransfer*>(model);
    heatmodels_[name] = heatmodel;
  }

  if( b_coupled_physics ) {
    string modelType = model->getType();
    if( modelType == "Length" ) {
      LengthModel = model;
    } else if( modelType == "Devolatilization" ) {
      DevolModel = model;
    } else if( modelType == "HeatTransfer" ) {
      HeatModel = model;
    } else if( modelType == "CharOxidation" ) {
      CharModel = model;
    } else {
      proc0cout << "WARNING: Arches: CoalModelFactory: Unrecognized model type "+name+" for coupled multiphysics particle algorithm!" << endl;
      proc0cout << "Continuing..." << endl;
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
  } else {
    string err_msg = "ERROR: Arches: CoalModelFactory: No model registered for " + name + "\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }

  // don't build the models here... they're already built in CoalModelFactory::register_model()

  //ModelBuilder* builder = ibuilder->second;
  //ModelBase* model = builder->build();
  //models_[name] = model;
  //return *model;
}

//---------------------------------------------------------------------------
// Method: Schedule calculation of all models
//---------------------------------------------------------------------------
void
CoalModelFactory::sched_coalParticleCalculation( const LevelP& level, 
                                                 SchedulerP& sched, 
                                                 int timeSubStep )
{
  if( b_coupled_physics ) {
    // require all internal coordinate and gas source variables
    // leave any other model-specific calculated variables for model to take care of
  } else {
    // Model evaluation order:
    // 1. Constant model
    // 2. Heat transfer model
    // 3. Devolatilization model
    // 4. Char oxidation model
    for( ModelMap::iterator iModel = models_.begin(); iModel != models_.end(); ++iModel ) {
      iModel->second->sched_computeModel( level, sched, timeSubStep );
    }
  }
}

void
CoalModelFactory::coalParticleCalculation(  const ProcessorGroup  * pc,
                                            const PatchSubset     * patches,
                                            const MaterialSubset  * matls,
                                            DataWarehouse         * old_dw,
                                            DataWarehouse         * new_dw )
{
  if( b_coupled_physics ) {
    for( int p = 0; p < patches->size(); ++p ) {
      //Ghost::GhostType  gaf = Ghost::AroundFaces;
      Ghost::GhostType  gac = Ghost::AroundCells;
      Ghost::GhostType  gn  = Ghost::None;

      const Patch* patch = patches->get(p);
      int archIndex = 0;
      int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

      constCCVariable<double> length_wa;
      old_dw->get( length_wa, d_Length_ICLabel, matlIndex, patch, gn, 0 );

      constCCVariable<double> gas_temperature;
      old_dw->get( gas_temperature, d_fieldLabels->d_tempINLabel, matlIndex, patch, gac, 1 );

      CCVariable<double> length_model;
      if( new_dw->exists( d_Length_ICLabel, matlIndex, patch ) ) {
        new_dw->getModifiable( length_model, d_Length_ICLabel, matlIndex, patch ); 
      } else {
        new_dw->allocateAndPut( length_model, d_Length_ICLabel, matlIndex, patch );
        length_model.initialize(0.0);
      }

      CCVariable<double> length_gas;

      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter; 
        // Put the CCVariables needed into temporary variables
  
        // get length internal coordinate value
        // calculate new particle temperature
        // calculate particle surface area
        // calculate heat capacity
        // calculate overall gas concentration
        // calculate coal devlatilization rate
        // calculate partial pressure and heat of devolatilization
        // enter iterative loop:
          // calculate particle temperature
          // calculate partial pressure
          // calculate heat ofvaporization
          // mass transfer blowing parameter
          // mass transfer coefficients
          // char reaction rates for each oxidizer
          // total char reaction rate
          // convective heat transfer
          // radiative heat transfer
          // liquid vaporization rate
          // total reaction rate (including liquid raction rate)
          // enthalpy of coal off-gas
          // enthalpy change of particle
          // particle enthalpy relati e to standard state
  
        // Put the temporary (final) values into the CCVariables
  
      }//end cells
    }//end patches
  }
}

