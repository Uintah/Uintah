#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h> 
#include <CCA/Components/Arches/CoalModels/ConstantModel.h> 
#include <CCA/Components/Arches/CoalModels/Size.h> 
#include <CCA/Components/Arches/CoalModels/Devolatilization.h> 
#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h> 
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h> 
#include <CCA/Components/Arches/CoalModels/SimpleHeatTransfer.h> 
#include <CCA/Components/Arches/CoalModels/ParticleVelocity.h> 
#include <CCA/Components/Arches/CoalModels/DragModel.h> 
#include <CCA/Components/Arches/CoalModels/Balachandar.h> 
#include <CCA/Components/Arches/CoalModels/CharOxidation.h> 
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>

//===========================================================================

using namespace Uintah;

CoalModelFactory::CoalModelFactory()
{
  d_labelSet = false;
  d_useParticleDensityModel = false;
  d_useParticleVelocityModel = false;
  yelem.resize(5);

  ParticleDensityModel.resize(numQuadNodes);
  ParticleVelocityModel.resize(numQuadNodes);
  SizeModel.resize(numQuadNodes);
  DevolModel.resize(numQuadNodes);
  HeatModel.resize(numQuadNodes);
  CharModel.resize(numQuadNodes);
}

CoalModelFactory::~CoalModelFactory()
{
  //  // delete the builders
  for( BuildMap::iterator i=builders_.begin(); i!=builders_.end(); ++i ){
      delete i->second;
    }

  // delete the models
  for( ModelMap::iterator i=models_.begin(); i!=models_.end(); ++i ){
      delete i->second;
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

  if( d_labelSet == false ) {
    std::string err_msg;
    err_msg = "ERROR: Arches: EqnFactory: You must set the EqnFactory field labels using setArchesLabel() before you run the problem setup method!";
    throw ProblemSetupException(err_msg, __FILE__, __LINE__);
  }

  // ----------------------------------------------
  // Step 1: CoalModelFactory problem setup

  // Grab coal properties from input file
  ProblemSpecP db_coalProperties = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
  if( db_coalProperties ) {
    db_coalProperties->require("C", yelem[0]);
    db_coalProperties->require("H", yelem[1]);
    db_coalProperties->require("N", yelem[2]);
    db_coalProperties->require("O", yelem[3]);
    db_coalProperties->require("S", yelem[4]);
  } else {
    //string err_msg="Missing <Coal_Properties> section in input file!";
    //throw ProblemSetupException(err_msg,__FILE__,__LINE__);
    for( int i=0; i<5; ++i ) {
      yelem[i] = 0.0;
    }
  }

  ProblemSpecP db_coalParticleIterator = db->findBlock("coalParticleIterator");
  if( !db_coalParticleIterator ) {
    d_coupled_physics = false;

  } else {
    
    // Coupled or separable physics calculations?
    string calculation_type;
    db_coalParticleIterator->getAttribute("type",calculation_type);
  
    if( calculation_type=="split" || calculation_type == "separable" ) {
      d_coupled_physics = false;
      proc0cout << endl << "DQMOM coal particle calculation: using separable multiphysics calculation." << endl << endl;
    } else if( calculation_type == "coupled" ) {
      d_coupled_physics = true;
      proc0cout << endl << "DQMOM coal particle calculation: using coupled multiphysics calculation." << endl << endl;
    } else {
      string err_msg = "ERROR: Arches: CoalModelFactory: Unrecognized <coalParticleIterator> type: " + calculation_type + ": should be 'coupled' or 'separable'.";
      throw ProblemSetupException(err_msg,__FILE__,__LINE__);
    }
  
    // Now grab specific internal coordinates (only if using coupled algorithm)
    b_useParticleTemperature = false;
    b_useParticleEnthalpy = false; 
    b_useMoisture = false;
    b_useAsh = false;
  }



  // ----------------------------------------------
  // Step 2: register all models with the CoalModelFactory
  ProblemSpecP models_db = db->findBlock("Models");
  DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self(); 
  
  proc0cout << endl;
  proc0cout << "******* Model Registration ********" << endl; 

  // There are three kind of variables to worry about:
  // 1) internal coordinates
  // 2) other "extra" scalars
  // 3) standard flow variables
  // We want the model to have access to all three.  
  // for 1) you specify this in the <ICVars> tag - set the internal coordinate name and the "_qn#" is attached (since models are reproduced qn times)
  // for 2) you specify this in the <scalarVars> tag
  // for 3) you specify this in the implementation of the model itself (ie, no user input)

  const int numQuadNodes = dqmom_factory.get_quad_nodes();  
  
  if (models_db) {
    for (ProblemSpecP model_db = models_db->findBlock("model"); model_db != 0; model_db = model_db->findNextBlock("model")){
      
      std::string model_name;
      model_db->getAttribute("label", model_name);
      
      std::string model_type;
      model_db->getAttribute("type", model_type);

      proc0cout << endl << "Found  a model: " << model_name << endl;

      vector<string> requiredICVarLabels;
      ProblemSpecP icvar_db = model_db->findBlock("ICVars"); 

      if ( icvar_db ) {
        proc0cout << "Requires the following internal coordinates: " << endl;
        for (ProblemSpecP var = icvar_db->findBlock("variable"); var !=0; var = var->findNextBlock("variable")){
          std::string label_name; 
          var->getAttribute("label", label_name);
          proc0cout << "label = " << label_name << endl; 

          // This map hold the labels that are required to compute this model term. 
          requiredICVarLabels.push_back(label_name);  
        }
      } else {
        proc0cout << "Model does not require any internal coordinates. " << endl;
      }
      
      // This section is not immediately useful.
      // However, if any new models start to depend on scalar variables, this must be commented in
      vector<string> requiredScalarVarLabels;
      /*
      ProblemSpecP scalarvar_db = model_db->findBlock("scalarVars");

      if ( scalarvar_db ) {
        proc0cout << "Requires the following scalar variables: " << endl;
        for (ProblemSpecP var = scalarvar_db->findBlock("variable");
             var != 0; var = var->findNextBlock("variable") ) {
          std::string label_name;
          var->getAttribute("label", label_name);

          proc0cout << "label = " << label_name << endl;
          // This map holds the scalar labels required to compute this model term
          requiredScalarVarLabels.push_back(label_name);
        }
      } else {
        proc0cout << "Model does not require any scalar variables. " << endl;
      }
      */
      
      // --- looping over quadrature nodes ---
      // This will make a model for each quadrature node. 

      for (int iqn = 0; iqn < numQuadNodes; iqn++){
        std::string temp_model_name = model_name; 
        std::string node;  
        std::stringstream out; 

        out << iqn; 
        node = out.str(); 
        temp_model_name += "_qn";
        temp_model_name += node; 
        
        ModelBuilder* modelBuilder;
        
        if ( model_type == "ConstantModel" ) {
          modelBuilder = scinew ConstantModelBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

        // Devolatilization
        } else if ( model_type == "BadHawkDevol" ) {
          //modelBuilder = scinew BadHawkDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn );
          throw InvalidValue("ERROR: Arches: CoalModelFactory: BadHawkDevol model is not supported.\n",__FILE__,__LINE__);

        } else if ( model_type == "KobayashiSarofimDevol" ) {
          modelBuilder = scinew KobayashiSarofimDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

        // Heat transfer
        } else if ( model_type == "SimpleHeatTransfer" ) {
          modelBuilder = scinew SimpleHeatTransferBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

        // Velocity model
        } else if (model_type == "Drag" ) {
          modelBuilder = scinew DragModelBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

        } else if (model_type == "Balachandar" ) {
          modelBuilder = scinew BalachandarBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

        } else {
          proc0cout << "For model named: " << temp_model_name << endl;
          proc0cout << "with type: " << model_type << endl;
          std::string errmsg;
          errmsg = model_type + ": This model type not recognized or not supported.";
          throw InvalidValue(errmsg, __FILE__, __LINE__);
        }

        CoalModelFactory::register_model( temp_model_name, modelBuilder, iqn );

        // ----------------------------------------------
        // Step 3: run ModelBase::problemSetup() for each model

        ModelMap::iterator iM = models_.find(temp_model_name);
        if (iM != models_.end() ) {
          ModelBase* a_model = iM->second;
          a_model->problemSetup( model_db );
        }

      }//end for each qn

    }//end for each model

  } else {
    proc0cout << "No models were found by CoalModelFactory." << endl;

  }//end if model block

  proc0cout << endl;



  // ----------------------------------------------
  // Step 4: check model types with associated internal coordinates (if coupled)

  // ----------------------------------------------
  // Step 5: Set CoalParticle objects/labels/variables/etc.



}

//---------------------------------------------------------------------------
// Method: Schedule initialization of models
//---------------------------------------------------------------------------
/* @details
This calls sched_initVars for each model, which in turn
 initializes the DQMOM-phase source terms (model terms G)
 and the gas-phase source terms. This must be done by the
 individual models, and not the Factory, because only the 
 model knows what variable type it is (e.g. double or Vector)
*/
void
CoalModelFactory::sched_modelInit( const LevelP& level, SchedulerP& sched)
{
  for( ModelMap::iterator iModel = models_.begin(); iModel != models_.end(); ++iModel ) {
    ModelBase* model = iModel->second;
    model->sched_initVars( level, sched );
  }
}


//---------------------------------------------------------------------------
// Method: Register a model  
//---------------------------------------------------------------------------
void
CoalModelFactory::register_model( const std::string name,
                                  ModelBuilder* builder,
                                  int quad_node )
{
  ASSERT( builder != NULL );

  BuildMap::iterator iBuilder = builders_.find( name );
  if( iBuilder == builders_.end() ){
    builders_[name] = builder;
  } else {
    string err_msg = "ERROR: Arches: CoalModelFactory: A duplicate ModelBuilder object named "+name+" was already loaded. This is forbidden.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }

  // build the models
  const ModelMap::iterator iModel = models_.find( name );
  if( iModel != models_.end() ) {
    string err_msg = "ERROR: Arches: CoalModelFactory: A duplicate ModelBase object named "+name+" was already built. This is forbidden.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }
  
  ModelBase* model = builder->build();
  models_[name] = model;

  if( model->getType() == "ParticleDensity" ) {
    ParticleDensityModel[quad_node] = dynamic_cast<ParticleDensity*>(model);
    d_useParticleDensityModel = true;
  }

  string modelType = model->getType();

  /*
  // this is for when CoalParticle is a class that's being used...
  if( d_coupled_physics ) {

    if( modelType == "Size" ) {
      coal_particle->setSizeModel(model);

    } else if( modelType == "Devolatilization" ) {
      coal_particle->setDevolModel(model);

    } else if( modelType == "HeatTransfer" ) {
      coal_particle->setHeatTransferModel(model);

    } else if( modelType == "CharOxidation" ) {
      coal_particle->setCharOxidationModel(model);

    } else if( modelType == "Drag" ) { // FIXME Drag
      coal_particle->setDragModel(model);

    } else if( modelType == "Evaporation" ) {
      coal_particle->setEvaporationModel(model);

    } else {
      proc0cout << "WARNING: Arches: CoalModelFactory: Unrecognized model type " << name << " for coupled particle iterator! This model will not be used in the iterative procedure." << endl;
      proc0cout << "Continuing..." << endl;

    }
  }
  */

  if( modelType == "Size" ) {
    SizeModel[quad_node] = dynamic_cast<Size*>(model);

  } else if( modelType == "ParticleVelocity" ) {
    ParticleVelocityModel[quad_node] = dynamic_cast<ParticleVelocity*>(model);
    d_useParticleVelocityModel = true;
    
  } else if( modelType == "Devolatilization" ) {
    DevolModel[quad_node] = dynamic_cast<Devolatilization*>(model);
  
  } else if( modelType == "HeatTransfer" ) {
    HeatModel[quad_node] = dynamic_cast<HeatTransfer*>(model);
  
  } else if( modelType == "CharOxidation" ) {
    CharModel[quad_node] = dynamic_cast<CharOxidation*>(model);
  
  } else {
    proc0cout << "WARNING: Arches: CoalModelFactory: Unrecognized model type "+name+" for coupled multiphysics particle algorithm!" << endl;
    proc0cout << "Continuing..." << endl;
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

}

void
CoalModelFactory::sched_computeVelocity( const LevelP& level,
                                         SchedulerP& sched,
                                         int timeSubStep )
{
  // schedule particle velocity model's computevelocity method();
}

//---------------------------------------------------------------------------
// Method: Schedule calculation of all models
//---------------------------------------------------------------------------
void
CoalModelFactory::sched_coalParticleCalculation( const LevelP& level, 
                                                 SchedulerP& sched, 
                                                 int timeSubStep )
{
  if( d_coupled_physics ) {
    // require all internal coordinate and gas source variables
    // leave any other model-specific calculated variables for model to take care of
  } else {
    
    if( d_useParticleDensityModel ) {
      // evaluate the density
      for( vector<ParticleDensity*>::iterator iPD = ParticleDensityModel.begin();
           iPD != ParticleDensityModel.end(); ++iPD ) {
        (*iPD)->sched_computeParticleDensity( level, sched, timeSubStep );
      }
    }
    
    // Model evaluation order matches order in input file
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
  /*
  if( d_coupled_physics ) {
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
  */
}

