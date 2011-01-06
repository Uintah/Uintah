#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h> 
#include <CCA/Components/Arches/CoalModels/ConstantModel.h> 
#include <CCA/Components/Arches/CoalModels/Size.h> 
#include <CCA/Components/Arches/CoalModels/Devolatilization.h> 
#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h> 
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h> 
#include <CCA/Components/Arches/CoalModels/CoalParticleHeatTransfer.h> 
#include <CCA/Components/Arches/CoalModels/ParticleVelocity.h> 
#include <CCA/Components/Arches/CoalModels/DragModel.h> 
#include <CCA/Components/Arches/CoalModels/Balachandar.h> 
#include <CCA/Components/Arches/CoalModels/ParticleDensity.h>
#include <CCA/Components/Arches/CoalModels/ConstantSizeCoal.h>
#include <CCA/Components/Arches/CoalModels/ConstantDensityCoal.h>
#include <CCA/Components/Arches/CoalModels/ConstantSizeInert.h>
#include <CCA/Components/Arches/CoalModels/ConstantDensityInert.h>
#include <CCA/Components/Arches/CoalModels/CharOxidation.h> 
#include <CCA/Components/Arches/CoalModels/GlobalCharOxidation.h> 
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>

//===========================================================================

using namespace Uintah;

CoalModelFactory::CoalModelFactory()
{
  d_labelSet = false;
  yelem.resize(5);
  
  d_useParticleVelocityModel = false;
  d_useParticleDensityModel = false;
  d_useHeatTransferModel = false;
  d_useDevolatilizationModel = false;

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
    err_msg = "ERROR: Arches: EqnFactory: You must set the CoalModelFactory field labels using CoalModelFactory::setArchesLabel() before you run the problem setup method!";
    throw ProblemSetupException(err_msg, __FILE__, __LINE__);
  }

  DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self(); 
  numQuadNodes = dqmom_factory.get_quad_nodes();  

  d_ParticleVelocityModel.resize(numQuadNodes);
  d_ParticleDensityModel.resize(numQuadNodes);
  d_HeatTransferModel.resize(numQuadNodes);
  d_DevolatilizationModel.resize(numQuadNodes);

  // ----------------------------------------------
  // Step 1: CoalModelFactory problem setup

  ProblemSpecP dqmom_db;
  if( params_root->findBlock("CFD") ) {
    if(params_root->findBlock("CFD")->findBlock("ARCHES") ) {
      if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM") ) {
        dqmom_db = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM");
        std::string which_dqmom; 
        dqmom_db->getAttribute( "type", which_dqmom ); 
        if ( which_dqmom == "unweightedAbs" ) {
          d_unweighted = true; 
        } else {
          d_unweighted = false; 
        }
      }
    }
  }

  // Grab coal properties from input file
  ProblemSpecP db_coalProperties;
  if( params_root->findBlock("CFD") ) {
    if( params_root->findBlock("CFD")->findBlock("ARCHES") ) {
      if( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties") ) {
        db_coalProperties = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
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
      }
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
  }



  // ----------------------------------------------
  // Step 2: register all models with the CoalModelFactory
  ProblemSpecP models_db = db->findBlock("Models");
  
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

  if (models_db) {
    for (ProblemSpecP model_db = models_db->findBlock("model"); model_db != 0; model_db = model_db->findNextBlock("model")){
      
      std::string model_name;
      model_db->getAttribute("label", model_name);
      
      std::string model_type;
      model_db->getAttribute("type", model_type);

      proc0cout << "Found a model: " << model_name << endl;

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

        // This ModelBuilder stuff is pointless here!
        // It was designed so that models were only built as needed, when needed
        // But we're building everything right away, there's no situation where we WOULDN'T build the model...
        // -cmr
        
//-------- Constant models
        if ( model_type == "ConstantModel" ) {
          modelBuilder = scinew ConstantModelBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

//-------- Devolatilization models
        } else if ( model_type == "BadHawkDevol" ) {
          //modelBuilder = scinew BadHawkDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn );
          throw InvalidValue("ERROR: Arches: CoalModelFactory: BadHawkDevol model is not supported.\n",__FILE__,__LINE__);

        } else if ( model_type == "KobayashiSarofimDevol" ) {
          modelBuilder = scinew KobayashiSarofimDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);
          //what about computedVarLabels?
          d_useDevolatilizationModel = true;

//-------- Heat transfer models
        } else if ( model_type == "CoalParticleHeatTransfer" ) {
          modelBuilder = scinew CoalParticleHeatTransferBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);
          d_useHeatTransferModel = true;

//-------- Velocity models
        } else if (model_type == "DragModel" ) {
          modelBuilder = scinew DragModelBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);
          d_useParticleVelocityModel = true;

        } else if (model_type == "Balachandar" ) {
          modelBuilder = scinew BalachandarBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);
          d_useParticleVelocityModel = true;
        
//-------- Char oxidation models
        } else if (model_type == "GlobalCharOxidation" ) {
          modelBuilder = scinew GlobalCharOxidationBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

//-------- Density models
        } else if (model_type == "ConstantSizeCoal" ) {
          modelBuilder = scinew ConstantSizeCoalBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

        } else if (model_type == "ConstantDensityCoal" ) {
          modelBuilder = scinew ConstantDensityCoalBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

        } else if (model_type == "ConstantSizeInert" ) {
          modelBuilder = scinew ConstantSizeInertBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

        } else if (model_type == "ConstantDensityInert" ) {
          modelBuilder = scinew ConstantDensityInertBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);

        } else {
          proc0cout << "For model named: " << temp_model_name << endl;
          proc0cout << "with type: " << model_type << endl;
          string errmsg;
          errmsg = "ERROR: CoalModelFactory: " + model_type + ": This model type not recognized or not supported.";
          throw ProblemSetupException(errmsg, __FILE__, __LINE__);
        }

        register_model( temp_model_name, modelBuilder, iqn );

        // ----------------------------------------------
        // Step 3: run ModelBase::problemSetup() for each model

        ModelMap::iterator iM = models_.find(temp_model_name);
        if (iM != models_.end() ) {
          ModelBase* a_model = iM->second;
          a_model->problemSetup( model_db );
        }

      }//end for each qn

      proc0cout << endl;

    }//end for each model

    // if a model using density is included, assert that a density model is specified
    if( !d_useParticleDensityModel ) {
      // no particle density model is specified
      // but a density model is required for some models 
      if( d_useParticleVelocityModel ) {
        string err = "ERROR: CoalModelFactory: You specified a coal model that requires density (particle velocity or heat transfer), but you did not specify a density model!  Please specify a density model before proceeding.";
        throw ProblemSetupException(err,__FILE__,__LINE__);
      }
    }

  } else {
    proc0cout << "No models were found by CoalModelFactory." << endl;

  }//end if model block

  proc0cout << endl;



  // ----------------------------------------------
  // Step 4: check model types with associated internal coordinates (if coupled)

  // ----------------------------------------------
  // Step 5: Set CoalParticle objects

  // ----------------------------------------------

}


//---------------------------------------------------------------------------
// Method: Register a model  
//---------------------------------------------------------------------------
/** @details
This method checks for a couple of different model types.  The reason for this is,
certain model types require a particle density model (these include particle velocity
and heat transfer models).  

Additionally, the factory checks to see if the mixture fraction, momentum, and mass 
solvers "should have" source terms (i.e. if models are being used that create source terms).
If there are, and there is no <src> tag in the corresponding block, a warning will be printed.
This happens in CoalModelFactory::problemSetup(), right after registration of all models.
*/
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

  model->setUnweightedAbscissas(d_unweighted);

  string modelType = model->getType();

  if( modelType == "ParticleDensity" ) {
    d_ParticleDensityModel[quad_node] = dynamic_cast<ParticleDensity*>(model);
    d_useParticleDensityModel = true;
  }

  if( modelType == "ParticleVelocity" ) {
    d_ParticleVelocityModel[quad_node] = dynamic_cast<ParticleVelocity*>(model);
    d_useParticleVelocityModel = true;
  }

  if( modelType == "HeatTransfer" ) {
    d_HeatTransferModel[quad_node] = dynamic_cast<HeatTransfer*>(model);
    d_useHeatTransferModel = true;
  }

  if( modelType == "Devolatilization" ) {
    d_DevolatilizationModel[quad_node] = dynamic_cast<Devolatilization*>(model);
    d_useDevolatilizationModel = true;
  }

  if( modelType == "CharOxidation" ) {
    d_useCharOxidationModel = true;
    CharOxidation* char_model = dynamic_cast<CharOxidation*>(model);
    char_model->setTabPropsInterface( d_TabPropsInterface );
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
  // schedule particle velocity model's compute velocity method
  if( d_useParticleVelocityModel ) {
    for( vector<ParticleVelocity*>::iterator iPV = d_ParticleVelocityModel.begin();
         iPV != d_ParticleVelocityModel.end(); ++iPV) {
      (*iPV)->sched_computeParticleVelocity(level, sched, timeSubStep);
      //(*iPV)->sched_dummyVel(level, sched, timeSubStep);
    }
  }
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
    string err = "ERROR: CoalModelFactory: Coupled physics particle calculation is not implemented yet!\n";
    throw InvalidValue(err,__FILE__,__LINE__);
  } else {
    
    if( d_useParticleDensityModel ) {
      // evaluate the density
      for( vector<ParticleDensity*>::iterator iPD = d_ParticleDensityModel.begin();
           iPD != d_ParticleDensityModel.end(); ++iPD ) {
        (*iPD)->sched_computeParticleDensity( level, sched, timeSubStep );
      }
    }

    // Model evaluation order matches order in input file
    for( ModelMap::iterator iModel = models_.begin(); iModel != models_.end(); ++iModel ) {
#ifdef DEBUG_MODELS
      proc0cout << "Scheduling model computation for model " << iModel->second->getModelName() << endl;
#endif
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
    iModel->second->sched_initVars( level, sched );
  }
}


//---------------------------------------------------------------------------
// Method: Schedule dummy initialization of models
//---------------------------------------------------------------------------
void
CoalModelFactory::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  for( ModelMap::iterator iModel = models_.begin(); iModel != models_.end(); ++iModel ) {
    iModel->second->sched_dummyInit( level, sched );
  }
}
