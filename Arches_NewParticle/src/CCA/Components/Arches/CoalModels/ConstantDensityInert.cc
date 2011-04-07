#include <CCA/Components/Arches/CoalModels/ConstantDensityInert.h>
#include <CCA/Components/Arches/CoalModels/ParticleDensity.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
ConstantDensityInertBuilder::ConstantDensityInertBuilder( const std::string         & modelName,
                                                        const vector<std::string> & reqICLabelNames,
                                                        const vector<std::string> & reqScalarLabelNames,
                                                        ArchesLabel         * fieldLabels,
                                                        SimulationStateP          & sharedState,
                                                        int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

ConstantDensityInertBuilder::~ConstantDensityInertBuilder(){}

ModelBase* ConstantDensityInertBuilder::build() {
  return scinew ConstantDensityInert( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

ConstantDensityInert::ConstantDensityInert( std::string modelName, 
                                          SimulationStateP& sharedState,
                                          ArchesLabel* fieldLabels,
                                          vector<std::string> icLabelNames, 
                                          vector<std::string> scalarLabelNames,
                                          int qn ) 
: ParticleDensity(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  d_useLength = false;
  d_useMass = false;
}

ConstantDensityInert::~ConstantDensityInert()
{
  if( d_constantLength ) {
    VarLabel::destroy(d_length_label);
  }
}

//-----------------------------------------------------------------------------
//Problem Setup
//-----------------------------------------------------------------------------
void 
ConstantDensityInert::problemSetup(const ProblemSpecP& params)
{
  // call parent's method first
  ParticleDensity::problemSetup(params);

  ProblemSpecP db = params; 

  string label_name;
  string role_name;
  string temp_label_name;
  
  string temp_ic_name;
  string temp_ic_name_full;
  
  std::stringstream out;
  out << d_quadNode; 
  string node = out.str();



  // -----------------------------------------------------------------
  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  if (db_icvars) {
    for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label",label_name);
      variable->getAttribute("role", role_name);

      temp_label_name = label_name;
      
      temp_label_name += "_qn";
      temp_label_name += node;

      // user specifies "role" of each internal coordinate
      // if it isn't an internal coordinate or a scalar, it's required explicitly
      // ( see comments in Arches::registerModels() for details )
      if( role_name == "particle_length" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useLength = true;
      } else if( role_name == "particle_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useMass = true;
      } else {
        string errmsg = "ERROR: Arches: ConstantDensityInert: Invalid variable role: must be \"particle_length\" or \"particle_mass\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }



  /*
  // -----------------------------------------------------------------
  // Look for required scalars
  ProblemSpecP db_scalarvars = params->findBlock("scalarVars");
  if (db_scalarvars) {
    for( ProblemSpecP variable = db_scalarvars->findBlock("variable");
         variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label", label_name);
      variable->getAttribute("role",  role_name);

      // user specifies "role" of each scalar
      if( role_name == "---" ) {
        LabelToRoleMap[temp_label_name] = role_name;
      } else {
        string errmsg = "ERROR: Arches: CoalParticleHeatTransfer: Invalid scalar variable role for Simple Heat Transfer model: must be \"particle_temperature\" or \"gas_temperature\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }
  */



  // -----------------------------------------------------------------
  // Look for constants used (for now, only length)
  //
  //  <ConstantVar label="length" role="particle_length">
  //    <constant qn="0" value="1.00" />
  //  </ConstantVar>
  for( ProblemSpecP db_constantvar = params->findBlock("ConstantVar");
       db_constantvar != 0; db_constantvar = params->findNextBlock("ConstantVar") ) {

    db_constantvar->getAttribute("label", label_name);
    db_constantvar->getAttribute("role",  role_name );

    temp_label_name = d_modelName;
    temp_label_name += "_";
    temp_label_name += label_name;
    temp_label_name += "_qn";
    temp_label_name += node;

    if (role_name == "particle_length") {
      LabelToRoleMap[temp_label_name] = role_name;
      d_useLength = true;
      d_constantLength = true;

      d_length_label = VarLabel::create( temp_label_name, CCVariable<double>::getTypeDescription() );
      d_length_scaling_constant = 1.0;

    } else {
      std::string errmsg;
      errmsg = "ERROR: Arches: ConstantDensityInert: Invalid constant role:";
      errmsg += "must be \"particle_length\", you specified \"" + role_name + "\".";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);

    }

    // Now grab the actual values of the constants
    for( ProblemSpecP db_constant = db_constantvar->findBlock("constant");
         db_constant != 0; db_constant = db_constantvar->findNextBlock("constant") ) {
      string s_tempQuadNode;
      db_constant->getAttribute("qn",s_tempQuadNode);
      int i_tempQuadNode = atoi( s_tempQuadNode.c_str() );

      if( i_tempQuadNode == d_quadNode ) {
        string s_constant;
        db_constant->getAttribute("value", s_constant);
        d_length_constant_value = atof( s_constant.c_str() );
      }
    }
  }



  // -----------------------------------------------------------------
  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {

    temp_ic_name        = (*iString);
    temp_ic_name_full   = temp_ic_name;

    std::stringstream out;
    out << d_quadNode;
    string node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }



  if(!d_useMass) {
    string errmsg = "ERROR: Arches: ConstantDensityInert: You did not specify a particle mass internal coordinate!\n";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  if(!d_useLength) {
    string errmsg = "ERROR: Arches: ConstantDensityCoal: You did not specify a particle length internal coordinate!\n";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  ///////////////////////////////////////////


  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  EqnFactory& eqn_factory = EqnFactory::self();

  // assign labels for each required internal coordinate
  for( map<string,string>::iterator iter = LabelToRoleMap.begin();
       iter != LabelToRoleMap.end(); ++iter ) {

    EqnBase* current_eqn;
    if( dqmom_eqn_factory.find_scalar_eqn(iter->first) ) {
      current_eqn = &(dqmom_eqn_factory.retrieve_scalar_eqn(iter->first));
    } else if( eqn_factory.find_scalar_eqn(iter->first) ) {
      current_eqn = &(eqn_factory.retrieve_scalar_eqn(iter->first));
    } else {
      if( !d_constantLength ) {
        string errmsg = "ERROR: Arches: ConstantDensityInert: Invalid variable \"" + iter->first + "\" given for \""+iter->second+"\" role, could not find in EqnFactory or DQMOMEqnFactory!";
        throw ProblemSetupException(errmsg,__FILE__,__LINE__);
      }
    }

    if( iter->second == "particle_length" ) {
      if( !d_constantLength ) {
        d_length_label = current_eqn->getTransportEqnLabel();
        d_length_scaling_constant = current_eqn->getScalingConstant();

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(current_eqn);
        dqmom_eqn->addModel( d_modelLabel );
      }

    } else if( iter->second == "particle_mass" ) {
      d_particle_mass_label = current_eqn->getTransportEqnLabel();
      d_mass_scaling_constant = current_eqn->getScalingConstant();
    } else {
      string errmsg = "ERROR: Arches: ConstantDensityInert: Could not identify specified variable role \""+iter->second+"\".";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }

  }
 
  //// set model clipping
  //db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
  //db->getWithDefault( "high_clip", d_highModelClip, 999999 );

  if( d_useMass ) {
    d_massLabels.push_back( d_particle_mass_label );
    d_massScalingConstants.push_back( d_mass_scaling_constant );
  }

}


//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
ConstantDensityInert::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "ConstantDensityInert::initVars";
  Task* tsk = scinew Task(taskname, this, &ConstantDensityInert::initVars);

  tsk->computes( d_modelLabel );
  tsk->computes( d_gasLabel   );
  tsk->computes( d_density_label );

  if( d_constantLength ) {
    tsk->computes( d_length_label );
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}


//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
ConstantDensityInert::initVars( const ProcessorGroup * pc, 
                                const PatchSubset    * patches, 
                                const MaterialSubset * matls, 
                                DataWarehouse        * old_dw, 
                                DataWarehouse        * new_dw )
{
  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();

  EqnBase* length_eqn = &dqmomFactory.retrieve_scalar_eqn(d_length_label->getName());
  EqnBase* mass_eqn   = &dqmomFactory.retrieve_scalar_eqn(d_particle_mass_label->getName());

  if(  length_eqn->getInitFcn() == "step"
    || length_eqn->getInitFcn() == "env_step" 
    || mass_eqn->getInitFcn() == "step"
    || mass_eqn->getInitFcn() == "env_step" ) {
    string errmsg = "ERROR: Arches: ConstantDensityInert: Internal coordinates cannot be initialized with a step function when using constant density models, otherwise no density value can be obtained.\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    double mass = mass_eqn->getInitializationConstant()*mass_eqn->getScalingConstant();
    double length = length_eqn->getInitializationConstant()*length_eqn->getScalingConstant();
    if( length < TINY ) {
      length = TINY;
    }
    d_density = mass / ( 4.0/3.0 * pi * pow(length/2.0,3) );

    CCVariable<double> model_value; 
    new_dw->allocateAndPut( model_value, d_modelLabel, matlIndex, patch ); 
    model_value.initialize(0.0);

    CCVariable<double> gas_value; 
    new_dw->allocateAndPut( gas_value, d_gasLabel, matlIndex, patch ); 
    gas_value.initialize(0.0);

    CCVariable<double> density;
    new_dw->allocateAndPut( density, d_density_label, matlIndex, patch );
    density.initialize(d_density);

  }
}



//---------------------------------------------------------------------------
//Schedule computation of the particle density
//---------------------------------------------------------------------------
/*
@details
The particle density calculation is scheduled before the other model term calculations are scheduled.
*/
void
ConstantDensityInert::sched_computeParticleDensity( const LevelP& level,
                                                    SchedulerP& sched,
                                                    int timeSubStep )
{
  std::string taskname = "ConstantDensityInert::computeParticleDensity";
  Task* tsk = scinew Task(taskname, this, &ConstantDensityInert::computeParticleDensity, timeSubStep );

  if( timeSubStep == 0 ) {
    tsk->computes( d_density_label );
  } else {
    tsk->modifies( d_density_label );
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}



//---------------------------------------------------------------------------
// Compute particle density
//---------------------------------------------------------------------------
/**
@details
Becuase the density is constant, the CCVariable is simply initialized to the constant value.
*/
void
ConstantDensityInert::computeParticleDensity( const ProcessorGroup* pc,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw,
                                              int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    // compute density
    CCVariable<double> density;
    if( timeSubStep == 0 ) {
      new_dw->allocateAndPut( density, d_density_label, matlIndex, patch );
    } else {
      new_dw->getModifiable( density, d_density_label, matlIndex, patch );
    }
    density.initialize(d_density);

  }//end patch loop
}

//---------------------------------------------------------------------------
//Schedule computation of the model
//---------------------------------------------------------------------------
/**
@details
The constant density model treats the density as constant,
so the length internal coordinate associated with the ConstantDensityInert model
must change accordingly. This model term will be added to the length
internal coordinate specified in the required variables block for the <model>.
*/
void 
ConstantDensityInert::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "ConstantDensityInert::computeModel";
  Task* tsk = scinew Task(taskname, this, &ConstantDensityInert::computeModel, timeSubStep );

  Ghost::GhostType gn = Ghost::None;

  d_timeSubStep = timeSubStep; 

  // require timestep label
  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label() );

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;
  }

  if( timeSubStep == 0 ) {

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 

    tsk->requires(Task::OldDW, d_weight_label, gn, 0 );
    tsk->requires(Task::OldDW, d_length_label, gn, 0 );
    tsk->requires(Task::OldDW, d_particle_mass_label, gn, 0 );

    if( d_constantLength ) {
      // this is required, because initializing variable to its constant value in NewDW
      tsk->computes(d_length_label);
    }

  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  

    tsk->requires(Task::NewDW, d_weight_label, gn, 0 );
    tsk->requires(Task::NewDW, d_length_label, gn, 0 );
    tsk->requires(Task::NewDW, d_particle_mass_label, gn, 0 );

  }

  // also need the source terms "G" for each mass internal coordinate
  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  DQMOMEqn* eqn;
  vector<const VarLabel*> models;
  for( vector<const VarLabel*>::iterator iL = d_massLabels.begin(); iL != d_massLabels.end(); ++iL ) {
    eqn = dynamic_cast<DQMOMEqn*>( &dqmomFactory.retrieve_scalar_eqn((*iL)->getName()) );
    models = eqn->getModelsList();
    for( vector<const VarLabel*>::iterator iS = models.begin(); iS != models.end(); ++iS ) {
      if( d_timeSubStep == 0 ) { 
        tsk->requires(Task::OldDW, (*iS), gn, 0);
      } else { 
        tsk->requires(Task::NewDW, (*iS), gn, 0);
      }
    }
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}


//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
/** 
@details
This method computes the model term for the length internal coordinate.
This term is defined as 
\f[
G_L = \frac{dL}{dt}
\f]
The calculation of this term is different in form than the calculation done
in the constant size case, since in that case the density is being set outright
(i.e. we're not calculating the rate of change of density).

The underling equation being used is (White Book, Chapter 4, p. 93):
\f[
\frac{d m_{p} }{ dt } = \rho_{p} \frac{\pi}{2} L_{p}^{2} \frac{ d L_{p} }{ dt}
\f]
When this is split up to find the term \f$ G_{L} \f$, it becomes:
\f[
G_{L} = \left( \dfrac{ m_{p}^{k+1} - m_{p}^{k} }{ \Delta t } \right) 
        \times 
        \left( \frac{ 2 }{ \rho_{p} \pi  \left( L_{p}^{k} \right)^{2}  } \right)
\f]
However, because the particle density is calculated before all of the other model terms,
the term \f$ m_{p}^{k+1} - m_{p}^{k} \f$ is actually \f$ m_{p}^{k} - m_{p}^{k-1} \f$.

In principle, this could be corrected by evaluating the reaction rate terms at time sub-step k+1,
most of which will (probably) not depend on density, and then evaluating the density at time sub-step k+1,
and finally evaluating any other model terms at k+1 that depend on density at k+1.

However, using dm/dt for timestep k (rather than k+1) is a reasonable approximation.
*/
void
ConstantDensityInert::computeModel( const ProcessorGroup * pc, 
                                    const PatchSubset    * patches, 
                                    const MaterialSubset * matls, 
                                    DataWarehouse        * old_dw, 
                                    DataWarehouse        * new_dw,
                                    int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    delt_vartype delta_t;
    old_dw->get( delta_t, d_fieldLabels->d_sharedState->get_delt_label() );

    CCVariable<double> model;
    CCVariable<double> model_gasSource; // always 0 for density

    constCCVariable<double> weight;
    constCCVariable<double> length;
    constCCVariable<double> mass;


    if( timeSubStep == 0 ) {

      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      new_dw->allocateAndPut( model_gasSource, d_gasLabel, matlIndex, patch ); 

      old_dw->get( weight,   d_weight_label,        matlIndex, patch, gn, 0 );
      old_dw->get(length, d_length_label,        matlIndex, patch, gn, 0 );
      old_dw->get(mass,   d_particle_mass_label, matlIndex, patch, gn, 0 );


    } else {

      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( model_gasSource, d_gasLabel, matlIndex, patch ); 
    
      new_dw->get( weight,   d_weight_label,        matlIndex, patch, gn, 0 );
      new_dw->get(length, d_length_label,        matlIndex, patch, gn, 0 );
      new_dw->get(mass,   d_particle_mass_label, matlIndex, patch, gn, 0 );

    }
    model.initialize(0.0);
    model_gasSource.initialize(0.0);

    // make a vector of all the mass internal coordinate model terms
    DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
    DQMOMEqn* eqn;
    vector<const VarLabel*> models;
    int z = 0;
    
    vector< constCCVariable<double>* > modelCCVars;

    // put the model terms into constCCVariables, and the constCCVariables into a vector
    for( vector<const VarLabel*>::iterator iL = d_massLabels.begin(); iL != d_massLabels.end(); ++iL, ++z ) {
      eqn = dynamic_cast<DQMOMEqn*>( &dqmomFactory.retrieve_scalar_eqn((*iL)->getName()) );
      models = eqn->getModelsList();
      for( vector<const VarLabel*>::iterator iS = models.begin(); iS != models.end(); ++iS ) {
        modelCCVars.push_back( scinew constCCVariable<double> );
        if( timeSubStep == 0 ) {
          old_dw->get( (*modelCCVars[z]), (*iS), matlIndex, patch, gn, 0 );
        } else {
          new_dw->get( (*modelCCVars[z]), (*iS), matlIndex, patch, gn, 0 );
        }
      }//end models for those i.c.s
    }//end mass labels

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      // weight - check if small
      bool weight_is_small = (weight[c] < d_w_small) || (weight[c] == 0.0);
      double model_sum = 0.0;
      int z=0;
      for( vector< constCCVariable<double>* >::iterator iM = modelCCVars.begin(); iM != modelCCVars.end(); ++iM, ++z ) {
        model_sum += (**iM)[c] * d_massScalingConstants[z] ;
      }

      if(!d_unweighted && weight_is_small) {

        model[c] = 0.0;

#ifdef DEBUG_MODELS
        //cmr
        if( c==IntVector(1,2,3) && d_quadNode==0 ) {
          cout << "ConstantDensityInert model QN " << d_quadNode << " has value " << model[c] << " (small weight)" << endl;
        }
#endif

      } else {
        
        double unscaled_length_old;
        if( d_unweighted ) {
          unscaled_length_old = length[c]*d_length_scaling_constant;
        } else {
          unscaled_length_old = (length[c]*d_length_scaling_constant)/weight[c];
        }

        // see the White Book, Ch. 4, p. 93
        if( unscaled_length_old < TINY ) {
          unscaled_length_old = TINY;
        }
        double unscaled_RHS = ( 2/(pi*d_density*unscaled_length_old*unscaled_length_old) )*( model_sum );
        double scaled_RHS   = unscaled_RHS/d_length_scaling_constant;

        model[c] = scaled_RHS;

      }
    }//end cell loop

    for( vector< constCCVariable<double>* >::iterator iM = modelCCVars.begin(); iM != modelCCVars.end(); ++iM ) {
      delete (*iM);
    }
  }//end patch loop
}

