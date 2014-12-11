#include <CCA/Components/Arches/PropertyModels/RadProperties.h>
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <Core/Containers/StaticArray.h>
using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
RadProperties::RadProperties( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 

  // Evaluated before or after table lookup: 
  _before_table_lookup = true; 

  int matlIndex = _shared_state->getArchesMaterial(0)->getDWIndex(); 
  _boundaryCond = scinew BoundaryCondition_new( matlIndex );



}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
RadProperties::~RadProperties( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    VarLabel::destroy( *iter ); 
  }
  if(_particlesOn){
    delete _ocalc;  
  }

  delete _boundaryCond;
  delete _calc; 
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void RadProperties::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 

  commonProblemSetup(db);

  std::string calculator_type; 
  ProblemSpecP db_calc = db->findBlock("calculator"); 
  if ( db_calc ){ 
    db_calc->getAttribute("type",calculator_type); 
  } else { 
    throw InvalidValue("Error: Calculator type not specified.",__FILE__, __LINE__); 
  }

  if ( calculator_type == "constant" ){ 
    _calc = scinew RadPropertyCalculator::ConstantProperties(); 
  } else if ( calculator_type == "burns_christon" ){ 
    _calc = scinew RadPropertyCalculator::BurnsChriston(); 
  } else if ( calculator_type == "hottel_sarofim"){
    _calc = scinew RadPropertyCalculator::HottelSarofim(); 
  } else if ( calculator_type == "radprops" ){
#ifdef HAVE_RADPROPS
    _calc = scinew RadPropertyCalculator::RadPropsInterface(); 
#else
    throw InvalidValue("Error: You haven't configured with the RadProps library (try configuring with --enable-wasatch_3p and --with-boost=DIR.)",__FILE__,__LINE__);
#endif
  } else { 
    throw InvalidValue("Error: Property calculator not recognized.",__FILE__, __LINE__); 
  } 

  if ( db_calc->findBlock("temperature")){ 
    db_calc->findBlock("temperature")->getAttribute("label", _temperature_name); 
  } else { 
    _temperature_name = "temperature"; 
  }

 _particlesOn = db_calc->findBlock("particles");

  bool complete; 
  complete = _calc->problemSetup( db_calc );
  if ( _particlesOn ){ 
    //------------ check to see if scattering is turned on --//
    std::string radiation_model;
    db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->getAttribute("type",radiation_model) ; 

    if (radiation_model == "do_radiation"){
      db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->findBlock("DORadiationModel")->getWithDefault("ScatteringOn" ,_scatteringOn,false) ; 
    }
    else if ( radiation_model == "rmcrt_radiation"){
      db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->findBlock("RMCRT")->getWithDefault("ScatteringOn" ,_scatteringOn,false) ; 
    }
    //-------------------------------------------------------//
    

    std::string particle_calculator_type; 
    db_calc->findBlock("particles")->getAttribute("type",particle_calculator_type); 
    if(particle_calculator_type == "basic"){
      _ocalc = scinew RadPropertyCalculator::basic(db_calc,_scatteringOn); 
    }else if(particle_calculator_type == "coal"){
      _ocalc = scinew RadPropertyCalculator::coalOptics(db_calc,_scatteringOn); 
    }else{
      throw InvalidValue("Particle radiative property model not found!! Name:"+particle_calculator_type,__FILE__, __LINE__); 
    }

    if (_ocalc->construction_success== false)
      throw InvalidValue("Error: Unable to setup optical radiation property calculator needed for scattering and absorption coefficients!",__FILE__, __LINE__); 
  }

  _nQn_part = 0;
  if ( db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM") && _particlesOn ){
    db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->require( "number_quad_nodes", _nQn_part ); 
    db->findBlock("calculator")->findBlock("particles")->getWithDefault( "part_temp_label", _base_temperature_label_name, "heat_pT" ); 
    db->findBlock("calculator")->findBlock("particles")->getWithDefault( "part_size_label", _base_size_label_name, "length" ); 
  }
  else if ( _particlesOn){
    throw InvalidValue("Error: No particle models found (DQMOM CQMOM Lagrangian?) ",__FILE__, __LINE__); 
  }

  if ( !complete )
    throw InvalidValue("Error: Unable to setup radiation property calculator: "+calculator_type,__FILE__, __LINE__); 


  if ( _prop_name == _calc->get_abskg_name()){ 
    std::ostringstream msg; 
    msg << "Error: The label defined for the radiation_property: " << _prop_name << " matches the gas-only absorption coefficient. " << std::endl << 
      "Please choose a different label for one or the other. " << std::endl;
    throw InvalidValue(msg.str(),__FILE__, __LINE__); 
  }


}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void RadProperties::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{


  std::string taskname = "RadProperties::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &RadProperties::computeProp, 
                           time_substep ); 

  _temperature_label = VarLabel::find(_temperature_name); 
  if ( _temperature_label == 0 ){ 
    throw ProblemSetupException("Error: Could not find the temperature label",__FILE__, __LINE__);
  }
  tsk->requires( Task::NewDW, VarLabel::find("volFraction"), Ghost::None, 0 ); 


  if ( time_substep == 0 ){ 

    tsk->modifies( _prop_label ); 
    tsk->computes( _calc->get_abskg_label() );
    tsk->requires( Task::OldDW, VarLabel::find(_temperature_name), Ghost::None, 0);

    if ( _particlesOn ){ 
      tsk->computes( _ocalc->get_abskp_label() ); 
      for( int i=0; i< _nQn_part; i++){
        tsk->computes( _ocalc->get_abskp_label_vector()[i] ); 
      }
      if ( _scatteringOn  ){
        tsk->computes( _ocalc->get_scatkt_label() ); 
      }
    } 



    //participating species from property calculator
    std::vector<std::string> part_sp = _calc->get_sp(); 

    for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){
      const VarLabel* label = VarLabel::find(*iter);
      if ( label != 0 ){ 
        tsk->requires(Task::OldDW, label, Ghost::None, 0 ); 
      } else { 
        throw ProblemSetupException("Error: Could not match species with varlabel: "+*iter,__FILE__, __LINE__);
      }
    }

  } else { 

    tsk->modifies( _prop_label ); 
    tsk->modifies( _calc->get_abskg_label() );
    tsk->requires( Task::NewDW, VarLabel::find(_temperature_name), Ghost::None, 0);

    if ( _particlesOn ){ 
      tsk->modifies( _ocalc->get_abskp_label() ); 
      for( int i=0; i< _nQn_part; i++){
        tsk->modifies( _ocalc->get_abskp_label_vector()[i] ); 
      }
      if ( _scatteringOn  ){ 
        tsk->modifies( _ocalc->get_scatkt_label() ); 
      }
    } 
      
    

    //participating species from property calculator
    std::vector<std::string> part_sp = _calc->get_sp(); 

    for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){
      const VarLabel* label = VarLabel::find(*iter);
      if ( label != 0 ){ 
        tsk->requires( Task::NewDW, label, Ghost::None, 0 ); 
      } else { 
        throw ProblemSetupException("Error: Could not match species with varlabel: "+*iter,__FILE__, __LINE__);
      }
    }
  }
 
  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 


  // Require DQMOM labels if needed 
  if (  _particlesOn){
    if( _nQn_part ==0){
      throw ProblemSetupException("Error: DQMOM must be used in combination with radiation properties for particles. Zero quadrature nodes found." ,__FILE__, __LINE__);
    }
    for ( int i = 0; i < _nQn_part; i++ ){
      std::string label_name_s = _base_size_label_name + "_qn"; 
      std::string label_name_t = _base_temperature_label_name + "_qn"; 
      std::string label_name_w =   "w_qn"; 
      std::stringstream out; 
      out << i; 
      label_name_s += out.str();  // size
      label_name_t += out.str();  // temperature
      label_name_w += out.str();  // weight


      tsk->requires( Task::OldDW, VarLabel::find( label_name_s ) , Ghost::None, 0 ); 
      tsk->requires( Task::OldDW, VarLabel::find( label_name_t ) , Ghost::None, 0 ); 
      tsk->requires( Task::OldDW, VarLabel::find( label_name_w ) , Ghost::None, 0 ); 
      tsk->requires( Task::NewDW, VarLabel::find( label_name_s ) , Ghost::None, 0 ); 
      tsk->requires( Task::NewDW, VarLabel::find( label_name_t ) , Ghost::None, 0 ); 
      tsk->requires( Task::NewDW, VarLabel::find( label_name_w ) , Ghost::None, 0 ); 

    }

    if (_particlesOn )  
      _ocalc->problemSetup(tsk, time_substep);
  }
}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void RadProperties::computeProp(const ProcessorGroup* pc, 
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
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 


    CCVariable<double> absk_tot; 
    //get other variables
    constCCVariable<double> vol_fraction; 
    constCCVariable<double> temperature; 

    new_dw->get( vol_fraction, VarLabel::find("volFraction"), matlIndex, patch, Ghost::None, 0 ); 

    DataWarehouse* which_dw; 
    

    CCVariable<double> abskg; 
    CCVariable<double> abskpt; 
    CCVariable<double> scatkt; 
    StaticArray< CCVariable<double> > abskp(_nQn_part); 

    if ( time_substep == 0 ) { 
      which_dw = old_dw; 
      new_dw->getModifiable( absk_tot, _prop_label, matlIndex, patch ); 
      new_dw->allocateAndPut( abskg, _calc->get_abskg_label(), matlIndex, patch );
      if ( _particlesOn ){ 
        new_dw->allocateAndPut( abskpt, _ocalc->get_abskp_label(), matlIndex, patch );
        for( int i=0; i< _nQn_part; i++){
          new_dw->allocateAndPut( abskp[i], _ocalc->get_abskp_label_vector()[i], matlIndex, patch );
        }
      } 
      old_dw->get( temperature, _temperature_label, matlIndex, patch, Ghost::None, 0 ); 
    } else { 
      which_dw = new_dw; 
      new_dw->getModifiable( absk_tot, _prop_label, matlIndex, patch ); 
      new_dw->getModifiable( abskg, _calc->get_abskg_label(), matlIndex, patch );
      if ( _particlesOn ){ 
        new_dw->getModifiable( abskpt, _ocalc->get_abskp_label(), matlIndex, patch );
        for( int i=0; i< _nQn_part; i++){
          new_dw->getModifiable( abskp[i], _ocalc->get_abskp_label_vector()[i], matlIndex, patch );
        }
      }  
      new_dw->get( temperature, _temperature_label, matlIndex, patch, Ghost::None, 0 ); 
    }


      
    //participating species from property calculator
    typedef std::vector<constCCVariable<double> > CCCV; 
    CCCV species; 
    std::vector<std::string> part_sp = _calc->get_sp(); 

    for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){
      const VarLabel* label = VarLabel::find(*iter);
      constCCVariable<double> spec; 
      which_dw->get( spec, label, matlIndex, patch, Ghost::None, 0 ); 
      species.push_back(spec); 
    }

    //initializing properties here.  This needs to be made consistent with BCs
    if ( time_substep == 0 ){ 
      abskg.initialize(1.0);           //walls, bcs, etc, are fulling absorbing 
      if ( _particlesOn ){
        abskpt.initialize(0.0); 
      }
    }

    //actually compute the properties
    _calc->compute_abskg( patch, vol_fraction, species, temperature, abskg ); 

    //copy the gas portion to the total: 
    absk_tot.copyData(abskg); 

    //sum in the particle contribution if needed

      if ( _particlesOn){ 
        // Create containers to be passed to function that populates abskp
        DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self(); // DQMOM singleton object
        CCCV pWeight;      // particle weights
        CCCV pSize;        // particle sizes
        CCCV pTemperature; // particle Temperatures
        typedef std::vector<const VarLabel*> CCCVL; // object used for iterating over quadrature nodes


        // Get labels and scaling constants for DQMOM size, temperature and weights
        std::vector<const VarLabel*> s_varlabels;     // DQMOM size label
        std::vector<const VarLabel*> w_varlabels;     // DQMOM weight label
        std::vector<const VarLabel*> t_varlabels;     // DQMOM Temperature label
        s_varlabels.resize(0);
        w_varlabels.resize(0);
        t_varlabels.resize(0);
        double s_scaling_constant; // scaling constant for sizes
        double w_scaling_constant; // scaling constant for weights

        for ( int i = 0; i < _nQn_part; i++ ){
          std::string label_name_s = _base_size_label_name + "_qn"; 
          std::string label_name_t = _base_temperature_label_name + "_qn"; 
          std::string label_name_w =   "w_qn"; 
          std::stringstream out; 
          out << i; 
          label_name_s += out.str(); 
          label_name_t += out.str(); 
          label_name_w += out.str(); 
          if (i == 0){
            s_scaling_constant = dqmom_eqn_factory.retrieve_scalar_eqn(label_name_s).getScalingConstant();
            w_scaling_constant = dqmom_eqn_factory.retrieve_scalar_eqn(label_name_w).getScalingConstant();
          }
          s_varlabels.push_back(VarLabel::find( label_name_s ));
          t_varlabels.push_back(VarLabel::find( label_name_t ) );
          w_varlabels.push_back(VarLabel::find( label_name_w ) );
        }

        ////size
        for ( CCCVL::iterator iterx = s_varlabels.begin(); iterx != s_varlabels.end(); iterx++ ){ 
          constCCVariable<double> var; 
          which_dw->get( var,*iterx, matlIndex, patch, Ghost::None, 0 ); 
          pSize.push_back( var ); 
        }
        /////--temperature
        for ( CCCVL::iterator iterx = t_varlabels.begin(); iterx != t_varlabels.end(); iterx++ ){ 
          constCCVariable<double> var; 
          which_dw->get( var, *iterx, matlIndex, patch, Ghost::None, 0 ); 
          pTemperature.push_back( var ); 
        } 

        //////--weight--
        for ( CCCVL::iterator iterx = w_varlabels.begin(); iterx != w_varlabels.end(); iterx++ ){ 
          constCCVariable<double> var; 
          which_dw->get( var, *iterx, matlIndex, patch, Ghost::None, 0 ); 
          pWeight.push_back( var ); 
        } 

        /////--Other required scalars needed to compute optical props
        std::vector< const VarLabel*> requiredLabels;
        requiredLabels =  _ocalc->getRequiresLabels();  
        StaticArray< constCCVariable<double> > RequiredScalars(requiredLabels.size());
        for (unsigned int i=0; i<requiredLabels.size(); i++){ // unsigned avoids compiler warning
          new_dw->get( RequiredScalars[i] , requiredLabels[i], matlIndex,patch, Ghost::None, 0);// This should be WhichDW, but I'm getting an error BEN??
        }
          ////--compute the complex index of refraction
          StaticArray<CCVariable<double> >complexIndexReal(_nQn_part);
          if(_ocalc->get_complexIndexBool()){
            if(time_substep==0) {
              for ( int i=0; i<_nQn_part; i++){ 
                new_dw->allocateAndPut(complexIndexReal[i], _ocalc->get_complexIndexReal_label()[i], matlIndex,patch);
                complexIndexReal[i].initialize(0.0);
              }
            }
            else{

              for ( int i=0; i<_nQn_part; i++){ 
                new_dw->getModifiable(complexIndexReal[i], _ocalc->get_complexIndexReal_label()[i], matlIndex,patch);
              }
            }
          }



          _ocalc->computeComplexIndex(patch, vol_fraction,RequiredScalars, complexIndexReal);

        _ocalc->compute_abskp( patch, vol_fraction, s_scaling_constant, pSize, pTemperature, 
            w_scaling_constant, pWeight, _nQn_part,  abskpt,abskp, complexIndexReal);

        _calc->sum_abs( absk_tot, abskpt, patch ); 

        if (_scatteringOn){
          //-----------------scattering props---------//
          StaticArray<CCVariable<double> >scatktQuad(_nQn_part);
          CCVariable<double> asymmetryParam;
          if(time_substep==0) {
            new_dw->allocateAndPut( scatkt, _ocalc->get_scatkt_label(), matlIndex, patch );
            new_dw->allocateAndPut(asymmetryParam  , _ocalc->get_asymmetryParam_label()  , matlIndex,patch);
            scatkt.initialize(0.0);  
            asymmetryParam.initialize(0.0);  
          }
          else{
            new_dw->getModifiable( scatkt, _ocalc->get_scatkt_label(), matlIndex, patch );
            new_dw->getModifiable(asymmetryParam  , _ocalc->get_asymmetryParam_label()  , matlIndex,patch);
          }
          _ocalc->compute_scatkt( patch, vol_fraction, s_scaling_constant, pSize, pTemperature, 
              w_scaling_constant, pWeight, _nQn_part,  scatkt, scatktQuad, complexIndexReal);

          _ocalc->computeAsymmetryFactor(patch,vol_fraction, scatktQuad, RequiredScalars, scatkt, asymmetryParam);

          _calc->sum_abs( absk_tot, scatkt, patch ); 
  //----------end of scattering props----------//
        }
      }

    // update absk_tot at the walls
    _boundaryCond->setScalarValueBC( pc, patch, absk_tot, _prop_name );
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void RadProperties::sched_initialize( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "RadProperties::initialize"; 

  Task* tsk = scinew Task(taskname, this, &RadProperties::initialize);

  tsk->computes(_prop_label);                     //the total 
  tsk->computes( _calc->get_abskg_label() );      //gas only

  if ( _particlesOn ){
    tsk->computes( _ocalc->get_abskp_label() );      //particle only
    for( int i=0; i< _nQn_part; i++){
      tsk->computes( _ocalc->get_abskp_label_vector()[i] ); 
    }
    if (_scatteringOn ){ 
      tsk->computes( _ocalc->get_scatkt_label() );      
      tsk->computes( _ocalc->get_asymmetryParam_label() );      
    }
  }



  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void RadProperties::initialize( const ProcessorGroup* pc, 
                                    const PatchSubset* patches, 
                                    const MaterialSubset* matls, 
                                    DataWarehouse* old_dw, 
                                    DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> prop; 
    CCVariable<double> abskg;

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 


    new_dw->allocateAndPut( abskg, _calc->get_abskg_label(), matlIndex, patch ); 
    prop.initialize(0.0); 
    abskg.initialize(0.0); 

    if ( _particlesOn ){ 
      CCVariable<double> abskpt;
      new_dw->allocateAndPut( abskpt, _ocalc->get_abskp_label(), matlIndex, patch ); 
      abskpt.initialize(0.0); 
      StaticArray< CCVariable<double> >abskp(_nQn_part);
      for( int i=0; i< _nQn_part; i++){
        new_dw->allocateAndPut( abskp[i], _ocalc->get_abskp_label_vector()[i], matlIndex, patch ); 
        abskp[i].initialize(0.0); 
      }
      if (_scatteringOn ){ 
        CCVariable<double> scatkt;
        CCVariable<double> asymmetryParam;
        new_dw->allocateAndPut( scatkt, _ocalc->get_scatkt_label(), matlIndex, patch ); 
        scatkt.initialize(0.0); 
        new_dw->allocateAndPut( asymmetryParam, _ocalc->get_asymmetryParam_label(), matlIndex, patch ); 
        asymmetryParam.initialize(0.0); 
      }
    }


    PropertyModelBase::base_initialize( patch, prop ); // generic initialization functionality 

    _boundaryCond->setScalarValueBC( pc, patch, prop, _prop_name );
  }
}
