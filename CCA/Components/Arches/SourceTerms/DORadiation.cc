#include <CCA/Components/Arches/SourceTerms/DORadiation.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>

using namespace std;
using namespace Uintah; 

DORadiation::DORadiation( std::string src_name, ArchesLabel* labels, MPMArchesLabel* MAlab,
                          BoundaryCondition* bc, 
                          vector<std::string> req_label_names, const ProcessorGroup* my_world, 
                          std::string type ) 
: SourceTermBase( src_name, labels->d_sharedState, req_label_names, type ), 
  _labels( labels ),
  _MAlab(MAlab), 
  _bc(bc), 
  _my_world(my_world)
{

  // NOTE: This boundary condition here is bogus.  Passing it for 
  // now until the boundary condition reference can be stripped out of 
  // the radiation model. 
  
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();

  _src_label = VarLabel::create( src_name, CC_double ); 

  // Add any other local variables here. 
  _radiationFluxELabel = VarLabel::create("radiationFluxE",  CC_double);
  _extra_local_labels.push_back(_radiationFluxELabel); 

  _radiationFluxWLabel = VarLabel::create("radiationFluxW",  CC_double);
  _extra_local_labels.push_back(_radiationFluxWLabel); 

  _radiationFluxNLabel = VarLabel::create("radiationFluxN",  CC_double);
  _extra_local_labels.push_back(_radiationFluxNLabel); 

  _radiationFluxSLabel = VarLabel::create("radiationFluxS",  CC_double);
  _extra_local_labels.push_back(_radiationFluxSLabel); 

  _radiationFluxTLabel = VarLabel::create("radiationFluxT",  CC_double);
  _extra_local_labels.push_back(_radiationFluxTLabel); 

  _radiationFluxBLabel = VarLabel::create("radiationFluxB",  CC_double);
  _extra_local_labels.push_back(_radiationFluxBLabel); 

  _radiationVolqLabel = VarLabel::create("radiationVolq",  CC_double);
  _extra_local_labels.push_back(_radiationVolqLabel); 

  //Declare the source type: 
  _source_grid_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

  _prop_calculator = 0;
  _using_prop_calculator = 0; 
  _DO_model = 0; 

}

DORadiation::~DORadiation()
{
  
  // source label is destroyed in the base class 

  for (vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
       iter != _extra_local_labels.end(); iter++) { 

    VarLabel::destroy( *iter ); 

  }

  delete _prop_calculator; 
  delete _DO_model; 

}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
DORadiation::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->getWithDefault( "calc_frequency",   _radiation_calc_freq, 3 ); 
  db->getWithDefault( "calc_on_all_RKsteps", _all_rk, false ); 
  db->getWithDefault( "co2_label", _co2_label_name, "CO2" ); 
  db->getWithDefault( "h2o_label", _h2o_label_name, "H2O" ); 
  db->getWithDefault( "T_label", _T_label_name, "temperature" ); 
  db->require( "soot_label",  _soot_label_name ); 
  db->getWithDefault( "psize_label", _size_label_name, "length");
  db->getWithDefault( "ptemperature_label", _pT_label_name, "heat_pT"); 

  //get the number of quadrature nodes and store it locally 
  _nQn_part = 0;
  if ( db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM") ){
    db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->require( "number_quad_nodes", _nQn_part ); 
  }
  
  proc0cout << " --- DO Radiation Model Summary: --- " << endl;
  proc0cout << "   -> calculation frequency:     " << _radiation_calc_freq << endl;
  proc0cout << "   -> co2 label name:            " << _co2_label_name << endl; 
  proc0cout << "   -> h20 label name:            " << _h2o_label_name << endl;
  proc0cout << "   -> T label name:              " << _T_label_name << endl;
  proc0cout << "   -> soot label name:           " << _soot_label_name << endl;
  proc0cout << " --- end DO Radiation Summary ------ " << endl;

  _DO_model = scinew DORadiationModel( _labels, _MAlab, _bc, _my_world ); 
  _DO_model->problemSetup( db ); 

  _prop_calculator = scinew RadPropertyCalculator(); 
  ProblemSpecP db_DORad = db->findBlock("DORadiationModel");
  _using_prop_calculator = _prop_calculator->problemSetup( db_DORad ); 

  if ( !_using_prop_calculator ){ 
    throw ProblemSetupException("Error: No valid property calculator found.",__FILE__, __LINE__);
  }

  _labels->add_species( _co2_label_name ); 
  _labels->add_species( _h2o_label_name ); 
  _labels->add_species( _T_label_name ); 

}
void 
DORadiation::extraSetup( GridP& grid, const ProblemSpecP& inputdb)
{

  _prop_calculator->extraProblemSetup( inputdb );  

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
DORadiation::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "DORadiation::computeSource";
  Task* tsk = scinew Task(taskname, this, &DORadiation::computeSource, timeSubStep);

  _perproc_patches = level->eachPatch(); 

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;

  _co2_label = VarLabel::find( _co2_label_name ); 
  _h2o_label = VarLabel::find( _h2o_label_name ); 
  _T_label   = VarLabel::find( _T_label_name ); 
  _soot_label = VarLabel::find( _soot_label_name ); 

  tsk->requires( Task::OldDW, _src_label, gn, 0 );
  
  if (timeSubStep == 0) { 

    tsk->computes(_src_label);

    std::vector<std::string> part_sp = _prop_calculator->get_participating_sp(); //participating species from property calculator
    _species_varlabels.resize(0);
    _size_varlabels.resize(0);
    _w_varlabels.resize(0);
    _T_varlabels.resize(0);

    for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){

      const VarLabel* label = VarLabel::find(*iter);
      _species_varlabels.push_back(label); 

      if ( label != 0 ){ 
        tsk->requires( Task::OldDW, label, gn, 0 ); 
      } else { 
        throw ProblemSetupException("Error: Could not match species with varlabel: "+*iter,__FILE__, __LINE__);
      }
    }

    for ( int i = 0; i < _nQn_part; i++ ){ 

      //--size--
      std::string label_name = _size_label_name + "_qn"; 
      std::stringstream out; 
      out << i; 
      label_name += out.str(); 

      const VarLabel* sizelabel = VarLabel::find( label_name ); 
      _size_varlabels.push_back( sizelabel ); 

      if ( sizelabel != 0 ){ 
        tsk->requires( Task::OldDW, sizelabel, gn, 0 ); 
      } else { 
        throw ProblemSetupException("Error: Could not find particle size quadrature node: " + label_name, __FILE__, __LINE__);
      }

      //--temperature--
      label_name = _pT_label_name + "_qn"; 
      label_name += out.str(); 

      const VarLabel* tlabel = VarLabel::find( label_name ); 
      _T_varlabels.push_back( tlabel ); 

      if ( tlabel != 0 ){ 
        tsk->requires( Task::OldDW, tlabel, gn, 0 ); 
      } else { 
        throw ProblemSetupException("Error: Could not find particle temperature quadrature node: " + label_name , __FILE__, __LINE__);
      }

      //--weight--
      label_name = "w_qn"+out.str(); 
      const VarLabel* wlabel = VarLabel::find( label_name ); 
      _w_varlabels.push_back( wlabel ); 

      if ( wlabel != 0 ){ 
        tsk->requires( Task::OldDW, wlabel, gn, 0 ); 
      } else { 
        throw ProblemSetupException("Error: Could not find particle weight quadrature node: w_qn"+out.str() , __FILE__, __LINE__);
      }
    } 

    tsk->requires( Task::OldDW, _T_label, gac, 1 ); 
    tsk->requires( Task::OldDW, _labels->d_volFractionLabel, gac, 1);

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
         iter != _extra_local_labels.end(); iter++){

      tsk->requires( Task::OldDW, *iter, gn, 0 ); 
      tsk->computes( *iter ); 

    }

    //properties: 
    tsk->computes( _prop_calculator->get_abskg_label() ); 
    tsk->requires( Task::OldDW, _prop_calculator->get_abskg_label(), gn, 0 ); 

    if ( _prop_calculator->has_abskp_local() ){
      tsk->computes( _prop_calculator->get_abskp_label() ); //derek
    } else { 
      tsk->requires( Task::OldDW, _prop_calculator->get_abskp_label(), gn, 0 ); 
    }

  } else {

    tsk->modifies(_src_label); 

    for ( std::vector<const VarLabel*>::iterator iter = _species_varlabels.begin();  iter != _species_varlabels.end(); iter++ ){ 
      tsk->requires( Task::NewDW, *iter, gn, 0 ); 
    } 

    for ( std::vector<const VarLabel*>::iterator iter = _size_varlabels.begin(); iter != _size_varlabels.end(); iter++) { 
      tsk->requires( Task::NewDW, *iter, gn, 0 ); 
    } 

    for ( std::vector<const VarLabel*>::iterator iter = _w_varlabels.begin(); iter != _w_varlabels.end(); iter++) { 
      tsk->requires( Task::NewDW, *iter, gn, 0 ); 
    } 

    for ( std::vector<const VarLabel*>::iterator iter = _T_varlabels.begin(); iter != _T_varlabels.end(); iter++) { 
      tsk->requires( Task::NewDW, *iter, gn, 0 ); 
    } 

    tsk->requires( Task::NewDW, _T_label, gac, 1 ); 
    tsk->requires( Task::NewDW, _labels->d_volFractionLabel, gac, 1 ); 

    for ( int i = 0; i < _nQn_part; i++ ){ 

      //--size--
      tsk->requires( Task::NewDW, _size_varlabels[i], gn, 0 ); 

      //--temperature--
      tsk->requires( Task::NewDW, _T_varlabels[i], gn, 0 ); 

      //--weight--
      tsk->requires( Task::NewDW, _w_varlabels[i], gn, 0 ); 

    } 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
         iter != _extra_local_labels.end(); iter++){

      tsk->modifies( *iter ); 

    }

    tsk->modifies( _prop_calculator->get_abskg_label() ); 

    if ( _prop_calculator->has_abskp_local() ){
      tsk->modifies( _prop_calculator->get_abskp_label() ); //derek
    } else { 
      tsk->requires( Task::NewDW, _prop_calculator->get_abskp_label(), gn, 0 ); 
    }

  }

  tsk->requires(Task::OldDW, _labels->d_cellTypeLabel, gac, 1 ); 
  tsk->requires(Task::NewDW, _labels->d_cellInfoLabel, gn);

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
DORadiation::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{
  _DO_model->d_linearSolver->matrixCreate( _perproc_patches, patches );

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int timestep = _labels->d_sharedState->getCurrentTopLevelTimeStep(); 

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, _labels->d_cellInfoLabel, matlIndex, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    CCVariable<double> divQ; 
    CCVariable<double> abskp_nonconst; 
    constCCVariable<double> abskp_const; 

    bool do_radiation = false; 
    if ( timestep%_radiation_calc_freq == 0 ) { 
      if ( _all_rk ) { 
        do_radiation = true; 
      } else if ( timeSubStep == 0 && !_all_rk ) { 
        do_radiation = true; 
      } 
    } 

    ArchesVariables radiation_vars; 
    ArchesConstVariables const_radiation_vars;
  //  Ghost::GhostType  gn = Ghost::None; // Not needed - Derek?
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None; 
    constCCVariable<double> mixT;
    constCCVariable<double> VolFractionBC;
    typedef std::vector<constCCVariable<double> > CCCV; 
    typedef std::vector<const VarLabel*> CCCVL; 

    CCCV species; 
    CCCV weights; 
    CCCV size;
    CCCV pT; 

    double weights_scaling_constant=1.0;
    double size_scaling_constant=1.0;

    DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
    string tlabelname;

    if ( timeSubStep == 0 ) { 

      //--species--
      for ( CCCVL::iterator iter = _species_varlabels.begin();  iter != _species_varlabels.end(); iter++ ){ 
        constCCVariable<double> var; 
        // std::cout<<"species_label="<<*iter<<", "; 
        old_dw->get( var, *iter, matlIndex, patch, gn, 0 ); 
        species.push_back( var ); 
      }
      // std::cout<<"species.size="<<species.size()<<endl; 

      //--size--
      for ( CCCVL::iterator iter = _size_varlabels.begin(); iter != _size_varlabels.end(); iter++ ){ 
        constCCVariable<double> var; 
        old_dw->get( var, *iter, matlIndex, patch, gn, 0 ); 
        size.push_back( var ); 
        //to get size scaling constant
        if(iter == _size_varlabels.begin()){
          tlabelname = (*iter)->getName();
          size_scaling_constant = dqmom_eqn_factory.retrieve_scalar_eqn(tlabelname).getScalingConstant();
        } 
      }

      //--temperature--
      for ( CCCVL::iterator iter = _T_varlabels.begin(); iter != _T_varlabels.end(); iter++ ){ 
        constCCVariable<double> var; 
        old_dw->get( var, *iter, matlIndex, patch, gn, 0 ); 
        pT.push_back( var ); 
      } 

      //--weight--
      for ( CCCVL::iterator iter = _w_varlabels.begin(); iter != _w_varlabels.end(); iter++ ){ 
        constCCVariable<double> var; 
        old_dw->get( var, *iter, matlIndex, patch, gn, 0 ); 
        weights.push_back( var ); 
        //to get weight scaling constant
        if(iter == _w_varlabels.begin()){
          tlabelname = (*iter)->getName();
          weights_scaling_constant = dqmom_eqn_factory.retrieve_scalar_eqn(tlabelname).getScalingConstant();
        } 
      } 

      old_dw->getCopy( radiation_vars.temperature, _T_label, matlIndex , patch , gac , 1 );
      old_dw->get( mixT, _T_label, matlIndex , patch , gac , 1 );
      old_dw->get( VolFractionBC, _labels->d_volFractionLabel , matlIndex , patch , gac , 1 );

      new_dw->allocateAndPut( radiation_vars.qfluxe , _radiationFluxELabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxw , _radiationFluxWLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxn , _radiationFluxNLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxs , _radiationFluxSLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxt , _radiationFluxTLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxb , _radiationFluxBLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.volq   , _radiationVolqLabel                 , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.ABSKG  , _prop_calculator->get_abskg_label() , matlIndex , patch );

      if ( _prop_calculator->has_abskp_local() ){ 
        new_dw->allocateAndPut( abskp_nonconst, _prop_calculator->get_abskp_label(), matlIndex, patch ); 
      } else { 
        old_dw->get( abskp_const, _prop_calculator->get_abskp_label(), matlIndex, patch, gn, 0 ); 
      }

      new_dw->allocateAndPut( divQ, _src_label, matlIndex, patch ); 
      old_dw->copyOut( divQ, _src_label, matlIndex, patch, gn, 0 ); 

      radiation_vars.ESRCG.allocate( patch->getExtraCellLowIndex(1), patch->getExtraCellHighIndex(1) );  
      radiation_vars.ESRCG.initialize(0.0); 

      // copy old solution into newly allocated variable
      old_dw->copyOut( radiation_vars.qfluxe , _radiationFluxELabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxw , _radiationFluxWLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxn , _radiationFluxNLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxs , _radiationFluxSLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxt , _radiationFluxTLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxb , _radiationFluxBLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.volq   , _radiationVolqLabel                 , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.ABSKG  , _prop_calculator->get_abskg_label() , matlIndex , patch , gn , 0 );

    } else { 

      //--species--
      for ( std::vector<const VarLabel*>::iterator iter = _species_varlabels.begin();  iter != _species_varlabels.end(); iter++ ){ 
        constCCVariable<double> var; 
        new_dw->get( var, *iter, matlIndex, patch, gn, 0 ); 
        species.push_back( var ); 
      }

      //--size--
      for ( CCCVL::iterator iter = _size_varlabels.begin(); iter != _size_varlabels.end(); iter++ ){ 
        constCCVariable<double> var; 
        new_dw->get( var, *iter, matlIndex, patch, gn, 0 ); 
        size.push_back( var ); 
      } 

      //--temperature--
      for ( CCCVL::iterator iter = _T_varlabels.begin(); iter != _T_varlabels.end(); iter++ ){ 
        constCCVariable<double> var; 
        new_dw->get( var, *iter, matlIndex, patch, gn, 0 ); 
        pT.push_back( var ); 
      } 

      //--weight--
      for ( CCCVL::iterator iter = _w_varlabels.begin(); iter != _w_varlabels.end(); iter++ ){ 
        constCCVariable<double> var; 
        new_dw->get( var, *iter, matlIndex, patch, gn, 0 ); 
        weights.push_back( var ); 
      } 

      new_dw->getCopy( radiation_vars.temperature, _T_label, matlIndex , patch , gac , 1 );
      new_dw->get( mixT, _T_label, matlIndex , patch , gac , 1 );
      new_dw->get( VolFractionBC, _labels->d_volFractionLabel, matlIndex , patch , gac , 1 );

      new_dw->getModifiable( radiation_vars.qfluxe , _radiationFluxELabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxw , _radiationFluxWLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxn , _radiationFluxNLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxs , _radiationFluxSLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxt , _radiationFluxTLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxb , _radiationFluxBLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.volq   , _radiationVolqLabel  , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.ABSKG  , _prop_calculator->get_abskg_label(), matlIndex , patch );
      if ( _prop_calculator->has_abskp_local() ){ 
        new_dw->getModifiable( abskp_nonconst, _prop_calculator->get_abskp_label(), matlIndex, patch ); 
      } else { 
        old_dw->get( abskp_const, _prop_calculator->get_abskp_label(), matlIndex, patch, gn, 0 ); 
      }

      new_dw->getModifiable( divQ, _src_label, matlIndex, patch ); 

      radiation_vars.ESRCG.allocate( patch->getExtraCellLowIndex(1), patch->getExtraCellHighIndex(1) );  
      radiation_vars.ESRCG.initialize(0.0); 

    } 

    old_dw->get( const_radiation_vars.cellType , _labels->d_cellTypeLabel, matlIndex, patch, gac, 1 ); 

    if ( do_radiation ){ 

      if ( timeSubStep == 0 ) {

        if ( _prop_calculator->does_scattering() ){ 

          //the property model is computing a abskp//
          _prop_calculator->compute( patch, VolFractionBC, species, size_scaling_constant, size, pT, 
                                     weights_scaling_constant, weights, _nQn_part, mixT, 
                                     radiation_vars.ABSKG, abskp_nonconst ); 

          _prop_calculator->sum_abs( radiation_vars.ABSKG, abskp_nonconst, patch ); 

        } else { 

          //abskp is computed elsewhere 
          _prop_calculator->compute( patch, VolFractionBC, species, mixT, radiation_vars.ABSKG);

          _prop_calculator->sum_abs( radiation_vars.ABSKG, abskp_const, patch ); 

        }

        //blackbody emissive flux
        for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){ 

          IntVector c = *iter;
          radiation_vars.ESRCG[c] = 1.0*5.67e-8/M_PI*radiation_vars.ABSKG[c]*pow(mixT[c],4);

        }

        _DO_model->boundarycondition( pc, patch, cellinfo, &radiation_vars, &const_radiation_vars ); 

        //Note: The final divQ is initialized (to zero) and set after the solve in the intensity solve itself.
        _DO_model->intensitysolve( pc, patch, cellinfo, &radiation_vars, &const_radiation_vars, divQ, BoundaryCondition::WALL ); 

      }
    }
  } // end patch loop
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
DORadiation::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DORadiation::initialize"; 
  Task* tsk = scinew Task(taskname, this, &DORadiation::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
       iter != _extra_local_labels.end(); iter++){

    tsk->computes(*iter); 
  }

  tsk->computes(_prop_calculator->get_abskg_label()); 

  if ( _prop_calculator->does_scattering()){ 
    tsk->computes( _prop_calculator->get_abskp_label()); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
DORadiation::initialize( const ProcessorGroup* pc, 
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

    CCVariable<double> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
         iter != _extra_local_labels.end(); iter++){

      CCVariable<double> temp_var; 
      new_dw->allocateAndPut(temp_var, *iter, matlIndex, patch ); 
      temp_var.initialize(0.0);
      
    }

    CCVariable<double> abskg; 
    new_dw->allocateAndPut( abskg, _prop_calculator->get_abskg_label(), matlIndex, patch ); 

    if ( _prop_calculator->does_scattering()){ 
      CCVariable<double> abskp; 
      new_dw->allocateAndPut( abskp, _prop_calculator->get_abskp_label(), matlIndex, patch ); 
      abskp.initialize(0.0); 
    }

    abskg.initialize(0.0); 

  }
}

