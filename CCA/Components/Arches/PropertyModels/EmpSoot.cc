
    if ( new_dw->exists( _prop_label, matlIndex, patch ) ){

      new_dw->getModifiable( soot_vf, _prop_label, matlIndex, patch ); 
      new_dw->getModifiable( absorp_coef, _absorp_label, matlIndex, patch ); 

      new_dw->get( temperature, _T_label, matlIndex, patch, Ghost::None, 0 ); 
      new_dw->get( density, _den_label, matlIndex, patch, Ghost::None, 0 ); 

    } else {

      new_dw->allocateAndPut( soot_vf, _prop_label, matlIndex, patch ); 
      new_dw->allocateAndPut( absorp_coef, _absorp_label, matlIndex, patch ); 
      soot_vf.initialize(0.0); 
      absorp_coef.initialize(0.0); 

      old_dw->get( temperature, _T_label, matlIndex, patch, Ghost::None, 0 ); 
      old_dw->get( density, _T_label, matlIndex, patch, Ghost::None, 0 ); 

    }

    CellIterator iter = patch->getCellIterator(); 

    for (iter.begin(); !iter.done(); iter++){

      IntVector c = *iter; 

      double bc = _carb_content * density[c];


      if ( temperature[c] > 1000.0 ) { 
        soot_vf[c] = _scale_factor * ( _c1 * bc * _cmw ) / _rho_soot; 
      } else { 
        soot_vf[c] = 0.0;
      } 
      
      absorp_coef[c] = 0.01 + std::min( 0.5, (4.0/_opl)*log( 1.0 + 
                       350.0 * soot_vf[c] * temperature[c] * _opl));

    }
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Dummy Initialization
//---------------------------------------------------------------------------
void EmpSoot::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "EmpSoot::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &EmpSoot::dummyInit);
  tsk->computes(_prop_label); 
  tsk->requires( Task::OldDW, _prop_label, Ghost::None, 0 ); 

  tsk->computes(_absorp_label); 
  tsk->requires( Task::OldDW, _absorp_label, Ghost::None, 0 ); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}

//---------------------------------------------------------------------------
//Method: Actually do the Dummy Initialization
//---------------------------------------------------------------------------
void EmpSoot::dummyInit( const ProcessorGroup* pc, 
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
    constCCVariable<double> old_prop; 

    CCVariable<double> absorp_coef; 
    constCCVariable<double> old_absorp_coef; 

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    old_dw->get( old_prop, _prop_label, matlIndex, patch, Ghost::None, 0); 

    new_dw->allocateAndPut( absorp_coef, _absorp_label, matlIndex, patch ); 
    old_dw->get( old_absorp_coef, _absorp_label, matlIndex, patch, Ghost::None, 0); 

    prop.copyData( old_prop );
    absorp_coef.copyData( old_absorp_coef ); 

  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void EmpSoot::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "EmpSoot::initialize"; 

  _den_label    = VarLabel::find( _den_label_name );
  _T_label      = VarLabel::find( _T_label_name );
  _absorp_label = VarLabel::find( _absorp_label_name );

  Task* tsk = scinew Task(taskname, this, &EmpSoot::initialize);
  tsk->computes(_prop_label); 
  tsk->computes(_absorp_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void EmpSoot::initialize( const ProcessorGroup* pc, 
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
    CCVariable<double> absorp_coef; 

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    prop.initialize(0.0); 

    new_dw->allocateAndPut( absorp_coef, _absorp_label, matlIndex, patch ); 
    absorp_coef.initialize(0.0); 

    PropertyModelBase::base_initialize( patch, prop ); // generic initialization functionality 

  }
}
