#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
//#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>


//===========================================================================

using namespace Uintah;
using namespace std;

PartVel::PartVel(ArchesLabel* fieldLabels ) : 
d_fieldLabels(fieldLabels)
{
}

PartVel::~PartVel()
{

}
//---------------------------------------------------------------------------
// Method: ProblemSetup
//---------------------------------------------------------------------------
void PartVel::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb; 
  ProblemSpecP dqmom_db = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM");

  _uname = ParticleTools::parse_for_role_to_label(db, "uvel"); 
  _vname = ParticleTools::parse_for_role_to_label(db, "vvel"); 
  _wname = ParticleTools::parse_for_role_to_label(db, "wvel"); 

  std::string which_dqmom; 
  dqmom_db->getAttribute( "type", which_dqmom ); 

  ProblemSpecP vel_db = db->findBlock("VelModel");
  if (vel_db) {

    std::string model_type;
    vel_db->getAttribute("type", model_type);

    if(model_type == "Dragforce") {
      d_drag = true;
    } else {
      throw InvalidValue( "Invalid type for Velocity Model must be Dragforce",__FILE__,__LINE__);
    }
  } else {
    throw InvalidValue( "A <VelModel> section is missing from your input file!",__FILE__,__LINE__);
  }

}



//---------------------------------------------------------------------------
// Method: Schedule the initialization of the particle velocities
//---------------------------------------------------------------------------
void 
PartVel::schedInitPartVel( const LevelP& level, SchedulerP& sched )
{
 
  string taskname = "PartVel::InitPartVel";
  Task* tsk = new Task(taskname, this, &PartVel::InitPartVel);

  // actual velocity we will compute
  for (ArchesLabel::PartVelMap::iterator i = d_fieldLabels->partVel.begin(); 
        i != d_fieldLabels->partVel.end(); i++){
    tsk->computes( i->second );
  }


  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());


  // Not needed since velocities are always used from new_dw
}


//---------------------------------------------------------------------------
// Method: Schedule the calculation of the particle velocities
//---------------------------------------------------------------------------
void 
PartVel::schedComputePartVel( const LevelP& level, SchedulerP& sched, const int rkStep )
{
 
  string taskname = "PartVel::ComputePartVel";
  Task* tsk = new Task(taskname, this, &PartVel::ComputePartVel, rkStep);

  Ghost::GhostType gn = Ghost::None;

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
  int N = dqmomFactory.get_quad_nodes();  

  Task::WhichDW which_dw; 

  if ( rkStep == 0 ){ 
    which_dw = Task::OldDW; 
  } else { 
    which_dw = Task::NewDW; 
  }

  //--New
  // actual velocity we will compute
  for (ArchesLabel::PartVelMap::iterator i = d_fieldLabels->partVel.begin(); 
        i != d_fieldLabels->partVel.end(); i++){
    if (rkStep < 1 )
      tsk->computes( i->second );
    else 
      tsk->modifies( i->second );  
    // also get the old one
    tsk->requires( Task::OldDW, i->second, gn, 0);
  }


    tsk->requires( which_dw, d_fieldLabels->d_CCVelocityLabel, gn, 0 );
    
    //for (int i = 0; i < N; i++){
      //std::string name = "w_qn";
      //std::string node;
      //std::stringstream out;
      //out << i;
      //node = out.str();
      //name += node;

      //EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( name );
      //const VarLabel* myLabel = eqn.getTransportEqnLabel();

      //tsk->requires( which_dw, myLabel, gn, 0 );
    //}
    
    for (int i = 0; i < N; i++){

      //U
      std::string name = get_env_name( _uname, i ); 
      const VarLabel* ulabel = VarLabel::find(name); 
      tsk->requires( which_dw, ulabel, gn, 0 );
      //V
      name = get_env_name( _vname, i ); 
      const VarLabel* vlabel = VarLabel::find(name); 
      tsk->requires( which_dw, vlabel, gn, 0 );
      //W
      name = get_env_name( _wname, i ); 
      const VarLabel* wlabel = VarLabel::find(name); 
      tsk->requires( which_dw, wlabel, gn, 0 );

    }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Copy old data into a newly allocated new data spot
//---------------------------------------------------------------------------
void PartVel::ComputePartVel( const ProcessorGroup* pc, 
                              const PatchSubset* patches, 
                              const MaterialSubset* matls, 
                              DataWarehouse* old_dw, 
                              DataWarehouse* new_dw, 
                              const int rkStep )
{

  for (int p=0; p < patches->size(); p++){

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
    Ghost::GhostType  gn  = Ghost::None;
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    DataWarehouse* which_dw; 
    if ( rkStep == 0 ){ 
      which_dw = old_dw; 
    } else { 
      which_dw = new_dw; 
    }

    constCCVariable<Vector> gasVel;

    which_dw->get( gasVel, d_fieldLabels->d_CCVelocityLabel, matlIndex, patch, gn, 0 );

    int N = dqmomFactory.get_quad_nodes();
    for ( int iqn = 0; iqn < N; iqn++){

      //U
      std::string name;
      name = get_env_name( _uname, iqn ); 
      const VarLabel* ulabel = VarLabel::find(name); 
      constCCVariable<double> u; 
      which_dw->get(u, ulabel, matlIndex, patch, gn, 0); 
      //V
      name = get_env_name( _vname, iqn ); 
      const VarLabel* vlabel = VarLabel::find(name); 
      constCCVariable<double> v; 
      which_dw->get(v, vlabel, matlIndex, patch, gn, 0); 
      //W
      name = get_env_name( _wname, iqn ); 
      const VarLabel* wlabel = VarLabel::find(name); 
      constCCVariable<double> w; 
      which_dw->get(w, wlabel, matlIndex, patch, gn, 0); 

      ArchesLabel::PartVelMap::iterator iter = d_fieldLabels->partVel.find(iqn);

      CCVariable<Vector> partVel;
      if (rkStep == 0){
        new_dw->allocateAndPut( partVel, iter->second, matlIndex, patch );
        partVel.initialize(Vector(0.,0.,0.));
      } else { 
        new_dw->getModifiable( partVel, iter->second, matlIndex, patch );
      }


      // now loop over all cells
      for (CellIterator iter=patch->getExtraCellIterator(0); !iter.done(); iter++){

        IntVector c = *iter;
        partVel[c] = Vector(u[c],v[c],w[c]);

      }
    } 
  } 
} // end ComputePartVel()

//---------------------------------------------------------------------------
// Method: Initialized partvel 
//---------------------------------------------------------------------------
void PartVel::InitPartVel( const ProcessorGroup* pc, 
                              const PatchSubset* patches, 
                              const MaterialSubset* matls, 
                              DataWarehouse* old_dw, 
                              DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++){

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    int N = dqmomFactory.get_quad_nodes();
    for ( int iqn = 0; iqn < N; iqn++){

      ArchesLabel::PartVelMap::iterator iter = d_fieldLabels->partVel.find(iqn);

      CCVariable<Vector> partVel;
      new_dw->allocateAndPut( partVel, iter->second, matlIndex, patch );

      partVel.initialize(Vector(0.,0.,0.));
    }
  }  
} // end ComputePartVel()
