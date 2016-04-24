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
  for ( std::map<int, const VarLabel*>::iterator i = _face_x_part_vel_labels.begin();
        i != _face_x_part_vel_labels.end(); i++)
  {
    VarLabel::destroy( i->second );
  }
  for ( std::map<int, const VarLabel*>::iterator i = _face_y_part_vel_labels.begin();
        i != _face_y_part_vel_labels.end(); i++)
  {
    VarLabel::destroy( i->second );
  }
  for ( std::map<int, const VarLabel*>::iterator i = _face_z_part_vel_labels.begin();
        i != _face_z_part_vel_labels.end(); i++)
  {
    VarLabel::destroy( i->second );
  }
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

  const int N = ParticleTools::get_num_env( db, ParticleTools::DQMOM );
  for ( int i = 0; i < N; i++ ){

    //x,y,z face velocities
    std::string name = ParticleTools::append_env("face_pvel_x",i);
    const VarLabel* label = VarLabel::create(name, SFCXVariable<double>::getTypeDescription() );
    _face_x_part_vel_labels[i] = label;
    name = ParticleTools::append_env("face_pvel_y",i);
    label = VarLabel::create(name, SFCYVariable<double>::getTypeDescription() );
    _face_y_part_vel_labels[i] = label;
    name = ParticleTools::append_env("face_pvel_z",i);
    label = VarLabel::create(name, SFCZVariable<double>::getTypeDescription() );
    _face_z_part_vel_labels[i] = label;

  }

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
  Task* tsk = scinew Task(taskname, this, &PartVel::InitPartVel);

  // actual velocity we will compute
  for (ArchesLabel::PartVelMap::iterator i = d_fieldLabels->partVel.begin();
        i != d_fieldLabels->partVel.end(); i++){
    tsk->computes( i->second );
  }

  for ( std::map<int, const VarLabel*>::iterator i = _face_x_part_vel_labels.begin();
        i != _face_x_part_vel_labels.end(); i++)
  {
    tsk->computes( i->second );
  }
  for ( std::map<int, const VarLabel*>::iterator i = _face_y_part_vel_labels.begin();
        i != _face_y_part_vel_labels.end(); i++)
  {
    tsk->computes( i->second );
  }
  for ( std::map<int, const VarLabel*>::iterator i = _face_z_part_vel_labels.begin();
        i != _face_z_part_vel_labels.end(); i++)
  {
    tsk->computes( i->second );
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());

  // Not needed since velocities are always used from new_dw
}

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

      //initialize the face velocities
      SFCXVariable<double> _x_vel;
      new_dw->allocateAndPut( _x_vel, _face_x_part_vel_labels[iqn], matlIndex, patch );
      _x_vel.initialize(0.0);
      SFCYVariable<double> _y_vel;
      new_dw->allocateAndPut( _y_vel, _face_y_part_vel_labels[iqn], matlIndex, patch );
      _y_vel.initialize(0.0);
      SFCZVariable<double> _z_vel;
      new_dw->allocateAndPut( _z_vel, _face_z_part_vel_labels[iqn], matlIndex, patch );
      _z_vel.initialize(0.0);

    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the particle velocities
//---------------------------------------------------------------------------
void
PartVel::schedComputePartVel( const LevelP& level, SchedulerP& sched, const int rkStep )
{

  string taskname = "PartVel::ComputePartVel";
  Task* tsk = scinew Task(taskname, this, &PartVel::ComputePartVel, rkStep);

  Ghost::GhostType gn = Ghost::None;
  Ghost::GhostType ga = Ghost::AroundCells;

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
    tsk->requires( which_dw, d_fieldLabels->d_areaFractionFXLabel, gn, 0 );
    tsk->requires( which_dw, d_fieldLabels->d_areaFractionFYLabel, gn, 0 );
    tsk->requires( which_dw, d_fieldLabels->d_areaFractionFZLabel, gn, 0 );

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

      if ( rkStep == 0 ){
        tsk->computes( _face_x_part_vel_labels[i]);
        tsk->computes( _face_y_part_vel_labels[i]);
        tsk->computes( _face_z_part_vel_labels[i]);
      } else {
        tsk->modifies( _face_x_part_vel_labels[i]);
        tsk->modifies( _face_y_part_vel_labels[i]);
        tsk->modifies( _face_z_part_vel_labels[i]);
      }

      std::string name = ParticleTools::append_env( _uname, i );
      const VarLabel* ulabel = VarLabel::find(name);
      tsk->requires( which_dw, ulabel, ga, 1 );

      name = ParticleTools::ParticleTools::append_env( _vname, i );
      const VarLabel* vlabel = VarLabel::find(name);
      tsk->requires( which_dw, vlabel, ga, 1 );

      name = ParticleTools::append_env( _wname, i );
      const VarLabel* wlabel = VarLabel::find(name);
      tsk->requires( which_dw, wlabel, ga, 1 );

      name = ParticleTools::append_qn_env( "w", i );
      const VarLabel* weightlabel = VarLabel::find(name);
      tsk->requires( which_dw, weightlabel, ga, 1);

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
    Ghost::GhostType  ga = Ghost::AroundCells;
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
    constSFCXVariable<double> af_x;
    constSFCYVariable<double> af_y;
    constSFCZVariable<double> af_z;

    which_dw->get( gasVel, d_fieldLabels->d_CCVelocityLabel, matlIndex, patch, gn, 0 );
    which_dw->get( af_x, d_fieldLabels->d_areaFractionFXLabel, matlIndex, patch, gn, 0 );
    which_dw->get( af_y, d_fieldLabels->d_areaFractionFYLabel, matlIndex, patch, gn, 0 );
    which_dw->get( af_z, d_fieldLabels->d_areaFractionFZLabel, matlIndex, patch, gn, 0 );

    int N = dqmomFactory.get_quad_nodes();
    for ( int iqn = 0; iqn < N; iqn++){

      //U
      std::string name;
      name = ParticleTools::append_env( _uname, iqn );
      const VarLabel* ulabel = VarLabel::find(name);
      constCCVariable<double> cc_uvel;
      which_dw->get(cc_uvel, ulabel, matlIndex, patch, ga, 1);
      //V
      name = ParticleTools::append_env( _vname, iqn );
      const VarLabel* vlabel = VarLabel::find(name);
      constCCVariable<double> cc_vvel;
      which_dw->get(cc_vvel, vlabel, matlIndex, patch, ga, 1);
      //W
      name = ParticleTools::append_env( _wname, iqn );
      const VarLabel* wlabel = VarLabel::find(name);
      constCCVariable<double> cc_wvel;
      which_dw->get(cc_wvel, wlabel, matlIndex, patch, ga, 1);

      name = ParticleTools::append_qn_env( "w", iqn );
      const VarLabel* weightlabel = VarLabel::find(name);
      constCCVariable<double> weight;
      which_dw->get(weight, weightlabel, matlIndex, patch, ga, 1);

      ArchesLabel::PartVelMap::iterator iter = d_fieldLabels->partVel.find(iqn);

      CCVariable<Vector> partVel;
      if (rkStep == 0){
        new_dw->allocateAndPut( partVel, iter->second, matlIndex, patch );
        partVel.initialize(Vector(0.,0.,0.));
      } else {
        new_dw->getModifiable( partVel, iter->second, matlIndex, patch );
      }

      SFCXVariable<double> x_face_vel;
      SFCYVariable<double> y_face_vel;
      SFCZVariable<double> z_face_vel;
      if ( rkStep == 0  ){
        new_dw->allocateAndPut( x_face_vel, _face_x_part_vel_labels[iqn], matlIndex, patch );
        new_dw->allocateAndPut( y_face_vel, _face_y_part_vel_labels[iqn], matlIndex, patch );
        new_dw->allocateAndPut( z_face_vel, _face_z_part_vel_labels[iqn], matlIndex, patch );
        x_face_vel.initialize(0.0);
        y_face_vel.initialize(0.0);
        z_face_vel.initialize(0.0);
      } else {
        new_dw->getModifiable( x_face_vel, _face_x_part_vel_labels[iqn], matlIndex, patch );
        new_dw->getModifiable( y_face_vel, _face_y_part_vel_labels[iqn], matlIndex, patch );
        new_dw->getModifiable( z_face_vel, _face_z_part_vel_labels[iqn], matlIndex, patch );
      }

      bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
      bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
      bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

      IntVector patchLo = patch->getCellLowIndex();
      IntVector patchHi = patch->getCellHighIndex();

      partVel.initialize(Vector(0,0,0));

      // now loop over all cells
      for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){

        IntVector c = *iter;

        IntVector cs = *iter - IntVector(1,0,0);
        x_face_vel[c] = ( cc_uvel[c] + cc_uvel[cs] ) / 2.0 * af_x[c];

        cs = *iter - IntVector(0,1,0);
        y_face_vel[c] = ( cc_vvel[c] + cc_vvel[cs] ) / 2.0 * af_y[c];

        cs = *iter - IntVector(0,0,1);
        z_face_vel[c] = ( cc_wvel[c] + cc_wvel[cs] ) / 2.0 * af_z[c];

        //NOTE: Assuming that the scaling factor on the velocity = 1
        partVel[c] = Vector(cc_uvel[c],cc_vvel[c],cc_wvel[c]);

        if ( xplus ){
          if ( c[0] == patchHi[0]-1 ){

            IntVector cp = c + IntVector(1,0,0);
            x_face_vel[cp] = ( cc_uvel[c] + cc_uvel[cp] ) / 2.0 * af_x[cp];

          }
        }

        if ( yplus ){
          if ( c[1] == patchHi[1]-1 ){

            IntVector cp = c + IntVector(0,1,0);
            y_face_vel[cp] = ( cc_vvel[c] + cc_vvel[cp] ) / 2.0 * af_y[cp];

          }
        }

        if ( zplus ){
          if ( c[2] == patchHi[2]-1 ){

            IntVector cp = c + IntVector(0,0,1);
            z_face_vel[cp] = ( cc_wvel[c] + cc_wvel[cp] ) / 2.0 * af_z[cp];

          }
        }
      }
      //boundary conditions for partVel:
      vector<Patch::FaceType>::const_iterator bc_iter;
      vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);

      for (bc_iter = bf.begin(); bc_iter !=bf.end(); bc_iter++){
        Patch::FaceType face = *bc_iter;
        Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
        CellIterator fiter=patch->getFaceIterator(face, MEC);
        IntVector f_dir = patch->getFaceDirection(face);

        for(; !fiter.done(); fiter++){
          IntVector c = *fiter;
          IntVector ci = c - f_dir;
          double bc_u = (cc_uvel[c] + cc_uvel[ci])/2.;
          double bc_v = (cc_vvel[c] + cc_vvel[ci])/2.;
          double bc_w = (cc_wvel[c] + cc_wvel[ci])/2.;
          double u = 2. * bc_u - cc_uvel[ci];
          double v = 2. * bc_v - cc_vvel[ci];
          double w = 2. * bc_w - cc_wvel[ci];

          partVel[c] = Vector(u, v, w);

        }
      }
    }
  }
} // end ComputePartVel()
