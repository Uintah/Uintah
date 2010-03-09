#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>


//===========================================================================

using namespace Uintah;

PartVel::PartVel(ArchesLabel* fieldLabels ) : 
d_fieldLabels(fieldLabels)
{
}

PartVel::~PartVel()
{
  delete d_boundaryCond; 
}
//---------------------------------------------------------------------------
// Method: ProblemSetup
//---------------------------------------------------------------------------
void PartVel::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb; 

  ProblemSpecP vel_db = db->findBlock("VelModel");
  if (vel_db) {

    std::string model_type;
    vel_db->getAttribute("type", model_type);

    if (model_type == "Balachandar") {
      d_bala = true;
      d_drag = false;
      int regime; 
      vel_db->getWithDefault("kinematic_viscosity",kvisc,1.e-5); 
      vel_db->getWithDefault("iter", d_totIter, 15); 
      vel_db->getWithDefault("tol",  d_tol, 1e-15); 
      vel_db->getWithDefault("rho_ratio",rhoRatio,1000);
      vel_db->getWithDefault("L",d_L, 1.0);
      vel_db->getWithDefault("eta", d_eta, 1e-5); 
      beta = 3. / (2.*rhoRatio + 1.); 
      vel_db->getWithDefault("regime",regime,1);      
      vel_db->getWithDefault("min_vel_ratio", d_min_vel_ratio, .1); 

      if (regime == 1)
        d_power = 1;
      else if (regime == 2) 
        d_power = 0.5;
      else if (regime == 3)
        d_power = 1./3.;
    } else if(model_type == "Dragforce") {
      d_bala = false;
      d_drag = true;
    } else {
      throw InvalidValue( "Invalid type for Velocity Model, must be Balachandar or Dragforce",__FILE__,__LINE__);
    }
  } else {
    throw InvalidValue( "A <VelModel> section is missing from your input file!",__FILE__,__LINE__);
  }

  vel_db->getWithDefault( "upper_limit_multiplier", d_upLimMult, 2.0 ); // d_highClip set using this factor (below)
  vel_db->getWithDefault( "clip_low", d_lowClip, 0.0 ); 

  vel_db->getWithDefault( "partvelBC_eq_gasvelBC", d_gasBC, false ); 

  d_boundaryCond = scinew BoundaryCondition_new( d_fieldLabels ); 
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

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
  int N = dqmomFactory.get_quad_nodes();  

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


  if (d_bala) {
    //--Old
    // fluid velocity
    tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0 );

    // requires weighted legnth and weight (to back out length)
    for (int i = 0; i < N; i++){
      std::string name = "length_qn"; 
      std::string node; 
      std::stringstream out; 
      out << i; 
      node = out.str(); 
      name += node; 

      EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( name ); 
      const VarLabel* myLabel = eqn.getTransportEqnLabel();  
    
      tsk->requires( Task::OldDW, myLabel, gn, 0 ); 
    }

    for (int i = 0; i < N; i++){
      std::string name = "w_qn"; 
      std::string node; 
      std::stringstream out; 
      out << i; 
      node = out.str(); 
      name += node; 

      EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( name ); 
      const VarLabel* myLabel = eqn.getTransportEqnLabel(); 
  
      tsk->requires( Task::OldDW, myLabel, gn, 0 ); 
    }

  } else if(d_drag) {
   
    tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0 );
    
    for (int i = 0; i < N; i++){
      std::string name = "w_qn";
      std::string node;
      std::stringstream out;
      out << i;
      node = out.str();
      name += node;

      EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( name );
      const VarLabel* myLabel = eqn.getTransportEqnLabel();

      tsk->requires( Task::OldDW, myLabel, gn, 0 );
    }
    
    for (int i = 0; i < N; i++){
      std::string name = "ux_qn";
      std::string node;
      std::stringstream out;
      out << i;
      node = out.str();
      name += node;

      EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( name );
      const VarLabel* myLabel = eqn.getTransportEqnLabel();

      tsk->requires( Task::OldDW, myLabel, gn, 0 );
    }
  
   for (int i = 0; i < N; i++){
      std::string name = "uy_qn";
      std::string node;
      std::stringstream out;
      out << i;
      node = out.str();
      name += node;

      EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( name );
      const VarLabel* myLabel = eqn.getTransportEqnLabel();

      tsk->requires( Task::OldDW, myLabel, gn, 0 );
    }
   
    for (int i = 0; i < N; i++){
      std::string name = "uz_qn";
      std::string node;
      std::stringstream out;
      out << i;
      node = out.str();
      name += node;

      EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( name );
      const VarLabel* myLabel = eqn.getTransportEqnLabel();

      tsk->requires( Task::OldDW, myLabel, gn, 0 );
    }
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
                              DataWarehouse* new_dw, const int rkStep )
{
   //patch loop
  for (int p=0; p < patches->size(); p++){
    
    //double pi = acos(-1.0);

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 


    if (d_bala) {

      constCCVariable<Vector> gasVel; 

      old_dw->get( gasVel, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 ); 

      // now loop for all qn's
      int N = dqmomFactory.get_quad_nodes();  
      for ( int iqn = 0; iqn < N; iqn++){
        std::string name = "length_qn"; 
        std::string node; 
        std::stringstream out; 
        out << iqn; 
        node = out.str(); 
        name += node; 

        EqnBase& t_eqn = dqmomFactory.retrieve_scalar_eqn( name );
        DQMOMEqn& eqn = dynamic_cast<DQMOMEqn&>(t_eqn);
      
        constCCVariable<double> wlength;  
        const VarLabel* mylLabel = eqn.getTransportEqnLabel();  
        old_dw->get(wlength, mylLabel, matlIndex, patch, gn, 0); 

        d_highClip = eqn.getScalingConstant()*d_upLimMult; // should figure out how to do this once and only once...

        name = "w_qn"; 
        name += node; 
        EqnBase& eqn2 = dqmomFactory.retrieve_scalar_eqn( name );
        constCCVariable<double> weight;  
        const VarLabel* mywLabel = eqn2.getTransportEqnLabel();  
        old_dw->get(weight, mywLabel, matlIndex, patch, gn, 0); 

        ArchesLabel::PartVelMap::iterator iter = d_fieldLabels->partVel.find(iqn);

        CCVariable<Vector> partVel;
        constCCVariable<Vector> old_partVel;  
        if (rkStep < 1) 
          new_dw->allocateAndPut( partVel, iter->second, matlIndex, patch );  
        else  
          new_dw->getModifiable( partVel, iter->second, matlIndex, patch ); 
        old_dw->get(old_partVel, iter->second, matlIndex, patch, gn, 0);

        partVel.initialize(Vector(0.,0.,0.));

        // set boundary conditions. 
        name = "vel_qn";
        name += node; 
        if ( d_gasBC )  // assume gas vel =  part vel on boundary 
          d_boundaryCond->setVectorValueBC( 0, patch, partVel, gasVel, name ); 
        else           // part vel set by user.  
          d_boundaryCond->setVectorValueBC( 0, patch, partVel, name );  
      

        // now loop over all cells
        for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
          IntVector c = *iter;
          IntVector cxm = *iter - IntVector(1,0,0); 

          double length;
          if( weight[c] < 1e-6 ) {
            length = 0;
          } else {
            length = (wlength[c]/weight[c])*eqn.getScalingConstant();
          }

        //if( length > d_highClip ) {
        //  length = d_highClip;
        //} else if( length < d_lowClip ) {
        //  length = d_lowClip;
        //}

          Vector sphGas = Vector(0.,0.,0.);
          Vector cartGas = gasVel[c]; 
        
          Vector sphPart = Vector(0.,0.,0.);
          Vector cartPart = old_partVel[c]; 

          sphGas = cart2sph( cartGas ); 
          sphPart = cart2sph( cartPart ); 

          double diff = sphGas.z() - sphPart.z(); 
          double prev_diff = 0.0;
          double newPartMag = 0.0;
          double length_ratio = 0.0;

          epsilon =  pow(sphGas.z(),3.0);
          epsilon /= d_L;

          length_ratio = length / d_eta;
          double uk = 0.0;
       
          if (length > 0.0) {
            uk = pow(d_eta/d_L, 1./3.);
            uk *= sphGas.z();
          }

          diff = 0.0;

          for ( int i = 0; i < d_totIter; i++) {

            prev_diff = diff; 
            double Re  = abs(diff)*length / kvisc;
            double phi = 1. + .15*pow(Re, 0.687);
            double t_p_by_t_k = (2*rhoRatio+1)/36*1.0/phi*pow(length_ratio,2);
            diff = uk*(1-beta)*pow(t_p_by_t_k, d_power);
            double error = abs(diff - prev_diff)/diff; 
            if ( abs(diff) < 1e-16 )
              error = 0.0;

            if (abs(error) < d_tol)
              break;

          }

          newPartMag = sphGas.z() - diff; 

          double vel_ratio = newPartMag / sphGas.z(); 

          if (vel_ratio < d_min_vel_ratio) 
            newPartMag = sphGas.z() * d_min_vel_ratio; 
  
          sphPart = Vector(sphGas.x(), sphGas.y(), newPartMag);

          // now convert back to cartesian
          Vector newcartPart = Vector(0.,0.,0.);
          newcartPart = sph2cart( sphPart ); 

          partVel[c] = newcartPart; 
        }
      }

    } else if (d_drag) {
     
      constCCVariable<Vector> gasVel;

      old_dw->get( gasVel, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );
      
      int N = dqmomFactory.get_quad_nodes();
      for ( int iqn = 0; iqn < N; iqn++){
    
        std::string node;
        std::stringstream out;
        out << iqn;
        node = out.str();

        std::string name = "w_qn";
        name += node;
        EqnBase& eqn2 = dqmomFactory.retrieve_scalar_eqn( name );
        constCCVariable<double> weight;
        const VarLabel* mywLabel = eqn2.getTransportEqnLabel();
        old_dw->get(weight, mywLabel, matlIndex, patch, gn, 0);
        
        name = "ux_qn";
        name += node;
        EqnBase& t_eqn3 = dqmomFactory.retrieve_scalar_eqn( name );
        DQMOMEqn& eqn3 = dynamic_cast<DQMOMEqn&>(t_eqn3);
        constCCVariable<double> vel_x;
        const VarLabel* myuxLabel = eqn3.getTransportEqnLabel();
        old_dw->get(vel_x, myuxLabel, matlIndex, patch, gn, 0);
        
        name = "uy_qn";
        name += node;
        EqnBase& t_eqn4 = dqmomFactory.retrieve_scalar_eqn( name );
        DQMOMEqn& eqn4 = dynamic_cast<DQMOMEqn&>(t_eqn4);
        constCCVariable<double> vel_y;
        const VarLabel* myuyLabel = eqn4.getTransportEqnLabel();
        old_dw->get(vel_y, myuyLabel, matlIndex, patch, gn, 0);

        name = "uz_qn";
        name += node;
        EqnBase& t_eqn5 = dqmomFactory.retrieve_scalar_eqn( name );
        DQMOMEqn& eqn5 = dynamic_cast<DQMOMEqn&>(t_eqn5);
        constCCVariable<double> vel_z;
        const VarLabel* myuzLabel = eqn5.getTransportEqnLabel();
        old_dw->get(vel_z, myuzLabel, matlIndex, patch, gn, 0);

        ArchesLabel::PartVelMap::iterator iter = d_fieldLabels->partVel.find(iqn);

        CCVariable<Vector> partVel;
        if (rkStep < 1)
          new_dw->allocateAndPut( partVel, iter->second, matlIndex, patch );
        else
          new_dw->getModifiable( partVel, iter->second, matlIndex, patch );
        partVel.initialize(Vector(0.,0.,0.));

        // set boundary conditions. 
        name = "vel_qn";
        name += node;
        if ( d_gasBC )  // assume gas vel =  part vel on boundary 
          d_boundaryCond->setVectorValueBC( 0, patch, partVel, gasVel, name );
        else           // part vel set by user.  
          d_boundaryCond->setVectorValueBC( 0, patch, partVel, name );

        // now loop over all cells
        for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
          IntVector c = *iter;
          double ux;
          double uy;
          double uz;
          if( weight[c] < 1e-20 ) {
            ux = 0;
            uy = 0;
            uz = 0;
          } else {
            ux = (vel_x[c]/weight[c])*eqn3.getScalingConstant();
            uy = (vel_y[c]/weight[c])*eqn4.getScalingConstant();
            uz = (vel_z[c]/weight[c])*eqn5.getScalingConstant();
          }

          partVel[c] = Vector(ux,uy,uz);
        }
      } 
    }  
  } 
}

