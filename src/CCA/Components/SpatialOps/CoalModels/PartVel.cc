#include <CCA/Components/SpatialOps/CoalModels/PartVel.h>
#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnBase.h>
#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>


//===========================================================================

using namespace Uintah;

PartVel::PartVel(Fields* fieldLabels ) : 
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

  ProblemSpecP vel_db = db->findBlock("VelModel");
  if (vel_db) {
    vel_db->getWithDefault("Re",Re,0);
    vel_db->getWithDefault("eta",eta,1); 
    vel_db->getWithDefault("rhof",rhof,0);
    vel_db->getWithDefault("beta",beta,0); 
    vel_db->getWithDefault("eps",eps,0);
    vel_db->getWithDefault("regime",regime,1);      
    vel_db->getWithDefault("part_mass", partMass,0);
  } else {
    throw InvalidValue( "A <VelModel> section is missing from your input file!",__FILE__,__LINE__);
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

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
  int N = dqmomFactory.get_quad_nodes();  
  //--New
  // actual velocity we will compute
  for (Fields::PartVelMap::iterator i = d_fieldLabels->partVel.begin(); 
        i != d_fieldLabels->partVel.end(); i++){
    if (rkStep < 1 )
      tsk->computes( i->second );
    else 
      tsk->modifies( i->second );  
  }

  //for (vector<const VarLabel>::iterator i=d_fieldLabels->partVel.begin(); i != d_fieldLabels->partVel.end(); i++){
  //  tsk->computes( *i ); // not sure why this doesn't work with the iterator?!  
  //} 
 
  //--Old
  // fluid velocity
  tsk->requires( Task::OldDW, d_fieldLabels->velocityLabels.ccVelocity, gn, 0 );
  // environments
  // right now assume that the velocity is a function of length.
  // requires that length be an ic (with that name)  
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

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allSpatialOpsMaterials());
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
    
    double pi = acos(-1.0);

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    constCCVariable<Vector> gasVel; 

    old_dw->get( gasVel, d_fieldLabels->velocityLabels.ccVelocity, matlIndex, patch, gn, 0 ); 

    // now loop for all qn's
    int N = dqmomFactory.get_quad_nodes();  
    for ( int iqn = 0; iqn < N; iqn++){
      std::string name = "length_qn"; 
      std::string node; 
      std::stringstream out; 
      out << iqn; 
      node = out.str(); 
      name += node; 

      EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( name );
      constCCVariable<double> wlength;  
      const VarLabel* mylLabel = eqn.getTransportEqnLabel();  
      old_dw->get(wlength, mylLabel, matlIndex, patch, gn, 0); 

      name = "w_qn"; 
      name += node; 
      EqnBase& eqn2 = dqmomFactory.retrieve_scalar_eqn( name );
      constCCVariable<double> weight;  
      const VarLabel* mywLabel = eqn2.getTransportEqnLabel();  
      old_dw->get(weight, mywLabel, matlIndex, patch, gn, 0); 

      Fields::PartVelMap::iterator iter = d_fieldLabels->partVel.find(iqn);

      CCVariable<Vector> partVel; 
      if (rkStep < 1) 
        new_dw->allocateAndPut( partVel, iter->second, matlIndex, patch );  
      else  
        new_dw->getModifiable( partVel, iter->second, matlIndex, patch ); 

      // now loop over all cells
      for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
  
        IntVector c = *iter;
        double length = wlength[c]/weight[c]; 

        // for now the density will only be a function of length.
        double rhop = partMass / ( 4./3.*pi*length*length*length );
        double denRatio = rhop/rhof;
        double rePow = pow( Re, 0.687 );
        double phi = 1 + 0.15*rePow; 
        double uk = eta*eps; 
        uk = pow( uk, 1./3. );  

        double u_comp = 0.0;
        double v_comp = 0.0;
        double w_comp = 0.0; 
        
        u_comp = gasVel[c].x() - uk * ( 2*denRatio + 1)/36*( 1 - beta )/phi * length/eta*length/eta ; 
#ifdef YDIM
        v_comp = gasVel[c].y() - uk * ( 2*denRatio + 1)/36*( 1 - beta )/phi * length/eta*length/eta ; 
#endif
#ifdef ZDIM
        z_comp = 0 - uk * ( 2*denRatio + 1)/36*( 1 - beta )/phi * length[c]/eta*length/eta ; 
#endif
        partVel[c] = Vector(u_comp,v_comp,w_comp);

        double L = u_comp*u_comp;
        L += v_comp*v_comp; 
        L += w_comp*w_comp; 
        L = pow(L,0.5);
        if (L > 10 )
          cout << " FUNNY VEL AT : " << c << endl;
      }
    }
  } 
}

