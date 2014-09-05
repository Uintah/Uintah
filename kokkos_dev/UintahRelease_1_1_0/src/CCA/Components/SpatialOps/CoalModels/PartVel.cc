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
    vel_db->getWithDefault("new",nnew,1); 
    vel_db->getWithDefault("eta",eta,1); 
    vel_db->getWithDefault("rho_ratio",rhoRatio,0);
    beta = 3. / (2.*rhoRatio + 1.); 
    vel_db->getWithDefault("eps",eps,0);
    vel_db->getWithDefault("regime",regime,1);      
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
    // also get the old one
    tsk->requires( Task::OldDW, i->second, gn, 0);
  }

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

    // get the initial value and store it. 
    double initValue = eqn.getInitValue();
    d_wlo.push_back(initValue);
    
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
    // get the initial value and store it. 
    double initValue = eqn.getInitValue();
    d_wo.push_back(initValue);

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
      constCCVariable<Vector> old_partVel;  
      if (rkStep < 1) 
        new_dw->allocateAndPut( partVel, iter->second, matlIndex, patch );  
      else  
        new_dw->getModifiable( partVel, iter->second, matlIndex, patch ); 
      old_dw->get(old_partVel, iter->second, matlIndex, patch, gn, 0);
      

      // now loop over all cells
      for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
  
        IntVector c = *iter;
        double length = d_wlo[iqn]/d_wo[iqn];

        Vector sphGas = Vector(0.,0.,0.);
        Vector cartGas = gasVel[c]; 
        Vector sphPart = Vector(0.,0.,0.);
        Vector cartPart = old_partVel[c]; 

        //cout << "carGas = " << cartGas << endl; 

        sphGas = cart2sph( cartGas ); 
        sphPart = cart2sph( cartPart ); 

        //cout << "sphGas = " << sphGas << endl;
        //cout << "old sphPart = " << sphPart << endl;
         
        double Re  = abs(sphGas.z() - sphPart.z())*length / nnew;
        double phi = 1. + .15*pow(Re, 0.687);
        double uk  = eta*eps; 
        uk = pow(uk,1./3.);

        double newPartMag = sphGas.z() - uk*(2*rhoRatio+1)/36*(1-beta)/phi*pow(length/eta,2);
        sphPart = Vector(sphGas.x(), sphGas.y(), newPartMag);

        //cout << "SPHPART NEW " << sphPart << endl;
        // now convert back to cartesian
        Vector newcartPart = Vector(0.,0.,0.);
        newcartPart = sph2cart( sphPart ); 

        //cout << "CARTPART = " << newcartPart << endl;

        partVel[c] = newcartPart; 

      }
    }
  } 
}

