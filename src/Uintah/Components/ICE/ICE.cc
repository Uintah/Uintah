/* ---------------------------------------------------------------------
GENERAL INFORMATION

 FILE NAME:  ICE.cc
 Purpose:    This is the main component for the Uintah ICE cfd code. 
.
History: 
Version   Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       02/22/99                               
.                                                                    
    Programming Conventions
        i, j, k         Loop indices for the x, y, z directions respectively
        f               is a loop index for face-centered values.
        m               Loop index for the different materials
.
                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(3)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (6) = back face
                                 (2)              (5) = front face
.
 STEPS:
    - Set some eviromnental variables required for PGPLOT
    - Initialize some variables that are mainly used in testing
    - MEMORY SECTION: Allocate the memory needed for all of the arrays
      For all of the face-centered arrays set equate the common face addresses
      [i][j][k][RIGHT][m] = [i-1][j][k][LEFT][m]
    - PROBLEM INITIALIZATION SECTION: Read in the input file, test the inputs,
      set the boundary condtions, generate the grid
    - MAIN LOOP
        to be filled in
    
 ---------------------------------------------------------------------*/   


#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/CellIterator.h>
#include <Uintah/Grid/SoleVariable.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/VarLabel.h>
using SCICore::Geometry::Vector;
using SCICore::Geometry::IntVector;

#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <stdlib.h>
#include <iostream>
using std::cerr;
using std::endl;

#include "nrutil+.h"
#include "ice_sm/Header_files/functionDeclare.h"
#include "ice_sm/Header_files/parameters.h"
#include "ice_sm/Header_files/switches.h"
#include "ice_sm/Header_files/macros.h"
#include "ice_sm/Header_files/cpgplot.h" /*must have this for plotting to work   */


extern "C" void audit();
#include <Uintah/Grid/VarTypes.h>

using Uintah::ICESpace::ICE;

ICE::ICE()
{
  delTLabel = 
    new VarLabel( "delT", delt_vartype::getTypeDescription() );
  vel_CCLabel = 
    new VarLabel( "vel_CC", CCVariable<Vector>::getTypeDescription() );
  press_CCLabel = 
    new VarLabel( "press_CC", CCVariable<double>::getTypeDescription() );
  rho_CCLabel = 
    new VarLabel( "rho_CC", CCVariable<double>::getTypeDescription() );
  temp_CCLabel = 
    new VarLabel( "temp_CC", CCVariable<double>::getTypeDescription() );
  cv_CCLabel = 
    new VarLabel( "cv_CC", CCVariable<double>::getTypeDescription() );

  // Face centered variables
  vel_FCLabel = 
    new VarLabel( "vel_FC", FCVariable<Vector>::getTypeDescription() );
  press_FCLabel = 
    new VarLabel( "press_FC", FCVariable<double>::getTypeDescription() );
  tau_FCLabel = 
    new VarLabel( "tau_FC", FCVariable<Vector>::getTypeDescription() );

  
#if 1
/*______________________________________________________________________ 
*                       MEMORY SECTION
* 
*  Allocate memory for the arrays                                          
*_______________________________________________________________________*/
    nMaterials=1;
audit();
#include "allocate_memory.i"
audit();
#endif

/*__________________________________
*   Plotting variables
*___________________________________*/
#if (switchDebug_main == 1|| switchDebug_main == 2 || switchDebug_main_input == 1)
    #include "plot_declare_vars.h"   
#endif
    stat = putenv("PGPLOT_DIR=/usr/people/jas/PSE/src/Uintah/Components/ICE/ice_sm/Libraries");
    stat = putenv("PGPLOT_I_AM_HERE=0");              
                                        /* tell the plotting routine that  */
                                        /* you're at the top of main       */      

    stat = putenv("PGPLOT_PLOTTING_ON_OFF=1");
    stat = putenv("PGPLOT_OPEN_NEW_WINDOWS=1");  

}

ICE::~ICE()
{
#if 1
/* -----------------------------------------------------------------------  
*  Free the memory                                                         
* -----------------------------------------------------------------------  */
    fprintf(stderr,"Now deallocating memory\n");
#include "free_memory.i"
#endif
}

void ICE::problemSetup(const ProblemSpecP& prob_spec, GridP&,
		       SimulationStateP&)
{

  // Read in the material properties
  double viscosity,thermal_conductivity,specific_heat,speed_of_sound;
  double ideal_gas_constant,d_gamma;

  ProblemSpecP mat_ps = prob_spec->findBlock("MaterialProperties");

  ProblemSpecP ice_mat_ps = mat_ps->findBlock("ICE");

  for (ProblemSpecP ps = ice_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    ps->require("viscosity",viscosity);
    ps->require("thermal_conductivity",thermal_conductivity);
    ps->require("specific_heat",specific_heat);
    ps->require("speed_of_sound",speed_of_sound);
    ps->require("ideal_gas_constant",ideal_gas_constant);
    ps->require("gamma",d_gamma);
  }
    
  cerr << "viscosity " << viscosity << endl;
  cerr << "thermal_conductivity " << thermal_conductivity << endl;
  cerr << "specific_heat " << specific_heat << endl;
  cerr << "speed_of_sound " << speed_of_sound << endl;
  cerr << "ideal_gas_constant " << ideal_gas_constant << endl;
  cerr << "gamma " << d_gamma << endl;
  


audit();
/*__________________________________
 *   Plotting variables
 *___________________________________*/
 putenv("PGPLOT_I_AM_HERE=0");              
 /* tell the plotting routine that  */
 /* you're at the top of main        */
 
 putenv("PGPLOT_PLOTTING_ON_OFF=1");
 putenv("PGPLOT_OPEN_NEW_WINDOWS=1"); 
 
 /*__________________________________
  * Now make sure that the face centered
  * values know about each other.
  * for example 
  * [i][j][k][RIGHT][m] = [i-1][j][k][LEFT][m]
  *___________________________________*/  
 
 audit();
 equate_ptr_addresses_adjacent_cell_faces(              
					  x_FC, y_FC,z_FC,
					  uvel_FC,vvel_FC, wvel_FC,
					  press_FC,
					  tau_X_FC, tau_Y_FC, tau_Z_FC,
					  nMaterials);   
 audit();
 /*______________________________________________________________________
  *
  *                       PROBLEM INITIALIZATION SECTION
  *   Initializing routines                                                  
  *   First read the problem input then test the inputs.                     
  * -----------------------------------------------------------------------  */
 
 
 
 int printSwitch = 1;
 t = 0.0;
 m = 1;
 fileNum = 1;

 readInputFile( &xLoLimit, &yLoLimit, &zLoLimit,     
		&xHiLimit,      &yHiLimit,      &zHiLimit,
		&delX,          &delY,          &delZ,
		uvel_CC,        vvel_CC,        wvel_CC, 
		Temp_CC,        press_CC,       rho_CC,
		scalar1_CC,     scalar2_CC,     scalar3_CC,
		viscosity_CC,   thermalCond_CC, cv_CC,
		R,              gamma,
		&t_final,       t_output_vars,  delt_limits,
		output_file_basename,           output_file_desc,       
		grav,           speedSound,
		BC_inputs,      BC_Values,      &CFL,
		&nMaterials);      

    
 testInputFile(xLoLimit,       yLoLimit,       zLoLimit,
	       xHiLimit,       yHiLimit,       zHiLimit,
	       delX,           delY,           delZ,
	       Temp_CC,        press_CC,       rho_CC,
	       viscosity_CC,   thermalCond_CC, cv_CC,
	       speedSound,      
	       t_final,        t_output_vars,  delt_limits,
	       BC_inputs,      printSwitch,    CFL,
	       nMaterials); 
 
 definition_of_different_physical_boundary_conditions(              
						      BC_inputs, BC_types, 
						      BC_float_or_fixed,
						      BC_Values, nMaterials);  
                        
 /*__________________________________
  * Now make sure that the face centered
  * values know about each other.
  * for example 
  * [i][j][k][RIGHT][m] = [i-1][j][k][LEFT][m]
  *___________________________________*/  
 
 equate_ptr_addresses_adjacent_cell_faces(              
					  x_FC, y_FC, z_FC,
					  uvel_FC, vvel_FC, wvel_FC,
					  press_FC,
					  tau_X_FC,tau_Y_FC,tau_Z_FC,
					  nMaterials);   

 /*__________________________________
  * Generate a grid
  *___________________________________*/ 
 generateGrid(xLoLimit,       yLoLimit,       zLoLimit,
	      xHiLimit,       yHiLimit,       zHiLimit,
	      delX,           delY,           delZ,
	      x_CC,           y_CC,           z_CC,   Vol_CC,  
	      x_FC,           y_FC,           z_FC );
 /*__________________________________
  *   zero the face-centered arrays
  *___________________________________*/
 zero_arrays_6d(
		xLoLimit,       yLoLimit,       zLoLimit,             
		xHiLimit,       yHiLimit,       zHiLimit,
		1,              N_CELL_FACES,
		1,              nMaterials,     
		7,             
		uvel_FC,        vvel_FC,        wvel_FC,
		press_FC,
		tau_X_FC,       tau_Y_FC,       tau_Z_FC);                         
 stat = putenv("PGPLOT_PLOTTING_ON_OFF=1");
                            
 
 /*__________________________________
  *   overide the initial conditions
  *___________________________________*/
#if switchOveride_Initial_Conditions                               
#include "ice_sm/Header_files/overide_initial_conds.i"
#endif 
 
 /*__________________________________
  *  If desired plot the inputs
  *___________________________________*/
#if switchDebug_main_input
#define switchInclude_main_1 1
#include "debugcode.i"
#undef switchInclude_main_1
#endif 
 
 /*__________________________________
  *   For the first time through
  *   set some variables
  *___________________________________*/
 delt    = delt_limits[3];              
 t       = delt;
 fprintf(stderr,"\nInitial time %f, timestep is %f\n",t,delt);
 
 
 
#if 0
 CCVariable<Vector> vel_CC;
 ds.put("vel_CC", vel_CCLabel,0,patch);
 
#else
 cerr << "put vel_CC not done\n";
#endif
 
}

void ICE::scheduleInitialize(const LevelP& level,
			     SchedulerP& sched,
			     DataWarehouseP& dw)
{
  Level::const_patchIterator iter;

  for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
    
    const Patch* patch=*iter;
    {
      Task* t = scinew Task("ICE::actuallyInitialize", patch, dw, dw,
			    this, &ICE::actuallyInitialize);
      t->computes(dw, vel_CCLabel,0,patch);
      t->computes(dw, press_CCLabel,0,patch);
      t->computes(dw, rho_CCLabel,0,patch);
      t->computes(dw, temp_CCLabel,0,patch);
      t->computes(dw, cv_CCLabel,0,patch);

      t->computes(dw, vel_FCLabel,0,patch);
      t->computes(dw, press_FCLabel,0,patch);
      t->computes(dw, tau_FCLabel,0,patch);



      sched->addTask(t);
    }
  }
  cerr << "ICE::scheduleInitialize not done\n";
}

void ICE::actuallyInitialize(const ProcessorGroup*,
			     const Patch* patch,
			     DataWarehouseP& /* old_dw */,
			     DataWarehouseP& new_dw)
{
  cerr <<"Doing actuallyInitialize . . ." << endl;
  CCVariable<Vector> vel_cc;
  CCVariable<double> press_cc,rho_cc,temp_cc,cv_cc;
  FCVariable<Vector> vel_fc,tau_fc;
  FCVariable<double> press_fc;
  
  new_dw->allocate(vel_cc,vel_CCLabel,0,patch);
#if 1
  convertNR_4dToUCF(patch,vel_cc,uvel_CC,vvel_CC,wvel_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,
		    nMaterials);
#endif
  new_dw->put(vel_cc,vel_CCLabel,0,patch);

  new_dw->allocate(press_cc,press_CCLabel,0,patch);
#if 1
  convertNR_4dToUCF(patch,press_cc,press_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,
		    nMaterials);
#endif
  new_dw->put(press_cc,press_CCLabel,0,patch);

  new_dw->allocate(rho_cc,rho_CCLabel,0,patch);
#if 1
  convertNR_4dToUCF(patch,rho_cc,rho_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,
		    nMaterials);
#endif
  new_dw->put(rho_cc,rho_CCLabel,0,patch);

  new_dw->allocate(temp_cc,temp_CCLabel,0,patch);
#if 1
  convertNR_4dToUCF(patch,temp_cc,Temp_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,
		    nMaterials);
#endif
  new_dw->put(temp_cc,temp_CCLabel,0,patch);

  new_dw->allocate(cv_cc,cv_CCLabel,0,patch);
#if 1
  convertNR_4dToUCF(patch,cv_cc,cv_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,
		    nMaterials);
#endif
  new_dw->put(cv_cc,cv_CCLabel,0,patch);

  new_dw->allocate(vel_fc,vel_FCLabel,0,patch);
  new_dw->put(vel_fc,vel_FCLabel,0,patch);
  new_dw->allocate(press_fc,press_FCLabel,0,patch);
  new_dw->put(press_fc,press_FCLabel,0,patch);
  new_dw->allocate(tau_fc,tau_FCLabel,0,patch);
  new_dw->put(tau_fc,tau_FCLabel,0,patch);


}

void ICE::scheduleComputeStableTimestep(const LevelP& level,
					SchedulerP& sched,
					DataWarehouseP& dw)
{

    for(Level::const_patchIterator iter=level->patchesBegin();
	iter != level->patchesEnd(); iter++){
	const Patch* patch=*iter;
	Task* t = scinew Task("ICE::computeStableTimestep", patch, dw, dw,
			      this, &ICE::actuallyComputeStableTimestep);
	t->requires(dw, vel_CCLabel, 0,patch, Ghost::None);
	//t->requires(dw, "params", ProblemSpec::getTypeDescription());
	t->computes(dw,delTLabel);
	t->usesMPI(false);
	t->usesThreads(false);
	//t->whatis the cost model?();
	sched->addTask(t);
    }

}

void ICE::actuallyComputeStableTimestep(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& fromDW,
					DataWarehouseP& toDW)
{
  cerr << "Doing actuallyComputeStableTimestep . . ." << endl;

  
  CCVariable<Vector> vel_cc;
  toDW->get(vel_cc, vel_CCLabel,0, patch,Ghost::None,0);

#if 0
  for (CellIterator iter = patch->getCellIterator(patch->getBox());
       !iter.done(); iter++) {
    cerr << "vel_cc = " << vel_cc[*iter] << endl;
  }
  

  for (int i = xLoLimit; i <= xHiLimit; i++) {
    for (int j = yLoLimit; j <= yHiLimit; j++) {
      for (int k = zLoLimit; k <= zHiLimit; k++) {
	for (int m = 1; m <= nMaterials; m++) {
	  IntVector idx(i-1,j-1,k-1);
	  vel_CC[idx]=Vector(uvel_CC[m][i][j][k], vvel_CC[m][i][j][k], 
			     wvel_CC[m][i][j][k]);
	  // cerr << "vel_ucf = " << vel_CC[idx] << endl;
	}
      }
    }
  }
#endif
  // Convert the data
#if 1
  convertUCFToNR_4d(patch,vel_cc,uvel_CC,vvel_CC,wvel_CC,xLoLimit,xHiLimit,
		    yLoLimit,yHiLimit,zLoLimit,zHiLimit,nMaterials);
#endif  
  /*__________________________________
   *   Find the new time step based on the
   *   Courant condition
   *___________________________________*/
  
  //double delt;
  find_delta_time_based_on_CC_vel(xLoLimit,        yLoLimit,      zLoLimit,
				  xHiLimit,        yHiLimit,      zHiLimit,
				  &delt,           delt_limits,
				  delX,            delY,          delZ,
				  uvel_CC,         vvel_CC,       wvel_CC,
				  speedSound,      CFL,           nMaterials );
  
#if 0
  int numMatls = d_sharedState->getNumMatls();

  double A,B, delt_CFL;

  for (int m = 0; m < numMatls; m++) {
    Material* matl = d_sharedState->getMaterial(m);
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if (ice_matl) {
      int matlindex = matl->getDWIndex();
      inf vfindex = matl->getVFIndex();

      // 
      CCVariable<Vector> vel_CC;
      toDW->get(vel_CC, vel_CCLabel,matlindex, patch,Ghost::None,0);

      for (CellIterator iter = patch->getCellIterator(patch->getBox()); 
	   !iter.done();
	   iter++) {

	// Get the patch spacing.
	delx = patch->dCell();
	dely = patch->dCell()
	  
	
	A = fudge_factor*CFL*delx/fabs(vel_CC[*iter].x() + SMALL_NUM);
	B = fudge_factor*CFL*delx/fabs(vel_CC[*iter].y() + SMALL_NUM);

	delt_CFL = DMIN(A,delt_CFL);
	delt_CFL = DMIN(B,delt_CFL);

	// Do other steps

      }
    }

  }
  
  delt = detl_CFL;
#endif
  
  cerr << "Del T is " << delt << endl;
  delt_vartype dt(delt);
  toDW->put(dt, delTLabel);

}


void ICE::scheduleTimeAdvance(double /*t*/, 
			      double /*delt*/,
			      const LevelP& level, 
			      SchedulerP& sched,
			      DataWarehouseP& old_dw, 
			      DataWarehouseP& new_dw)
{

    for(Level::const_patchIterator iter=level->patchesBegin();
	iter != level->patchesEnd(); iter++){
	const Patch* patch=*iter;
	{
	// Step 0 -- Plot initialization stuff

	Task* t = scinew Task("ICE::step0", patch, old_dw, new_dw,
			      this, &ICE::actuallyStep0);
	t->requires(old_dw, vel_CCLabel, 0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
	//	t->computes(new_dw, vel_CCLabel,0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
	//t->whatis the cost model?();
	sched->addTask(t);
	}	

	{
	// Step 1

	Task* t = scinew Task("ICE::step1", patch, old_dw, new_dw,
			      this, &ICE::actuallyStep1);
	t->requires(old_dw, vel_CCLabel, 0,patch, Ghost::None);
	//	t->requires(old_dw, press_CCLabel, 0,patch, Ghost::None);
	//t->requires(old_dw, rho_CCLabel, 0,patch, Ghost::None);
	t->requires(old_dw, temp_CCLabel, 0,patch, Ghost::None);
	t->requires(old_dw, cv_CCLabel, 0,patch, Ghost::None);

//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
	//	t->computes(new_dw, vel_CCLabel,0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
	//t->whatis the cost model?();
	sched->addTask(t);
	}

	{
	// Step 2

	Task* t = scinew Task("ICE::step2", patch, old_dw, new_dw,
			      this, &ICE::actuallyStep2);
	t->requires(old_dw, vel_CCLabel, 0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
//	t->computes(new_dw, vel_CCLabel,0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
	//t->whatis the cost model?();
	sched->addTask(t);
	}

	{
	// Step 3

	Task* t = scinew Task("ICE::step3", patch, old_dw, new_dw,
			      this, &ICE::actuallyStep3);
	t->requires(old_dw, vel_CCLabel, 0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
	//	t->computes(new_dw, vel_CCLabel,0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
	//t->whatis the cost model?();
	sched->addTask(t);
	}

	{
	// Step 4

	Task* t = scinew Task("ICE::step4", patch, old_dw, new_dw,
			      this, &ICE::actuallyStep4);
	t->requires(old_dw, vel_CCLabel, 0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
	//	t->computes(new_dw, vel_CCLabel,0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
	//t->whatis the cost model?();
	sched->addTask(t);
	}
	
	{
	// Step 5

	Task* t = scinew Task("ICE::step5", patch, old_dw, new_dw,
			      this, &ICE::actuallyStep5);
	t->requires(old_dw, vel_CCLabel, 0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
	//	t->computes(new_dw, vel_CCLabel,0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
	//t->whatis the cost model?();
	sched->addTask(t);
	}

	{
	// Step 6

	Task* t = scinew Task("ICE::step6", patch, old_dw, new_dw,
			      this, &ICE::actuallyStep6);
	t->requires(old_dw, vel_CCLabel, 0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
	//	t->computes(new_dw, vel_CCLabel,0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
	//t->whatis the cost model?();
	sched->addTask(t);
	}

	{
	// Step 7

	Task* t = scinew Task("ICE::step7", patch, old_dw, new_dw,
			      this, &ICE::actuallyStep7);
	t->requires(old_dw, vel_CCLabel, 0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
	//	t->computes(new_dw, vel_CCLabel,0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
	//t->whatis the cost model?();
	sched->addTask(t);
	}

	{
	Task* t = scinew Task("ICE::timeStep", patch, old_dw, new_dw,
			      this, &ICE::actuallyTimeStep);
	t->requires(old_dw, vel_CCLabel, 0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
  	t->computes(new_dw, vel_CCLabel,0,patch);
	t->computes(new_dw, temp_CCLabel,0,patch);
	t->computes(new_dw, cv_CCLabel,0,patch);
	t->computes(new_dw, press_CCLabel,0,patch);
	t->computes(new_dw, rho_CCLabel,0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
	//t->whatis the cost model?();
	sched->addTask(t);
	}
    }

    this->cheat_t=t;
    this->cheat_delt=delt;


}
void ICE::actuallyStep0(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{

  cerr << "Actually doing step 0" << endl;
  
  /*__________________________________
   *   Plotting variables
   *___________________________________*/
#if (switchDebug_main == 1|| switchDebug_main == 2 || switchDebug_main_input == 1)
#include "plot_declare_vars.h"   
#endif
  stat = putenv("PGPLOT_DIR=/usr/people/jas/PSE/src/Uintah/Components/ICE/ice_sm/Libraries");
  stat = putenv("PGPLOT_I_AM_HERE=0");              
  /* tell the plotting routine that  */
  /* you're at the top of main       */      
  
  stat = putenv("PGPLOT_PLOTTING_ON_OFF=1");
  stat = putenv("PGPLOT_OPEN_NEW_WINDOWS=1");  
  
  
  /*__________________________________
   * update the physical boundary conditions
   * and initialize some arrays
   *___________________________________*/                        
  update_CC_FC_physical_boundary_conditions( 
					    xLoLimit, yLoLimit, zLoLimit,      
					    xHiLimit,yHiLimit, zHiLimit,
					    delX,delY, delZ,
					    BC_types, BC_float_or_fixed,
					    BC_Values, 
					    nMaterials, 3,                 
					    uvel_CC, UVEL, uvel_FC,
					    vvel_CC, VVEL, vvel_FC,
					    wvel_CC, WVEL, wvel_FC);
  
    update_CC_physical_boundary_conditions( 
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,
                        BC_types,       BC_float_or_fixed,
                        BC_Values, 
                        nMaterials,     3,                 
                        Temp_CC,TEMP,   rho_CC,DENSITY, press_CC,PRESS);
                        
    zero_arrays_4d(
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,
                        1,              nMaterials,     8,             
                        mass_source,    delPress_CC,    int_eng_source,  
                        xmom_source,    ymom_source,    zmom_source,
                        Vol_L_CC,       mass_CC);


    /*__________________________________
    *   Find the new time step based on the
    *   Courant condition
    *___________________________________*/        
    find_delta_time_based_on_CC_vel(
                        xLoLimit,        yLoLimit,      zLoLimit,
                        xHiLimit,        yHiLimit,      zHiLimit,
                        &delt,           delt_limits,
                        delX,            delY,          delZ,
                        uvel_CC,         vvel_CC,       wvel_CC,
                        speedSound,      CFL,           nMaterials );                      



}

void ICE::actuallyStep1(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
  
  /*__________________________________
   *   S  T  E  P     1 
   *  Use the equation of state to get
   *  P at the cell center
   *___________________________________*/
  // Inputs = 
  // Outputs = 

  CCVariable<double> press_cc;
  CCVariable<double> rho_cc;
  CCVariable<double> temp_cc;
  CCVariable<double> cv_cc;

  old_dw->get(press_cc,press_CCLabel, 0, patch, Ghost::None, 0);
  old_dw->get(rho_cc,rho_CCLabel, 0, patch, Ghost::None, 0);
  old_dw->get(temp_cc,temp_CCLabel, 0, patch, Ghost::None, 0);
  old_dw->get(cv_cc,cv_CCLabel, 0, patch, Ghost::None, 0);
#if 1
  convertUCFToNR_4d(patch,press_cc,press_CC,xLoLimit,xHiLimit,
		    yLoLimit,yHiLimit,zLoLimit,zHiLimit,nMaterials);

  convertUCFToNR_4d(patch,rho_cc,rho_CC,xLoLimit,xHiLimit,
		    yLoLimit,yHiLimit,zLoLimit,zHiLimit,nMaterials);

  convertUCFToNR_4d(patch,temp_cc,Temp_CC,xLoLimit,xHiLimit,
		    yLoLimit,yHiLimit,zLoLimit,zHiLimit,nMaterials);

  convertUCFToNR_4d(patch,cv_cc,cv_CC,xLoLimit,xHiLimit,
		    yLoLimit,yHiLimit,zLoLimit,zHiLimit,nMaterials);
#endif


#if switch_step1_OnOff
  cerr << "Actually doing step 1" << endl;
  equation_of_state(
		    xLoLimit,       yLoLimit,       zLoLimit,
		    xHiLimit,       yHiLimit,       zHiLimit,
		    R,
		    press_CC,       rho_CC,         Temp_CC,
		    cv_CC,          nMaterials   );
  
  speed_of_sound(
		 xLoLimit,       yLoLimit,       zLoLimit,       
		 xHiLimit,       yHiLimit,       zHiLimit,       
		 gamma,          R,              Temp_CC,     
		 speedSound,     nMaterials   );
#endif
  
  // convert NR stuff back to ucf and store in data warehouse

}

void ICE::actuallyStep2(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{

  CCVariable<Vector> vel_cc;
  old_dw->get(vel_cc, vel_CCLabel, 0, patch, Ghost::None,0);
  // Convert from UCF to NR
#if 1
  convertUCFToNR_4d(patch,vel_cc,uvel_CC,vvel_CC,wvel_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,nMaterials);
#endif
    
  cerr << "Actually doing step 2" << endl;
    /*__________________________________
    *    S  T  E  P     2 
    *   Use Euler's equation thingy to solve
    *   for the n+1 Lagrangian press (CC)
    *   and the n+1 face centered fluxing
    *   velocity
    *___________________________________*/ 
     /*__________________________________
    *   Take (*)vel_CC and interpolate it to the 
    *   face-center.  Advection operator needs
    *   uvel_FC and so does the pressure solver
    *___________________________________*/ 

  // Inputs = 
  // Outputs = 


  stat = putenv("PGPLOT_PLOTTING_ON_OFF=1"); 

  compute_face_centered_velocities( 
				   xLoLimit, yLoLimit, zLoLimit,
				   xHiLimit, yHiLimit, zHiLimit,
				   delX,delY, delZ,
				   delt,     
				   BC_types, BC_float_or_fixed,
				   BC_Values,
				   rho_CC, grav, press_CC,
				   uvel_CC, vvel_CC, wvel_CC,
				   uvel_FC, vvel_FC, wvel_FC,
				   nMaterials ); 
  
  
  divergence_of_face_centered_velocity(  
				       xLoLimit,yLoLimit,zLoLimit,
				       xHiLimit,yHiLimit, zHiLimit,
				       delX,delY, delZ,
				       uvel_FC, vvel_FC,wvel_FC,
				       div_velFC_CC,nMaterials); 

  stat = putenv("PGPLOT_PLOTTING_ON_OFF=1");


#if switch_step2_OnOff                        
  
  explicit_delPress(  
		    xLoLimit,       yLoLimit,       zLoLimit,
		    xHiLimit,       yHiLimit,       zHiLimit,
		    delX,           delY,           delZ,
		    div_velFC_CC,
		    delPress_CC,    press_CC,
		    rho_CC,         delt,           speedSound,
		    nMaterials );
  
  update_CC_physical_boundary_conditions( 
					 xLoLimit,yLoLimit,zLoLimit,
					 xHiLimit,yHiLimit, zHiLimit,
					 delX, delY,delZ,
					 BC_types,BC_float_or_fixed,
					 BC_Values, nMaterials, 1,           
					 delPress_CC,    DELPRESS);
  
#endif     
  // Convert back to ucf format and store in data warehouse


}

void ICE::actuallyStep3(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{

    /* ______________________________   
    *    S  T  E  P     3    
    *   Compute the face-centered pressure
    *   using the "continuity of acceleration"
    *   principle                     
    * ______________________________   */

#if switch_step3_OnOff                                  
  cerr << "Actually doing step 3" << endl;
        press_face(         
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        BC_types,       BC_float_or_fixed, BC_Values,
                        press_CC,       press_FC,       rho_CC, 
                        nMaterials );
#endif
}

void ICE::actuallyStep4(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{

  CCVariable<double> mass_cc;
  CCVariable<double> temp_cc;
  CCVariable<double> rho_cc;
  CCVariable<Vector> tau_cc;
  CCVariable<double> viscosity_cc;
  CCVariable<double> del_press_cc;

    /* ______________________________  
    *    S  T  E  P     4                               
    *   Compute sources of mass, momentum and energy
    *   For momentum, there are sources
    *   due to mass conversion, gravity
    *   pressure, divergence of the stress
    *   and momentum exchange
    * ______________________________   */
#if (switch_step4_OnOff == 1 && switch_Compute_burgers_eq == 0) 
  cerr << "Actually doing step 4" << endl;
  accumulate_momentum_source_sinks(
                        xLoLimit,       yLoLimit,       zLoLimit,                  
                        xHiLimit,       yHiLimit,       zHiLimit,                  
                        delt,                      
                        delX,           delY,           delZ,                      
                        grav,                  
                        mass_CC,        rho_CC,         press_FC,            
                        Temp_CC,        cv_CC,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        tau_X_FC,       tau_Y_FC,       tau_Z_FC,               
                        viscosity_CC,              
                        xmom_source,    ymom_source,    zmom_source,           
                        nMaterials   ); 

 
   accumulate_energy_source_sinks(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delt,            
                        delX,           delY,           delZ,    
                        grav,           mass_CC,        rho_CC,          
                        press_CC,       delPress_CC,    Temp_CC,         
                        cv_CC,          speedSound,     
                        uvel_CC,        vvel_CC,        wvel_CC,
                        div_velFC_CC,         
                        int_eng_source,  
                        nMaterials   );

    #endif

}

void ICE::actuallyStep5(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{

     /*__________________________________
    *    S  T  E  P     5                        
    *   Compute Lagrangian values for the volume 
    *   mass, momentum and energy.
    *   Lagrangian values are the sum of the time n
    *   values and the sources computed in 4
    *___________________________________*/
 cerr << "Actually doing step 5" << endl;
#if switch_step5_OnOff 
    lagrangian_vol(     xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        delt,           
                        Vol_L_CC,       Vol_CC,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        nMaterials);
                        
    calc_flux_or_primitive_vars(    -1,           
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        rho_CC,         Vol_CC,         
                        uvel_CC,        vvel_CC,        wvel_CC,        
                        xmom_CC,        ymom_CC,        zmom_CC,
                        cv_CC,          int_eng_CC,     Temp_CC,
                        nMaterials );                       
                        
    lagrangian_values(  
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        Vol_L_CC,       Vol_CC,         rho_CC,
                        rho_L_CC,
                        xmom_CC,        ymom_CC,        zmom_CC,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        xmom_L_CC,      ymom_L_CC,      zmom_L_CC,
                        mass_L_CC,      mass_source,    
                        xmom_source,    ymom_source,    zmom_source,
                        int_eng_CC,     int_eng_L_CC,   int_eng_source,
                        nMaterials);
    #endif  

}

void ICE::actuallyStep6(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{

    /*_________________________________   
    *    S  T  E  P     6                            
    *   Compute the advection of mass,
    *   momentum and energy.  These
    *   quantities are advected using the face
    *   centered velocities velocities from 2
    *                  
    *    S  T  E  P     7 
    *   Compute the time advanced values for
    *   mass, momentum and energy.  "Time advanced"
    *   means the sum of the "Lagrangian" values,
    *   found in 5 and the advection contribution
    *   from 6                      
    *______________________________ */  
    #if (switch_step7_OnOff== 1 || switch_step6_OnOff == 1)
  cerr << "Actually doing step 6" << endl;
     advect_and_advance_in_time(   
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        Vol_CC,         rho_CC,
                        xmom_CC,        ymom_CC,        zmom_CC,
                        Vol_L_CC,       rho_L_CC,       mass_L_CC,
                        xmom_L_CC,      ymom_L_CC,      zmom_L_CC,
                        int_eng_CC,     int_eng_L_CC,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        delt,           nMaterials);

         
    /*__________________________________
    *   Backout the velocities from the 
    *   the momentum
    *___________________________________*/                        
    calc_flux_or_primitive_vars(    1,           
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        rho_CC,         Vol_CC,         
                        uvel_CC,        vvel_CC,        wvel_CC,        
                        xmom_CC,        ymom_CC,        zmom_CC,
                        cv_CC,          int_eng_CC,     Temp_CC,
                        nMaterials ); 
    #endif

}

void ICE::actuallyStep7(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{

  cerr << "Actually doing step 7" << endl;
}



void ICE::actuallyTimeStep(const ProcessorGroup*,
			   const Patch* patch,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw)
{


  cerr << "Actually doing the time step" << endl;
#if 1
  double t = this->cheat_t;
  double delt = this->cheat_delt;

#if 1
  /*__________________________________
   *   Plotting variables
   *___________________________________*/
#if (switchDebug_main == 1|| switchDebug_main == 2 || switchDebug_main_input == 1)
#include "plot_declare_vars.h"   
#endif
  stat = putenv("PGPLOT_DIR=/usr/people/jas/PSE/src/Uintah/Components/ICE/ice_sm/Libraries");
  stat = putenv("PGPLOT_I_AM_HERE=0");              
  /* tell the plotting routine that  */
  /* you're at the top of main       */      
  
  stat = putenv("PGPLOT_PLOTTING_ON_OFF=1");
  stat = putenv("PGPLOT_OPEN_NEW_WINDOWS=1");  
#endif
  
  /*______________________________________________________________________
   *   M  A  I  N     A  D  V  A  N  C  E     L  O  O  P 
   *_______________________________________________________________________*/                      
  cerr << "Beginning of while loop " << endl;
  cerr << "t = " << t << " t_final = " << endl;
  //    while( t <= t_final)
  {
    should_I_write_output = Is_it_time_to_write_output( t, t_output_vars  );
    /* fprintf(stderr, "should _ I write_output %i\n",should_I_write_output); */
    
#if 0
    /*__________________________________
     * update the physical boundary conditions
     * and initialize some arrays
     *___________________________________*/                        
    update_CC_FC_physical_boundary_conditions( 
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,
                        BC_types,       BC_float_or_fixed,
                        BC_Values, 
                        nMaterials,     3,                 
                        uvel_CC,        UVEL,           uvel_FC,
                        vvel_CC,        VVEL,           vvel_FC,
                        wvel_CC,        WVEL,           wvel_FC);
                        
    update_CC_physical_boundary_conditions( 
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,
                        BC_types,       BC_float_or_fixed,
                        BC_Values, 
                        nMaterials,     3,                 
                        Temp_CC,TEMP,   rho_CC,DENSITY, press_CC,PRESS);
                        
    zero_arrays_4d(
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,
                        1,              nMaterials,     8,             
                        mass_source,    delPress_CC,    int_eng_source,  
                        xmom_source,    ymom_source,    zmom_source,
                        Vol_L_CC,       mass_CC);


    /*__________________________________
     *   Find the new time step based on the
     *   Courant condition
     *___________________________________*/        
    find_delta_time_based_on_CC_vel(
                        xLoLimit,        yLoLimit,      zLoLimit,
                        xHiLimit,        yHiLimit,      zHiLimit,
                        &delt,           delt_limits,
                        delX,            delY,          delZ,
                        uvel_CC,         vvel_CC,       wvel_CC,
                        speedSound,      CFL,           nMaterials );                      
#endif

#if 0
    /*__________________________________
     *   S  T  E  P     1 
     *  Use the equation of state to get
     *  P at the cell center
     *___________________________________*/
#if switch_step1_OnOff
    equation_of_state(
		      xLoLimit,       yLoLimit,       zLoLimit,
		      xHiLimit,       yHiLimit,       zHiLimit,
		      R,
		      press_CC,       rho_CC,         Temp_CC,
		      cv_CC,          nMaterials   );
    
    speed_of_sound(
		   xLoLimit,       yLoLimit,       zLoLimit,       
		   xHiLimit,       yHiLimit,       zHiLimit,       
		   gamma,          R,              Temp_CC,     
		   speedSound,     nMaterials   );
#endif
#endif

#if 0
    /*__________________________________
     *    S  T  E  P     2 
     *   Use Euler's equation thingy to solve
     *   for the n+1 Lagrangian press (CC)
     *   and the n+1 face centered fluxing
     *   velocity
     *___________________________________*/ 
    /*__________________________________
     *   Take (*)vel_CC and interpolate it to the 
     *   face-center.  Advection operator needs
     *   uvel_FC and so does the pressure solver
     *___________________________________*/ 
    stat = putenv("PGPLOT_PLOTTING_ON_OFF=1"); 
    compute_face_centered_velocities( 
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        delt,           
                        BC_types,       BC_float_or_fixed,
                        BC_Values,
                        rho_CC,         grav,           press_CC,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        nMaterials ); 
                        
                        
    divergence_of_face_centered_velocity(  
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        div_velFC_CC,   nMaterials); 
    stat = putenv("PGPLOT_PLOTTING_ON_OFF=1");


#if switch_step2_OnOff                        
    
    explicit_delPress
      (  
       xLoLimit,       yLoLimit,       zLoLimit,
       xHiLimit,       yHiLimit,       zHiLimit,
       delX,           delY,           delZ,
       div_velFC_CC,
       delPress_CC,    press_CC,
       rho_CC,         delt,           speedSound,
       nMaterials );
                
    update_CC_physical_boundary_conditions( 
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,
                        BC_types,       BC_float_or_fixed,
                        BC_Values, 
                        nMaterials,     1,                 
                        delPress_CC,    DELPRESS);
                                            
#endif     
   
#endif

#if 0
    /* ______________________________   
    *    S  T  E  P     3    
    *   Compute the face-centered pressure
    *   using the "continuity of acceleration"
    *   principle                     
    * ______________________________   */

    #if switch_step3_OnOff                                  
        press_face(         
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        BC_types,       BC_float_or_fixed, BC_Values,
                        press_CC,       press_FC,       rho_CC, 
                        nMaterials );
    #endif

#endif


#if 0
    /* ______________________________  
    *    S  T  E  P     4                               
    *   Compute sources of mass, momentum and energy
    *   For momentum, there are sources
    *   due to mass conversion, gravity
    *   pressure, divergence of the stress
    *   and momentum exchange
    * ______________________________   */
    #if (switch_step4_OnOff == 1 && switch_Compute_burgers_eq == 0) 
    accumulate_momentum_source_sinks(
                        xLoLimit,       yLoLimit,       zLoLimit,                  
                        xHiLimit,       yHiLimit,       zHiLimit,                  
                        delt,                      
                        delX,           delY,           delZ,                      
                        grav,                  
                        mass_CC,        rho_CC,         press_FC,            
                        Temp_CC,        cv_CC,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        tau_X_FC,       tau_Y_FC,       tau_Z_FC,               
                        viscosity_CC,              
                        xmom_source,    ymom_source,    zmom_source,           
                        nMaterials   ); 

 
   accumulate_energy_source_sinks(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delt,            
                        delX,           delY,           delZ,    
                        grav,           mass_CC,        rho_CC,          
                        press_CC,       delPress_CC,    Temp_CC,         
                        cv_CC,          speedSound,     
                        uvel_CC,        vvel_CC,        wvel_CC,
                        div_velFC_CC,         
                        int_eng_source,  
                        nMaterials   );

    #endif

#endif

#if 0

    /*__________________________________
    *    S  T  E  P     5                        
    *   Compute Lagrangian values for the volume 
    *   mass, momentum and energy.
    *   Lagrangian values are the sum of the time n
    *   values and the sources computed in 4
    *___________________________________*/
    #if switch_step5_OnOff 
    lagrangian_vol(     xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        delt,           
                        Vol_L_CC,       Vol_CC,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        nMaterials);
                        
    calc_flux_or_primitive_vars(    -1,           
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        rho_CC,         Vol_CC,         
                        uvel_CC,        vvel_CC,        wvel_CC,        
                        xmom_CC,        ymom_CC,        zmom_CC,
                        cv_CC,          int_eng_CC,     Temp_CC,
                        nMaterials );                       
                        
    lagrangian_values(  
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        Vol_L_CC,       Vol_CC,         rho_CC,
                        rho_L_CC,
                        xmom_CC,        ymom_CC,        zmom_CC,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        xmom_L_CC,      ymom_L_CC,      zmom_L_CC,
                        mass_L_CC,      mass_source,    
                        xmom_source,    ymom_source,    zmom_source,
                        int_eng_CC,     int_eng_L_CC,   int_eng_source,
                        nMaterials);
    #endif  

#endif

#if 0
                                     
    /*_________________________________   
    *    S  T  E  P     6                            
    *   Compute the advection of mass,
    *   momentum and energy.  These
    *   quantities are advected using the face
    *   centered velocities velocities from 2
    *                  
    *    S  T  E  P     7 
    *   Compute the time advanced values for
    *   mass, momentum and energy.  "Time advanced"
    *   means the sum of the "Lagrangian" values,
    *   found in 5 and the advection contribution
    *   from 6                      
    *______________________________ */  
    #if (switch_step7_OnOff== 1 || switch_step6_OnOff == 1)
     advect_and_advance_in_time(   
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        Vol_CC,         rho_CC,
                        xmom_CC,        ymom_CC,        zmom_CC,
                        Vol_L_CC,       rho_L_CC,       mass_L_CC,
                        xmom_L_CC,      ymom_L_CC,      zmom_L_CC,
                        int_eng_CC,     int_eng_L_CC,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        delt,           nMaterials);

         
    /*__________________________________
    *   Backout the velocities from the 
    *   the momentum
    *___________________________________*/                        
    calc_flux_or_primitive_vars(    1,           
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        rho_CC,         Vol_CC,         
                        uvel_CC,        vvel_CC,        wvel_CC,        
                        xmom_CC,        ymom_CC,        zmom_CC,
                        cv_CC,          int_eng_CC,     Temp_CC,
                        nMaterials ); 
    #endif

#endif

    /*__________________________________
    *    T  E  C  P  L  O  T  
    *___________________________________*/     
     
    #if tecplot    
    if ( should_I_write_output == YES)
    {                     
        tecplot_CC(         
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        x_CC,           y_CC,           z_CC,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        press_CC,       Temp_CC,        rho_CC,
                        scalar1_CC,     scalar2_CC,     scalar3_CC,
                        fileNum,        output_file_basename,       output_file_desc,
                        nMaterials);

        tecplot_FC(         
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        x_FC,           y_FC,           z_FC,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        fileNum,        output_file_basename,       output_file_desc,
                        nMaterials );
                            
        fileNum ++;
    } 
    #endif 
    /*__________________________________
     *  P  L  O  T  T  I  N  G     S  E  C  T  I  O  N 
     *___________________________________*/
#if switchDebug_main
    if ( should_I_write_output == NO)
      {
#define switchInclude_main 1
#include "debugcode.i"
#undef switchInclude_main 
      }
#endif
    /*__________________________________
     *  Clean up the plotting windows 
     *___________________________________*/
    putenv("PGPLOT_I_AM_HERE=1");              
    /* tell the plotting routine that   */
    /* you're at the bottom of main     */
    putenv("PGPLOT_OPEN_NEW_WINDOWS=1"); 
    
    
    /*__________________________________
     *    A  D  V  A  N  C  E     I  N     T  I  M  E 
     *___________________________________*/
    
    t = t + delt;
    fprintf(stderr,"\nTime is %f, timestep is %f\n",t,delt);
    std::cerr << "Time is " << t << " timestep is " << delt << std::endl;
    
  }

  
  // Added by Steve for sanity checking
  double sumRho=0;
  double sumEng=0;
  int m=1;
  for ( int i = xLoLimit; i <= xHiLimit; i++){
    for ( int j = yLoLimit; j <= yHiLimit; j++){
      for ( int k = zLoLimit; k <= zHiLimit; k++){ 
	sumRho += rho_CC[m][i][j][k];
	sumEng += int_eng_CC[m][i][j][k];
      }
    }
  }
  cerr << "sum rho=" << sumRho << '\n';
  cerr << "sum eng=" << sumEng << '\n';
  cerr << "ii=" << int_eng_CC[5][5][1][1] << '\n';
#endif
  
  // Allocate and store the new vel_CC stuff
  
  CCVariable<Vector> new_vel_cc;
  new_dw->allocate(new_vel_cc,vel_CCLabel,0,patch);
  
#if 1
  convertNR_4dToUCF(patch,new_vel_cc,uvel_CC,vvel_CC,wvel_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,
		    nMaterials);
#endif
  new_dw->put(new_vel_cc,vel_CCLabel,0,patch);     

  // temperature
  CCVariable<double> new_temp_cc;
  new_dw->allocate(new_temp_cc,temp_CCLabel,0,patch);
  
#if 1
  convertNR_4dToUCF(patch,new_temp_cc,Temp_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,
		    nMaterials);
#endif
  new_dw->put(new_temp_cc,temp_CCLabel,0,patch);     

  // cv 
  CCVariable<double> new_cv_cc;
  new_dw->allocate(new_cv_cc,cv_CCLabel,0,patch);
  
#if 1
  convertNR_4dToUCF(patch,new_cv_cc,cv_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,
		    nMaterials);
#endif
  new_dw->put(new_cv_cc,cv_CCLabel,0,patch);     

// pressure 
  CCVariable<double> new_press_cc;
  new_dw->allocate(new_press_cc,press_CCLabel,0,patch);
  
#if 1
  convertNR_4dToUCF(patch,new_press_cc,press_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,
		    nMaterials);
#endif
  new_dw->put(new_press_cc,press_CCLabel,0,patch);     

// rho
  CCVariable<double> new_rho_cc;
  new_dw->allocate(new_rho_cc,rho_CCLabel,0,patch);
  
#if 1
  convertNR_4dToUCF(patch,new_rho_cc,rho_CC,xLoLimit,
		    xHiLimit,yLoLimit,yHiLimit,zLoLimit,zHiLimit,
		    nMaterials);
#endif
  new_dw->put(new_rho_cc,rho_CCLabel,0,patch);     

}

void ICE::convertNR_4dToUCF(const Patch* patch,CCVariable<Vector>& vel_ucf, 
			  double ****uvel_CC,
			  double ****vvel_CC,
			  double **** wvel_CC,
			  int xLoLimit,
			  int xHiLimit,
			  int yLoLimit,
			  int yHiLimit,
			  int zLoLimit,
			  int zHiLimit,
			  int nMaterials)
{

  
  for (CellIterator iter = patch->getCellIterator(patch->getBox()); 
       !iter.done(); iter++) {
    // Do something
  }

  CellIterator iter = patch->getCellIterator(patch->getBox());
  cerr << "CC iterator begin = " << iter.begin() << " end = " << iter.end() 
       << endl;
  
  cerr << "CC variables limits " << vel_ucf.getLowIndex() << " " 
       << vel_ucf.getHighIndex() << endl;

  cerr << "NR limits: [" << xLoLimit << " " << yLoLimit << " " << zLoLimit << "] [ " 
       << xHiLimit << " " << yHiLimit << " " << zHiLimit << "]" << endl;  

  for (int i = xLoLimit; i <= xHiLimit; i++) {
    for (int j = yLoLimit; j <= yHiLimit; j++) {
      for (int k = zLoLimit; k <= zHiLimit; k++) {
	for (int m = 1; m <= nMaterials; m++) {
	  // Do something
	  //  cerr << "uvel = " << uvel_CC[m][i][j][k] 
	  //     << " vvel = " << vvel_CC[m][i][j][k] 
	  //     << " wvel = " << wvel_CC[m][i][j][k] << endl;
	  IntVector idx(i-1,j-1,k-1);
	  vel_ucf[idx]=Vector(uvel_CC[m][i][j][k], vvel_CC[m][i][j][k], 
			      wvel_CC[m][i][j][k]);
	  //cerr << "vel_ucf = " << vel_ucf[idx] << endl;
	}
      }
    }
  }

  return;

}


void ICE::convertNR_4dToUCF(const Patch* patch,CCVariable<double>& scalar_ucf, 
			  double ****scalar_CC,
			  int xLoLimit,
			  int xHiLimit,
			  int yLoLimit,
			  int yHiLimit,
			  int zLoLimit,
			  int zHiLimit,
			  int nMaterials)
{

  
  for (CellIterator iter = patch->getCellIterator(patch->getBox()); !iter.done(); iter++) {
  }

  CellIterator iter = patch->getCellIterator(patch->getBox());
  cerr << "CC iterator begin = " << iter.begin() << " end = " << iter.end() << endl;
  
  cerr << "CC variables limits " << scalar_ucf.getLowIndex() << " " 
       << scalar_ucf.getHighIndex() << endl;

  cerr << "NR limits: [" << xLoLimit << " " << yLoLimit << " " << zLoLimit << "] [ " 
       << xHiLimit << " " << yHiLimit << " " << zHiLimit << "]" << endl;  

  for (int i = xLoLimit; i <= xHiLimit; i++) {
    for (int j = yLoLimit; j <= yHiLimit; j++) {
      for (int k = zLoLimit; k <= zHiLimit; k++) {
	for (int m = 1; m <= nMaterials; m++) {
	  // Do something
	  //  cerr << "scalar = " << scalar_CC[m][i][j][k] << endl;
	  IntVector idx(i-1,j-1,k-1);
	  scalar_ucf[idx]=scalar_CC[m][i][j][k];
	  //cerr << "scalar_ucf = " << scalar_ucf[idx] << endl;
	}
      }
    }
  }

  return;

}


void ICE::convertUCFToNR_4d(const Patch* patch,CCVariable<Vector>& vel_ucf, 
			  double ****uvel_CC,
			  double ****vvel_CC,
			  double **** wvel_CC,
			  int xLoLimit,
			  int xHiLimit,
			  int yLoLimit,
			  int yHiLimit,
			  int zLoLimit,
			  int zHiLimit,
			  int nMaterials)
{

  
  for (CellIterator iter = patch->getCellIterator(patch->getBox()); !iter.done(); iter++) {
  }

  CellIterator iter = patch->getCellIterator(patch->getBox());
  cerr << "CC iterator begin = " << iter.begin() << " end = " << iter.end() 
       << endl;
  
  cerr << "CC variables limits " << vel_ucf.getLowIndex() << " " 
       << vel_ucf.getHighIndex() << endl;

  cerr << "NR limits: [" << xLoLimit << " " << yLoLimit << " " << zLoLimit
       << "] [ " 
       << xHiLimit << " " << yHiLimit << " " << zHiLimit << "]" << endl;  

  for (int i = xLoLimit; i <= xHiLimit; i++) {
    for (int j = yLoLimit; j <= yHiLimit; j++) {
      for (int k = zLoLimit; k <= zHiLimit; k++) {
	for (int m = 1; m <= nMaterials; m++) {
	  // Do something
	  //  cerr << "uvel = " << uvel_CC[m][i][j][k] 
	  //     << " vvel = " << vvel_CC[m][i][j][k] 
	  //     << " wvel = " << wvel_CC[m][i][j][k] << endl;
	  IntVector idx(i-1,j-1,k-1);
	  uvel_CC[m][i][j][k]= vel_ucf[idx].x();
	  vvel_CC[m][i][j][k]= vel_ucf[idx].y();
	  wvel_CC[m][i][j][k]= vel_ucf[idx].z();
	}
      }
    }
  }

  return;

}


void ICE::convertUCFToNR_4d(const Patch* patch,CCVariable<double>& scalar_ucf, 
			  double ****scalar_CC,
			  int  xLoLimit,
			  int xHiLimit,
			  int yLoLimit,
			  int yHiLimit,
			  int zLoLimit,
			  int zHiLimit,
			  int nMaterials)
{

  
  for (CellIterator iter = patch->getCellIterator(patch->getBox()); !iter.done(); iter++) {
  }

  CellIterator iter = patch->getCellIterator(patch->getBox());
  cerr << "CC iterator begin = " << iter.begin() << " end = " << iter.end() 
       << endl;
  
  cerr << "CC variables limits " << scalar_ucf.getLowIndex() << " " 
       << scalar_ucf.getHighIndex() << endl;

  cerr << "NR limits: [" << xLoLimit << " " << yLoLimit << " " << zLoLimit
       << "] [ " 
       << xHiLimit << " " << yHiLimit << " " << zHiLimit << "]" << endl;  

  for (int i = xLoLimit; i <= xHiLimit; i++) {
    for (int j = yLoLimit; j <= yHiLimit; j++) {
      for (int k = zLoLimit; k <= zHiLimit; k++) {
	for (int m = 1; m <= nMaterials; m++) {
	  // Do something
	  //  cerr << "scalar = " << scalar_CC[m][i][j][k] << endl;
	  IntVector idx(i-1,j-1,k-1);
	  scalar_CC[m][i][j][k]= scalar_ucf[idx];
	}
      }
    }
  }

  return;

}
