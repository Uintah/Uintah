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
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/SoleVariable.h>
#include <SCICore/Geometry/Vector.h>
using SCICore::Geometry::Vector;

#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <stdlib.h>
#include <iostream>
using std::cerr;
#if 0
#include "ICE/Header_files/nrutil+.h"
#include "ICE/Header_files/functionDeclare.h"
#include "ICE/Header_files/parameters.h"
#include "ICE/Header_files/switches.h"
#include "ICE/Header_files/macros.h"
#include "ICE/Header_files/cpgplot.h"            /*must have this for plotting to work   */
#endif

extern "C" void audit();

using Uintah::Components::ICE;

ICE::ICE()
{
#if 0
/*______________________________________________________________________ 
*                       MEMORY SECTION
* 
*  Allocate memory for the arrays                                          
*_______________________________________________________________________*/
    nMaterials=1;
audit();
#include "ICE/Header_files/allocate_memory.i"
audit();
#endif
}

ICE::~ICE()
{
#if 0
/* -----------------------------------------------------------------------  
*  Free the memory                                                         
* -----------------------------------------------------------------------  */
    fprintf(stderr,"Now deallocating memory\n");
#include "ICE/Header_files/free_memory.i"
#endif
}

void ICE::problemSetup(const ProblemSpecP& params, GridP& grid,
		       DataWarehouseP& ds)
{
#if 0
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
                        x_FC,           y_FC,           z_FC,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        press_FC,
                        tau_x_FC,       tau_y_FC,       tau_z_FC,
                        nMaterials);   
audit();
/*______________________________________________________________________
*
*                       PROBLEM INITIALIZATION SECTION
*   Initializing routines                                                  
*   First read the problem input then test the inputs.                     
* -----------------------------------------------------------------------  */

    double t_final;
    int printSwitch = 1;
    readInputFile(   &xLoLimit,      &yLoLimit,      &zLoLimit,     
		     &xHiLimit,      &yHiLimit,      &zHiLimit,
		     &delX,          &delY,          &delZ,
		     uvel_CC,        vvel_CC,        wvel_CC, 
		     Temp_CC,        press_CC,       rho_CC,
		     scalar1_CC,     scalar2_CC,     scalar3_CC,
		     viscosity_CC,   thermalCond_CC, cv_CC,
		     R,
		     &t_final,       t_output_vars,  delt_limits,
		     output_file_basename,           output_file_desc,       
		     grav,           speedSound,
		     BC_inputs,      BC_Values,      &nMaterials);      
    
    testInputFile(      xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,      
                        t_final,        t_output_vars,  delt_limits,
                        BC_inputs,      printSwitch,    nMaterials); 
                   
    definition_of_different_boundary_conditions(              
                        BC_inputs,      BC_types,       BC_float_or_fixed,
                        BC_Values,      nMaterials );  
                        
                        
    update_boundary_conditions( 
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,                 
                        uvel_CC,        vvel_CC,        wvel_CC,          
                        press_CC,       Temp_CC,        rho_CC,        
                        uvel_FC,        vvel_FC,        wvel_FC,
                        press_FC,       
                        BC_types,       BC_float_or_fixed, 
                        BC_Values,      nMaterials        );
     
    generateGrid(       xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        x_CC,           y_CC,           z_CC,   Vol_CC,  
                        x_FC,           y_FC,           z_FC );

                         
int m=1;
     for ( int i = xLoLimit-1; i <= xHiLimit+1; i++){
	 for ( int j = yLoLimit-1; j <= yHiLimit+1; j++){
	     for ( int k = zLoLimit-1; k <= zHiLimit+1; k++){ 
		 mass_source[i][j][k][m]=0;
		 xmom_source[i][j][k][m]=ymom_source[i][j][k][m]=zmom_source[i][j][k][m]=0;
		 int_eng_source[i][j][k][m]=0;
		 delPress_CC[i][j][k][m]=0;
	     }
	 }
     }
    putenv("PGPLOT_PLOTTING_ON_OFF=1");
                            
/*______________________________________________________________________
*   Plot the inputs (MUST HARDWIRE WHAT YOU WANT TO VIEW)
*   To keep the code clean I moved the code to another file
*_______________________________________________________________________*/
#if switchDebug_main_input
    #define switchInclude_main_1 1
    #include "debugcode.i"
    #undef switchInclude_main_1
#endif     

/*______________________________________________________________________
*  TESTING: HARDWIRE SOME OF THE INPUTS
*       HARDWIRE FOR NOW
*_______________________________________________________________________*/
                               
#include "ICE/Header_files/advection_test.i"
#if 0
    CCVariable<Vector> vel_CC_var(...);
    ds.put("vel_CC", vel_CC_var);
#else
    cerr << "put vel_CC not done\n";
#endif
#endif
}

void ICE::scheduleStableTimestep(const LevelP& level,
				 SchedulerP& sched, DataWarehouseP& dw)
{
#if 0
    for(Level::const_regionIterator iter=level->regionsBegin();
	iter != level->regionsEnd(); iter++){
	const Region* region=*iter;
	Task* t = new Task("ICE::computeStableTimestep", region, dw, dw,
			   this, ICE::actuallyComputeStableTimestep);
	t->requires(dw, "vel_CC", region, 0, 
		    CCVariable<Vector>::getTypeDescription());
	t->requires(dw, "params", ProblemSpec::getTypeDescription());
	t->computes(dw, "delt", SoleVariable<double>::getTypeDescription());
	t->usesMPI(false);
	t->usesThreads(false);
	t->subregionCapable(true);
	//t->whatis the cost model?();
	sched->addTask(t);
    }
#endif
}

void ICE::actuallyComputeStableTimestep(const ProcessorContext*,
					const Region* region,
					const DataWarehouseP& fromDW,
					DataWarehouseP& toDW)
{
#if 0
#if 0
    CCVariable<Vector> vel_CC;
    fromDW->get(vel_CC, "vel_CC", region);
#endif

    /*__________________________________
     *   Find the new time step based on the
     *   Courant condition
     *___________________________________*/
    double delt;
    find_delta_time(
		    xLoLimit,        yLoLimit,      zLoLimit,
		    xHiLimit,        yHiLimit,      zHiLimit,
		    &delt,           delt_limits,
		    delX,            delY,          delZ,
		    uvel_CC,         vvel_CC,       wvel_CC,
		    nMaterials );

    //toDW->put("delt", delt);
#endif
}

void ICE::scheduleTimeAdvance(double t, double delt,
			      const LevelP& level, SchedulerP& sched,
			      const DataWarehouseP& old_dw, DataWarehouseP& new_dw)
{
#if 0
    for(Level::const_regionIterator iter=level->regionsBegin();
	iter != level->regionsEnd(); iter++){
	const Region* region=*iter;
	Task* t = new Task("ICE::timeStep", region, old_dw, new_dw,
			   this, ICE::actuallyTimeStep);
	t->requires(old_dw, "vel_CC", region, 0, 
		    CCVariable<Vector>::getTypeDescription());
	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
	t->computes(new_dw, "vel_CC",
		    CCVariable<Vector>::getTypeDescription());
	t->usesMPI(false);
	t->usesThreads(false);
	t->subregionCapable(false);
	//t->whatis the cost model?();
	sched->addTask(t);
    }
    this->cheat_t=t;
    this->cheat_delt=delt;
#endif
}

void ICE::actuallyTimeStep(const ProcessorContext*,
			   const Region* region,
			   const DataWarehouseP& fromDW,
			   DataWarehouseP& toDW)
{
#if 0
    double t = this->cheat_t;
    double delt = this->cheat_delt;
    int should_I_write_output = Is_it_time_to_write_output( t, t_output_vars  );
    /*______________________________________________________________________
     *                        MAIN ADVANCE LOOP
     *_______________________________________________________________________*/                       
    /*__________________________________
     *  STEP 1
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
#endif
    /*__________________________________
     *   STEP 2
     *   Use Euler's equation thingy to solve
     *   for the n+1 Lagrangian press (CC)
     *   and the n+1 face centered fluxing
     *   velocity
     *___________________________________*/ 
    /*__________________________________
     *   Take (*)vel_CC and interpolate it to the 
     *   face-center.  Advection operator needs
     *   uvel_FC and so does the first iteration in the
     *   pressure solver
     *___________________________________*/ 
    double switch_press_term_coeff         = 1.0;
    int comput_vel_FC_only_ghostcells   = 0;  
    compute_face_centered_velocities( 
		        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        delt,           rho_CC,         grav,
                        press_CC,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        switch_press_term_coeff,        comput_vel_FC_only_ghostcells,
                        nMaterials );  
#if switch_step2_OnOff                        
    pressure_iteration(    
		       xLoLimit,       yLoLimit,       zLoLimit,
		       xHiLimit,       yHiLimit,       zHiLimit,
		       delX,           delY,           delZ,
		       uvel_FC,        vvel_FC,        wvel_FC,
		       uvel_CC,        vvel_CC,        wvel_CC,
		       press_CC,       delPress_CC,      
		       rho_CC,         delt,           
		       grav,           BC_types,       BC_Values,
		       speedSound,     nMaterials);
                            
#endif                           


    /* ______________________________   
     *   STEP 3 
     *   Compute the face-centered pressure
     *   using the "continuity of acceleration"
     *   principle                     
     * ______________________________   */
#if switch_step3_OnOff                                  
    press_face(         
	       xLoLimit,       yLoLimit,       zLoLimit,
	       xHiLimit,       yHiLimit,       zHiLimit,
	       press_CC,       press_FC,       rho_CC, 
	       nMaterials );
#endif 
    /* ______________________________  
     *   STEP 4                          
     *   Compute ssources of mass, momentum and energy
     *   For momentum, there are sources
     *   due to mass conversion, gravity
     *   pressure, divergence of the stress
     *   and momentum exchange
     * ______________________________   */
#if switch_step4_OnOff 
    accumulate_momentum_source_sinks(
                        xLoLimit,       yLoLimit,       zLoLimit,                  
                        xHiLimit,       yHiLimit,       zHiLimit,                  
                        delt,                      
                        delX,           delY,           delZ,                      
                        grav,                  
                        mass_CC,        rho_CC,         press_FC,            
                        Temp_CC,        cv_CC,
                        uvel_CC,        vvel_CC,        wvel_CC,               
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
                        int_eng_source,  
                        nMaterials   );
#endif 

    /*__________________________________
     *   STEP 5                     
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
    /*_________________________________   
     *   STEP 6                           
     *   Compute the advection of mass,
     *   momentum and energy.  These
     *   quantities are advected using the face
     *   c	entered velocities velocities from 2
     *                  
     *   STEP 7
     *   Compute the time advanced values for
     *   mass, momentum and energy.  "Time advanced"
     *   means the sum of the "Lagrangian" values,
     *   found in 5 and the advection contribution
     *   from 6                      
     *______________________________ */  
     advect_and_advance_in_time(   
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        Vol_CC,         rho_CC,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        xmom_CC,        ymom_CC,        zmom_CC,
                        Vol_L_CC,       rho_L_CC,
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
                        
     update_boundary_conditions( 
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,                 
                        uvel_CC,        vvel_CC,        wvel_CC,          
                        press_CC,       Temp_CC,        rho_CC,        
                        uvel_FC,        vvel_FC,        wvel_FC,
                        press_FC,   
                        BC_types,       BC_float_or_fixed, 
                        BC_Values,      nMaterials        );


     /*__________________________________
      * If gradient boundary conditions are being
      * used extract what the ghostcell varialbew
      * should be set to
      *___________________________________*/                                                        
/*                             
            set_Neumann_BC(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        press_CC,       BC_types,       BC_Values,
                        m); */

    
     
     /*__________________________________
      *   Write to tecplot files
      *___________________________________*/     
     
#if write_tecplot_files
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
/*______________________________________________________________________
*   DEBUGGING SECTION
*_______________________________________________________________________*/
#if switchDebug_main
     if ( should_I_write_output == YES)
     {
#if (switchDebug_main == 1 || switchDebug_main_input == 1)
    #include "ICE/Header_files/plot_declare_vars.h"   
#endif
         #define switchInclude_main 1
         #include "ICE/Header_files/debugcode.i"
         #undef switchInclude_main
     }
     /*__________________________________
      *  Clean up the plotting windows 
      *___________________________________*/
     fprintf(stderr,"\npress return to continue\n"); 
     getchar();  
     cpgend(); 
     putenv("PGPLOT_I_AM_HERE=1");              
     /* tell the plotting routine that   */
     /* you're at the bottom of main     */
     putenv("PGPLOT_OPEN_NEW_WINDOWS=1"); 
#endif

     // Added by Steve for sanity checking
     double sumRho=0;
     double sumEng=0;
     int m=1;
     for ( int i = xLoLimit; i <= xHiLimit; i++){
	 for ( int j = yLoLimit; j <= yHiLimit; j++){
	     for ( int k = zLoLimit; k <= zHiLimit; k++){ 
		 sumRho += rho_CC[i][j][k][m];
		 sumEng += int_eng_CC[i][j][k][m];
	     }
	 }
     }
     cerr << "sum rho=" << sumRho << '\n';
     cerr << "sum eng=" << sumEng << '\n';
     cerr << "ii=" << int_eng_CC[5][5][1][1] << '\n';
#endif
}

