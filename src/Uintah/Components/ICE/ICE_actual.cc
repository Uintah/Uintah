
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

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
#include <Uintah/Grid/VarTypes.h>

#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
#include "cpgplot.h" /*must have this for plotting to work   */

using SCICore::Geometry::Vector;
using SCICore::Geometry::IntVector;
using std::cerr;
using std::endl;






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
extern "C" void audit();


using Uintah::ICESpace::ICE;

ICE::ICE()
{
    delTLabel     =   new VarLabel( "delT",       delt_vartype::getTypeDescription() );
    vel_CCLabel   =   new VarLabel( "vel_CC",     CCVariable<Vector>::getTypeDescription() );
    press_CCLabel =   new VarLabel( "press_CC",   CCVariable<double>::getTypeDescription() );
    press_CCLabel_1 = new VarLabel( "press_CC_1", CCVariable<double>::getTypeDescription() );

    rho_CCLabel   =   new VarLabel( "rho_CC",     CCVariable<double>::getTypeDescription() );
    temp_CCLabel  =   new VarLabel( "temp_CC",    CCVariable<double>::getTypeDescription() );
    cv_CCLabel    =   new VarLabel( "cv_CC",      CCVariable<double>::getTypeDescription() );

    // Face centered variables
    vel_FCLabel   =   new VarLabel( "vel_FC",     FCVariable<Vector>::getTypeDescription() );
    press_FCLabel =   new VarLabel( "press_FC",   FCVariable<double>::getTypeDescription() );
    tau_FCLabel   =   new VarLabel( "tau_FC",     FCVariable<Vector>::getTypeDescription() );

/*__________________________________
*   Plotting variables
*___________________________________*/
    stat = putenv("PGPLOT_DIR=/usr/people/harman/Csafe/PSE/src/Uintah/Components/ICE/ice_sm/Libraries");
    stat = putenv("PGPLOT_I_AM_HERE=0");              
    stat = putenv("PGPLOT_PLOTTING_ON_OFF=1");
    stat = putenv("PGPLOT_OPEN_NEW_WINDOWS=1");  
    
/*__________________________________
*   Allocate memory for the arrays
*___________________________________*/
    #include "allocate_memory.i"
   

}





ICE::~ICE()
{
    /*__________________________________
    *   Now deallocate memory
    *___________________________________*/
    fprintf(stderr,"Now deallocating memory");
    #include "free_memory.i"
}




/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  ICE::problemSetup--
 Filename:  ICE_actual.cc
 Purpose:   

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt      06/23/00                              
_____________________________________________________________________*/
void ICE::problemSetup(const ProblemSpecP& prob_spec, GridP&,
		       SimulationStateP&)
{

  double 
    viscosity,
    thermal_conductivity,
    specific_heat,
    speed_of_sound,
    ideal_gas_constant,
    d_gamma;
    
    printSwitch = 1;    
    t           = 0.0;  
    m           = 1;
    fileNum     = 1;
/*__________________________________
*   Read in from the spec file
*___________________________________*/
    ProblemSpecP mat_ps = prob_spec->findBlock("MaterialProperties");

    ProblemSpecP ice_mat_ps = mat_ps->findBlock("ICE");

    for (ProblemSpecP ps = ice_mat_ps->findBlock("material"); ps != 0;
         ps = ps->findNextBlock("material") ) 
   {
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

/*______________________________________________________________________
*
*  P  R  O  B  L  E  M     I  N  I  T  I  A  L  I  Z  A  T  I  O  N  
*  - read input file
*   - test the input variables
*   - Equate the address of the face centered variables
*   - Generate a grid
*   - zero all of the face-centered arrays
*   
*                  
* -----------------------------------------------------------------------  */
                                        
       readInputFile(   &xLoLimit,      &yLoLimit,      &zLoLimit,     
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
    
    testInputFile(      xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        Temp_CC,        press_CC,       rho_CC,
                        viscosity_CC,   thermalCond_CC, cv_CC,
                        speedSound,      
                        t_final,        t_output_vars,  delt_limits,
                        BC_inputs,      printSwitch,    CFL,
                        nMaterials); 
                   
    definition_of_different_physical_boundary_conditions(              
                        BC_inputs,      BC_types,       BC_float_or_fixed,
                        BC_Values,      nMaterials  );  
                        
/*__________________________________
* Now make sure that the face centered
* values know about each other.
* for example 
* [i][j][k][RIGHT][m] = [i-1][j][k][LEFT][m]
*___________________________________*/  

    equate_ptr_addresses_adjacent_cell_faces(              
                        x_FC,           y_FC,           z_FC,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        press_FC,
                        tau_X_FC,       tau_Y_FC,       tau_Z_FC,
                        nMaterials);   

    /*__________________________________
    * Generate a grid
    *___________________________________*/ 
    generateGrid(       xLoLimit,       yLoLimit,       zLoLimit,
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
      #include "overide_initial_conds.i"
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





/* --------------------------------------------------------------------- 
GENERAL INFORMATION
 Function:  ICE::actuallyInitialize--
 Filename:  ICE_actual.cc
 Purpose:   -allocate variables in the new DW
            - Convert the NR array data into the UCF format
            - Put the data into the DW
            

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/ 
void ICE::actuallyInitialize(
    const ProcessorGroup*,
    const Patch* patch,
    DataWarehouseP& /* old_dw */,
    DataWarehouseP& new_dw)
{
    int include_ghost_cells = YES;
    cerr <<"Doing actuallyInitialize . . ." << endl;
    CCVariable<Vector> vel_cc;
    CCVariable<double> press_cc,    press_cc_1,     rho_cc, temp_cc,    cv_cc;
    FCVariable<Vector> vel_fc,      tau_fc;
    FCVariable<double> press_fc;
    /*__________________________________
    *  Allocate variable in the new dw
    *___________________________________*/
    new_dw->allocate(   vel_cc,     vel_CCLabel,    0,patch);
    new_dw->allocate(   press_cc,   press_CCLabel,  0,patch);
    new_dw->allocate(   press_cc_1, press_CCLabel_1,0,patch);
    new_dw->allocate(   rho_cc,     rho_CCLabel,    0,patch);
    new_dw->allocate(   temp_cc,    temp_CCLabel,   0,patch);
    new_dw->allocate(   cv_cc,      cv_CCLabel,     0,patch);
    new_dw->allocate(   vel_fc,     vel_FCLabel,    0,patch);
    new_dw->allocate(   press_fc,   press_FCLabel,  0,patch);
    new_dw->allocate(   tau_fc,     tau_FCLabel,    0,patch);
    /*__________________________________
    *   Convert the array data into ucf format
    *___________________________________*/

  
    ICE::convertNR_4dToUCF(patch,
                        press_cc,       press_CC,
                        include_ghost_cells,
                        xLoLimit,       xHiLimit,       
                        yLoLimit,       yHiLimit,
                        zLoLimit,       zHiLimit,       
                        nMaterials);
  
    ICE::convertNR_4dToUCF(patch,
                        rho_cc,         rho_CC,
                        include_ghost_cells,
                        xLoLimit,       xHiLimit,       
                        yLoLimit,       yHiLimit,
                        zLoLimit,       zHiLimit,       
                        nMaterials);
   
    ICE::convertNR_4dToUCF(patch,
                        temp_cc,        Temp_CC,
                        include_ghost_cells,
                        xLoLimit,       xHiLimit,       
                        yLoLimit,       yHiLimit,
                        zLoLimit,       zHiLimit,       
                        nMaterials);
 
    ICE::convertNR_4dToUCF(patch,
                        cv_cc,          cv_CC,      
                        include_ghost_cells,
                        xLoLimit,       xHiLimit,       
                        yLoLimit,       yHiLimit,
                        zLoLimit,       zHiLimit,       
                        nMaterials);
                        
    ICE::convertNR_4dToUCF(patch,         vel_cc,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        include_ghost_cells,
                        xLoLimit,       xHiLimit,       
                        yLoLimit,       yHiLimit,
                        zLoLimit,       zHiLimit,       
                        nMaterials);

    /*__________________________________
    *   Now put the data into the dw
    *___________________________________*/                        
    new_dw->put(      vel_cc,     vel_CCLabel,    0,patch);
    new_dw->put(      press_cc,   press_CCLabel,  0,patch);
    new_dw->put(      press_cc_1, press_CCLabel_1,0,patch);
    new_dw->put(      rho_cc,     rho_CCLabel,    0,patch);
    new_dw->put(      temp_cc,    temp_CCLabel,   0,patch);
    new_dw->put(      cv_cc,      cv_CCLabel,     0,patch);
    new_dw->put(      vel_fc,     vel_FCLabel,    0,patch);
    new_dw->put(      press_fc,   press_FCLabel,  0,patch);
    new_dw->put(      tau_fc,     tau_FCLabel,    0,patch);
}


/* --------------------------------------------------------------------- 
GENERAL INFORMATION
 Function:  ICE::actuallyComputeStableTimestep--
 Filename:  ICE_actual.cc
 Purpose:   Compute the stable time step based on the courant condition
            
History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/ 
void ICE::actuallyComputeStableTimestep(const ProcessorGroup*,
    const Patch* patch,
    DataWarehouseP& fromDW,
    DataWarehouseP& toDW)
{
    int include_ghost_cells = NO;
  cerr << "Doing actuallyComputeStableTimestep . . ." << endl;

    /*__________________________________
    * convert UCF data into NR arrays 
    *___________________________________*/
    CCVariable<Vector> vel_cc;
  
    toDW->get(vel_cc, vel_CCLabel,0, patch,Ghost::None,0);

    ICE::convertUCFToNR_4d(patch,       vel_cc,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        include_ghost_cells,
                        xLoLimit,       xHiLimit,
		          yLoLimit,       yHiLimit,
                        zLoLimit,       zHiLimit,
                        nMaterials);
  
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
  
  delt_vartype dt(delt);
  toDW->put(dt, delTLabel);
}



/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  ICE::actually_Top_of_main_loop--
 Filename:  ICE_actual.cc
 Purpose:   - Include the pgplot variables and set environmental vars
            - Update the face and cell centered variables in the ghost cells
            - Before you enter the main loop find the time step
            

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/ 
void ICE::actually_Top_of_main_loop(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
    int should_I_write_output;
    
    fprintf(stderr,"\n\n________________________________\n");
    cerr << "Actually doing step 0" << endl;
  
    should_I_write_output = Is_it_time_to_write_output( t, t_output_vars  ); 
  
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
    /*__________________________________
    *   Quite full warn remarks
    *___________________________________*/
    should_I_write_output = should_I_write_output;                     
}





/* ---------------------------------------------------------------------
                                                        S  T  E  P     1 
GENERAL INFORMATION
 Function:  ICE::actuallyStep1--
 Filename:  ICE_actual.cc
 Purpose:   STEP 1 
            compute the cell-centered pressure using the equation of state
            and the speed of sound 

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/ 
void ICE::actuallyStep1(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
    int include_ghost_cells = NO;
    /*__________________________________
    * get data from the data warehouse and
    *  convert it to NR arrays
    *___________________________________*/
#if 1
    CCVariable<double> press_cc; 
    CCVariable<double> rho_cc;
    CCVariable<double> temp_cc;
    CCVariable<double> cv_cc;

    old_dw->get(  press_cc,   press_CCLabel,  0, patch, Ghost::None, 0);  
    old_dw->get(  rho_cc,     rho_CCLabel,    0, patch, Ghost::None, 0);
    old_dw->get(  temp_cc,    temp_CCLabel,   0, patch, Ghost::None, 0); 
    old_dw->get(  cv_cc,      cv_CCLabel,     0, patch, Ghost::None, 0);

     ICE::convertUCFToNR_4d(patch,
          press_cc,   press_CC,
          include_ghost_cells,
          xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
          nMaterials); 

    ICE::convertUCFToNR_4d(patch,
          rho_cc,     rho_CC,
          include_ghost_cells,
          xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
          nMaterials); 

    ICE::convertUCFToNR_4d(patch,
          temp_cc,    Temp_CC,
          include_ghost_cells,
          xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
          nMaterials); 

    ICE::convertUCFToNR_4d(patch,
          cv_cc,      cv_CC,
          include_ghost_cells,
          xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
          nMaterials); 
#endif
    
    
     /*__________________________________
     *  Use the equation of state to get
     *  P at the cell center
     *___________________________________*/
    #if switch_step1_OnOff
    cerr << "Actually step 1 " << endl;
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
     
  // ICE::convert NR stuff back to ucf and store in data warehouse
/*    CCVariable<double> press_cc_1;
     old_dw->allocate(   press_cc_1,   press_CCLabel_1,  0,patch); 
     old_dw->get(  press_cc_1,   press_CCLabel_1,  0, patch, Ghost::None, 0);
     ICE::convertNR_4dToUCF(patch,
          press_cc_1,   press_CC,
          include_ghost_cells,
          xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
          nMaterials);
          
    old_dw->put(        press_cc_1,   press_CCLabel_1,  0,patch); */             
    /*__________________________________
    *   Quite full warn remarks
    *___________________________________*/
    include_ghost_cells = include_ghost_cells; 

}






/* ---------------------------------------------------------------------
                                                        S  T  E  P     2 
GENERAL INFORMATION
 Function:  ICE::actuallyStep2--
 Filename:  ICE_actual.cc
 Purpose:   - Compute the face-centered velocities using time n data
            - Compute the divergence of the face centered velocities
            - Compute the change in the cell centered pressure (explicit)   
History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/
void ICE::actuallyStep2(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
    int include_ghost_cells = NO;
    /*__________________________________
    * get data from the data warehouse and
    *  convert it to NR arrays
    *___________________________________*/
#if 0
    CCVariable<Vector> vel_cc;
   
  
    old_dw->get(vel_cc, vel_CCLabel, 0, patch, Ghost::None,0);
  
    ICE::convertUCFToNR_4d(patch,
        vel_cc,     uvel_CC,    vvel_CC,    wvel_CC,
        include_ghost_cells,
        xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
        nMaterials);
        
        
/*      CCVariable<double> press_cc_1;    
    old_dw->get(press_cc_1, press_CCLabel_1, 0, patch, Ghost::None,0);
  
    ICE::convertUCFToNR_4d(patch,
        press_cc_1,             press_CC,
        include_ghost_cells,
        xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
        nMaterials); */
#endif
    
    cerr << "Actually doing step 2" << endl;


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
  // ICE::convert back to ucf format and store in data warehouse
    /*__________________________________
    *   Quite full warn remarks
    *___________________________________*/
    include_ghost_cells = include_ghost_cells;

}


/* ---------------------------------------------------------------------
                                                        S  T  E  P     3 
GENERAL INFORMATION
 Function:  ICE::actuallyStep3--
 Filename:  ICE_actual.cc
 Purpose:   - Compute the face centered pressure
             
History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/
void ICE::actuallyStep3(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
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





/* ---------------------------------------------------------------------
                                                        S  T  E  P     4 
GENERAL INFORMATION
 Function:  ICE::actuallyStep4--
 Filename:  ICE_actual.cc
 Purpose:   Compute sources of mass, momentum and energy
            Sources due to mass conversion, gravity
            pressure, divergence of the stressv and momentum exchange
            
             
History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/
void ICE::actuallyStep4(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
    /*__________________________________
    * get data from the data warehouse and
    *  convert it to NR arrays
    *___________________________________*/
    CCVariable<double> mass;
    CCVariable<double> temp;
    CCVariable<double> rho;
    CCVariable<Vector> tau;
    CCVariable<double> viscosity;
    CCVariable<double> del_pres;
 
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

/* ---------------------------------------------------------------------
                                                        S  T  E  P     5 
GENERAL INFORMATION
 Function:  ICE::actuallyStep5--
 Filename:  ICE_actual.cc
 Computes:  Lagrangian volume (currently not used in algoritm)
            - Converts primative variables into flux form
            - computes lagrangian mass, momentum and energy   
            
History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/
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





/* ---------------------------------------------------------------------
                                            S  T  E  P     6     &     7  
GENERAL INFORMATION
 Function:  ICE::actuallyStep6--
 Filename:  ICE_actual.cc
 Computes:  Advects the mass, momentum and energy 
            Computes time advances mass, momentum and energy
            coverts flux mass, momentum and energy into the 
            primative varibles   
            
History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/
void ICE::actuallyStep6and7(const ProcessorGroup*,
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


/* ---------------------------------------------------------------------
                                              
GENERAL INFORMATION
 Function:  ICE::actually_Bottom_of_main_loop--
 Filename:  ICE_actual.cc
 Purpose:   - output tecplot files
            - PGPLOT output
            - increment time
History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/
void ICE::actually_Bottom_of_main_loop(const ProcessorGroup*,
                    const Patch* patch,
		      DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw)
{
  double t      = this->cheat_t;
  double delt   = this->cheat_delt;
  int   should_I_write_output,
        include_ghost_cells = NO;
  /*__________________________________
   *   Plotting variables
   *___________________________________*/
#if (switchDebug_main == 1|| switchDebug_main == 2 || switchDebug_main_input == 1)
    #include "plot_declare_vars.h"   
#endif
    should_I_write_output = Is_it_time_to_write_output( t, t_output_vars  );
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
     *  Plotting section
     *___________________________________*/
    #if switchDebug_main
    if ( should_I_write_output == YES)
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
     *   Advance time
     *___________________________________*/
    t = t + delt;
    fprintf(stderr,"\nTime is %f, timestep is %f\n",t,delt);  
    
    fprintf(stderr, "press return to continue \n");
    getchar();
  

/*______________________________________________________________________
*   leftovers from steve
*_______________________________________________________________________*/
    // Added by Steve for sanity checking
#if 0
    double sumRho=0;
    double sumEng=0;
    int m=1;
    for ( int i = xLoLimit; i <= xHiLimit; i++){
      for ( int j = yLoLimit; j <= yHiLimit; j++){
        for ( int k = zLoLimit; k <= zHiLimit; k++){ 
	  cerr << "rho["<<i<<"]["<<j<<"]["<<k<<"] = " << rho_CC[m][i][j][k] << endl;
	  sumRho += rho_CC[m][i][j][k];
	  sumEng += int_eng_CC[m][i][j][k];
        }
      }
    }
    cerr << "sum rho=" << sumRho << '\n';
    cerr << "sum eng=" << sumEng << '\n';
    cerr << "ii=" << int_eng_CC[1][5][5][1] << '\n';
#endif

    /*__________________________________
    *   - Allocate memory for the new dw
    *   - Convert NR arrays into UCF format
    *   - put the UCF formatted arrays into the dw
    *___________________________________*/
  
    CCVariable<Vector>  new_vel_cc;
    CCVariable<double>  new_temp_cc;
    CCVariable<double>  new_rho_cc;
    CCVariable<double>  new_press_cc;
    CCVariable<double>  new_cv_cc;
    
    new_dw->allocate(   new_vel_cc,     vel_CCLabel,    0,patch);
    new_dw->allocate(   new_temp_cc,    temp_CCLabel,   0,patch);
    new_dw->allocate(   new_rho_cc,     rho_CCLabel,    0,patch);
    new_dw->allocate(   new_press_cc,   press_CCLabel,  0,patch);
    new_dw->allocate(   new_cv_cc,      cv_CCLabel,     0,patch);
    
    ICE::convertNR_4dToUCF(patch,
        new_vel_cc, uvel_CC,    vvel_CC,    wvel_CC,
        include_ghost_cells,
        xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
	 nMaterials);
  
    ICE::convertNR_4dToUCF(patch,
        new_temp_cc,            Temp_CC,
        include_ghost_cells,
        xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
	 nMaterials);
        
    ICE::convertNR_4dToUCF(patch,
        new_rho_cc,             rho_CC,
        include_ghost_cells,
        xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
	 nMaterials);
        
    ICE::convertNR_4dToUCF(patch,
        new_press_cc,           press_CC,
        include_ghost_cells,
        xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
	 nMaterials);
        
    ICE::convertNR_4dToUCF(patch,
        new_cv_cc,              cv_CC,
        include_ghost_cells,
        xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
	 nMaterials);
                  
    new_dw->put(        new_cv_cc,      cv_CCLabel,     0,patch);
    new_dw->put(        new_temp_cc,    temp_CCLabel,   0,patch); 
    new_dw->put(        new_vel_cc,     vel_CCLabel,    0,patch); 
    new_dw->put(        new_press_cc,   press_CCLabel,  0,patch); 
    new_dw->put(        new_rho_cc,     rho_CCLabel,    0,patch);    
    
    /*__________________________________
    *   Quite full warn remarks
    *___________________________________*/
    include_ghost_cells = include_ghost_cells;

}














