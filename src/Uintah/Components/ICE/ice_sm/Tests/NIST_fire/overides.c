 
/* 
======================================================================*/
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <stdlib.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
#include "cpgplot.h"            /*must have this for plotting to work   */

#include <ieeefp.h>            /* needed by Steve Parker's malloc Library*/
/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  main--Main program
 Filename:  main.c 
 Purpose:    This is the main program for the Uintah ICE cfd code. 

History: 
Version   Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       02/22/99                               
                                                                    
    Programming Conventions
        i, j, k         Loop indices for the x, y, z directions respectively
        f               is a loop index for face-centered values.
        m               Loop index for the different materials

                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(3)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (6) = back face
                                 (2)              (5) = front face

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
        
        
____________________________________________________________________
 C  U  S  T  O  M  I  Z  A  T  I  O  N  S  : 
This version of ICE_SM is a test bed for the NIST fire code.
_____________________________________________________________________*/ 
main()
{
    int i, j, k, m,  
        xLoLimit,                       /* x array lower limits             */
        yLoLimit,                       /* y array lower limits             */
        zLoLimit,
        xHiLimit,
        yHiLimit,
        zHiLimit,
        printSwitch,
        should_I_write_output,          /* flag for dumping output          */
        fileNum,                        /* tecplot file number              */
        stat,                           /* status of putenv and getenv      */
        **BC_inputs,                    /* BC_types[wall][m] that contains  */
                                        /* the users boundary condition     */
                                        /* selection for each wall          */
        ***BC_types,                    /* each variable can have a Neuman, */
                                        /* or Dirichlet type BC             */
                                        /* BC_types[wall][variable][m]=type */
        ***BC_float_or_fixed,           /* BC_float_or_fixed[wall][variable][m]*/
                                        /* Variable on boundary is either   */
                                        /* fixed or it floats during the    */
                                        /* compuation                       */
        nMaterials;                     /* Number of materials              */

                                        
/* ______________________________   
*  Geometry                        
* ______________________________   */     
     double  delX,                      /* Cell width                       */
             delY,                      /* Cell Width in the y dir          */
             delZ,                      /* Cell width in the z dir          */
             delt,                      /* time step                        */
             CFL,                       /* Courant-Friedrichs and Lewy      */
             t_final,                   /* final problem time               */
            *t_output_vars,             /* array holding output timing info */
                                        /* t_output_vars[1] = t_initial     */
                                        /* t_output_vars[2] = t final       */
                                        /* t_output_vars[3] = delta t       */
            *delt_limits,               /* delt_limits[1]   = delt_minimum  */
                                        /* delt_limits[2]   = delt_maximum  */
                                        /* delt_limits[3]   = delt_initial  */
             t,                         /* current time                     */
             ***x_CC,                   /* x-coordinate of cell center      */
             ***y_CC,                   /* y-coordinate of cell center      */
             ***z_CC,                   /* z-coordinate of cell center      */
             ***Vol_CC,                 /* vol of the cell at the cellcenter*/
                                        /* (x, y, z)                        */
            /*------to be treated as pointers---*/
             *****x_FC,                 /* x-coordinate of face center      */
                                        /* x_FC(i,j,k,face)                 */
                                        /* cell i,j,k                       */
             *****y_FC,                 /* y-coordinate of face center      */
                                        /* y_FC(i,j,k,face)                 */
                                        /* of cell i,j,k                    */
             *****z_FC;                 /* z-coordinate of face center      */
                                        /* z_FC(i,j,k,face)                 */
                                        /* of cell i,j,k                    */
            /*----------------------------------*/ 

/* ______________________________   
*  Cell-centered and Face centered                      
* ______________________________   */ 
    double                              /* (x,y,z,material                  */
            ****uvel_CC,                /* u-cell-centered velocity         */
            ****vvel_CC,                /*  v-cell-centered velocity        */
            ****wvel_CC,                /* w cell-centered velocity         */
            ****delPress_CC,            /* cell-centered change in pressure */                                                                                       
            ****press_CC,               /* Cell-centered pressure           */
            ****Temp_CC,                /* Cell-centered Temperature        */
            ****rho_CC,                 /* Cell-centered density            */
            ****viscosity_CC,           /* Cell-centered Viscosity          */ 
            ****thermalCond_CC,         /* Cell-centered thermal conductivity*/
            ****cv_CC,                  /* Cell-centered specific heat      */ 
            ****mass_CC,                /* total mass, cell-centered        */
            ****xmom_CC,                /* x-dir momentum cell-centered     */
            ****ymom_CC,                /* y-dir momentum cell-centered     */
            ****zmom_CC,                /* z-dir momentum cell-centered     */
            ****int_eng_CC,             /* Internal energy cell-centered    */
            ****total_eng_CC,           /* Total energy cell-centered       */
            ****div_velFC_CC,           /* Divergence of the face centered  */
                                        /* velocity that lives at CC        */    
            ****scalar1_CC,             /* Cell-centered scalars            */   
            ****scalar2_CC,             /* (x, y, z, material)              */
            ****scalar3_CC,
            /*------to be treated as pointers---*/
                                        /*______(x,y,z,face, material)______*/
            ******uvel_FC,              /* u-face-centered velocity         */
            ******vvel_FC,              /* *v-face-centered velocity        */
            ******wvel_FC,              /* w face-centered velocity         */
            ******press_FC,             /* face-centered pressure           */
            ******tau_X_FC,             /* *x-stress component at each face */
            ******tau_Y_FC,             /* *y-stress component at each face */
            ******tau_Z_FC,             /* *z-stress component at each face */
            /*----------------------------------*/                                              
           *grav,                       /* gravity (dir)                    */
                                        /* x-dir = 1, y-dir = 2, z-dir = 3  */
            *gamma,
            ****speedSound;             /* speed of sound (x,y,z, material) */    
             
/* ______________________________   
*  Lagrangian Variables            
* ______________________________   */ 
    double                              /*_________(x,y,z,material)_________*/
            ****rho_L_CC,               /* Lagrangian cell-centered density */
            ****mass_L_CC,              /* Lagrangian cell-centered mass    */
            ****Temp_L_CC,              /* Lagrangian cell-centered Temperature */
            ****press_L_CC,             /* Lagrangian cell-centered pressure*/            
            ****xmom_L_CC,              /* Lagrangian cell-centered momentum*/
            ****ymom_L_CC,              /* Lagrangian cell-centered momentum*/
            ****zmom_L_CC,              /* Lagrangian cell-centered momentum*/
            ****int_eng_L_CC,           /* Lagrangian cc internal energy    */
            ****Vol_L_CC,               /* Lagrangian cell-centered volume  */
/*__________________________________
* source terms
*___________________________________*/
            ****mass_source,            /* Mass source term (x,y,z, material */
            ****xmom_source,            /* momentum source terms            */
                                        /* (x, y, z, material)              */
            ****ymom_source,
            ****zmom_source,
            ****int_eng_source;         /* internal energy source           */
            
/*__________________________________
*   NIST FIRE VARIABLE
*   See reference 
*   Blueprint for putting fire in ICE-CFD
*   code Single Material by Ruddy Mell
*___________________________________*/
    double  *t_inject_TE_parms,          /* array holding output timing inf0 */
                                        /*  [1] = When to begin injection   */
                                        /*  [2] = When to stop injecting    */
                                        /*  [3] = delta t injection         */
            *t_inject_TE,               /* time TE was injected             */
            Q_fire,                     /* Heat Release rate of fire        */
            del_H_fuel,                 /* heat of combustion of fuel [j/kg]*/
            *x_TE,                      /* x position of TE (n)             */
            *y_TE,                      /* y position of TE (n)             */
            *z_TE,                      /* z position of TE (n)             */
            *Q_TE,                      /* Vol. heat release from TE        */
            ***Q_chem,                  /* Sum(Q_TE * (1 - rad_coeff)   )   */
            *scalar_TE,                 /* scalar used to mark the TE (test)*/ 
            t_burnout_TE,               /* burnout time of TE               */
            u_inject_TE,                /* velocity which the TE are injected*/           
            rho_fuel;                   /* density of fuel                  */
    
    int     nThermalElements,           /* number of thermal elements       */
            N_inject_TE,                /* number of TE injected at any time*/
            *index_inject_TE,           /* indices where the particles are  */
                                        /* injected [1] = xlo, [2] = xHi,
                                                    [3] = yLo, [4] = yHi,
                                                    [5] = zLo, [6] = zHi    */
                                        
            **cell_index_TE;            /* array contains which cell TE(n)  */
                                        /* lives in                         */            
/*__________________________________
*   MISC Variables
*___________________________________*/ 
    double      
            ***BC_Values,                /* BC values BC_values[wall][variable][m]*/  
            *R;                         /* gas constant R[material]          */ 
                             
    double  residual,                   /* testing*/            
            temp1, temp2;
            
    char    output_file_basename[30],   /* Tecplot filename description     */
            output_file_desc[50];       /* Title used in tecplot stuff      */



/*__________________________________
*   Plotting variables
*___________________________________*/
#if (switchDebug_main == 1|| switchDebug_main == 2 || switchDebug_main_input == 1)
    #include "plot_declare_vars.h"   
#endif
    stat = putenv("PGPLOT_DIR=" PGPLOT_DIR);
    stat = putenv("PGPLOT_I_AM_HERE=0");              
                                        /* tell the plotting routine that  */
                                        /* you're at the top of main       */      

    stat = putenv("PGPLOT_PLOTTING_ON_OFF=1");
    stat = putenv("PGPLOT_OPEN_NEW_WINDOWS=1");  

/*______________________________________________________________________
*   Initialize variables
*_______________________________________________________________________*/ 
    printSwitch = 1;    
    t           = 0.0;  
    m           = 1;
    fileNum     = 1;

/*______________________________________________________________________ 
*    M  E  M  O  R  Y     S  E  C  T  I  O  N 
*   - Allocate memory for the arrays                                          
*_______________________________________________________________________*/
#include "allocate_memory.i"

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
                        t_inject_TE_parms,&Q_fire,      &N_inject_TE,
                        &del_H_fuel,    &rho_fuel,      index_inject_TE,
                        &nMaterials);      
    
    testInputFile(      xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        Temp_CC,        press_CC,       rho_CC,
                        viscosity_CC,   thermalCond_CC, cv_CC,
                        speedSound,      
                        t_final,        t_output_vars,  delt_limits,
                        BC_inputs,      printSwitch,    CFL,
                        t_inject_TE_parms,Q_fire,       N_inject_TE,
                        del_H_fuel,     rho_fuel,       index_inject_TE,
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
    
    
    
/*______________________________________________________________________
*   M  A  I  N     A  D  V  A  N  C  E     L  O  O  P 
*_______________________________________________________________________*/                       
    while( t <= t_final)
    {
         should_I_write_output = Is_it_time_to_write_output( t, t_output_vars  );
        /* fprintf(stderr, "should _ I write_output %i\n",should_I_write_output); */
         

       
    /*__________________________________
    * update the physical boundary conditions
    * and initialize some arrays
    *___________________________________*/                        
/*     update_CC_FC_physical_boundary_conditions( 
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,
                        BC_types,       BC_float_or_fixed,
                        BC_Values, 
                        nMaterials,     3,                 
                        uvel_CC,        UVEL,           uvel_FC,
                        vvel_CC,        VVEL,           vvel_FC,
                        wvel_CC,        WVEL,           wvel_FC); */
                        
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



    /* ______________________________  
    *    S  T  E  P     4                               
    *   Compute sources of mass, momentum and energy
    *   For momentum, there are sources
    *   due to mass conversion, gravity
    *   pressure, divergence of the stress
    *   and momentum exchange
    * ______________________________   */
    #if (switch_step4_OnOff == 1 && switch_Compute_burgers_eq == 0) 
    
/*`==========TESTING==========*/ 
    /*______________________________________________________________________
    *   RUDDY's playground
    *_______________________________________________________________________*/
    /*__________________________________
    *   NIST FIRE
    *___________________________________*/
#if switch_step4_NIST_fire
    compute_initial_constants_NIST_TE(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        Temp_CC,        rho_CC,         grav,                     
                        del_H_fuel,     rho_fuel,       Q_fire,                   
                        &u_inject_TE,   &t_burnout_TE,   m );                    
    
    
    /* Step 1       */
    update_TE_position(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        nThermalElements,cell_index_TE, 
                        x_TE,           y_TE,           z_TE,           
                        uvel_CC,        vvel_CC,        wvel_CC,     
                        delt,           1);  
                        
     /* Step 2 update heat release rate of all of the particles.
        This depend on the time that the TE was injected        */
    update_Q_TE_and_add_Q_cell(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        t,                    
                        x_TE,           y_TE,           z_TE,
                        Q_fire,               
                        t_inject_TE,    t_inject_TE_parms,      t_burnout_TE,         
                        N_inject_TE,    nThermalElements,
                        Q_TE,           Q_chem,         m);                    
                         
    /* Step 3       */
    inject_TE(
                        index_inject_TE,N_inject_TE,    t_inject_TE_parms,
                        t,        
                        delX,           delY,           delZ,
                        u_inject_TE,    t_inject_TE,    
                        x_TE,           y_TE,           z_TE,               
                        &nThermalElements );
#endif        
    /*______________________________________________________________________
    *
    *_______________________________________________________________________*/
 /*==========TESTING==========`*/    
    
 
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
                        div_velFC_CC,   Q_chem,         
                        int_eng_source,  
                        nMaterials   );

    #endif


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
                        
/*`==========TESTING==========*/ 
   /*  zero_arrays_4d(
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,
                        1,              nMaterials,     1,             
                        vvel_CC); */
 /*==========TESTING==========`*/

/*`==========TESTING==========*/ 
    update_CC_physical_boundary_conditions( 
                    xLoLimit,       yLoLimit,       zLoLimit,             
                    xHiLimit,       yHiLimit,       zHiLimit,             
                    delX,           delY,           delZ,
                    BC_types,       BC_float_or_fixed,
                    BC_Values, 
                    nMaterials,     3,                 
                    Temp_CC,TEMP,   rho_CC,DENSITY, press_CC,PRESS);
                    
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

 /*==========TESTING==========`*/    
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
    *    A  D  V  A  N  C  E     I  N     T  I  M  E 
    *___________________________________*/
                        
        find_delta_time_based_on_CC_vel(
                        xLoLimit,        yLoLimit,      zLoLimit,
                        xHiLimit,        yHiLimit,      zHiLimit,
                        &delt,           delt_limits,
                        delX,            delY,          delZ,
                        uvel_CC,         vvel_CC,       wvel_CC,
                        speedSound,      CFL,           nMaterials );
           
        t = t + delt;
        fprintf(stderr,"\nTime is %f, timestep is %f\n",t,delt);
 
 }
/* -----------------------------------------------------------------------  
*   F  R  E  E     T  H  E     M  E  M  O  R  Y                                                     
* -----------------------------------------------------------------------  */
    fprintf(stderr,"Now deallocating memory");
    #include "free_memory.i"
   
/*__________________________________
*   Quite fullwarn compiler remarks
*___________________________________*/
    i = i;      j = j;      k = k;    
    residual    = residual;
    temp1       = temp1;
    temp2       = temp2;
    QUITE_FULLWARN(stat);                       
    QUITE_FULLWARN(fileNum); 

    return(1);
/*STOP_DOC*/
}
