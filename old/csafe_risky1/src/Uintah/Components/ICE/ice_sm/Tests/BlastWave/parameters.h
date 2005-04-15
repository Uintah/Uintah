#ifndef __PARAMETERS_H
#define __PARAMETERS_H
/*__________________________________
*   GEOMETRY DEFINITIONS    
*___________________________________*/
#define X_MAX_LIM       55              /* Max. array size in the x dir.  xHiLimit + 2 */
#define Y_MAX_LIM       55              /* Max. array size in the y dir.  yHiLimit + 2 */
#define Z_MAX_LIM       6               /* Max. array size in the z dir. Must be >=3    */


#define N_CELL_FACES    6               /* Number of faces on a cell                    */
#define N_GHOSTCELLS    1               /* Number of ghost cells padding the domain     */
#define N_MATERIAL      1               /*  Maximum Number of different materials       */
#define N_CELL_VERTICES 4               /* Number of cell vertices                      */
#define N_DOMAIN_WALLS  6               /* Number of walls surrounding the domain       */
#define TOP             1               /* index used to designate the top cell face    */
#define BOTTOM          2               /* index used to designate the bottom cell face */
#define RIGHT           3               /* index used to designate the right cell face  */
#define LEFT            4               /* index used to designate the left cell face   */
#define FRONT           5               /* index used to designate the front cell face  */
#define BACK            6               /* index used to designate the back cell face   */
#define N_DIMENSIONS    2               /* Number of dimensions in the problem          */
#define XDIR            1               /* used to designate the x direction            */
#define YDIR            2               /* used to designate the y direction            */
#define ZDIR            3               /* used to designate the z direction            */
/*__________________________________
*   Compute delta time based on 
*   convective velocity or speed
*   of sound
*___________________________________*/
#define compute_delt_based_on_velocity 3      
                                            /* =1 convective velocity                       */
                                            /* =2 speed of sound                            */
                                            /* =3 fabs(velocity) + speed of sounce          */
#define N_ITERATIONS_TO_STABILIZE     10    /* number of iteration for the solution to      */
                                            /* stabilize.  The CFL number linearly increases*/
                                            /* during the these iterations                  */
/*__________________________________
* RELATED TO BOUNDARY CONDITIONS
* This is mainly to associate names
* with integer numbers
*___________________________________*/
#define N_VARIABLE_BC       7               /* Number of varibles that require BC           */
#define UVEL                1               /* index to desingate the x-component of vel    */
#define VVEL                2               /* index to desingate the y-component of vel    */
#define WVEL                3               /* index to desingate the z-component of vel    */
#define PRESS               4               /* index to designate that the pressure on a    */
                                            /* wall has been set                            */
#define TEMP                5               /* index to designate that the temperature on a */
                                            /* wall has been set                            */
#define DENSITY             6               /* index to designate that the density on a     */
                                            /* wall has been set                            */
#define DELPRESS            7               /* index for change in delta pressure           */
                                            
        /* DIFFERENT TYPES OF INPUT BOUNDARY CONDITIONS */
#define N_DIFFERENT_BCS     7               /* Number of different types of input BCs       */
#define NO_SLIP             1               /* No slip boundary condition                   */
#define SUBSONIC_INFLOW     2               /* subsonic inflow                              */
#define SUBSONIC_OUTFLOW    3               /* subsonic outflow                             */
#define SUBSONIC_OUTFLOW_V2 4               /* subsonic outflow version 2                   */
#define REFLECTIVE          5               /* Reflective                                   */
#define ALL_NEUMAN          6               /* Normal gradients for all variables = 0       */
#define ALL_PERIODIC        7               /* periodic boundary conditions                 */          
      /* TYPE OF BOUNDARY CONDITION */
#define DIRICHLET           0               /* index to designate DIRICHLET BC's            */
#define NEUMANN             1               /* index to designate Neumann BC's              */
#define PERIODIC            2               /* index to designate Periodic BC's             */
#define FIXED               0               /* a boundary condition that is fixed2          */
#define FLOAT               1               /* a boundary condition that floats             */

/*__________________________________
*   NUMERICS
*___________________________________*/
#define SMALL_NUM           1.0e-100        /* small number used in bulletproofing          */
#define BIG_NUM             1.0e100         /* Big number used in bulletproofing            */
/*__________________________________
*   PRESSURE SOLVER SWITCHES
*   see http://www-unix.mcs.anl.gov/petsc/docs/manualpages/KSP/KSPSetTolerances.html
*   for the definition of the tolerances
*___________________________________*/
#define MAX_ITERATION       100             /* max iteration allowed in pressure solve      */
#define RELATIVE_TOLERANCE  0.0             /* Relative tolerence used in the pressure solver*/
#define ABSOLUTE_TOLERANCE  1e-100          /* Absolute tolerance used in the pressure solver*/
#define DIV_TOLERANCE       0.001           /* amount residual can increase before the solver*/
                                            /* concludes that the method is diverging       */ 
#define NO                  -9              /* Answers to questions                         */
#define YES                 1
/*__________________________________
*   OUTPUT
*___________________________________*/
#define write_tecplot_files 0               /* = 1 to write tecplot files, see input file   */
                                            /* for timing specifications                    */
/*__________________________________
*   ADVECTIONS PARAMETERS
*___________________________________*/
/*______________________________________________________________________

                        q_outflux(TOP)
                        ______________________
                        |   |             |  |
  q_outflux_CF(TOP_L)   | + |      +      | +| q_outflux_CF(TOP_R)
                        |---|----------------|
                        |   |             |  |
  q_outflux(LEFT)       | + |     i,j,k   | +| q_outflux(RIGHT)
                        |   |             |  |
                        |---|----------------|
 q_outflux_CF(BOT_L)    | + |      +      | +| q_outflux_CF(BOT_R)
                        ----------------------
                         q_outflux(BOTTOM) 
*_______________________________________________________________________*/
#define TOP_R   1                           /* top right corner flux index                  */
#define TOP_L   2                           /* top left corner flux index                   */
#define BOT_L   3                           /* bottom left corner flux index                */
#define BOT_R   4                           /* bottom right corner flux index               */        

#define LIMIT_GRADIENT_FLAG 0               /* Flag for selecting different                 */
                                            /* of gradient limiters                         */
                                            /*  0 = No limiter                              */
                                            /*  1 = Comptible Flux paper                    */
                                            /*  2 = CFDLIB limiter                          */
#define SECOND_ORDER_ADVECTION 0            /*  0 = First order advection                   */
                                            /*  1 = 2nd order advection                     */
                                        
                                        
/*__________________________________
*   PLOTTING VARIABLES
*___________________________________*/
#define PLOT_NEAR_ZERO_CUTOFF 1E-7          /* How sensitive the plots are when ymax and    */
                                            /* ymin are near zero                           */ 
#define NUM_COLORS          100             /* number of colors in contour plot             */
#define PLOT_MAX_LIM        10000           /* Maximum length for a array in a 2-d          */
                                            /* scatter plot                                 */
#define filepath "/usr/people/harman/Csafe/Uintah_cfd_code/Results/" 
#define show_grid           1               /* = 1 To see the overlying grid                */
#define contourplot_type    2               /* = 1 for a normal contour plot.               */
                                            /* = 2 for a checkerboard type contour plot     */
#define GRAPHDESC  ""
#define GRAPHDESC2 ""
#define GRAPHDESC3 ""
#define GRAPHDESC4 ""


#endif      /*__PARAMETERS_H*/  
