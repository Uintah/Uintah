/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"

/* ---------------------------------------------------------------------
 Function:  definition_of_different_physical_boundary_conditions--BOUNDARY CONDITIONS: Define in-terms-of Neumann and Dirchlet, no slip, subsonic inflow and subsonic outflow.
 Filename:  boundary_cond.c
 Purpose:
            This function is used to define the BC_type( Neuman or Dirichlet)
            and whether it floats or is fixed throughout the compuation, for each
            dependent variable.
              
            This routine doesn't actually set the boundary conditions, 
            it is just used to define what a particular boundary condition means
            with regards to neuman and dirichlet 
            
            BC_inputs:
            -no slip
            -subsonic input     (velocity components and rho are specified)    
            -subsonic outflow   (normal component of velocity)
            implement different flavors of each boundary condition
            
            BC_types:
                Neuman
                Dirichlet

references:
    APACHE: A generalized-Mesh Eulerian Computer Code for Multicomponent
    Chemically Reactive Fluid Flow, LA-7427, 1979
            
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/03/99    
 ---------------------------------------------------------------------  */
 
 void definition_of_different_physical_boundary_conditions( 
    int     **BC_inputs,                /* What the user has defined as     (INPUT) */
                                        /* boundary conditions              */ 
    int     ***BC_types,                /* array containing the different   (OUTPUT) */
                                        /* types of boundary conditions     */
                                        /* BC_types[wall][variable]=type    */
    int     ***BC_float_or_fixed,       /* BC_float_or_fixed[wall][var][m]  (OUTPUT)*/
                                        /* Variable on boundary is either   */
                                        /* fixed or it floats during the    */
                                        /* compuation                       */
    double ***BC_Values,                /* Values of the variables at the   (INPUT) */
                                        /* boundaries                       */ 
    int       nMaterials        )

 {
    int     m,   
            wall,                       /* wall index                       */
            wallLo, wallHi;             /* lower and upper wall indices     */                                   
      
/*__________________________________
*   Set looping
*   indices for the number of walls.
*___________________________________*/
    
#if (N_DIMENSIONS == 1)
    wallLo = LEFT;  wallHi = RIGHT;
#endif  
#if (N_DIMENSIONS == 2)
    wallLo = TOP;   wallHi = LEFT;
#endif
#if (N_DIMENSIONS == 3)
    wallLo = TOP;   wallHi = BACK;
#endif  
/*______________________________________________________________________
*   Loop over all of the outer walls and set what it means to have noslip
*   subsonic inflow or subsonic outflow boundary condition
*_______________________________________________________________________*/
    for(m = 1; m <= nMaterials; m++)
    {
        for( wall = wallLo; wall <= wallHi; wall ++)
        {

            /*__________________________________
            *   No slip boundary condition
            *___________________________________*/
            if(BC_inputs[wall][m] == NO_SLIP)
            {
                BC_types[wall][UVEL][m]             = DIRICHLET;
                BC_types[wall][VVEL][m]             = DIRICHLET;
                BC_types[wall][WVEL][m]             = DIRICHLET;
                BC_types[wall][TEMP][m]             = NEUMANN;
                BC_types[wall][PRESS][m]            = NEUMANN;
                BC_types[wall][DENSITY][m]          = NEUMANN;
                BC_types[wall][DELPRESS][m]         = NEUMANN;

                BC_Values[wall][UVEL][m]            = 0.0;
                BC_Values[wall][VVEL][m]            = 0.0;
                BC_Values[wall][WVEL][m]            = 0.0;
                BC_Values[wall][PRESS][m]           = 0.0;
                BC_Values[wall][TEMP][m]            = 0.0;
                BC_Values[wall][DENSITY][m]         = 0.0;
                BC_Values[wall][DELPRESS][m]        = 0.0; 

                BC_float_or_fixed[wall][UVEL][m]    = FIXED;
                BC_float_or_fixed[wall][VVEL][m]    = FIXED;
                BC_float_or_fixed[wall][WVEL][m]    = FIXED;
                BC_float_or_fixed[wall][PRESS][m]   = FIXED;
                BC_float_or_fixed[wall][TEMP][m]    = FIXED;
                BC_float_or_fixed[wall][DENSITY][m] = FIXED;
                BC_float_or_fixed[wall][DELPRESS][m]= FIXED;
            }
            /*__________________________________
            *   Subsonic inflow
            *___________________________________*/
             if(BC_inputs[wall][m] == SUBSONIC_INFLOW)
            {
                BC_types[wall][UVEL][m]             = DIRICHLET;
                BC_types[wall][VVEL][m]             = DIRICHLET;
                BC_types[wall][WVEL][m]             = DIRICHLET;
                BC_types[wall][TEMP][m]             = DIRICHLET;
                BC_types[wall][PRESS][m]            = NEUMANN;  
                BC_types[wall][DENSITY][m]          = DIRICHLET;
                BC_types[wall][DELPRESS][m]         = DIRICHLET;                

                BC_Values[wall][UVEL][m]            = BC_Values[wall][UVEL][m];
                BC_Values[wall][VVEL][m]            = BC_Values[wall][VVEL][m];
                BC_Values[wall][WVEL][m]            = BC_Values[wall][WVEL][m];
                BC_Values[wall][PRESS][m]           = 0.0;
                BC_Values[wall][TEMP][m]            = BC_Values[wall][TEMP][m];
                BC_Values[wall][DENSITY][m]         = BC_Values[wall][DENSITY][m];
                BC_Values[wall][DELPRESS][m]        = 0.0; 

                BC_float_or_fixed[wall][UVEL][m]    = FIXED;
                BC_float_or_fixed[wall][VVEL][m]    = FIXED;
                BC_float_or_fixed[wall][WVEL][m]    = FIXED;
                BC_float_or_fixed[wall][PRESS][m]   = FLOAT;         
                BC_float_or_fixed[wall][TEMP][m]    = FIXED;
                BC_float_or_fixed[wall][DENSITY][m] = FIXED;
                BC_float_or_fixed[wall][DELPRESS][m]= FIXED;
            }
            /*__________________________________
            *   subsonic outflow
            *   Velocity is set and pressure
            *   density and temperature can float
            *___________________________________*/
             if(BC_inputs[wall][m] == SUBSONIC_OUTFLOW)
            {
                BC_types[wall][UVEL][m]             = DIRICHLET;
                BC_types[wall][VVEL][m]             = DIRICHLET;
                BC_types[wall][WVEL][m]             = DIRICHLET;
                BC_types[wall][TEMP][m]             = NEUMANN;
                BC_types[wall][PRESS][m]            = NEUMANN;
                BC_types[wall][DENSITY][m]          = NEUMANN;
                BC_types[wall][DELPRESS][m]         = DIRICHLET;
                
                /* set the values of the neumann condition*/
                BC_Values[wall][UVEL][m]            = BC_Values[wall][UVEL][m];
                BC_Values[wall][VVEL][m]            = BC_Values[wall][VVEL][m];
                BC_Values[wall][WVEL][m]            = BC_Values[wall][WVEL][m];
                BC_Values[wall][PRESS][m]           = 0.0;
                BC_Values[wall][TEMP][m]            = 0.0;
                BC_Values[wall][DENSITY][m]         = 0.0;
                BC_Values[wall][DELPRESS][m]        = 0.0; 
                
                /* set the fixed or float flags             */
                BC_float_or_fixed[wall][UVEL][m]    = FIXED;
                BC_float_or_fixed[wall][VVEL][m]    = FIXED;
                BC_float_or_fixed[wall][WVEL][m]    = FIXED;
                BC_float_or_fixed[wall][PRESS][m]   = FLOAT;
                BC_float_or_fixed[wall][TEMP][m]    = FLOAT;
                BC_float_or_fixed[wall][DENSITY][m] = FLOAT;
                BC_float_or_fixed[wall][DELPRESS][m]= FIXED;
                
            }
            
            /*__________________________________
            *   subsonic outflow
            *   Velocity is set and pressure
            *   density and temperature can float
            *___________________________________*/
             if(BC_inputs[wall][m] == SUBSONIC_OUTFLOW_V2)
            {
                BC_types[wall][UVEL][m]             = NEUMANN;
                BC_types[wall][VVEL][m]             = NEUMANN;
                BC_types[wall][WVEL][m]             = NEUMANN;
                BC_types[wall][TEMP][m]             = NEUMANN;
                BC_types[wall][PRESS][m]            = DIRICHLET;
                BC_types[wall][DENSITY][m]          = NEUMANN;
                BC_types[wall][DELPRESS][m]         = DIRICHLET;
                
                /* set the values of the neumann condition*/
                BC_Values[wall][UVEL][m]            = 0.0;
                BC_Values[wall][VVEL][m]            = 0.0;
                BC_Values[wall][WVEL][m]            = 0.0;
                BC_Values[wall][PRESS][m]           = BC_Values[wall][PRESS][m];
                BC_Values[wall][TEMP][m]            = 0.0;
                BC_Values[wall][DENSITY][m]         = 0.0;
                BC_Values[wall][DELPRESS][m]        = 0.0; 
                
                /* set the fixed or float flags             */
                BC_float_or_fixed[wall][UVEL][m]    = FLOAT;
                BC_float_or_fixed[wall][VVEL][m]    = FLOAT;
                BC_float_or_fixed[wall][WVEL][m]    = FLOAT;
                BC_float_or_fixed[wall][PRESS][m]   = FIXED;
                BC_float_or_fixed[wall][TEMP][m]    = FLOAT;
                BC_float_or_fixed[wall][DENSITY][m] = FLOAT;
                BC_float_or_fixed[wall][DELPRESS][m]= FIXED; 
            }
            /*__________________________________
            *   REFLECTIVE
            *   Velocity,pressure
            *   density and temperature all float
            *___________________________________*/

             if(BC_inputs[wall][m] == REFLECTIVE)
            {
                /*__________________________________
                *   Left and right walls
                *___________________________________*/
                if (wall == LEFT || wall == RIGHT)
                {
                    BC_types[wall][UVEL][m]         = DIRICHLET;
                    BC_types[wall][VVEL][m]         = NEUMANN;
                    BC_types[wall][WVEL][m]         = NEUMANN;
                    
                    BC_float_or_fixed[wall][UVEL][m]= FIXED;
                    BC_float_or_fixed[wall][VVEL][m]= FLOAT;
                    BC_float_or_fixed[wall][WVEL][m]= FLOAT;
                }
                /*__________________________________
                *   Top and bottom walls
                *___________________________________*/
                if (wall == TOP || wall == BOTTOM)
                {
                    BC_types[wall][UVEL][m]         = NEUMANN;
                    BC_types[wall][VVEL][m]         = DIRICHLET;
                    BC_types[wall][WVEL][m]         = NEUMANN;
                    
                    BC_float_or_fixed[wall][UVEL][m]= FLOAT;
                    BC_float_or_fixed[wall][VVEL][m]= FIXED;
                    BC_float_or_fixed[wall][WVEL][m]= FLOAT;
                } 

                BC_types[wall][TEMP][m]             = NEUMANN;
                BC_types[wall][PRESS][m]            = NEUMANN;
                BC_types[wall][DENSITY][m]          = NEUMANN;
                BC_types[wall][DELPRESS][m]         = DIRICHLET;
                
                /* set the values of the neumann and dirichlet condition*/
                BC_Values[wall][UVEL][m]            = 0.0;
                BC_Values[wall][VVEL][m]            = 0.0;
                BC_Values[wall][WVEL][m]            = 0.0;
                BC_Values[wall][PRESS][m]           = 0.0;
                BC_Values[wall][TEMP][m]            = 0.0;
                BC_Values[wall][DENSITY][m]         = 0.0;
                BC_Values[wall][DELPRESS][m]        = 0.0; 
                
                /* set the fixed or float flags             */
                BC_float_or_fixed[wall][PRESS][m]   = FLOAT;
                BC_float_or_fixed[wall][TEMP][m]    = FLOAT;
                BC_float_or_fixed[wall][DENSITY][m] = FLOAT;
                BC_float_or_fixed[wall][DELPRESS][m]= FIXED;
            }            

/*`==========TESTING==========*/ 
            /*__________________________________
            *   All_NEUMANN
            *   Velocity,pressure
            *   density and temperature all float
            *___________________________________*/
             if(BC_inputs[wall][m] == ALL_NEUMAN)
            {
                BC_types[wall][UVEL][m]             = NEUMANN;
                BC_types[wall][VVEL][m]             = NEUMANN;
                BC_types[wall][WVEL][m]             = NEUMANN;
                BC_types[wall][TEMP][m]             = NEUMANN;
                BC_types[wall][PRESS][m]            = NEUMANN;
                BC_types[wall][DENSITY][m]          = NEUMANN;
                BC_types[wall][DELPRESS][m]         = DIRICHLET;
                
                /* set the values of the neumann condition*/
                BC_Values[wall][UVEL][m]            = BC_Values[wall][UVEL][m];
                BC_Values[wall][VVEL][m]            = BC_Values[wall][VVEL][m];
                BC_Values[wall][WVEL][m]            = BC_Values[wall][WVEL][m];
                BC_Values[wall][PRESS][m]           = BC_Values[wall][PRESS][m];
                BC_Values[wall][TEMP][m]            = BC_Values[wall][TEMP][m];
                BC_Values[wall][DENSITY][m]         = BC_Values[wall][DENSITY][m];
                BC_Values[wall][DELPRESS][m]        = 0;                

                /* set the fixed or float flags             */
                BC_float_or_fixed[wall][UVEL][m]    = FLOAT;
                BC_float_or_fixed[wall][VVEL][m]    = FLOAT;
                BC_float_or_fixed[wall][WVEL][m]    = FLOAT;
                BC_float_or_fixed[wall][PRESS][m]   = FLOAT;
                BC_float_or_fixed[wall][TEMP][m]    = FLOAT;
                BC_float_or_fixed[wall][DENSITY][m] = FLOAT;
                BC_float_or_fixed[wall][DELPRESS][m]= FIXED; 
                /*__________________________________
                *   Left and right walls
                *___________________________________*/
                if (wall == LEFT )
                   BC_Values[LEFT][DELPRESS][m]      = -1.0; 
                if (wall == RIGHT )
                   BC_Values[RIGHT][DELPRESS][m]     = 1.0;
            }
 /*==========TESTING==========`*/
 
 /*`==========TESTING==========*/ 
            /*__________________________________
            *   All_PERIODIC
            *   Velocity,pressure
            *   density and temperature all float
            *    equal to the the value on the opposing 
            *   wall
            *___________________________________*/
             if(BC_inputs[wall][m] == ALL_PERIODIC)
            {
                BC_types[wall][UVEL][m]             = PERIODIC;
                BC_types[wall][VVEL][m]             = PERIODIC;
                BC_types[wall][WVEL][m]             = PERIODIC;
                BC_types[wall][TEMP][m]             = PERIODIC;
                BC_types[wall][PRESS][m]            = NEUMANN;
                BC_types[wall][DENSITY][m]          = NEUMANN;
                BC_types[wall][DELPRESS][m]         = DIRICHLET;
                
                /* set the values of the neumann condition*/
                BC_Values[wall][UVEL][m]            = 0.0;
                BC_Values[wall][WVEL][m]            = 0.0;
                BC_Values[wall][PRESS][m]           = 0.0;
                BC_Values[wall][TEMP][m]            = 0.0;
                BC_Values[wall][DENSITY][m]         = 0.0;
                BC_Values[wall][DENSITY][m]         = 0.0;
                BC_Values[wall][DELPRESS][m]        = 0.0;              

                /* set the fixed or float flags             */
                BC_float_or_fixed[wall][UVEL][m]    = FLOAT;
                BC_float_or_fixed[wall][VVEL][m]    = FLOAT;
                BC_float_or_fixed[wall][WVEL][m]    = FLOAT;
                BC_float_or_fixed[wall][PRESS][m]   = FLOAT;
                BC_float_or_fixed[wall][TEMP][m]    = FLOAT;
                BC_float_or_fixed[wall][DENSITY][m] = FLOAT;
                BC_float_or_fixed[wall][DELPRESS][m]= FIXED; 
 /*==========TESTING==========`*/
                
            }
        }
    }       
}

/*STOP_DOC*/



/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include <stdarg.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"
#include "nrutil+.h"
/* ---------------------------------------------------------------------
 Function:  update_CC_physical_boundary_conditions--BOUNDARY CONDITIONS: Visit all the ghost cells and reset the BC for each cell-centered variable.
 Filename:  boundary_cond.c
 Purpose:
            This function is used to set the values for each variable
            along the walls based on it's type (Neuman, Dirichlet) and whether
            it is a fixed or floating variable.  This function sets the cell-centered
            quantities ONLY.
            Currently this is setup so that the boundary conditions 
            are uniform for the entire wall, mixed boundary conditions 
            have not been implemented.
            
            The user can select from the following boundary conditons
            -no slip
            -subsonic input     (velocity components and rho are specified)    
            -subsonic outflow   (normal component of velocity)
            
Note on implementation:
            This function uses a variable length argument list.  For each data_array
            in the argument list the boundary conditions are set.
            The reason for doing it this way is simple.  In some instances we
            only need the velocity boundary conditions updated while other times we 
            need the pressure tweaked. 

Sticky Point:
            Each array that is passed into MUST also have a variable type
            passed in with it.  For example 
            
            Temp_CC,    TEMP,  
            rho_CC,     DENSITY
            ....
            
           Warning: type checking isn't done on the variables in the ... of the
           parameter list.
            
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/09/99    
 ---------------------------------------------------------------------  */
 
 void update_CC_physical_boundary_conditions( 
              
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
    double  delX,                       /* cell size in x-direction         (INPUT) */
    double  delY,                       /* cell size in y-direction         (INPUT) */
    double  delZ,                       /* cell size in z-directin          (INPUT) */
    int     ***BC_types,                /* array containing the different   (INPUT) */
                                        /* types of boundary conditions             */
                                        /* BC_types[wall][variable]=type            */
    int     ***BC_float_or_fixed,       /* array that designates which variable is  */
                                        /* either fixed or floating on each wall of */
                                        /* the compuational domain                  */
    double  ***BC_Values,               /* Values of the variables at the   (INPUT) */
                                        /* boundaries                       */ 
    int     nMaterials,                 /* number of materials              */
    int     n_data_arrays,              /* number of data arrays            (INPUT) */
            ...)                    /* used to designate whether        (INPUT) */
                                        /* the input array is UVEL,TEMP....         */
                       
 {
    va_list ptr_data_array;             /* pointer to each data array       */
    int     m,  
            var,                        /* used to designate whether the    */
                                        /* input array is UVEL, TEMP....    */               
            array;                      /* array number                     */
    double ****data_CC;                 /* cell-centered data array         */
/*__________________________________
*   double check inputs.
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
    
/*______________________________________________________________________
*   Now set the  physical boundary condtions for each dependent variable
*_______________________________________________________________________*/
    va_start(ptr_data_array, n_data_arrays);
    array = 0;
    /*__________________________________
    *   Loop through each data array in the
    *   argument list
    *___________________________________*/ 
    for (array = 1; array <=n_data_arrays; array++)
    {
        /*__________________________________
        *   Get the data array and var from
        *   the variable length argument list
        *___________________________________*/
        data_CC = va_arg(ptr_data_array, double****); 
        var     = va_arg(ptr_data_array, int);
        for (m = 1; m <= nMaterials; m++)
        { 
            set_Dirichlet_BC(          
                                xLoLimit,       yLoLimit,      zLoLimit,
                                xHiLimit,       yHiLimit,      zHiLimit,
                                data_CC,        var,           
                                BC_types,       BC_Values,     m        );
            set_Neumann_BC(          
                                xLoLimit,       yLoLimit,      zLoLimit,
                                xHiLimit,       yHiLimit,      zHiLimit,
                                delX,           delY,          delZ,
                                data_CC,        var,          
                                BC_types,       BC_Values,     
                                m        );
            set_Periodic_BC(          
                                xLoLimit,        yLoLimit,      zLoLimit,
                                xHiLimit,        yHiLimit,      zHiLimit,
                                data_CC,         var,          
                                BC_types,        m );

            set_corner_cells_BC( 
                                xLoLimit,       yLoLimit,       zLoLimit,       
                                xHiLimit,       yHiLimit,       zHiLimit,       
                                data_CC,      
                                m        );
        }
                 
/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
    #if switchDebug_update_CC_physical_boundary_conditions
        fprintf(stderr,"****************************************************************************\n");
        fprintf(stderr,"            UPDATE_CC_PHYSICAL_BOUNDARY_CONDITIONS\n");
        fprintf(stderr,"****************************************************************************\n");  
        for (m = 1; m <= nMaterials; m++)
        {       
            fprintf(stderr,"\t Material %i \n",m);
            printData_4d(       GC_LO(xLoLimit),     GC_LO(yLoLimit),       GC_LO(zLoLimit),
                                GC_HI(xHiLimit),     GC_HI(yHiLimit),       GC_HI(zHiLimit),
                                m,                  m,
                               "update_CC_physical_boundary_conditions",     
                               "data with ghost cells",                  data_CC);
        }

        fprintf(stderr,"****************************************************************************\n");         

        fprintf(stderr,"press return to continue\n");
        getchar();            
    #endif  

    }     

    va_end(ptr_data_array);                     /* clean up when done   */            
/*__________________________________
*   Quite fullwarn remarks is a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(BC_float_or_fixed);
}
/*STOP_DOC*/


/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include <stdarg.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"
#include "nrutil+.h"
/* ---------------------------------------------------------------------
 Function:  update_CC_FC_physical_boundary_conditions--BOUNDARY CONDITIONS: Visit all the ghost cells and reset the BC for each cell-centered and face-centered variable.
 Filename:  boundary_cond.c
 Purpose:
            This function is used to set the values for each variable
            along the walls based on it's type (Neuman, Dirichlet) and whether
            it is a fixed or floating variable.  This function sets both
            the cell and face-centered quantities.
            Currently this is setup so that the boundary conditions 
            are uniform for the entire wall, mixed boundary conditions 
            have not been implemented.
            
            The user can select from the following boundary conditons
            -no slip
            -subsonic input     (velocity components and rho are specified)    
            -subsonic outflow   (normal component of velocity)
            
Note on implementation:
            This function uses a variable length argument list.  For each data_array
            in the argument list the boundary conditions are set.
            The reason for doing it this way is simple.  In some instances we
            only need the velocity boundary conditions updated while other times we 
            need the pressure tweaked. 

Sticky Point:
            Each array that is passed into MUST also have a variable type
            passed in with it.  For example 
            
            uvel_CC,    UVEL,   uvel_FC,   
            vvel_CC,    VVEL,   vvel_FC,
            wvel_CC,    WVEL,   wvel_FC,
            ....
            
           Warning: type checking isn't done on the variables in the ... of the
           parameter list.
            
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/09/99    
 ---------------------------------------------------------------------  */
 
 void update_CC_FC_physical_boundary_conditions( 
              
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
    double  delX,                       /* cell size in x-direction         (INPUT) */
    double  delY,                       /* cell size in y-direction         (INPUT) */
    double  delZ,                       /* cell size in z-directin          (INPUT) */
    int     ***BC_types,                /* array containing the different   (INPUT) */
                                        /* types of boundary conditions             */
                                        /* BC_types[wall][variable]=type            */
    int     ***BC_float_or_fixed,       /* array that designates which variable is  */
                                        /* either fixed or floating on each wall of */
                                        /* the compuational domain                  */
    double ***BC_Values,                /* Values of the variables at the   (INPUT) */
                                        /* boundaries                       */ 
    int    nMaterials,                  /* number of materials              */
    int    n_data_arrays,               /* number of data arrays            */
         ...)          
 {
    va_list ptr_data_array;             /* pointer to each data array       */
    int    m,
           array,                       /* array number                     */
           var;
    double ****data_CC,
           ******data_FC;
/*__________________________________
*   double check inputs.
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
    
/*______________________________________________________________________
*   Now set the  physical boundary condtions for each dependent variable
*_______________________________________________________________________*/
    va_start(ptr_data_array, n_data_arrays);
    array = 0;
    /*__________________________________
    *   Loop through each data array in the
    *   argument list
    *___________________________________*/ 
    for (array = 1; array <=n_data_arrays; array++)
    {
        /*__________________________________
        *   Get the data array and var from
        *   the variable length argument list
        *___________________________________*/
        data_CC = va_arg(ptr_data_array, double****); 
        var     = va_arg(ptr_data_array, int);
        data_FC = va_arg(ptr_data_array, double******);

        for (m = 1; m <= nMaterials; m++)
        { 
            set_Dirichlet_BC(          
                                xLoLimit,        yLoLimit,      zLoLimit,
                                xHiLimit,        yHiLimit,      zHiLimit,
                                data_CC,         var,          
                                BC_types,        BC_Values,     m        );
            set_Neumann_BC(          
                                 xLoLimit,       yLoLimit,      zLoLimit,
                                 xHiLimit,       yHiLimit,      zHiLimit,
                                 delX,           delY,          delZ,
                                 data_CC,        var,          
                                 BC_types,       BC_Values,     
                                 m        );

            set_Periodic_BC(          
                                xLoLimit,        yLoLimit,      zLoLimit,
                                xHiLimit,        yHiLimit,      zHiLimit,
                                data_CC,         var,          
                                BC_types,        m );


            set_corner_cells_BC( 
                                xLoLimit,       yLoLimit,       zLoLimit,       
                                xHiLimit,       yHiLimit,       zHiLimit,       
                                data_CC,      
                                m        );

            set_Dirichlet_BC_FC(          
                                xLoLimit,       yLoLimit,       zLoLimit,
                                xHiLimit,       yHiLimit,       zHiLimit,
                                data_FC,        var,             
                                BC_types,       BC_Values,      BC_float_or_fixed,     
                                m        );

            /*__________________________________
            *   set the boundary conditions for
            *   both Neuman and Periodic
            *___________________________________*/
           set_Neumann_BC_FC(          
                                xLoLimit,       yLoLimit,       zLoLimit,
                                xHiLimit,       yHiLimit,       zHiLimit,
                                delX,           delY,           delZ,
                                data_CC,        data_FC,        var,
                                BC_types,       BC_Values,      BC_float_or_fixed,     
                                m        );
      }  
/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
        #if switchDebug_update_CC_FC_physical_boundary_conditions
            fprintf(stderr,"****************************************************************************\n");
            fprintf(stderr,"            UPDATE_CC_FC_PHYSICAL_BOUNDARY_CONDITIONS\n");
            fprintf(stderr,"****************************************************************************\n");
           for (m = 1; m <= nMaterials; m++)
           {          
                fprintf(stderr,"\t Material %i \n",m);
                printData_4d(       GC_LO(xLoLimit),     GC_LO(yLoLimit),       GC_LO(zLoLimit),
                                    GC_HI(xHiLimit),     GC_HI(yHiLimit),       GC_HI(zHiLimit),
                                    m,                  m,
                                   "update_CC_FC_physical_boundary_conditions",     
                                   "data with ghost cells",     
                                                data_CC);

                printData_6d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),       (zLoLimit),
                                    GC_HI(xHiLimit),    GC_HI(yHiLimit),       (zHiLimit),
                                    TOP,                LEFT,
                                    m,                  m,
                                   "update_CC_FC_physical_boundary_conditions",     
                                   "data_FC with ghost cells",                  data_FC,        0);
            }

            fprintf(stderr,"****************************************************************************\n");         

            fprintf(stderr,"press return to continue\n");
            getchar();            
        #endif  

    }     

    va_end(ptr_data_array);                     /* clean up when done   */            
}
/*STOP_DOC*/

/* 
 ======================================================================*/
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  set_Dirichlet_BC--BOUNDARY CONDITIONS: Set the Dirichlet BC for each variable at the cell center.
 Filename:  boundary_cond.c
 Purpose:
            This function sets the boundary conditions along the walls
            of the computational domain for dependent variable.  This does
            NOT include the corner cells.  

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       08/22/99    
 ---------------------------------------------------------------------  */
 
 void set_Dirichlet_BC( 
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
    double  ****data_CC,                /* cell-centered data               (IN/OUT)*/
                                        /* data_CC(x,y,z, material)                 */
    int     var,                        /* variable (TEMP,PRESS,UVEL,VVEL,DENSITY...*/
    int     ***BC_types,                /* defines which boundary conditions(INPUT) */
                                        /* have been set on each wall               */
    double  ***BC_Values,               /* BC values BC_values[wall][variable(INPUT) */
    int        nMaterials        )

 {
    int     i,j,k,m,                     /* indices                         */
            wall,       
            xLo,        xHi, 
            yLo,        yHi, 
            zLo,        zHi,
            wallLo,     wallHi,
            should_I_leave;
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
/*START_DOC*/
/*__________________________________
*   Determine the looping indices
*   for multidimensional problems
*___________________________________*/
#if (N_DIMENSIONS == 1)  
        wallLo = LEFT;  wallHi = RIGHT;
#endif

#if (N_DIMENSIONS == 2) 
        wallLo = TOP;   wallHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
        wallLo = TOP;   wallHi = BACK;
#endif

/*__________________________________
*   Test to see if you should be in this function
*___________________________________*/
    should_I_leave = YES;
    for(m = 1; m <= nMaterials; m++)
    {
        for( wall = wallLo; wall <= wallHi; wall ++)
        {
            if(BC_types[wall][var][m] == DIRICHLET ) should_I_leave = NO;
           
        }
    }
    if (should_I_leave == YES) return;
/*______________________________________________________________________
*   Now loop over all of the walls and set the appropriate BC
*   You need to find what the appropriate looping limits are for each 
*   different wall.
*_______________________________________________________________________*/
            
    for( wall = wallLo; wall <= wallHi; wall ++)
    {
        
        for(m = 1; m <= nMaterials; m++)
        {
            if( BC_types[wall][var][m] == DIRICHLET)
            {
                 find_loop_index_limits_at_domain_edges(                
                            xLoLimit,                  yLoLimit,                   zLoLimit,
                            xHiLimit,                  yHiLimit,                   zHiLimit,
                            &xLo,                      &yLo,                       &zLo,
                            &xHi,                      &yHi,                       &zHi,
                            wall    );
               /*__________________________________
               *   For each variable set the boundary condition
               *   if appropriate
               *___________________________________*/

                for ( k = zLo; k <= zHi; k++)
                {
                    for ( j = yLo; j <= yHi; j++)
                    {
                        for ( i = xLo; i <= xHi; i++)
                        {
                            /*__________________________________
                            *   Velocity BC
                            *___________________________________*/
                            data_CC[m][i][j][k] = BC_Values[wall][var][m];

                        }
                    }
                }
            }
        }
    }
/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
#if switchDebug_set_Dirichlet_BC
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        SET_Dirichlet_BC\n");
    fprintf(stderr,"****************************************************************************\n");         
    for (m = 1; m <= nMaterials; m++)
    {
        fprintf(stderr,"\t Material %i \n",m);
        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),        GC_LO(zLoLimit),
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),        GC_HI(zHiLimit),
                            m,                  m,
                           "set_Dirichlet_BC",     
                           "data_CC with ghost cells",                  uvel_CC);
    }

    fprintf(stderr,"****************************************************************************\n");             
    fprintf(stderr,"press return to continue\n");
    getchar();            
#endif

 }
 /*STOP_DOC*/
 

 /* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  set_Neumann_BC--BOUNDARY CONDITIONS: Set the Neuman BC for each variable at the cell center.
 Filename:  boundary_cond.c
 Purpose:
            This function backs out the proper value for the dependent variable
            from the specified Neuman condition.  

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       09/17/99    

    Currently the differences are 1st order approximations
 ---------------------------------------------------------------------  */
 
 void set_Neumann_BC( 
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* cell size in x direction         (INPUT) */
        double  delY,                   /* cell size in y direction         (INPUT) */
        double  delZ,                   /* cell size in z direction         (INPUT) */
                                        /* (*)vel_CC(x,y,z,material)                */
        double  ****data_CC,            /* cell-centered data               (IN/OUT)*/
        int     var,                    /* variable type (TEMP,PRESS,RHO,UVEL...... */         
        int     ***BC_types,            /* defines which boundary conditions(INPUT) */
                                        /* have been set on each wall               */
        double  ***BC_Values,           /* BC values BC_values[wall][variable(INPUT)*/
        int     nMaterials        )

 {
    int     m,                          /* indices                           */
            wall,      
            xLo,        xHi, 
            yLo,        yHi, 
            zLo,        zHi,
            wallLo,     wallHi,
            should_I_leave;

/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
/*START_DOC*/
/*__________________________________
*   Determine the looping indices
*   for multidimensional problems
*___________________________________*/
#if (N_DIMENSIONS == 1) 
        wallLo = LEFT;  wallHi = RIGHT;
#endif
#if (N_DIMENSIONS == 2) 
        wallLo = TOP;   wallHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
        wallLo = TOP;   wallHi = BACK;
#endif
/*__________________________________
*   Test to see if you should be in this function
*___________________________________*/
    should_I_leave = YES;
    
    for(m = 1; m <= nMaterials; m++)
    {
        for( wall = wallLo; wall <= wallHi; wall ++)
        {
            if(BC_types[wall][var][m] == NEUMANN ) should_I_leave = NO;
        }
    }
    if (should_I_leave == YES) return;
    
/*______________________________________________________________________
*   Now loop over all of the walls and set the appropriate BC
*   You need to find what the looping limits are for each 
*   side of the computational domain.  These limits don't include the 
*   corner cells
*_______________________________________________________________________*/
           
    for( wall = wallLo; wall <= wallHi; wall ++)
    {
        for(m = 1; m <= nMaterials; m++)
        {        
            if( BC_types[wall][var][m] == NEUMANN)
            {
                find_loop_index_limits_at_domain_edges(
                       xLoLimit,                  yLoLimit,                   zLoLimit,
                       xHiLimit,                  yHiLimit,                   zHiLimit,
                       &xLo,                      &yLo,                       &zLo,
                       &xHi,                      &yHi,                       &zHi,
                       wall    );
                /*__________________________________
                *   Now set the boundary conditions
                *___________________________________*/

                     neumann_BC_diffenence_formula(
                        xLo,            yLo,            zLo,
                        xHi,            yHi,            zHi,
                        wall,
                        delX,            delY,          delZ,
                        data_CC,        BC_Values,      var,
                        m        );

            }
        }
    }
/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
#if switchDebug_set_Neumann_BC
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        SET_NEUMANN_BC\n");
    fprintf(stderr,"****************************************************************************\n");         
    for (m = 1; m <= nMaterials; m++)
    {
        fprintf(stderr,"\t Material %i \n",m);
        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),       (zLoLimit),
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),       (zHiLimit),
                            m,                  m,
                           "set_Neumann_BC",     
                           "data_FC with ghost cells",                  data_CC);
    }

    fprintf(stderr,"****************************************************************************\n");         
    
    fprintf(stderr,"press return to continue\n");
    getchar();            
#endif
/*__________________________________
*   Quite fullwarn remarks is a way that
*   is compiler independent
*___________________________________*/
           QUITE_FULLWARN(delZ);

 }
 /*STOP_DOC*/
 
 
/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  neumann_BC_diffenence_formula--BOUNDARY CONDITIONS: Definition of the differencing formula used when computing Neumann BC.
 Filename:  boundary_cond.c
 Purpose:   This function does the actual computation for the set_neuman_BC
            function  

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/7/99    

    Currently the differences are 1st order approximations
 ---------------------------------------------------------------------  */
 
 void neumann_BC_diffenence_formula( 
         int     xLo,             
         int     yLo,             
         int     zLo,             
         int     xHi,             
         int     yHi,             
         int     zHi,
         int     wall,             
         double  delX,                  /* cell size in x direction         (INPUT) */
         double  delY,                  /* cell size in y direction         (INPUT) */
         double  delZ,                  /* cell size in z direction         (INPUT) */
         double  ****data_CC,           /* cell-centered data(i,j,k,m)      (IN/OUT)*/
         double  ***BC_Values,          /* BC values BC_values[wall][var][m](INPUT)*/
         int     var,                   /* name of the field dependent variable     */
                                        /* UVEL, VVEL, WVEL,PRESS, TEMP, DENSITY    */
         int     nMaterials        )
{         
    int i, j, k, m; 
    m = nMaterials;

    
   for ( i = xLo; i <= xHi; i++ )
    {
        for ( j = yLo; j <= yHi; j++ )
        {
            for ( k = zLo; k <= zHi; k++ )
            {

               if ( wall == LEFT )
               {
                    data_CC[m][i][j][k] = data_CC[m][i+1][j][k]
                        - BC_Values[wall][var][m] * delX;
                }
               if ( wall == RIGHT )
               {
                    data_CC[m][i][j][k] = data_CC[m][i-1][j][k]
                        + BC_Values[wall][var][m] * delX;
                }
               if ( wall == TOP )
               {
                    data_CC[m][i][j][k] = data_CC[m][i][j-1][k]
                        + BC_Values[wall][var][m] * delY;
                }
                if ( wall == BOTTOM )
               {
                    data_CC[m][i][j][k] = data_CC[m][i][j+1][k]
                        - BC_Values[wall][var][m] * delY;
                }

            }
        }
    }
/*__________________________________
*   Quite fullwarn remarks is a way that
*   is compiler independent
*___________________________________*/
    delZ =delZ; 
}
/*STOP_DOC*/


/* 
 ======================================================================*/
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  set_Periodic_BC--BOUNDARY CONDITIONS: Set the Periodic BC for each variable at the cell center.
 Filename:  boundary_cond.c
 Purpose:
            This function sets the boundary conditions along the walls
            of the computational domain for dependent variable.  This does
            NOT include the corner cells.  

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/10/99    
 ---------------------------------------------------------------------  */
 
 void set_Periodic_BC( 
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
    double  ****data_CC,                /* cell-centered data               (IN/OUT)*/
                                        /* data_CC(x,y,z, material)                 */
    int     var,                        /* variable (TEMP,PRESS,UVEL,VVEL,DENSITY...*/
    int     ***BC_types,                /* defines which boundary conditions(INPUT) */
                                        /* have been set on each wall               */
    int        nMaterials        )

 {
    int     i,j,k,m,                     /* indices                         */
            wall,       
            xLo,        xHi, 
            yLo,        yHi, 
            zLo,        zHi,
            wallLo,     wallHi,
            should_I_leave;
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
/*START_DOC*/
/*__________________________________
*   Determine the looping indices
*   for multidimensional problems
*___________________________________*/
#if (N_DIMENSIONS == 1)  
        wallLo = LEFT;  wallHi = RIGHT;
#endif

#if (N_DIMENSIONS == 2) 
        wallLo = TOP;   wallHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
        wallLo = TOP;   wallHi = BACK;
#endif

/*__________________________________
*   Test to see if you should be in this function
*___________________________________*/
    should_I_leave = YES;
    for(m = 1; m <= nMaterials; m++)
    {
        for( wall = wallLo; wall <= wallHi; wall ++)
        {
            if(BC_types[wall][var][m] == PERIODIC ) should_I_leave = NO;
           
        }
    }
    if (should_I_leave == YES) return;
/*______________________________________________________________________
*   Now loop over all of the walls and set the appropriate BC
*   You need to find what the appropriate looping limits are for each 
*   different wall.
*_______________________________________________________________________*/
            
    for( wall = wallLo; wall <= wallHi; wall ++)
    {
        
        for(m = 1; m <= nMaterials; m++)
        {
            if( BC_types[wall][var][m] == PERIODIC)
            {
                 find_loop_index_limits_at_domain_edges(                
                            xLoLimit,                  yLoLimit,                   zLoLimit,
                            xHiLimit,                  yHiLimit,                   zHiLimit,
                            &xLo,                      &yLo,                       &zLo,
                            &xHi,                      &yHi,                       &zHi,
                            wall    );
               /*__________________________________
               *   For each variable set the boundary condition
               *    I'm doing twice as much work as I need to
               *___________________________________*/

                for ( k = zLo; k <= zHi; k++)
                {
                    for ( j = yLo; j <= yHi; j++)
                    {
                        for ( i = xLo; i <= xHi; i++)
                        {
                            if ( wall == LEFT )
                            {
                                 data_CC[m][i][j][k] = data_CC[m][xHiLimit][j][k];
                             }
                            if ( wall == RIGHT )
                            {
                                 data_CC[m][i][j][k] = data_CC[m][xLoLimit][j][k];
                             } 
                             if ( wall == TOP )
                            {
                                 data_CC[m][i][j][k] = data_CC[m][i][yLoLimit][k];
                             }
                             if ( wall == BOTTOM )
                            {
                                 data_CC[m][i][j][k] = data_CC[m][i][yHiLimit][k];
                             } 
                            

                        }
                    }
                }
            }
        }
    }
/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
#if switchDebug_set_Periodic_BC
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        SET_PERIODIC_BC\n");
    fprintf(stderr,"****************************************************************************\n");         
    for (m = 1; m <= nMaterials; m++)
    {
        fprintf(stderr,"\t Material %i \n",m);
        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),        GC_LO(zLoLimit),
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),        GC_HI(zHiLimit),
                            m,                  m,
                           "set_Periodic_BC",     
                           "data_CC with ghost cells",                  data_CC);
    }

    fprintf(stderr,"****************************************************************************\n");             
    fprintf(stderr,"press return to continue\n");
    getchar();            
#endif

 }
 /*STOP_DOC*/
 
/*
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  set_corner_cells_BC--BOUNDARY CONDITIONS: Set the cell-center 
            BC in the corner ghostcells.
 Filename:  boundary_cond.c
 Purpose:
            This function sets the primative cell-centered boundary conditions 
            in the corner ghost cells.  The value in the corner cell is assumed
            to be the average value of the intersecting walls

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       08/22/99    


CAVEAT:     This function must be called after set_Dirichlet_BC and
            set_Neuman_BC.
 ---------------------------------------------------------------------  */
 
 void set_corner_cells_BC( 
              
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
    double  ****data_CC,                /* cell-centered data               (IN/OUT)*/
    int        m        )

 {
    int     i,j,k,                     /* indices                           */
            zLo, zHi;
/*__________________________________
*   double check inputs 
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
/*START_DOC*/
/*______________________________________________________________________
*   NEED TO ADD THE 3D
*_______________________________________________________________________*/
    i   = GC_LO(xLoLimit);
    j   = GC_HI(yHiLimit);
    zLo = GC_LO(zLoLimit);
    zHi = GC_HI(zHiLimit);
    for (k = zLo; k <= zHi; k++)
    {
        data_CC[m][i][j][k]     = ( data_CC[m][i][j-1][k] + data_CC[m][i+1][j][k]) /2.0;
    }
/*__________________________________
*   Upper right ghostcell corner
*___________________________________*/
    i = GC_HI(xHiLimit);
    j = GC_HI(yHiLimit);
    for (k = zLo; k <= zHi; k++)
    {
        data_CC[m][i][j][k]     = ( data_CC[m][i][j-1][k] + data_CC[m][i-1][j][k]) /2.0;
    }
    
/*__________________________________
*   Lower right ghostcell corner
*___________________________________*/
    i = GC_HI(xHiLimit);
    j = GC_LO(yLoLimit);
    k = zLoLimit;
    for (k = zLo; k <= zHi; k++)
    {
        data_CC[m][i][j][k]     = ( data_CC[m][i][j+1][k] + data_CC[m][i-1][j][k]) /2.0;
    }
/*__________________________________
*   Lower left ghostcell corner
*___________________________________*/
    i = GC_LO(xLoLimit);
    j = GC_LO(yLoLimit);
    for (k = zLo; k <= zHi; k++)
    {   
        data_CC[m][i][j][k]     = ( data_CC[m][i][j+1][k] + data_CC[m][i+1][j][k]) /2.0;
    }
}
/*STOP_DOC*/
 
 
 
 
