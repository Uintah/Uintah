/*  
======================================================================*/
#include "macros.h"
#include "parameters.h"
#include "functionDeclare.h"
#include  <math.h>
/*
 Function:  compute_initial_constants_NIST_TE( 
 Filename:  nist_file.c
 Purpose:
            Compute constants that are needed throughout the NIST fire
            code. 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/22/00   written   
       
 ---------------------------------------------------------------------- */
 void   compute_initial_constants_NIST_TE(
        int     xLoLimit,
        int     yLoLimit,
        int     zLoLimit,
        int     xHiLimit,
        int     yHiLimit,
        int     zHiLimit,
        double  ****Temp_CC,                /* Cell-centered Temperature        */
        double  ****rho_CC,                 /* Cell-centered density            */  
        double  *grav,                      /* gravity                          */
        double  del_H_fuel,                 /* heat of combution of fuel        */
        double  rho_fuel,                   /* density of fuel                  */
        double  Q_fire,                     /* heat release of fire             */ 
        double  *u_inject_TE,               /* velocity which TE are injected   */
        double  *t_burnout_TE,              /* burnout time of TE               */         
        int     m )                         /* material index                   */          
{     
        double  D_star,                     /* characteristic dia. fire         */
                T_inf,                      /* Temperature at infinity          */
                cp_inf,                     /* cp at infinity                   */
                rho_inf,                    /* rho at infinity                  */
                A_fire,                     /* Area of fire                     */
                g;                          /* Temp. variable                   */
        static int
                first_time_through;         /* flag                             */       
    /*__________________________________
    *   Bullet proof the inputs
    *___________________________________*/                
    if (fabs(grav[2]) < SMALL_NUM)
        Message(1,"File: nist_fire","Function: compute_initial_constants_NIST_TE",
       "Error: Gravity = 0.0, double check the input file");
                
                
    /*__________________________________
    * Hardwire some crap for now
    *___________________________________*/           
    cp_inf  = 716;
    T_inf   = 300;
    A_fire  = 1.0;
    rho_inf = 1.0;
    g = fabs(grav[2]);    
     
    if (first_time_through == 0) first_time_through = YES;       
    if(first_time_through == YES)
    {        
        /*__________________________________
        *   Compute D_star
        *___________________________________*/        
        D_star = (Q_fire/(cp_inf * rho_inf * T_inf * sqrt(g)));
        D_star = pow (D_star, 0.40);

        /*__________________________________
        *   time burnout
        *___________________________________*/
        
        *t_burnout_TE = 1.5 * sqrt(D_star/g);

        /*__________________________________
        *   u_eject_TE
        *___________________________________*/
        *u_inject_TE = Q_fire/(rho_fuel * del_H_fuel * A_fire);
        
        first_time_through = NO;
    }     
    
    
    /*__________________________________
    *   QUITE FULL WARNINGS
    *   I anticipate that these will eventually
    *   get used.
    *___________________________________*/
    QUITE_FULLWARN(Temp_CC[1][0][0][0]);        
    QUITE_FULLWARN(rho_CC[1][0][0][0]);
    xLoLimit = xLoLimit;        yLoLimit = yLoLimit;        zLoLimit = zLoLimit;
    xHiLimit = xHiLimit;        yHiLimit = yHiLimit;        zHiLimit = zHiLimit;
    m = m;     
}



/*  
======================================================================*/
#include <stdlib.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "inline_NIST.h"
#include "nrutil+.h"
#include <math.h>
/*
 Function:  update_Q_TE 
 Filename:  nist_file.c
 Purpose:
            update the heat release rate for each thermal element. 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/22/00   written   
       
  Note that having and if-then inside of th nThermalElements loop is slow

 ---------------------------------------------------------------------- */
 void   update_Q_TE_and_add_Q_cell(
        int     xLoLimit,
        int     yLoLimit,
        int     zLoLimit,
        int     xHiLimit,
        int     yHiLimit,
        int     zHiLimit,
        double  delX,
        double  delY,
        double  delZ,
        double  t,                      /* current time                     (INPUT) */
        double  *x_TE,                  /* position of the TE               (INPUT) */
        double  *y_TE,
        double  *z_TE,
        double  Q_fire,                 /* Heat release rate of the fire    (INPUT) */
        double  *t_inject_TE,           /* time TE was injected             (INPUT) */
        double  *t_inject_TE_parms,     
        double  t_burnout_TE,           /* burn-out time of TE              (INPUT) */
        int     N_inject_TE,            /* number of TE injected            (INPUT) */
        int     nThermalElements,       /* number of thermal elements       (INPUT) */
        double  *Q_TE,                  /* heat release from TE             (OUTPUT)*/
        double  ***Q_chem,              /* Heat transfered from TE to cell  (OUTPUT)*/
        int     m)                      /* material                         (INPUT) */
        
          
{ 
        int     i, j, k, n;
        double  vol,                    /* cell volume                      */
                bullet_proof_test;
        char    should_I_write_output;
    /*__________________________________
    *   Plotting variables
    *___________________________________*/
    #if (switchDebug_update_Q_TE_and_add_Q_cell == 1)
        #include "plot_declare_vars.h"   
    #endif 
    
     
   /*__________________________________
   *    zero out Q_chem
   *___________________________________*/
    zero_arrays_3d(
            xLoLimit,       yLoLimit,       zLoLimit,
            xHiLimit,       yHiLimit,       zHiLimit,
            1,              Q_chem );

    /*______________________________________________________________________
    *   M  A  I  N     L  O  O  P
    *_______________________________________________________________________*/
     for (n = 1 ; n <= nThermalElements; n++)
     {
        /*__________________________________
        * find the cell index that the TE
        * lives in 
        *___________________________________*/
        i = xLoLimit + x_TE[n]/delX - 1;
        j = yLoLimit + y_TE[n]/delY - 1;
        k = zLoLimit + z_TE[n]/delZ - 1;

                        
        if ( (i > GC_HI(xHiLimit) || i < GC_LO(xLoLimit) )     ||
             (j > GC_HI(yHiLimit) || j < GC_LO(yLoLimit) )     ||
             (k > GC_HI(zHiLimit) || k < GC_LO(zLoLimit) ) ) 
        {
            /*__________________________________
            *   If your outside the domain then
            *   set the Q_TE = 0.0
            * This is commented out so that the 
            *   plotting scale is correct
            *___________________________________*/
            /* Q_TE[n] = 0.0; */
        } else
        {
            /*__________________________________
            * Update the heat release rate of the TE
            * if t - t_inject_TE< t_burnout_TE
            *___________________________________*/
            if (t - t_inject_TE[n] < t_burnout_TE ) 
            {
            
                vol     = delX * delY * delZ;
                Q_TE[n] = ( Q_fire * t_inject_TE_parms[3] )/
                          ( N_inject_TE * vol * (t_burnout_TE/2.0) );
                          
                Q_TE[n] = Q_TE[n] * burnout_function((t - t_inject_TE[n]), t_burnout_TE);
                Q_chem[i][j][k] = Q_chem[i][j][k] + Q_TE[n];          
            }
        }
        bullet_proof_test = DMIN(bullet_proof_test, Q_TE[n]);
    }
    /*__________________________________
    *  Bulletproofing
    *___________________________________*/
    if (bullet_proof_test < 0)
        Message(1,"File: nist_fire","Function: update_Q_TE_and_add_Q_cell",
       "Error: The thermal element heat release was negative \n probably due to t_burnout");
    
    
/*______________________________________________________________________
*       P  L  O  T  T  I  N  G 
*_______________________________________________________________________*/    
#if (switchDebug_update_Q_TE_and_add_Q_cell == 1)
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");  
    if ( should_I_write_output == '1')
    {       
        /*__________________________________
         * Define plotting variables 
         *___________________________________*/
         plot_type           = 1;
         Number_sub_plots    = 1;
         strcpy(file_basename,"TE");
         outputfile_type     = 1;
         x_axis_origin       = (xLoLimit);
         y_axis_origin       = (yLoLimit);
         x_axis              = (xHiLimit);
         y_axis              = (yHiLimit);
         outline_ghostcells  = 1;
         strcpy(x_label,"cell\0");
         strcpy(y_label,"cell\0");                       
         /*__________________________________
         *   MAIN WINDOW 1 Particle data
         *___________________________________*/ 
         sprintf(graph_label, "update_Q_TE_and_add_Q_cell, Q_TE\0");

         plot_particles(  
                         xLoLimit,       xHiLimit,           
                         yLoLimit,       yHiLimit,             
                         delX,           delY,                 
                         x_TE,           y_TE,           Q_TE,                
                         x_label,        y_label,        graph_label,        
                         outline_ghostcells,             Number_sub_plots,          
                         nThermalElements,file_basename, outputfile_type);


             outputfile_type     = 0;
             strcpy(file_basename,"TE");
             sprintf(graph_label, "update_Q_TE_and_add_Q_cell, Q_chem");                   
             data_array1    = convert_darray_3d_to_vector(
                                     Q_chem,
                                    (xLoLimit),       (xHiLimit),       (yLoLimit),
                                    (yHiLimit),       (zLoLimit),       (zHiLimit),
                                     &max_len);
             PLOT;
             free_vector_nr(    data_array1,       1, max_len);
    }
    
#endif
    /*__________________________________
    *   QUITE FULL WARNS
    *___________________________________*/
    m = m;
}




/*  
======================================================================*/
#include "macros.h"
#include "parameters.h"
#include "functionDeclare.h"
#define INDEX(x) ( fabs(x)>(1.0)?SIGN(x)1:(0) )
/*
 Function:  update_TE_position  
 Filename:  nist_file.c
 Purpose:
            update the position of each thermal element. 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/18/00   written   

 ---------------------------------------------------------------------- */
 void   update_TE_position(
        int     xLoLimit,
        int     yLoLimit,
        int     zLoLimit,
        int     xHiLimit,
        int     yHiLimit,
        int     zHiLimit,
        double  delX,
        double  delY,
        double  delZ,
        int     nThermalElements,       /* number of thermal elements       (INPUT) */
        int     **cell_index_TE,        /* array that contains which cell   */
                                        /* TE(n) lives in                   (INPUT) */
        double  *x_TE,                  /* x position of TE (n)             (IN/OUT)*/
        double  *y_TE,                  /* y position of TE (n)             (IN/OUT)*/
        double  *z_TE,                  /* z position of TE (n)             (IN/OUT)*/  
        double  ****uvel_CC,            /* u-cell-centered velocity         (INPUT) */
        double  ****vvel_CC,            /* v-cell-centered velocity         (INPUT) */
        double  ****wvel_CC,            /* v-cell-centered velocity         (INPUT) */
        double  delt,                   /* time increment                   (INPUT) */
        int     m)                      /* material                         (INPUT) */
          
{ 

        int i, j, k, n;
       
     for (n = 1 ; n <= nThermalElements; n++)
     {
        /*__________________________________
        * find the cell index that the TE
        * lives in 
        *___________________________________*/
        i = xLoLimit + x_TE[n]/delX - 1;
        j = yLoLimit + y_TE[n]/delY - 1;
        k = zLoLimit + z_TE[n]/delZ - 1;
                        
        if ( (i > GC_HI(xHiLimit) || i < GC_LO(xLoLimit) )     ||
             (j > GC_HI(yHiLimit) || j < GC_LO(yLoLimit) )     ||
             (k > GC_HI(zHiLimit) || k < GC_LO(zLoLimit) ) ) 
        {
            /*__________________________________
            *   If your outside the domain then
            *   set the position = 0.0
            *___________________________________*/
            x_TE[n] = 0.0;
            y_TE[n] = 0.0;
            z_TE[n] = 0.0;
        } else
        {
            /*__________________________________
            * Update the position of the TE
            *___________________________________*/
            x_TE[n] = x_TE[n] + delt * uvel_CC[m][i][j][k];
            y_TE[n] = y_TE[n] + delt * vvel_CC[m][i][j][k];
            z_TE[n] = z_TE[n] + delt * wvel_CC[m][i][j][k];
        }
    }
    /*__________________________________
    *   QUITE FULL WARN
    *___________________________________*/
    QUITE_FULLWARN(cell_index_TE[1][1]);
}
 

 /*  
======================================================================*/
#include <stdlib.h>
#include "parameters.h"
#include "functionDeclare.h"
/*
 Function:  inject_TE  
 Filename:  nist_file.c
 Purpose:
            distribute the particles in the injection region.
 Steps:
            - Compute the size of the injection region
            - Find the number of injectd thermal elements per cell
            - Randomly place the elements in each cell in the 
              injection region. 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/18/00   written   
       
 Implementation:  This is currently called everytime step and the 
  number of thermal elements continues to increase.

 ---------------------------------------------------------------------- */
 void   inject_TE(
    int     *index_inject_TE,       /* indices where the particles are  (INPUT) */
                                    /* injected [1] = xlo, [2] = xHi,           */
                                    /*          [3] = yLo, [4] = yHi,           */
                                    /*          [5] = zLo, [6] = zHi            */
    int     N_inject_TE,            /* TE's injected                    (INPUT) */
    double  *t_inject_TE_parms,     /* array holding output timing info (INPUT) */
                                    /*  [1] = When to begin injection           */
                                    /*  [2] = When to stop injecting            */
                                    /*  [3] = delta t injection                 */
    double  t,                      /* current time                             */
    double  delX,                   /* cell spacing                     (INPUT) */               
    double  delY,
    double  delZ,
    double  u_inject_TE,            /* injection velocity for TE        (INPUT) */    
    double  *t_inject_TE,           /* time that TE was injected        (OUTPUT)*/       
    double  *x_TE,                  /* x position of TE[n]              (OUTPUT)*/
    double  *y_TE,                  /* y position of TE[n]              (OUTPUT)*/
    double  *z_TE,                  /* z position of TE[n]              (OUTPUT)*/  
    int     *nThermalElements )     /* number of thermal elements       (OUTPUT)*/ 
          
{ 
    int     i, j, n,
            time_to_inject,         /* flag for deciding if time to inject      */  
            counter,                /* temporary counter                        */
            x_cells, y_cells, z_cells,  /* #cells in injection region           */
            total_cells,            /* total number of cells in region          */
            TE_per_cell;            /* TE's per cell                            */
            
    double  factor;                 /* factor * del(*) + i * del(*)     */
    
    
    time_to_inject = Is_it_time_to_inject( t, t_inject_TE_parms  );   
    if (time_to_inject == YES)
    {     
        /*__________________________________
        * Test to see if there is enough
        * memory allocated 
        *___________________________________*/

        if (*nThermalElements + N_inject_TE > N_THERMAL_ELEMENTS)
            Message(0,"File: nist_fire","Function: inject_TE",
           "Error: Exceeded the allocated memory space, no more TE will be injected");

        else
        { 
            /*__________________________________
            *   Find the number of cells in the
            *   injection region and compute
            *   the number of TE per cell
            *___________________________________*/    
            x_cells = index_inject_TE[2] - index_inject_TE[1] + 1;
            y_cells = index_inject_TE[4] - index_inject_TE[3] + 1;
            z_cells = index_inject_TE[6] - index_inject_TE[5] + 1;
            total_cells = x_cells * y_cells * z_cells;

            TE_per_cell = N_inject_TE/total_cells;

            if(TE_per_cell == 0)
            {
                fprintf(stderr,"Number of cells in the injection region: \t %i \n",total_cells);
                Message(1,"WARNING:  The number of thermal elements that will be injected in any cell = zero",
                "",  "Increase N_inject_TE in the input file to at least 1 per cell in the injection region");
            }

            counter = *nThermalElements;
            /*__________________________________
            *   Need to do something with u_inject_TE here
            *___________________________________*/
            
            
            /*__________________________________
            *   Now loop over the injection region
            *   and dump TE_per_cell in each cell
            *___________________________________*/
            for ( j = (index_inject_TE[3]); j <= (index_inject_TE[4]); j++)
            {
                for ( i = (index_inject_TE[1]); i <= (index_inject_TE[2]); i++)
                {      
                    for(n = 1; n <= TE_per_cell; n++)
                    {
                        counter ++;
                        t_inject_TE[counter] = t;
                        factor = (double)rand()/RAND_MAX;
                        x_TE[counter] = i * delX + factor * delX;
                        y_TE[counter] = j * delY + factor * delY;
                        z_TE[counter] = delZ;
                    }

                }
            }
            *nThermalElements = counter;
        }
    }   
    
    /*__________________________________
    *   QUITE FULLWARN remarks
    *___________________________________*/  
    u_inject_TE = u_inject_TE;
}   



/*---------------------------------------------------------------------*/
#include <stdlib.h>
#include <math.h>
#include "parameters.h"
#include "functionDeclare.h"
#define  EPSILON 1.0e-6

/*
 Function:  Is_it_time_to_write_eject
 Filename:  nist_fire.c
 Purpose:
   Test to see if it is time to eject particles.
   If it is then return YES if not then return NO.  
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/18/00    
 
 ---------------------------------------------------------------------  */ 
 int Is_it_time_to_inject(
        double  t,                      /* current time                     */
        double  *t_inject_TE_parms)     /* array holding output timing info */
                                        /*  [1] = When to begin injection   */
                                        /*  [2] = When to stop injecting    */
                                        /*  [3] = delta t injection         */
{
    static double   told;
    static int      first_time_through;
    
    first_time_through ++; 
/*__________________________________
* Do this only the first time through
*___________________________________*/
     if( t >= t_inject_TE_parms[1] && 
         t <= t_inject_TE_parms[2] &&
         first_time_through == 1  )
    {
        told = t;
        return YES;
    }
    

/*__________________________________
*   For all other passes through routine
*   Add epsilon to the difference of t - told
*   because of roundoff error.
*___________________________________*/   
    if( t >= t_inject_TE_parms[1] && 
        t <= t_inject_TE_parms[2] &&
         ((fabs(t - told) + EPSILON ) >= t_inject_TE_parms[3]) &&
        first_time_through != 1  )
    {
        told = t;
        return YES;
    }
    else
    {
        return NO;
    } 
}
/*STOP_DOC*/




    
