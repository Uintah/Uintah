
/*  
======================================================================*/
#include <stdio.h>
#include "parameters.h"
#include "functionDeclare.h"
#include "switches.h"
#include "macros.h"
#define  MAXSTR 120              /* maximum length string length     */
/*
 Function:  readInputFile--INPUT: Controller for reading in the input file.
 Filename:  input.c
 Purpose:
   Read the problem input file   

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99   

 ---------------------------------------------------------------------- */

    void readInputFile(    
           int     *xLoLimit,
           int     *yLoLimit,
           int     *zLoLimit,
           int     *xHiLimit,
           int     *yHiLimit,
           int     *zHiLimit,
           double  *delX,
           double  *delY,
           double  *delZ,               /* (*)_CC (i, j, k, material)       */
           double  ****uvel_CC,         /* u-cell-centered velocity         */
           double  ****vvel_CC,         /* v-cell-centered velocity         */
           double  ****wvel_CC,         /* w-cell-centered velocity         */
           double  ****Temp_CC,         /* Cell-centered temperature        */
           double  ****Press_CC,        /* Cell-centered pressure           */
           double  ****rho_CC,          /* Cell-centered density            */
           double  ****scalar1_CC,
           double  ****scalar2_CC,
           double  ****scalar3_CC,
           double  ****viscosity_CC,    /* Cell-centered Viscosity          */
           double  ****thermalCond_CC,  /* Cell-centered thermal conductivity*/
           double  ****cv_CC,           /* Cell-centered constant cp        */
           double  *R,                  /* Gas constant R[m]                */
           double  *gamma,              /* ratio of specific heats          */
           double  *t_final,            /* Time final                       */
           double  *t_output_vars,      /* array holding output timing info */
                                        /* t_output_vars[1] = t_initial     */
                                        /* t_output_vars[2] = t final       */
                                        /* t_output_vars[3] = delta t       */
           double  *delt_limits,        /* delt_limits[1] = delt_min        */
                                        /* delt_limits[2] = delt_max        */
                                        /* delt_limits[3] = delt_initial interation*/
           char    output_file_basename[],
           char    output_file_desc[],
           double  *grav,               /* gravity (dir)                    */
           double  ****speedSound,      /* speed of sound (x, y, z, material*/
           int     **BC_inputs,         /* array containing the different   */
                                        /* types of boundary conditions     */   
                                        /* at each wall [wall][m]           */
           double  ***BC_Values,        /* BC values BC_values[wall][variable][m]*/
           double  *CFL,
            int    *nMaterials         )
            
{

    int     i,j,k,m,
            counter;                    /* used in debugging code           */      

    char    filename[]="if";      
    FILE    *fp;
/*______________________________________________________________________
* Initialize local variables
*_______________________________________________________________________*/
    counter =0;
/*__________________________________
* Open the file
*___________________________________*/
/*      printf("Input the input file name\n");
    scanf("%s",filename);  */
    
    
    
    fp = fopen(filename,"r");
    if(fp == NULL)
        Message(1,"File: input.c","Subroutine: input","Error: Couldn't open the input data file");
/*__________________________________
* Now read the file description
* 
*___________________________________*/
       
    readstring(fp,output_file_basename," output file basename",switchDebug_readInputFile);
    readstring(fp,output_file_desc," output file desc",switchDebug_readInputFile);  
/*__________________________________
* Geometry Section
*___________________________________*/           
   *delX                                            = readdouble(fp," Delta_X",switchDebug_readInputFile);
   *delY                                            = readdouble(fp," Delta_Y",switchDebug_readInputFile);
   *delZ                                            = readdouble(fp," Delta_Z",switchDebug_readInputFile);
 
   *xLoLimit                                        = readint(fp," Initial index x-dir",switchDebug_readInputFile);
   *yLoLimit                                        = readint(fp," Initial index y-dir",switchDebug_readInputFile);
   *zLoLimit                                        = readint(fp," Initial index z-dir",switchDebug_readInputFile);
 
   *xHiLimit                                        = readint(fp," n _cells x-dir.",switchDebug_readInputFile);
   *yHiLimit                                        = readint(fp," n _cells y-dir.",switchDebug_readInputFile);
   *zHiLimit                                        = readint(fp," n _cells z-dir.",switchDebug_readInputFile);
   *nMaterials                                      = readint(fp," Number of Materials",switchDebug_readInputFile);
   *CFL                                             = readdouble(fp," CFL",switchDebug_readInputFile);

/*__________________________________
* Time variables
*___________________________________*/
    *t_final                                        = readdouble(fp," Final time",switchDebug_readInputFile);
    t_output_vars[1]                                = readdouble(fp," t_initial_output",switchDebug_readInputFile);
    t_output_vars[2]                                = readdouble(fp," t_final_output",switchDebug_readInputFile);
    t_output_vars[3]                                = readdouble(fp," t_delta_output",switchDebug_readInputFile);
    delt_limits[1]                                  = readdouble(fp," delt_minimum",switchDebug_readInputFile);
    delt_limits[2]                                  = readdouble(fp," delt_maximum",switchDebug_readInputFile);
    delt_limits[3]                                  = readdouble(fp," delt_initial_iter",switchDebug_readInputFile);

/*__________________________________
*   Body Force  
*___________________________________*/
    grav[1]                                         = readdouble(fp," gravity x-dir",switchDebug_readInputFile);  
    grav[2]                                         = readdouble(fp," gravity y-dir",switchDebug_readInputFile);
    grav[3]                                         = readdouble(fp," gravity z-dir",switchDebug_readInputFile);
    
/*__________________________________
*  Do some preliminary bullet proofing
*  but save the majority of bullet proofing
*   to testInputFile function
*___________________________________*/
    if( (*xLoLimit - N_GHOSTCELLS) < 0 || (*yLoLimit - N_GHOSTCELLS) < 0 || (*zLoLimit - N_GHOSTCELLS) < 0)
        Message(1, "Input File error:", "The lower array limits are invalid", "(*)LoLimit - N_GHOSTCELLS <0 ");
    if( *xLoLimit < 0 || *xLoLimit > X_MAX_LIM)
        Message(1, "Input File Error:", "x Array limits are invalid", "xLoLimit < 0 || xLoLimit > X_MIN_LIM");      
    if( *xHiLimit < 0 || *xHiLimit > X_MAX_LIM)
        Message(1, "Input File Error:", "x Array limits are invalid", "xHiLimit < 0 || xHiLimit > X_MAX_LIM");
    if( *yLoLimit < 0 || *yLoLimit > Y_MAX_LIM)
        Message(1, "Input File Error:", "y Array limits are invalid", "yLoLimit < 0 || yLoLimit > Y_MIN_LIM");      
    if( *yHiLimit < 0 || *yHiLimit > Y_MAX_LIM)
        Message(1, "Input File Error:", "y Array limits are invalid", "yHiLimit < 0 || yHiLimit > Y_MAX_LIM");
    if( *zLoLimit < 0 || *zLoLimit > Z_MAX_LIM)
        Message(1, "Input File Error:", "z Array limits are invalid", "zLoLimit < 0 || zLoLimit > Z_MIN_LIM");      
    if( *zHiLimit < 0 || *zHiLimit > Z_MAX_LIM)
        Message(1, "Input File Error:", "z Array limits are invalid", "zHiLimit < 0 || zHiLimit > Z_MAX_LIM");
    if( *nMaterials < 0 || *nMaterials > N_MATERIAL)
        Message(1, "Input File Error:", "Number of material spec is invalid", "*nMaterials < 0 || *nMaterials > N_MATERIAL");
        
/*______________________________________________________________________
*   MATERIAL PROPERTIES
*_______________________________________________________________________*/
    for (m = 1; m <= *nMaterials; m++)
    {
    /*__________________________________
    * Initial primitive variables
    *___________________________________*/
       uvel_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]      = readdouble(fp," U velocity",switchDebug_readInputFile);
       vvel_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]      = readdouble(fp," V velocity",switchDebug_readInputFile);
       wvel_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]      = readdouble(fp," W velocity",switchDebug_readInputFile);

       rho_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]       = readdouble(fp," Density",switchDebug_readInputFile);               
       Temp_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]      = readdouble(fp," Temperature",switchDebug_readInputFile);
       Press_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]     = readdouble(fp," Pressure",switchDebug_readInputFile);

       scalar1_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]   = readdouble(fp," Scalar 1",switchDebug_readInputFile);
       scalar2_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]   = readdouble(fp," Scalar 2",switchDebug_readInputFile);
       scalar3_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]   = readdouble(fp," Scalar 3",switchDebug_readInputFile);
    /*__________________________________
    * Material properties
    * Thermodynamic and transport properties
    *___________________________________*/ 
       viscosity_CC[m][*xLoLimit][*yLoLimit][*zLoLimit] = readdouble(fp," Viscosity",switchDebug_readInputFile);
       thermalCond_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]= readdouble(fp," Thermal conductivity",switchDebug_readInputFile);
       cv_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]        = readdouble(fp," Constant Specific heat",switchDebug_readInputFile);
       speedSound[m][*xLoLimit][*yLoLimit][*zLoLimit]   = readdouble(fp," Speed of sound",switchDebug_readInputFile);
       R[m]                                             = readdouble(fp," Ideal Gas constant",switchDebug_readInputFile);  
       gamma[m]                                         = readdouble(fp," Ratio of specific heats",switchDebug_readInputFile);  
    /*__________________________________
    *   Top face boundary conditions
    *___________________________________*/
        BC_inputs[TOP][m]                               = readint(fp," BC_inputs[TOP] ",switchDebug_readInputFile);
        BC_Values[TOP][UVEL][m]                         = readdouble(fp," BC_Values[TOP][UVEL][m]",switchDebug_readInputFile);
        BC_Values[TOP][VVEL][m]                         = readdouble(fp," BC_Values[TOP][VVEL][m]",switchDebug_readInputFile);
        BC_Values[TOP][WVEL][m]                         = readdouble(fp," BC_Values[TOP][WVEL][m]",switchDebug_readInputFile);
        BC_Values[TOP][PRESS][m]                        = readdouble(fp," BC_Values[TOP][PRESS_BC][m]",switchDebug_readInputFile);
        BC_Values[TOP][TEMP][m]                         = readdouble(fp," BC_Values[TOP][TEMP_BC][m]",switchDebug_readInputFile);
        BC_Values[TOP][DENSITY][m]                      = readdouble(fp," BC_Values[TOP][DENSITY][m]",switchDebug_readInputFile);

    /*__________________________________
    * left and right face BC
    *___________________________________*/
        BC_inputs[LEFT][m]                              = readint(fp," BC_inputs[LEFT]",switchDebug_readInputFile);
        BC_inputs[RIGHT][m]                             = readint(fp," BC_inputs[RIGHT]",switchDebug_readInputFile);
        BC_Values[LEFT][UVEL][m]                        = readdouble(fp," BC_Values[LEFT][UVEL][m]",switchDebug_readInputFile);
        BC_Values[RIGHT][UVEL][m]                       = readdouble(fp," BC_Values[RIGHT][UVEL][m]",switchDebug_readInputFile);
        BC_Values[LEFT][VVEL][m]                        = readdouble(fp," BC_Values[LEFT][VVEL][m]",switchDebug_readInputFile);
        BC_Values[RIGHT][VVEL][m]                       = readdouble(fp," BC_Values[RIGHT][VVEL][m]",switchDebug_readInputFile);
        BC_Values[LEFT][WVEL][m]                        = readdouble(fp," BC_Values[LEFT][WVEL][m]",switchDebug_readInputFile);
        BC_Values[RIGHT][WVEL][m]                       = readdouble(fp," BC_Values[RIGHT][WVEL][m]",switchDebug_readInputFile);
        BC_Values[LEFT][PRESS][m]                       = readdouble(fp," BC_Values[LEFT][PRESS][m]",switchDebug_readInputFile);
        BC_Values[RIGHT][PRESS][m]                      = readdouble(fp," BC_Values[RIGHT][PRESS][m]",switchDebug_readInputFile);     
        BC_Values[LEFT][TEMP][m]                        = readdouble(fp," BC_Values[LEFT][TEMP][m]",switchDebug_readInputFile);
        BC_Values[RIGHT][TEMP][m]                       = readdouble(fp," BC_Values[RIGHT][TEMP][m]",switchDebug_readInputFile);
        BC_Values[LEFT][DENSITY][m]                     = readdouble(fp," BC_Values[LEFT][DENSITY][m]",switchDebug_readInputFile);
        BC_Values[RIGHT][DENSITY][m]                    = readdouble(fp," BC_Values[RIGHT][DENSITY][m]",switchDebug_readInputFile);

    /*__________________________________
    *   BOTTOM FACE
    *___________________________________*/
        BC_inputs[BOTTOM][m]                            = readint(fp," BC_inputs[BOTTOM]",switchDebug_readInputFile);
        BC_Values[BOTTOM][UVEL][m]                      = readdouble(fp," BC_Values[BOTTOM][UVEL][m]",switchDebug_readInputFile);
        BC_Values[BOTTOM][VVEL][m]                      = readdouble(fp," BC_Values[BOTTOM][VVEL][m]",switchDebug_readInputFile);
        BC_Values[BOTTOM][WVEL][m]                      = readdouble(fp," BC_Values[BOTTOM][WVEL][m]",switchDebug_readInputFile);
        BC_Values[BOTTOM][PRESS][m]                     = readdouble(fp," BC_Values[BOTTOM][PRESS][m]",switchDebug_readInputFile);
        BC_Values[BOTTOM][TEMP][m]                      = readdouble(fp," BC_Values[BOTTOM][TEMP][m]",switchDebug_readInputFile);
        BC_Values[BOTTOM][DENSITY][m]                   = readdouble(fp," BC_Values[BOTTOM][DENSITY][m]",switchDebug_readInputFile);

    /*__________________________________
    * 3D
    *___________________________________*/
        BC_inputs[FRONT][m]                             = readint(fp," BC_inputs[FRONT][m]",switchDebug_readInputFile);
        BC_inputs[BACK][m]                              = readint(fp," BC_inputs[BACK][m]",switchDebug_readInputFile);

        BC_Values[FRONT][UVEL][m]                       = readdouble(fp," BC_Values[FRONT][UVEL][m]",switchDebug_readInputFile);
        BC_Values[BACK][UVEL][m]                        = readdouble(fp," BC_Values[BACK][UVEL][m]",switchDebug_readInputFile);

        BC_Values[FRONT][VVEL][m]                       = readdouble(fp," BC_Values[FRONT][VVEL][m]",switchDebug_readInputFile);
        BC_Values[BACK][VVEL][m]                        = readdouble(fp," BC_Values[BACK][VVEL][m]",switchDebug_readInputFile);

        BC_Values[FRONT][WVEL][m]                       = readdouble(fp," BC_Values[FRONT][WVEL][m]",switchDebug_readInputFile);
        BC_Values[BACK][WVEL][m]                        = readdouble(fp," BC_Values[BACK][WVEL][m]",switchDebug_readInputFile);

        BC_Values[FRONT][PRESS][m]                      = readdouble(fp," BC_Values[FRONT][PRESS][m]",switchDebug_readInputFile);
        BC_Values[BACK][PRESS][m]                       = readdouble(fp," BC_Values[BACK][PRESS][m]",switchDebug_readInputFile);

        BC_Values[FRONT][TEMP][m]                       = readdouble(fp," BC_Values[FRONT][TEMP][m]",switchDebug_readInputFile);
        BC_Values[BACK][TEMP][m]                        = readdouble(fp," BC_Values[BACK][TEMP][m]",switchDebug_readInputFile);
        
        BC_Values[FRONT][DENSITY][m]                    = readdouble(fp," BC_Values[FRONT][DENSITY][m]",switchDebug_readInputFile);
        BC_Values[BACK][DENSITY][m]                     = readdouble(fp," BC_Values[BACK][DENSITY][m]",switchDebug_readInputFile); 
     }
    fclose(fp);

/*__________________________________
*  Fill the array with the inputs
*___________________________________*/
    for (m = 1; m <= *nMaterials; m++)
    { 
        for(k = GC_LO(*zLoLimit); k <= GC_HI(*zHiLimit); k++)
        {
             for(j = GC_LO(*yLoLimit); j <= GC_HI(*yHiLimit); j++)
             {
                 for(i = GC_LO(*xLoLimit); i <= GC_HI(*xHiLimit); i++)
                 { 

                     counter = counter +1;
                     uvel_CC[m][i][j][k]        = uvel_CC[m][*xLoLimit][*yLoLimit][*zLoLimit];
                     vvel_CC[m][i][j][k]        = vvel_CC[m][*xLoLimit][*yLoLimit][*zLoLimit];
                     wvel_CC[m][i][j][k]        = wvel_CC[m][*xLoLimit][*yLoLimit][*zLoLimit];
                     rho_CC[m][i][j][k]         = rho_CC[m][*xLoLimit][*yLoLimit][*zLoLimit];
                     Temp_CC[m][i][j][k]        = Temp_CC[m][*xLoLimit][*yLoLimit][*zLoLimit];                
                     Press_CC[m][i][j][k]       = Press_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]; 
                     scalar1_CC[m][i][j][k]     = scalar1_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]; 
                     scalar2_CC[m][i][j][k]     = scalar2_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]; 
                     scalar3_CC[m][i][j][k]     = scalar3_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]; 
                     viscosity_CC[m][i][j][k]   = viscosity_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]; 
                     thermalCond_CC[m][i][j][k] = thermalCond_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]; 
                     cv_CC[m][i][j][k]          = cv_CC[m][*xLoLimit][*yLoLimit][*zLoLimit]; 
                     speedSound[m][i][j][k]     = speedSound[m][*xLoLimit][*yLoLimit][*zLoLimit]; 
                 }
             }
         }
    }
    
#if switchDebug_readInputFile                 
        fprintf(stderr,"\n\n Now leaving readInputFile\n");
#endif        
}
/*STOP_DOC*/
     
/*
 ======================================================================*/
 #include <stdio.h>
 #include <math.h>
 #include "parameters.h"
 #include "functionDeclare.h"
 #include "switches.h"
 #include "macros.h"

/*
 Function:  testInputFile--INPUT: Tests the inputs to insure that they are reasonable.
 Filename:  input.c
 
 Purpose:
   Bullet proof the data that is entered in the readInputFile 
   
 Called by:    Main
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    
 ---------------------------------------------------------------------  */      

    void testInputFile( 
         int     xLoLimit,
        int     yLoLimit,
        int     zLoLimit,
        int     xHiLimit,
        int     yHiLimit,
        int     zHiLimit,    
        double  delX,
        double  delY,
        double  delZ,
        double  ****Temp_CC,            /*Cell-centered temperature        */
        double  ****Press_CC,           /* Cell-centered pressure           */
        double  ****rho_CC,             /* Cell-centered density            */
        double  ****viscosity_CC,       /* Cell-centered Viscosity          */
        double  ****thermalCond_CC,     /* Cell-centered thermal conductivity*/
        double  ****cv_CC,              /* Cell-centered constant cp        */
        double  ****speedSound,         /* speed of sound (x, y, z, material*/
        double  t_final,
        double  *t_output_vars,         /* array holding output timing info */
                                        /* t_output_vars[1] = t_initial     */
                                        /* t_output_vars[2] = t final       */
                                        /* t_output_vars[3] = delta t       */
        double  *delt_limits,           /* delt_limits[1] = delt_minimum    */
                                        /* delt_limits[2] = delt_maximum    */
        int     **BC_inputs,            /* array containing the different   */
                                        /* types of boundary conditions     */   
                                        /* at each wall [wall][m]           */
        int     printSwitch,
        double  CFL,
        int     nMaterials     )
{   
   int     m,
            wall,       
            wallLo,     wallHi;
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
      
/*______________________________________________________________________
 Check grid quantites
_______________________________________________________________________*/
    if ((xLoLimit - N_GHOSTCELLS < 0) || 
        (yLoLimit - N_GHOSTCELLS < 0) || 
        (zLoLimit - N_GHOSTCELLS < 0) ) 
        Message(1,"File: input.f","Subroutine: testInputFile",
        "Error: x,y or z LoLimit - N_GHOSTCELLS < 0");

    if ( (xHiLimit < xLoLimit) || (yHiLimit < yLoLimit) 
         || (zHiLimit < zLoLimit))  
        Message(1,"File: input.f","Subroutine: testInputFile",
        "Error: (x,y,z)HiLimit < (x,y,z)LoLimit");

    if ( (delX <= 0.0) || (delY <= 0.0) )  
        Message(1,"File: input.f","Subroutine: testInputFile",
       "Error: delX or delY <= 0.0");
/*__________________________________
*   Test CFL
*___________________________________*/       
    if ( (CFL <= 0.0) || (CFL > 1.0) )  
        Message(1,"File: input.f","Subroutine: testInputFile",
       "Error: (CFL <= 0.0) || (CFL > 1.0)");
/*__________________________________
*   Test time related inputs
*___________________________________*/
    if (t_final<= 0.0)  
        Message(1,"File: if","Subroutine: testInputFile",
       "Error: t_final <= 0.0");
       
    if (t_output_vars[1]< 0.0)  
        Message(1,"File: if","You've specified a negative initial ouput time",
       "Error: t_output_vars[1]< 0.0");
       
    if (t_output_vars[2]< t_output_vars[1])  
        Message(1,"File: if","The final output time is < the initial output time",
       "Error: t_output_vars[2]< t_output_vars[1]");
       
    if (t_output_vars[3]< 0.0)  
        Message(1,"File: if","You've specified a negative delta ouput time",
       "Error: t_output_vars[3]< 0.0");
       
    if (delt_limits[1] <0.0)
        Message(1,"File: if","You've specified a negative delt_minimum",
       "Error: delt_limits[1]< 0.0");
       
    if (delt_limits[2] <0.0)
        Message(1,"File: if","You've specified a negative delt_max",
       "Error: delt_limits[2]< 0.0");
       
    if (delt_limits[2] <SMALL_NUM)
        Message(1,"File: if","You've specified delt_max = 0.0",
       "Error: delt_limits[2] =  0.0");    
       
    if (delt_limits[2] <delt_limits[1])
        Message(1,"File: if","The max. allowable time step is less than the min. allowable time step",
       "Error: delt_limits[2]< delt_limits[1]");  
       
    if (delt_limits[3] <SMALL_NUM)
        Message(1,"File: if","You've specified delt_initial <=0.0",
       "Error: delt_limits[3] <=  0.0");    
       
/*__________________________________
*   Test material properties
*___________________________________*/   
    for( m = 1; m<= nMaterials; m++)
    {
    
        if( rho_CC[m][xLoLimit][yLoLimit][zLoLimit] < 0.0 || rho_CC[m][xLoLimit][yLoLimit][zLoLimit] > BIG_NUM)
            Message(1,"File: if","rho_CC has been set to either < 0 or > BIG_NUM","now Exiting");
            
        if( Temp_CC[m][xLoLimit][yLoLimit][zLoLimit] < 0.0 || Temp_CC[m][xLoLimit][yLoLimit][zLoLimit] > BIG_NUM)
            Message(1,"File: if","Temp_CC has been set to either < 0 or > BIG_NUM","now Exiting");
            
        if( Press_CC[m][xLoLimit][yLoLimit][zLoLimit] < 0.0 || Press_CC[m][xLoLimit][yLoLimit][zLoLimit] > BIG_NUM)
            Message(1,"File: if","Press_CC has been set to either < 0 or > BIG_NUM","now Exiting");
         
        if( speedSound[m][xLoLimit][yLoLimit][zLoLimit] < 0.0 || speedSound[m][xLoLimit][yLoLimit][zLoLimit] > BIG_NUM)
            Message(1,"File: if","Speed of Sound has been set to either < 0 or > BIG_NUM","now Exiting");
        
        if( cv_CC[m][xLoLimit][yLoLimit][zLoLimit] < 0.0 || cv_CC[m][xLoLimit][yLoLimit][zLoLimit] > BIG_NUM)
            Message(1,"File: if","Specific Heat cv has been set to either < 0 or > BIG_NUM","now Exiting");
         
        if( thermalCond_CC[m][xLoLimit][yLoLimit][zLoLimit] < 0.0 || thermalCond_CC[m][xLoLimit][yLoLimit][zLoLimit] > BIG_NUM)
            Message(1,"File: if","Thermal Conductivity has been set to either < 0 or > BIG_NUM","now Exiting"); 
            
            
        if( viscosity_CC[m][xLoLimit][yLoLimit][zLoLimit] < 0.0 || viscosity_CC[m][xLoLimit][yLoLimit][zLoLimit] > BIG_NUM)
            Message(1,"File: if","Viscosity has been set to either < 0 or > BIG_NUM", "now Exiting"); 
    }          
/*__________________________________
*   Test that at least one boundary
*   condition is set on every wall
*___________________________________*/  
    for( m = 1; m<= nMaterials; m++)
    {  
        for( wall = wallLo; wall <= wallHi; wall ++)
        { 
            if (BC_inputs[wall][m] < 1 || BC_inputs[wall][m] > N_DIFFERENT_BCS)
                Message(1,"File: if","One wall doesn't have a valid boundary condition set",
                "now Exiting");               
        }
    }
    
    if(printSwitch == 1)
    { 
        fprintf(stderr,"****************************************************************************\n");
        fprintf(stderr,"                       NOW LEAVING TESTINPUTFILE\n");
        fprintf(stderr,"****************************************************************************\n");
    } 
/*__________________________________
*   Quite all fullwarn compiler remarks
*___________________________________*/
    QUITE_FULLWARN(delZ);                       
}
/*STOP_DOC*/

/* 
 ======================================================================*/
#include "functionDeclare.h"
#include <string.h>
#define     MAXSTR 120              /* maximum length string length     */

/*
 Function:  readdouble--INPUT: Reads in a variable of type (double).
 Filename:  Input.c
 
 Purpose:
   Read a double value from the input file and returns it
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

 Alogrithm:  search for the character ~ and then read in the double  
 
 ---------------------------------------------------------------------  */      

    double readdouble(
            FILE    *fp,
            char    var_name[],  /* Name of the variable to be read     */
            int     printSwitch) /* switch for printing                 */
    
                    
    
{
    int     num;                /* used for error checking              */
            fpos_t  pos;        /* file position pointer                */
    char    c,
                                /* error err if something goes wrong    */
            err[1024]="Error: Read error";
    double  number=-999;        /* number read from input file          */
    
/*______________________________________________________________________
*
*_______________________________________________________________________*/                  
    strcat(err,var_name);
   
    fgetpos(fp,&pos);
                                /* search line until "~" is found       */
    while ( (c = fgetc(fp) ) != '~'){
        if(printSwitch ==1)
            fprintf(stderr,"%c",c);
    }
                               
    num = fscanf(fp,"%lg",&number);
    
/*__________________________________
* Bullet proofing
*___________________________________*/    
    if (num!=1)
        Message(1,"File: input.f","Subroutine: readdouble",err);
        
    if(printSwitch ==1)     
        fprintf(stderr,"%lg ",number);              
 return number;                    
}
/*STOP_DOC*/


/* 
 ======================================================================*/
#include "functionDeclare.h"
#include <string.h>
#define  MAXSTR 120                 /* maximum length string length     */

/*
 Function:  readfloat--INPUT: Reads in a variable of type (float).
 Filename:  Input.c
 
 Purpose:
   Read in a float value from the input file and returns it
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

 Alogrithm:  search for the character ~ and then read in the double  
 
 ---------------------------------------------------------------------  */      

    float readfloat(   
            FILE    *fp,
            char    var_name[], /* Name of the variable to be read     */
            int     printSwitch)/* switch for printing                 */
    
{
    int     num;                /* used for error checking              */
            fpos_t  pos;        /* file position pointer                */
    char    c,
                                /* error err if something goes wrong    */
            err[1024]="Error: Read error";

    float  number;              /* number read from input file          */
           number = -999;
/*______________________________________________________________________
*
*_______________________________________________________________________*/                  

    strcat(err,var_name);
   
    fgetpos(fp,&pos);
                                /* search line until "~" is found       */
    while ( (c = fgetc(fp) ) != '~'){
      if(printSwitch ==1)
            fprintf(stderr,"%c",c);
    }
                               
    if(printSwitch ==1)
        num = fscanf(fp,"%f",&number);
    
/*__________________________________
* Bullet proofing
*___________________________________*/    
    if (num!=1)
        Message(1,"File: input.f","Subroutine: readfloat", err); 
        
    fprintf(stderr,"%f \n",number);             
 return number;                    
}
/*STOP_DOC*/

/* 
 ======================================================================*/
#include "functionDeclare.h"
#include <string.h>
#define  MAXSTR 120                 /* maximum length string length     */

/*

 Function:  readint--INPUT: Reads in a variable of type (int).
 Filename:  Input.c

 Purpose:
   Read in a integer value from the input file and returns it
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    
       
 Alogrithm:  search for the character ~ and then read in the double  
 
 ---------------------------------------------------------------------  */      
 
    int readint( 
            FILE    *fp,
            char    var_name[], /* Name of the variable to be read     */
            int     printSwitch)/* switch for printing                 */                
    
{
    int     num,number;         /* used for error checking              */
            fpos_t  pos;        /* file position pointer                */
    char    c,
                                /* error err if something goes wrong    */
            err[1024]="Error: Read error";        
            
/*______________________________________________________________________
*
*_______________________________________________________________________*/                  
    strcat(err,var_name); 
    
    fgetpos(fp,&pos);
                                 /*  search line until "~" is found     */      
    while ( (c = fgetc(fp) ) != '~'){
    
     if(printSwitch ==1)
        fprintf(stderr,"%c",c);
    } 
                                                 
    num = fscanf(fp,"%d",&number);   
     
/*__________________________________
* Bullet proofing
*___________________________________*/  
   
      if (num!=1)
        Message(1,"File: input.f","Subroutine: readint",err); 
        
      if(printSwitch ==1)    
        fprintf(stderr,"%d ",number);
                        
 return number;                    
}
/*STOP_DOC*/

/* 
 ======================================================================*/
#include "functionDeclare.h"
#include <string.h>
#define  MAXSTR 120                 /* maximum length string length     */

/*

 Function:  readstring--INPUT: Reads in a variable of type (char[]).
 Filename:  Input.c

 Purpose:
   Read in a string from the input file and returns it
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       10/12/99    

 Alogrithm:  search for the character ~ and then read in the string  
 
 ---------------------------------------------------------------------  */      
 
    void readstring( 
            FILE    *fp,
            char    string[],
            char    var_name[], /* Name of the variable to be read     */
            int     printSwitch)/* switch for printing                 */                
    
{
    int     num;                /* used for error checking              */
            fpos_t  pos;        /* file position pointer                */
    char    c,
                                /* error err if something goes wrong    */
            err[1024]="Error: Read error";      
            
/*______________________________________________________________________
*
*_______________________________________________________________________*/                  
    strcat(err,var_name); 
    
    fgetpos(fp,&pos);
                                 /*  search line until "~" is found     */      
    while ( (c = fgetc(fp) ) != '~')
    {
    
     if(printSwitch ==1)
        fprintf(stderr,"%c",c);
    } 
                                                 
    num = fscanf(fp,"%s",string);   
     
/*__________________________________
* Bullet proofing
*___________________________________*/  
   
      if (num!=1)
        Message(1,"File: input.f","Subroutine: readint",err); 
        
      if(printSwitch ==1)    
        fprintf(stderr,"%s",string);                                          
}
/*STOP_DOC*/
