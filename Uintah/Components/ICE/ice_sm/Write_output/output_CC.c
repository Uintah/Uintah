/* 
 ======================================================================*/
#include <stdio.h>
#include <string.h>
#include "TECIO.h"
#include "functionDeclare.h"
#include "switches.h"
#include "macros.h"
/*
 Function:  tecplot_CC--TECPLOT: controlling function for dumping tecplot files.
 Filename:  output_CC.c
 
 Purpose:
   Write a binary file of all of the cell-centered variables
   that can be read by tecplot
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    
 
 ---------------------------------------------------------------------  */      

    void tecplot_CC(

        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  ***x_CC,                /* x-coordinate of cell center      (INPUT) */
        double  ***y_CC,                /* y-coordinate of cell center      (INPUT) */
        double  ***z_CC,                /* z-coordinate of cell center      (INPUT) */
        double  ****uvel_CC,            /* u-cell-centered velocity         (INPUT) */
        double  ****vvel_CC,            /*  v-cell-centered velocity        (INPUT) */
        double  ****wvel_CC,            /* w cell-centered velocity         (INPUT) */
        double  ****Press_CC,           /* Cell-centered pressure           (INPUT) */
        double  ****Temp_CC,            /* Cell-centered Temperature        (INPUT) */
        double  ****rho_CC,             /* Cell-centered density            (INPUT) */
        double  ****scalar1_CC,         /* Cell-centered scalars            (INPUT) */
        double  ****scalar2_CC,
        double  ****scalar3_CC,
        int     fileNum,                /* num to add to the filename       (INPUT) */
        char    output_file_basename[],
                                        /* Description to put in the filename(INPUT)*/
        char    title[],                /* title to put on plots            (INPUT) */
        int     nMaterials          )   /* number of materials              (INPUT) */
{
/*__________________________________
*   Local variables
*___________________________________*/    
    char    fileName[20], 
            extension[] = ".plt",
            num[5];
    
    int     Debug, 
            I,                                                            
            DIsDouble, 
            VIsDouble,
            xHi, 
            yHi, 
            zHi;
        
/*__________________________________
* Define local variables
*___________________________________*/
     Debug       = 0;
#if (switchDebug_output_CC || switchDebug_output_CC_MM)
     Debug       = 1;                    /* 1= debug, 0 = no debugging       */
#endif
     VIsDouble   = 1;                   /* Double precision =1              */
     DIsDouble   = 1;                   /* Double precision =1              */
/*______________________________________________________________________
* Main code
* First put together a file name, then open the file
*_______________________________________________________________________*/
    /* strcpy(fileName,"./Results/"); */
    sprintf(num,"%03d",fileNum);
    strcat(fileName,num);
    strcat(fileName,output_file_basename);
    strcat(fileName,"_CC");
    strcat(fileName,extension);           
    I = TECINI(title,
             "X Y Z T P RHO U V W",
             fileName,
             ".",
             &Debug,
             &VIsDouble);
 /*__________________________________
 * Write the zone header information
 *___________________________________*/
    xHi = xHiLimit - xLoLimit + 1;
    yHi = yHiLimit - yLoLimit + 1;
    zHi = zHiLimit - zLoLimit + 1; 
          
    I = TECZNE("Cell-Centered Data",
             &xHi,
             &yHi,
             &zHi,
             "BLOCK",
             NULL);

/*__________________________________
* Write out the Cell-center coordinates
*___________________________________*/
    dumpArrayTecplotCC(        xLoLimit,       yLoLimit,   zLoLimit,
                               xHiLimit,       yHiLimit,   zHiLimit,
                               x_CC );
 
    dumpArrayTecplotCC(        xLoLimit,       yLoLimit,   zLoLimit,
                               xHiLimit,       yHiLimit,   zHiLimit,
                               y_CC );
 
    dumpArrayTecplotCC(        xLoLimit,       yLoLimit,   zLoLimit,
                               xHiLimit,       yHiLimit,   zHiLimit,
                               z_CC );
/*__________________________________
* Write out the Cell-center temperature
* pressure and density
*___________________________________*/
    dumpArrayTecplotCC_MM(     xLoLimit,        yLoLimit,   zLoLimit,
                               xHiLimit,        yHiLimit,   zHiLimit,
                               Temp_CC,          nMaterials);
                            
    dumpArrayTecplotCC_MM(     xLoLimit,        yLoLimit,   zLoLimit,
                               xHiLimit,        yHiLimit,   zHiLimit,
                               Press_CC ,       nMaterials);
                                
    dumpArrayTecplotCC_MM(     xLoLimit,        yLoLimit,   zLoLimit,
                               xHiLimit,        yHiLimit,   zHiLimit,
                               rho_CC,          nMaterials );
/*__________________________________
* Write out the Cell-center velocities
*___________________________________*/
    dumpArrayTecplotCC_MM(     xLoLimit,        yLoLimit,   zLoLimit,
                               xHiLimit,        yHiLimit,   zHiLimit,
                               uvel_CC,         nMaterials );
                            
    dumpArrayTecplotCC_MM(     xLoLimit,        yLoLimit,   zLoLimit,
                               xHiLimit,        yHiLimit,   zHiLimit,
                               vvel_CC,         nMaterials );
                                
    dumpArrayTecplotCC_MM(     xLoLimit,        yLoLimit,   zLoLimit,
                               xHiLimit,        yHiLimit,   zHiLimit,
                               wvel_CC,         nMaterials ); 
/*__________________________________
* close file
*___________________________________*/
    I = TECEND();
    if (I == -1)
        Message(1,"ERROR: tecplot_CC()","There was a problem detected while writing out ",
       "the tecplot files");
         
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                       Wrote Tecplot file %s\n",fileName);
    fprintf(stderr,"****************************************************************************\n");
    
#if switchDebug_output_CC
        fprintf(stderr,"\n Exiting tecplotCC\n");
#endif

/*__________________________________
*   Quite fullwarn remarks
*___________________________________*/
    QUITE_FULLWARN(scalar1_CC[1][0][0][0]);     QUITE_FULLWARN(scalar2_CC[1][0][0][0]);
    QUITE_FULLWARN(scalar3_CC[1][0][0][0]);
    DIsDouble = DIsDouble;                      I = I;
 }
    
/*STOP_DOC*/    
 /* 
 ======================================================================*/
#include <string.h>
#include "TECIO.h"
#include "nrutil+.h"
#include "functionDeclare.h"
#include "switches.h"

/*
 Function:  dumpArrayTecplotCC--TECPLOT: Write a cell-centered,single material, array to a tecplot file.
 Filename:  output_CC.c
 Purpose:
   copy a cell-centered array to a dummy array that tecplot can read and
   then dump it to the tecplot file.  Note that this only works for single
   fluid arrays, NOT, multifluid.  
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    
 Prerequisites:  The functions TECINI and TECZNE must have been previously
 called.
 ---------------------------------------------------------------------  */      

    void dumpArrayTecplotCC( 
            int     xLoLimit,           /* x-array lower limit              */
            int     yLoLimit,           /* y-array lower limit              */
            int     zLoLimit,           /* z-array lower limit              */
            int     xHiLimit,           /* x-array upper limit              */
            int     yHiLimit,           /* y-array upper limit              */
            int     zHiLimit,           /* z-array upper limit              */
 
            double  ***data_array )     /* array to dump to file            (INPUT) */
{
    int     i,j,k,
            xlim,ylim,zlim,          /* temp array limits                   */
            I,
            III,                     /* number of values in that array      */
            DIsDouble   = 1,         /* Array is double precision =1        */
            counter     = 0;
    double  ***x2; 
/*______________________________________________________________________
*  Allocate memory for the dummy array and write the data to it 
*_______________________________________________________________________*/  
    zlim = zHiLimit - zLoLimit ;
    ylim = yHiLimit - yLoLimit ;
    xlim = xHiLimit - xLoLimit;                 
    x2= darray_3d(0, zlim, 0, ylim, 0, xlim ); 
    
    for(k = 0; k <= zlim; k++)
    {
        for(j = 0; j <= ylim; j++)
        {
            for(i = 0; i <= xlim; i++)
            {
                counter = counter + 1;
                 x2[k][j][i]= data_array[xLoLimit+i][yLoLimit+j][zLoLimit+k];
            }
        }
    }   
/*__________________________________
* Write out the field data to the open
* tecplot file
*___________________________________*/
    III = counter; 
     I  = TECDAT(&III,&x2[0][0][0],&DIsDouble);
/*__________________________________
* deallocate the memory
*___________________________________*/
    free_darray_3d(x2,0, zlim, 0, ylim, 0, xlim );
/*______________________________________________________________________
*   DEBUGGING
*_______________________________________________________________________*/    
#if switchDebug_output_CC  
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        DUMPARRAYTECPLOTCC\n");
    fprintf(stderr,"****************************************************************************\n");
           
     fprintf(stderr,"\n");
     for(k = 0; k <= zlim; k++)
     {
            for(j = 0; j <= ylim; j++)
            {
                for(i = 0; i <= xlim; i++)
                { 
                 fprintf(stderr,"[%d][%d][%d]=%lg  ",k,j,i,x2[k][j][i]); 
                }
                fprintf(stderr,"\n");
            }
            fprintf(stderr,"\n");
        }
#endif
/*__________________________________
*   Quite fullwarn remarks is a way that
*   is compiler independent
*___________________________________*/
    I = I;    

}
/*STOP_DOC*/
 /* 
 ======================================================================*/
#include <stdio.h>
#include <string.h>
#include "TECIO.h"
#include "nrutil+.h"
#include "functionDeclare.h"
#include "switches.h"

/*
 Function:  dumpArrayTecplotCC_MM--TECPLOT: write a cell-centered, multimaterial array to a tecplot files.
 Filename:  output_CC.c
 Purpose:
   copy a cell-centered array to a dummy array that tecplot can read and
   then dump it to the tecplot file.  Note that this works for multiple fluids.  
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

 Prerequisites:  The functions TECINI and TECZNE must have been previously
 called.
 ---------------------------------------------------------------------  */      

    void dumpArrayTecplotCC_MM( 
            int     xLoLimit,           /* x-array lower limit              */
            int     yLoLimit,           /* y-array lower limit              */
            int     zLoLimit,           /* z-array lower limit              */
            int     xHiLimit,           /* x-array upper limit              */
            int     yHiLimit,           /* y-array upper limit              */
            int     zHiLimit,           /* z-array upper limit              */
            double  ****data_array,     /* array to dump to file            (INPUT) */
            int     nMaterials)         /* number of materials              */     
{
    int     i,j,k,m,                
            xlim,ylim,zlim,          /* temp array limits                   */
            I,
            III,                     /* number of values in that array      */
            DIsDouble   = 1,         /* Array is double precision =1        */
            counter     = 0;
    double  ***x2; 
/*______________________________________________________________________
*  Allocate memory for the dummy array and write the data to it 
*_______________________________________________________________________*/  
    zlim = zHiLimit - zLoLimit ;
    ylim = yHiLimit - yLoLimit ;
    xlim = xHiLimit - xLoLimit;                 
    x2= darray_3d(0, zlim, 0, ylim, 0, xlim ); 
    
    for(m = 1; m <= nMaterials; m++)
    {
        for(k = 0; k <= zlim; k++)
        {
            for(j = 0; j <= ylim; j++)
            {
                for(i = 0; i <= xlim; i++)
                {
                    counter = counter + 1;
                     x2[k][j][i]= data_array[m][xLoLimit+i][yLoLimit+j][zLoLimit+k];
                }
            }
        }
    }

/*__________________________________
* Write out the field data to the open
* tecplot file
*___________________________________*/
    III  = counter; 
     I   = TECDAT(&III,&x2[0][0][0],&DIsDouble);
/*__________________________________
* deallocate the memory
*___________________________________*/
    free_darray_3d(x2,0, zlim, 0, ylim, 0, xlim );
/*______________________________________________________________________
*   DEBUGGING
*_______________________________________________________________________*/    
#if(switchDebug_output_CC_MM )
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        DUMPARRAYTECPLOTCC_MM\n");
    fprintf(stderr,"****************************************************************************\n");
              
    fprintf(stderr,"\n");
     for(k = 0; k <= zlim; k++)
     {
         for(j = 0; j <= ylim; j++)
         {
             for(i = 0; i <= xlim; i++)
             {
              fprintf(stderr,"[%d][%d][%d]=%lg  ",k,j,i,x2[k][j][i]);
             }
             fprintf(stderr,"\n");
         }
         fprintf(stderr,"\n");
    }
#endif
/*__________________________________
*   Quite fullwarn remarks is a way that
*   is compiler independent
*___________________________________*/
    I = I;       
}
/*STOP_DOC*/
