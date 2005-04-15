/* 
 ======================================================================*/
#include <stdio.h>
#include <string.h>
#include "TECIO.h"
#include "macros.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "switches.h"
/*
 Function:  tecplot_FC--TECPLOT: Main controlling code for write face-centered variables to a tecplot file. 
 Filename:  output_FC.c
 
 Purpose:
   Write a binary file of all of the Face-centered variables
   that can be read by tecplot
   
 References: see the tecplot manual for details on the dump routines
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

POTENTIAL IMPROVEMENT:
 THIS IS STUPID IN THAT IT DUMPS ALL OF THE FACES.
IT SHOULD BE TWEAKED SO THAT ONLY THE RIGHT AND LEFT FACES ARE DUMPED
IN 1-D, LEFT, RIGHT,TOP AND BOTTOM FACES FOR 2D AND ALL FACES FOR 3D.

 ---------------------------------------------------------------------  */      

    void tecplot_FC(
            int     xLoLimit,           /* x-array lower limit              */
            int     yLoLimit,           /* y-array lower limit              */
            int     zLoLimit,           /* z-array lower limit              */
            int     xHiLimit,           /* x-array upper limit              */
            int     yHiLimit,           /* y-array upper limit              */
            int     zHiLimit,           /* z-array upper limit              */
                                        /*-----------pointers---------------*/
            double  *****x_FC,          /* x-coordinate of cell center      */
            double  *****y_FC,          /* y-coordinate of cell center      */
            double  *****z_FC,          /* z-coordinate of cell center      */
                                        /* (x,y,z,face,material)            */
                                        /*----------------------------------*/
            double  ******uvel_FC,      /* u-cell-centered velocity (ptr)   */
            double  ******vvel_FC,      /*  v-cell-centered velocity(ptr)   */
            double  ******wvel_FC,      /* w cell-centered velocity (ptr)   */
            int     fileNum,            /* num to add to the filename       */
            char    output_file_basename[],         /* Description to put in the filename*/
            char    title[],            /* title to put on plots            */
            int     nMaterials      )   /* number of materials              */
{
/*__________________________________
*   Local variables
*___________________________________*/    
    char    fileName[20], 
            extension[] = ".plt",
            num[5];
    
    int     xHi, yHi, zHi,
            n_cell_faces;
    int     Debug, 
            I,
            DIsDouble, 
            VIsDouble;
    
/*__________________________________
* Define local variables
*___________________________________*/
    Debug        = 0;
#if( switchDebug_output_FC || switch_output_FC_MM)  
     Debug       = 1;          /* 1= debug, 0 = no debugging       */
#endif
     VIsDouble   = 1;          /* Double precision =1              */
     DIsDouble   = 1;          /* Double precision =1              */
     

/*______________________________________________________________________
* Main code
* First put together a file name, then open the file
*_______________________________________________________________________*/
    sprintf(num,"%03d",fileNum);
    strcat(fileName,num);
    strcat(fileName,output_file_basename);
    strcat(fileName,"_FC");
    strcat(fileName,extension);           
            /*  "X_FC Y_FC Z_FC U_FC V_FC W_FC", */ 
    I = TECINI(title,
             "X_f_c Y_f_c Z_f_c U_f_c V_f_c W_f_c",
             fileName,
             ".",
             &Debug,
             &VIsDouble);
 /*__________________________________
 * Write the zone header information
 *___________________________________*/
        xHi = (xHiLimit - xLoLimit + 1);
        yHi = (yHiLimit - yLoLimit + 1);
        zHi = (zHiLimit - zLoLimit + 1);     
#if (N_DIMENSIONS == 1)  
       n_cell_faces = 2;
       xHi = (xHiLimit - xLoLimit + 1) * 2;
#endif

#if (N_DIMENSIONS == 2) 
        n_cell_faces = 4;
        xHi = (xHiLimit - xLoLimit + 1) * 2;
        yHi = (yHiLimit - yLoLimit + 1) * 2;
#endif
#if (N_DIMENSIONS == 3) 
        n_cell_faces = 6;
        xHi = (xHiLimit - xLoLimit + 1) * 2;
        yHi = (yHiLimit - yLoLimit + 1) * 2;
        zHi = (zHiLimit - zLoLimit + 1) * 2; 
#endif    
    
    
    
    I = TECZNE("Face-Centered Data",
               &xHi,          
               &yHi,      
               &zHi,
               "BLOCK",        NULL);

/*__________________________________
* Write out the Face-center coordinates
*___________________________________*/
    dumpArrayTecplotFC(     xLoLimit,       yLoLimit,   zLoLimit,
                            xHiLimit,       yHiLimit,   zHiLimit,
                            x_FC );
                            
    dumpArrayTecplotFC(     xLoLimit,       yLoLimit,   zLoLimit,
                            xHiLimit,       yHiLimit,   zHiLimit,
                            y_FC);
                                
    dumpArrayTecplotFC(     xLoLimit,       yLoLimit,   zLoLimit,
                            xHiLimit,       yHiLimit,   zHiLimit,
                            z_FC); 

/*__________________________________
* Write out the Face-center velocities
*___________________________________*/
     dumpArrayTecplotFC_MM( xLoLimit,       yLoLimit,   zLoLimit,
                            xHiLimit,       yHiLimit,   zHiLimit,
                            uvel_FC,        nMaterials);
                            
    dumpArrayTecplotFC_MM(  xLoLimit,       yLoLimit,   zLoLimit,
                            xHiLimit,       yHiLimit,   zHiLimit,
                            vvel_FC,        nMaterials);
                                
    dumpArrayTecplotFC_MM(  xLoLimit,       yLoLimit,   zLoLimit,
                            xHiLimit,       yHiLimit,   zHiLimit,
                            wvel_FC,        nMaterials);  
/*__________________________________
* close file
*___________________________________*/
    I = TECEND();  
    if (I == -1)
        Message(1,"ERROR: tecplot_FC()","There was a problem detected while writing out ",
       "the tecplot files, Now exiting");
           
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                       Wrote Tecplot file %s\n",fileName);
    fprintf(stderr,"****************************************************************************\n");
    
#if switchDebug_output_FC
        fprintf(stderr,"\n Exiting tecplotFC\n");
#endif

/*__________________________________
*   Quite all fullwarn compiler remarks
*___________________________________*/
    QUITE_FULLWARN(I);                  QUITE_FULLWARN(DIsDouble);
    QUITE_FULLWARN(n_cell_faces);

 }
/*STOP_DOC*/    
    
 /* 
 ======================================================================*/
#include <stdio.h>
#include <string.h>
#include "TECIO.h"
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"

/*
 Function:  dumpArrayTecplotFC--TECPLOT: Write a single material, face-centered array to a tecplot file.
 Filename:  output_FC.c
 
 Purpose:
   copy a Face-centered array to a dummy array that tecplot can read and
   then dump it to the tecplot file, single fluid.  
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(3)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (6) = back face
                                 (2)              (5) = front face
_______________________________________________________________________ 
 Prerequisites:  The functions TECINI and TECZNE MUST have been previously
 called.
 ---------------------------------------------------------------------  */      

    void dumpArrayTecplotFC(
            int     xLoLimit,           /* x-array lower limit              */
            int     yLoLimit,           /* y-array lower limit              */
            int     zLoLimit,           /* z-array lower limit              */
            int     xHiLimit,           /* x-array upper limit              */
            int     yHiLimit,           /* y-array upper limit              */
            int     zHiLimit,           /* z-array upper limit              */
            double  *****data_array  )
{
    int     i,j,k,f,                     /* Loop variables                   */
            lim,
            faceLo,                     /* face looping indices             */ 
            faceHi,
            n_cell_faces,
                                                                            /*REFERENCED*/
            I,
            III,                        /* number of values in that array   */
            DIsDouble   = 1,            /* Array is double precision =1     */
            counter     = 0;            /* counts the total number          */
                                        /* of entries written to the x2     */ 
    double  *x2; 
/*______________________________________________________________________
*  Code
*   Initialize the loop steps and array limits
*_______________________________________________________________________*/
#if (N_DIMENSIONS == 1)  
        faceLo = LEFT;  faceHi = RIGHT;
        n_cell_faces = 2;
#endif
#if (N_DIMENSIONS == 2) 
        faceLo = TOP;   faceHi = LEFT;
        n_cell_faces = 4;
#endif
#if (N_DIMENSIONS == 3) 
        faceLo = TOP;   faceHi = BACK;
        n_cell_faces = 6;
#endif

    lim     =   (xHiLimit - xLoLimit +1) * 
                (yHiLimit - yLoLimit +1) * 
                (zHiLimit - zLoLimit +1) * n_cell_faces;
/*__________________________________
*  Allocate memory for the dummy array
*___________________________________*/                
    x2= dvector_nr(0, lim ); 

/*__________________________________
* Fill the dummy array
*___________________________________*/    
  
    for(k = zLoLimit; k <= zHiLimit ; k++)
    {
        for(j = yLoLimit; j <= yHiLimit; j++)
        {
            for(i = xLoLimit; i <= xHiLimit; i++)
            {
                for(f = faceLo; f <=faceHi; f++)
                {
                    x2[counter]= *data_array[i][j][k][f];
                    
                 }
            }
        }
    }    
/*__________________________________
* Print debugging if requested
*___________________________________*/ 
#if switchDebug_output_FC 
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        DUMPARRAYTECPLOTFC\n");
    fprintf(stderr,"****************************************************************************\n");

 /*    printData_FC    (       xLoLimit,       yLoLimit,       zLoLimit,
                            xHiLimit,       yHiLimit,       zHiLimit,
                           "dumpArrayTecplotFC,  "",        data_array); */
#endif

/*__________________________________
* Write out the field data to the open
* tecplot file
*___________________________________*/
    III = counter; 
     I   = TECDAT(&III,&x2[1],&DIsDouble);
/*__________________________________
* deallocate the memory
*___________________________________*/
    free_dvector_nr(x2, 0, lim );
    
/*__________________________________
*   Quite all fullwarn compiler remarks
*___________________________________*/
    QUITE_FULLWARN(I);

    
}
/*STOP_DOC*/
 /* 
 ======================================================================*/
#include <stdio.h>
#include <string.h>
#include "TECIO.h"
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"

/*
 Function:  dumpArrayTecplotFC_MM--TECPLOT: Write a multimaterial, face-centered array to a tecplot file.
 Filename:  output_FC.c
 
 Purpose:
   copy a Face-cented array to a dummy array that tecplot can read and
   then dump it to the tecplot file, multiple materials  
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(3)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (6) = back face
                                 (2)              (5) = front face
_______________________________________________________________________ 
 Prerequisites:  The functions TECINI and TECZNE MUST have been previously
 called.
 ---------------------------------------------------------------------  */      

    void dumpArrayTecplotFC_MM( 
            int     xLoLimit,           /* x-array lower limit              */
            int     yLoLimit,           /* y-array lower limit              */
            int     zLoLimit,           /* z-array lower limit              */
            int     xHiLimit,           /* x-array upper limit              */
            int     yHiLimit,           /* y-array upper limit              */
            int     zHiLimit,           /* z-array upper limit              */
            double  ******data_array,
            int     nMaterials     )    /* number of materials              */

{
    int     i,j,k,f,m,                  /* Loop variables                   */
            lim,
            faceLo,                     /* face looping indices             */ 
            faceHi,
            n_cell_faces,                                                   
            I,
            III,                        /* number of values in that array   */
            DIsDouble   = 1,            /* Array is double precision =1     */
            counter     = 0;            /* counts the total number          */
                                        /* of entries written to the x2     */ 
    double  *x2; 
/*______________________________________________________________________
*  Code
*   Initialize the loop steps and array limits
*_______________________________________________________________________*/
#if (N_DIMENSIONS == 1)  
        faceLo = LEFT;  faceHi = RIGHT;
        n_cell_faces = 2;
#endif
#if (N_DIMENSIONS == 2) 
        faceLo = TOP;   faceHi = LEFT;
        n_cell_faces = 4;
#endif
#if (N_DIMENSIONS == 3) 
        faceLo = TOP;   faceHi = BACK;
        n_cell_faces = 6;
#endif

    lim     =   (xHiLimit - xLoLimit +1) * 
                (yHiLimit - yLoLimit +1) * 
                (zHiLimit - zLoLimit +1) * n_cell_faces;
/*__________________________________
*  Allocate memory for the dummy array
*___________________________________*/                
    x2= dvector_nr(0, lim ); 

/*__________________________________
* Fill the dummy array
*___________________________________*/    
    for( m = 1; m <= nMaterials; m++)
    { 
        for(k = zLoLimit; k <= zHiLimit ; k++)
        {
            for(j = yLoLimit; j <= yHiLimit; j++)
            {
                for(i = xLoLimit; i <= xHiLimit; i++)
                {
                    for(f = faceLo; f <=faceHi; f++)
                    {
                         x2[counter]= *data_array[i][j][k][f][m];
                     }
                }
            }
        }
    }
/*__________________________________
* Print debugging if requested
*___________________________________*/ 
#if switchDebug_output_FC_MM
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        DUMPARRAYTECPLOT_FC_MM\n");
    fprintf(stderr,"****************************************************************************\n");

    printData_FC_MF(   xLoLimit,       yLoLimit,       zLoLimit,
                       xHiLimit,       yHiLimit,       zHiLimit,
                       "dumpArrayTecplot_FC_MM",     "", data_array, m);
#endif 

/*__________________________________
* Write out the field data to the open
* tecplot file
*___________________________________*/
    III = counter; 
     I   = TECDAT(&III,&x2[1],&DIsDouble);
/*__________________________________
* deallocate the memory
*___________________________________*/
    free_dvector_nr(x2, 0, lim );
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    I = I;
    
}
/*STOP_DOC*/
