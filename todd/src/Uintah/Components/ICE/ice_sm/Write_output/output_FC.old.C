/* 
 ======================================================================*/
#include <stdio.h>
#include <string.h>
#include "./Header_files/TECIO.h"
#include "./Header_files/parameters.h"
#include "./Header_files/functionDeclare.h"
#include "./Header_files/switches.h"
/*
 Filename: output.c
 Name:     tecplot_FC

 Purpose:
   Write a binary file of all of the Face-centered variables
   that can be read by tecplot
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

 IN args/commons         Units      Description
 ---------------         -----      -----------  
 
 ---------------------------------------------------------------------  */      

    void tecplot_FC(xLoLimit,       yLoLimit,   zLoLimit,
                    xHiLimit,       yHiLimit,   zHiLimit,
                    x_FC,           y_FC,       z_FC,  
                    uvel_FC,        vvel_FC,    wvel_FC,
                    fileNum,        fileDesc,    title )
                    
    int     xLoLimit,                   /* x-array lower limit              */
            yLoLimit,                   /* y-array lower limit              */
            zLoLimit,                   /* z-array lower limit              */
            xHiLimit,                   /* x-array upper limit              */
            yHiLimit,                   /* y-array upper limit              */
            zHiLimit,                   /* z-array upper limit              */
            fileNum;                    /* num to add to the filename       */
                
    double  ****x_FC,                   /* x-coordinate of cell center      */
            ****y_FC,                   /* y-coordinate of cell center      */
            ****z_FC,                   /* z-coordinate of cell center      */ 
            ****uvel_FC,                /* u-cell-centered velocity         */
                                        /* uvel_FC(x,y,z)                   */
            ****vvel_FC,                /*  v-cell-centered velocity        */
                                        /* vvel_FC(x,y,z)                   */
            ****wvel_FC;                /* w cell-centered velocity         */
                                        /* wvel_FC(x,y,z)                   */
                                 
    char    fileDesc[],                 /* Description to put in the filename*/
            title[];                    /* title to put on plots            */   
{
/*__________________________________
*   Local variables
*___________________________________*/    
    char    fileName[] = "", extension[] = ".plt";
    int     i,j,k, xlim, ylim, zlim;
    
    int     Debug, I, III, DIsDouble, VIsDouble;
    
/*__________________________________
* Define local variables
*___________________________________*/  
     Debug       = 1;          /* 1= debug, 0 = no debugging       */
     VIsDouble   = 1;          /* Double precision =1              */
     DIsDouble   = 1;          /* Double precision =1              */
/*______________________________________________________________________
* Main code
* First put together a file name, then open the file
*_______________________________________________________________________*/
    sprintf(fileName,"%03d",fileNum);
    strcat(fileName,fileDesc);
    strcat(fileName,extension);           
 
    I = TECINI(title,
             "X_FC Y_FC Z_FC U_FC V_FC W_FC",
             fileName,
             ".",
             &Debug,
             &VIsDouble);
 /*__________________________________
 * Write the zone header information
 *___________________________________*/
    xlim = xHiLimit * (int)N_CELL_FACES;
    ylim = yHiLimit * (int)N_CELL_FACES;
    zlim = zHiLimit * (int)N_CELL_FACES;
 
    I = TECZNE("Face-Centered Data",
                &xlim,
                &ylim,
                &zlim,
                "BLOCK",
                NULL);

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
    dumpArrayTecplotFC(     xLoLimit,       yLoLimit,   zLoLimit,
                            xHiLimit,       yHiLimit,   zHiLimit,
                            uvel_FC);
                            
    dumpArrayTecplotFC(     xLoLimit,       yLoLimit,   zLoLimit,
                            xHiLimit,       yHiLimit,   zHiLimit,
                            vvel_FC);
                                
    dumpArrayTecplotFC(     xLoLimit,       yLoLimit,   zLoLimit,
                            xHiLimit,       yHiLimit,   zHiLimit,
                            wvel_FC); 
/*__________________________________
* close file
*___________________________________*/
    I = TECEND();  
    
    if(switchDebug_output_FC ==1)
        fprintf(stderr,"\n Exiting tecplotFC\n");

 }
    
    
 /* 
 ======================================================================*/
#include <stdio.h>
#include <string.h>
#include "./Header_files/TECIO.h"
#include "./Header_files/nrutil+.h"
#include "./Header_files/functionDeclare.h"
#include "./Header_files/switches.h"

/*
 Filename: output_FC.c
 Name:     dump_array_tecplot_FC

 Purpose:
   copy a Face-cented array to a dummy array that tecplot can read and
   then dump it to the tecplot file.  
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

 IN args/commons         Units      Description
 ---------------         -----      -----------
 ***data_array          double      3D data array to dumped to tecplot
                                    data_array(i, j, k, face)
                                    face varies from 1 to 6


                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(2)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (5) = back face
                                 (3)              (6) = front face
_______________________________________________________________________ 
 Prerequisites:  The functions TECINI and TECZNE must have been previously
 called.
 ---------------------------------------------------------------------  */      

    void dumpArrayTecplotFC(xLoLimit,       yLoLimit,   zLoLimit,
                            xHiLimit,       yHiLimit,   zHiLimit,
                            data_array )   
    int     xLoLimit,                   /* x-array lower limit              */
            yLoLimit,                   /* y-array lower limit              */
            zLoLimit,                   /* z-array lower limit              */
            xHiLimit,                   /* x-array upper limit              */
            yHiLimit,                   /* y-array upper limit              */
            zHiLimit;                   /* z-array upper limit              */


    double  ****data_array;
{
    int     i,j,k,                      /* Loop variables                   */
            istep,  jstep,  kstep,      /* Step used in loops               */
            xlim,   ylim,   zlim,       /* temp array limits                */
            iface,  jface,  kface,      /* array indices used for dumping   */
                                        /* face values                      */
            I,
            III,                        /* number of values in that array   */
            DIsDouble   = 1,            /* Array is double precision =1     */
            counter     = 0,            /* counts the total number          */
                                        /* of entries written to the x2     */ 
    double  ***x2; 
/*______________________________________________________________________
*  Code
*   Initialize the loop steps and array limits
*_______________________________________________________________________*/
    istep   = 1;
    jstep   = 1;
    kstep   = 1;
    iface   = 0;
    jface   = 0;
    kface   = 0;
    xlim    = xHiLimit - xLoLimit ;         
    zlim    = zHiLimit - zLoLimit ;
    ylim    = yHiLimit - yLoLimit ;
/*__________________________________
* Set Array limits and loop step
*___________________________________*/    
    if (face == 1){                     /* change x face array and step     */
        xlim = 2*(xHiLimit - xLoLimit);
        if(xlim == 0) xlim = 1;
        istep = 2;
     }
    if (face == 2){                     /* change y face array and step     */
        ylim = 2*(yHiLimit - yLoLimit);
        if(ylim == 0) ylim = 1;
        jstep = 2;
     }
    if (face == 3){                    /* change z face array and step      */
        zlim = 2*(zHiLimit - zLoLimit);
        if(zlim == 0) zlim = 1;        
        kstep = 2;
     }
/*__________________________________
*  Allocate memory for the dummy array
*___________________________________*/                
    x2= darray_3d(0, zlim, 0, ylim, 0, xlim ); 

/*__________________________________
* Fill the dummy array
*___________________________________*/    
/*`Testing*/ 
    kface = -1;    
    for(k = 0; k <= zlim; k+=kstep)
    {
        kface ++;
        jface = -1;
        for(j = 0; j <= ylim; j+=jstep)
        {
            jface ++;
            iface = -1;
            for(i = 0; i <= xlim; i+=istep)
            {
                for(f = 1; f <=N_CELL_FACES; f++)
                {
                    counter = counter + 1;
                                        /* x face data                      */
                    if( face == 1) 
                    {
                     iface++;
                        x2[k][j][iface]    = data_array[xLoLimit+iface][yLoLimit+j][zLoLimit+k][1];
                        x2[k][j][iface+1]  = data_array[xLoLimit+index][yLoLimit+j][zLoLimit+k][2];
                    }
                                        /* y face data                      */
                    if( face == 2) 
                    {
                        x2[k][jface][i]    = data_array[xLoLimit+i][yLoLimit+jface][zLoLimit+k][1];
                        x2[k][jface+1][i]  = data_array[xLoLimit+i][yLoLimit+jface][zLoLimit+k][2]; 
                     }
                                        /* z face data                      */
                    if( face == 3) 
                    {    
                        x2[kface][j][i]    = data_array[xLoLimit+i][yLoLimit+j][zLoLimit+kface][1];
                        x2[kface+1][j][i]  = data_array[xLoLimit+i][yLoLimit+j][zLoLimit+kface][2];
                    }
                 }
            }
        }
    }    /*`*/
/*__________________________________
* Print debugging if requested
*___________________________________*/ 


/*__________________________________
* Write out the field data to the open
* tecplot file
*___________________________________*/
    III = counter; 
     I   = TECDAT(&III,&x2[0][0][0],&DIsDouble);
/*__________________________________
* deallocate the memory
*___________________________________*/
    free_darray_3d(x2,0, zlim, 0, ylim, 0, xlim );
    
    if(switchDebug_output_FC ==1)
        fprintf(stderr," \nExiting dumpArrayTecplotFC\n");
    

}
