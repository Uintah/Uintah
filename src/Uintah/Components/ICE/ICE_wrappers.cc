#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Components/ICE/ICE.h>
using Uintah::ICESpace::ICE;
/* ---------------------------------------------------------------------
                                       
GENERAL INFORMATION
 Function:  ICE::before_each_step_wrapper--
 Filename:  ICE_wrappers.cc
 Computes:  This wrapper converts all datawarehouse data structure for the
            press_CC, Temp_CC, rho_CC and vel_CC into
            Numerical Recipes in C 4 dimensional arrays.  
            
History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     Todd Harman       06/23/00                              
_____________________________________________________________________*/
void ICE::before_each_step_wrapper(
    const Patch* patch,
    CCVariable<double>& press_cc,
    double              ****press_CC,
    CCVariable<double>& rho_cc,
    double              ****rho_CC,
    CCVariable<double>& temp_cc,
    double              ****Temp_CC,
    CCVariable<Vector>& vel_cc,
    double              ****uvel_CC,
    double              ****vvel_CC,
    double              ****wvel_CC)
{   
    bool    include_ghost_cells = true;
    /*__________________________________
    *   UCF NR
    *___________________________________*/

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
            vel_cc,     uvel_CC,    vvel_CC,    wvel_CC,
            include_ghost_cells,
            xLoLimit,   xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
            nMaterials);  

}



/* ---------------------------------------------------------------------
                                       
GENERAL INFORMATION
 Function:  ICE::before_each_step_wrapper--
 Filename:  ICE_wrappers.cc
 Computes:  This wrapper converts all Numerical Recipes in C arrays,
            press_CC, Temp_CC, rho_CC and vel_CC into
            UCF data structures.  It then zeros all the NR's arrays.  
            
History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     Todd Harman       06/23/00                              
_____________________________________________________________________*/
void ICE::after_each_step_wrapper(
    const Patch* patch,
    CCVariable<double>& press_cc,
    double              ****press_CC,
    CCVariable<double>& rho_cc,
    double              ****rho_CC,
    CCVariable<double>& temp_cc,
    double              ****Temp_CC,
    CCVariable<Vector>& vel_cc,
    double              ****uvel_CC,
    double              ****vvel_CC,
    double              ****wvel_CC)
{   
    bool    include_ghost_cells = true;
    /*__________________________________
    *   UCF NR
    *___________________________________*/
    ICE::convertNR_4dToUCF(patch,
        vel_cc,     uvel_CC,    vvel_CC,    wvel_CC,
        include_ghost_cells,
        xLoLimit,       xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
	 nMaterials);
  
    ICE::convertNR_4dToUCF(patch,
        temp_cc,               Temp_CC,
        include_ghost_cells,
        xLoLimit,       xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
	 nMaterials);
        
    ICE::convertNR_4dToUCF(patch,
        rho_cc,                 rho_CC,
        include_ghost_cells,
        xLoLimit,       xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
	 nMaterials);
        
    ICE::convertNR_4dToUCF(patch,
        press_cc,               press_CC,
        include_ghost_cells,
        xLoLimit,       xHiLimit,   yLoLimit,   yHiLimit,   zLoLimit,   zHiLimit,
	 nMaterials); 
        
#if 0        
   /*__________________________________
   *    ZERO OUT ALL THE NR DATA
   *___________________________________*/ 
   for ( int m = 1; m <= nMaterials; m++)
   {
        for ( int k = (zLoLimit); k <= (zHiLimit); k++)
        {
            for ( int j = (yLoLimit); j <= (yHiLimit); j++)
            {
                for ( int i = (xLoLimit); i <= (xHiLimit); i++)
                { 
                    Temp_CC[m][i][j][k]    = 0.0;
                    press_CC[m][i][j][k]   = 0.0;
                    rho_CC[m][i][j][k]     = 0.0;
                    uvel_CC[m][i][j][k]    = 0.0;
                    vvel_CC[m][i][j][k]    = 0.0;
                    wvel_CC[m][i][j][k]    = 0.0;
                }
            }
        }
    } 
#endif
}
