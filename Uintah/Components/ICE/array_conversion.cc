
#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/CellIterator.h>
#include <Uintah/Interface/Scheduler.h>
#include "ICE.h"

#include "macros.h"
#include "parameters.h"
using Uintah::ICESpace::ICE;

/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  ICE::convertNR_4dToUCF
 Filename:  array_conversion.cc
 Purpose:   Convert Numerical recipes 4d arrays into the UCF 

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/ 
void ICE::convertNR_4dToUCF(const Patch* patch,CCVariable<Vector>& vel_ucf, 
        double  ****uvel_CC,
        double  ****vvel_CC,
        double  ****wvel_CC,
        int     include_ghostcells,
        int     xLoLimit,
        int     xHiLimit,
        int     yLoLimit,
        int     yHiLimit,
        int     zLoLimit,
        int     zHiLimit,
        int     nMaterials)
{
    int i, j, k, m,
        xLo, yLo, zLo,
        xHi, yHi, zHi;
    /*__________________________________
    *   Looping Limits
    *___________________________________*/
    if (include_ghostcells == YES)
    {
        xLo = GC_LO(xLoLimit);  yLo = GC_LO(yLoLimit);  zLo = GC_LO(zLoLimit);
        xHi = GC_HI(xHiLimit);  yHi = GC_HI(yHiLimit);  zHi = GC_HI(zHiLimit);
    } else
    {
        xLo = xLoLimit;         yLo = yLoLimit;         zLo = zLoLimit;
        xHi = xHiLimit;         yHi = yHiLimit;         zHi = zHiLimit;
    }

    for (CellIterator iter = patch->getCellIterator(patch->getBox()); 
         !iter.done(); iter++) 
    {
      // Do something
    }

    CellIterator iter = patch->getCellIterator(patch->getBox());
    cerr << "CC iterator begin = " << iter.begin() << " end = " << iter.end() 
         << endl;

    cerr << "CC variables limits " << vel_ucf.getLowIndex() << " " 
         << vel_ucf.getHighIndex() << endl;

    cerr << "NR s: [" << xLo << " " << yLo << " " << zLo << "] [ " 
         << xHi << " " << yHi << " " << zHi << "]" << endl;  

    for (i = xLo; i <= xHi; i++) 
    {
        for (j = yLo; j <= yHi; j++) 
        {
            for (k = zLo; k <= zHi; k++) 
            {
	        for (m = 1; m <= nMaterials; m++) 
               {
	          // Do something
	          //  cerr << "uvel = " << uvel_CC[m][i][j][k] 
	          //     << " vvel = " << vvel_CC[m][i][j][k] 
	          //     << " wvel = " << wvel_CC[m][i][j][k] << endl;
	          IntVector idx(i-xLo,j-yLo,k-zLo);
	          vel_ucf[idx]=Vector(uvel_CC[m][i][j][k], vvel_CC[m][i][j][k], 
			              wvel_CC[m][i][j][k]);
	          //cerr << "vel_ucf = " << vel_ucf[idx] << endl;
	        }
            }
        }
    }

  return;

}

/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  ICE::convertNR_4dToUCF
 Filename:  array_conversion.cc
 Purpose:   Convert Numerical recipes 4d arrays into the UCF 
            

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/ 
void ICE::convertNR_4dToUCF(const Patch* patch,CCVariable<double>& scalar_ucf, 
        double  ****scalar_CC,
        int     include_ghostcells,
        int     xLoLimit,
        int     xHiLimit,
        int     yLoLimit,
        int     yHiLimit,
        int     zLoLimit,
        int     zHiLimit,
        int     nMaterials)
{
    int i, j, k, m,
        xLo, yLo, zLo,
        xHi, yHi, zHi;
    /*__________________________________
    *   Looping Limits
    *___________________________________*/
    if (include_ghostcells == YES)
    {
        xLo = GC_LO(xLoLimit);  yLo = GC_LO(yLoLimit);  zLo = GC_LO(zLoLimit);
        xHi = GC_HI(xHiLimit);  yHi = GC_HI(yHiLimit);  zHi = GC_HI(zHiLimit);
    } else
    {
        xLo = xLoLimit;         yLo = yLoLimit;         zLo = zLoLimit;
        xHi = xHiLimit;         yHi = yHiLimit;         zHi = zHiLimit;
    }
  
    for (CellIterator iter = patch->getCellIterator(patch->getBox()); !iter.done(); iter++) 
    {
    }

    CellIterator iter = patch->getCellIterator(patch->getBox());
    cerr << "CC iterator begin = " << iter.begin() << " end = " << iter.end() << endl;

    cerr << "CC variables limits " << scalar_ucf.getLowIndex() << " " 
         << scalar_ucf.getHighIndex() << endl;

    cerr << "NR s: [" << xLo << " " << yLo << " " << zLo << "] [ " 
         << xHi << " " << yHi << " " << zHi << "]" << endl;  

    for (i = xLo; i <= xHi; i++) 
    {
        for (j = yLo; j <= yHi; j++) 
        {
            for (k = zLo; k <= zHi; k++) 
            {
               for (m = 1; m <= nMaterials; m++) 
               {
	          // Do something
	          //  cerr << "scalar = " << scalar_CC[m][i][j][k] << endl;
	          IntVector idx(i-xLo,j-yLo,k-zLo);
	          scalar_ucf[idx]=scalar_CC[m][i][j][k];
	          //cerr << "scalar_ucf = " << scalar_ucf[idx] << endl;
               }
            }
        }
    }
    return;
}

/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  ICE::convertUCFToNR_4d
 Filename:  array_conversion.cc
 Purpose:   Convert Numerical recipes 4d arrays into the UCF 
            for vector data

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/ 
void ICE::convertUCFToNR_4d(const Patch* patch,CCVariable<Vector>& vel_ucf, 
        double  ****uvel_CC,
        double  ****vvel_CC,
        double  ****wvel_CC,
        int     include_ghostcells,
        int     xLoLimit,
        int     xHiLimit,
        int     yLoLimit,
        int     yHiLimit,
        int     zLoLimit,
        int     zHiLimit,
        int     nMaterials)
{
    int i, j, k, m,
        xLo, yLo, zLo,
        xHi, yHi, zHi;
    /*__________________________________
    *   Looping Limits
    *___________________________________*/
    if (include_ghostcells == YES)
    {
        xLo = GC_LO(xLoLimit);  yLo = GC_LO(yLoLimit);  zLo = GC_LO(zLoLimit);
        xHi = GC_HI(xHiLimit);  yHi = GC_HI(yHiLimit);  zHi = GC_HI(zHiLimit);
    } else
    {
        xLo = xLoLimit;         yLo = yLoLimit;         zLo = zLoLimit;
        xHi = xHiLimit;         yHi = yHiLimit;         zHi = zHiLimit;
    }
  
    for (CellIterator iter = patch->getCellIterator(patch->getBox()); !iter.done(); iter++) 
    {
    }

    CellIterator iter = patch->getCellIterator(patch->getBox());
    cerr << "CC iterator begin = " << iter.begin() << " end = " << iter.end() 
         << endl;

    cerr << "CC variables limits " << vel_ucf.getLowIndex() << " " 
         << vel_ucf.getHighIndex() << endl;

    cerr << "NR s: [" << xLo << " " << yLo << " " << zLo
         << "] [ " 
         << xHi << " " << yHi << " " << zHi << "]" << endl;  

    for (i = xLo; i <= xHi; i++) 
    {
        for (j = yLo; j <= yHi; j++) 
        {
            for (k = zLo; k <= zHi; k++) 
            {
	         for (m = 1; m <= nMaterials; m++) 
                {
	           // Do something
	           //  cerr << "uvel = " << uvel_CC[m][i][j][k] 
	           //     << " vvel = " << vvel_CC[m][i][j][k] 
	           //     << " wvel = " << wvel_CC[m][i][j][k] << endl;
	           IntVector idx(i-xLo,j-yLo,k-zLo);
	           uvel_CC[m][i][j][k]= vel_ucf[idx].x();
	           vvel_CC[m][i][j][k]= vel_ucf[idx].y();
	           wvel_CC[m][i][j][k]= vel_ucf[idx].z();
	         }
           }
       }
    }
    return;
}

/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  ICE::convertUCFToNR_4d
 Filename:  array_conversion.cc
 Purpose:   Convert UCF data format into Numerical Recipes 4d arrays 
            for scalar data

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/ 
void ICE::convertUCFToNR_4d(const Patch* patch,CCVariable<double>& scalar_ucf, 
        double  ****scalar_CC,
        int     include_ghostcells,
        int     xLoLimit,
        int     xHiLimit,
        int     yLoLimit,
        int     yHiLimit,
        int     zLoLimit,
        int     zHiLimit,
        int     nMaterials)
{
    int i, j, k, m,
        xLo, yLo, zLo,
        xHi, yHi, zHi;
    /*__________________________________
    *   Looping Limits
    *___________________________________*/
    if (include_ghostcells == YES)
    {
        xLo = GC_LO(xLoLimit);  yLo = GC_LO(yLoLimit);  zLo = GC_LO(zLoLimit);
        xHi = GC_HI(xHiLimit);  yHi = GC_HI(yHiLimit);  zHi = GC_HI(zHiLimit);
    } else
    {
        xLo = xLoLimit;         yLo = yLoLimit;         zLo = zLoLimit;
        xHi = xHiLimit;         yHi = yHiLimit;         zHi = zHiLimit;
    }
  
    for (CellIterator iter = patch->getCellIterator(patch->getBox()); !iter.done(); iter++) 
    {
    }

    CellIterator iter = patch->getCellIterator(patch->getBox());
    cerr << "CC iterator begin = " << iter.begin() << " end = " << iter.end() 
         << endl;

    cerr << "CC variables limits " << scalar_ucf.getLowIndex() << " " 
         << scalar_ucf.getHighIndex() << endl;

    cerr << "NR s: [" << xLo << " " << yLo << " " << zLo
         << "] [ " 
         << xHi << " " << yHi << " " << zHi << "]" << endl;  

    for (i = xLo; i <= xHi; i++) 
    {
        for (j = yLo; j <= yHi; j++) 
        {
            for (k = zLo; k <= zHi; k++) 
            {
                   for (m = 1; m <= nMaterials; m++) 
                   {
	              // Do something
	              //  cerr << "scalar = " << scalar_CC[m][i][j][k] << endl;
	              IntVector idx(i-xLo,j-yLo,k-zLo);
	              scalar_CC[m][i][j][k]= scalar_ucf[idx];
                   }
              }
        }
    }
    return;
}

