#include <Packages/Uintah/CCA/Components/MPMICE/MPMICE.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <iostream>
#include <fstream>

using std::ifstream;
using std::cerr;
using namespace SCIRun;
using namespace Uintah;

//______________________________________________________________________
//
void    MPMICE::printData(const Patch* patch, int include_EC,
        char    message1[],           
        char    message2[],     
        const NCVariable<double>& q_NC)
{
 //__________________________________
 // Limit when we dump
  d_dbgTime= dataArchiver->getCurrentTime();   
  if ( d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;        
    IntVector low, high; 

    if (include_EC == 1)  { 
      low   = patch->getNodeLowIndex();
      high  = patch->getNodeHighIndex();
    }
    if (include_EC == 0) {
      low   = patch->getInteriorNodeLowIndex();
      high  = patch->getInteriorNodeHighIndex();
    }
    
    d_ice->adjust_dbg_indices( d_ice->d_dbgBeginIndx, d_ice->d_dbgEndIndx, 
                               low, high);
    cerr << "______________________________________________\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n"; 
    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(16);
        
    for(int k = low.z(); k < high.z(); k++)  {
      for(int j = low.y(); j < high.y(); j++) {
        for(int i = low.x(); i < high.x(); i++) {
         IntVector idx(i, j, k);
          cerr << "[" << i << "," << j << "," << k << "]~ " 
               << q_NC[idx] << "  ";
         /*  cerr << "\n"); */
        }
        cerr << "\n";
      }
      cerr << "\n";
    }
    cerr <<" ______________________________________________\n";
    cerr.setf(ios::scientific, ios::floatfield);
  }
}


//______________________________________________________________________
//
void    MPMICE::printNCVector(const Patch* patch, int include_EC,
        char    message1[],             
        char    message2[],             
        int     component,              /*  x = 0,y = 1, z = 2          */
        const NCVariable<Vector>& q_NC)
{
 //__________________________________
 // Limit when we dump
  d_dbgTime= dataArchiver->getCurrentTime();   
  if ( d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;             
    IntVector low, high; 
    
    if (include_EC == 1)  { 
      low   = patch->getNodeLowIndex();
      high  = patch->getNodeHighIndex();
    }
    if (include_EC == 0) {
      low   = patch->getInteriorNodeLowIndex();
      high  = patch->getInteriorNodeHighIndex();
    }
    
    d_ice->adjust_dbg_indices( d_ice->d_dbgBeginIndx, d_ice->d_dbgEndIndx, 
                               low, high);
    cerr << "______________________________________________\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n"; 
    
    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(16);
        
    for(int k = low.z(); k < high.z(); k++)  {
      for(int j = low.y(); j < high.y(); j++) {
        for(int i = low.x(); i < high.x(); i++) {
         IntVector idx(i, j, k);
         cerr << "[" << i << "," << j << "," << k << "]~ " 
              << q_NC[idx](component) << "  ";  
         /*  cerr << "\n"; */
        }
        cerr << "\n";
      }
      cerr << "\n";
    }
    cerr <<" ______________________________________________\n";
    cerr.setf(ios::scientific, ios::floatfield);
  }
}
