#include <Packages/Uintah/CCA/Components/MPMICE/MPMICE.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

#include <iostream>
#include <fstream>

using std::ifstream;
using std::cerr;
using namespace SCIRun;
using namespace Uintah;

//______________________________________________________________________
//
void    MPMICE::printData(int matl,
                          const Patch* patch, 
                          int include_EC,
                          const string&    message1,        
                          const string&    message2,      
                          const NCVariable<double>& q_NC)
{
  //__________________________________
  // Limit when we dump
  bool dumpThisMatl = false;
  for (int m = 0; m<(int) d_dbgMatls.size(); m++) {
    if (matl == d_dbgMatls[m]) {
      dumpThisMatl = true;
    }
  } 
  
  d_dbgTime= dataArchiver->getCurrentTime();  
  if ( dumpThisMatl == true        &&
       d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;        
    IntVector low, high; 
    
    d_ice->adjust_dbg_indices( include_EC, patch, d_ice->d_dbgBeginIndx, 
                                d_ice->d_dbgEndIndx, low, high);
    cerr << "______________________________________________\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n"; 
    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(d_dbgSigFigs);
        
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
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls");
  }
}


//______________________________________________________________________
//
void    MPMICE::printNCVector(int matl,
                              const Patch* patch, 
                              int include_EC,
                              const string&    message1,        
                              const string&    message2,              
                              int     /*component*/, 
                              const NCVariable<Vector>& q_NC)
{
  //__________________________________
  // Limit when we dump
  bool dumpThisMatl = false;
  for (int m = 0; m<(int) d_dbgMatls.size(); m++) {
    if (matl == d_dbgMatls[m]) {
      dumpThisMatl = true;
    }
  } 
  
  d_dbgTime= dataArchiver->getCurrentTime();  
  if ( dumpThisMatl == true        &&
       d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;             
    IntVector low, high; 
    
    d_ice->adjust_dbg_indices( include_EC, patch, d_ice->d_dbgBeginIndx, 
                                d_ice->d_dbgEndIndx, low, high); 
    
    string var_name;
    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(d_dbgSigFigs); 
    
    for (int dir = 0; dir < 3 ; dir ++ ) { 
      if (dir == 0 ) {
        var_name="X_" + message2;
      }
      if (dir == 1 ) {
        var_name="Y_" + message2;
      }
      if (dir == 2 ) {
        var_name="Z_" + message2;
      }
      cerr << "______________________________________________\n";
      cerr << "$" << message1 << "\n";
      cerr << "$" << var_name << "\n"; 
      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {
          for(int i = low.x(); i < high.x(); i++) {
           IntVector idx(i, j, k);
           cerr << "[" << i << "," << j << "," << k << "]~ " 
                << q_NC[idx][dir] << "  ";  
           /*  cerr << "\n"; */
          }
          cerr << "\n";
        }
        cerr << "\n";
      }
    }
    cerr <<" ______________________________________________\n";
    cerr.setf(ios::scientific, ios::floatfield);
  }
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls");
  }
}
