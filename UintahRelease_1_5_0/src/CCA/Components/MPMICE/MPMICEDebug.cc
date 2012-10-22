/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPMICE/MPMICE.h>
#include <CCA/Components/ICE/ICE.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <iostream>
#include <fstream>

using namespace std;
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
  for (int m = 0; m<(int) d_ice->d_dbgMatls.size(); m++) {
    if (matl == d_ice->d_dbgMatls[m]) {
      dumpThisMatl = true;
    }
  } 
  
  const Level* level = patch->getLevel();
  int levelIndx = level->getIndex();
    
  bool onRightLevel = false;
  for (int l = 0; l<(int) d_ice->d_dbgLevel.size(); l++) {
    if(levelIndx == d_ice->d_dbgLevel[l] || d_ice->d_dbgLevel[l] == -9) {
      onRightLevel = true;
    }
  }
  
  if ( onRightLevel && dumpThisMatl == true && d_ice->d_dbgTime_to_printData ) {        
    IntVector low, high; 
    
    d_ice->adjust_dbg_indices( include_EC, patch, 
                               d_ice->d_dbgBeginIndx[levelIndx],
                               d_ice->d_dbgEndIndx[levelIndx], low, high);
    //__________________________________
    // spew to stderr
    if ( high.x() > low.x() && high.y() > low.y() && high.z() > low.z() ) {      
      cerr << "____________________________________________L-"<<levelIndx<<"\n";
      cerr << "$" << message1 << "\n";
      cerr << "$" << message2 << "\n"; 
      cerr.setf(ios::scientific,ios::floatfield);
      cerr.precision(d_ice->d_dbgSigFigs);

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
  //__________________________________
  //  bullet proof
  if (d_ice->d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls", __FILE__, __LINE__);
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
  for (int m = 0; m<(int) d_ice->d_dbgMatls.size(); m++) {
    if (matl == d_ice->d_dbgMatls[m]) {
      dumpThisMatl = true;
    }
  } 
  
  const Level* level = patch->getLevel();
  int levelIndx = level->getIndex();
    
  bool onRightLevel = false;
  for (int l = 0; l<(int) d_ice->d_dbgLevel.size(); l++) {
    if(levelIndx == d_ice->d_dbgLevel[l] || d_ice->d_dbgLevel[l] == -9) {
      onRightLevel = true;
    }
  }
  
  if ( onRightLevel && dumpThisMatl == true && d_ice->d_dbgTime_to_printData ) {            
    IntVector low, high; 
    
    d_ice->adjust_dbg_indices(  include_EC, patch, 
                               d_ice->d_dbgBeginIndx[levelIndx],
                               d_ice->d_dbgEndIndx[levelIndx], low, high); 
    
    string var_name;
    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(d_ice->d_dbgSigFigs); 
    
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
  if (d_ice->d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls", __FILE__, __LINE__);
  }
}
