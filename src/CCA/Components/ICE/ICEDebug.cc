/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/ICE/ICE.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FileUtils.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#ifndef _WIN32
#include <dirent.h>
#endif
using std::ifstream;
using std::cerr;
using std::cout;
using std::endl;
using namespace Uintah;

//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "ICE_NORMAL_COUT:+"
//  ICE_NORMAL_COUT:  dumps out during problemSetup
static DebugStream cout_norm("ICE_NORMAL_COUT", false);
/*_______________________________________________________________________
 Function:  printData_problemSetup-
_______________________________________________________________________ */
void ICE::printData_problemSetup( const ProblemSpecP& prob_spec)
{

  // defaults
  d_dbgStartTime = 9;
  d_dbgStopTime  = 9;
  d_dbgOutputInterval = 9;
  d_dbgSymPlanes= IntVector(0,0,0);
  d_dbgSymmetryTest = false;
  d_dbgSym_relative_tol = 1e-6;
  d_dbgSym_absolute_tol = 1e-9;
  d_dbgSym_cutoff_value = 1e-12;
  d_dbgTime_to_printData = false;
  
  // Turn off all the debuging switches
  switchDebug_Initialize           = false;
  switchDebug_equil_press         = false;
  switchDebug_vel_FC              = false;
  switchDebug_Temp_FC             = false;
  switchDebug_PressDiffRF         = false;
  switchDebug_Exchange_FC         = false;
  switchDebug_explicit_press      = false;
  switchDebug_setupMatrix         = false;
  switchDebug_setupRHS            = false;
  switchDebug_updatePressure      = false;
  switchDebug_computeDelP         = false;
  switchDebug_PressFC             = false;
  switchDebug_LagrangianValues     = false;
  switchDebug_LagrangianSpecificVol= false;
  switchDebug_LagrangianTransportedVars = false;
  switchDebug_MomentumExchange_CC       = false; 
  switchDebug_Source_Sink               = false; 
  switchDebug_advance_advect            = false;
  switchDebug_conserved_primitive       = false;
  
  switchDebug_AMR_refine          = false;
  switchDebug_AMR_refineInterface = false;
  switchDebug_AMR_coarsen         = false;
  switchDebug_AMR_reflux          = false;
  
  //__________________________________
  // Find the switches
  ProblemSpecP debug_ps = prob_spec->findBlock("Debug");
  if (debug_ps) {
    IntVector orig(0,0,0);
    debug_ps->getWithDefault("dbg_GnuPlot",       d_dbgGnuPlot, false);
    debug_ps->getWithDefault("dbg_var1",          d_dbgVar1, 0);   
    debug_ps->getWithDefault("dbg_var2",          d_dbgVar2, 0);  
    debug_ps->getWithDefault("dbg_SigFigs",       d_dbgSigFigs, 5 );
    debug_ps->get("dbg_Level",                    d_dbgLevel);
    debug_ps->get("dbg_timeStart",                d_dbgStartTime);
    debug_ps->get("dbg_timeStop",                 d_dbgStopTime);
    debug_ps->get("dbg_outputInterval",           d_dbgOutputInterval);
    debug_ps->get("dbg_BeginIndex",               d_dbgBeginIndx);
    debug_ps->get("dbg_EndIndex",                 d_dbgEndIndx);
    debug_ps->get("dbg_Matls",                    d_dbgMatls);
    debug_ps->get("dbg_SymmetryPlanes",           d_dbgSymPlanes);
    debug_ps->get("dbg_Sym_absolute_tol",         d_dbgSym_absolute_tol); 
    debug_ps->get("dbg_Sym_relative_tol",         d_dbgSym_relative_tol);
    debug_ps->get("dbg_Sym_cutoff_value",         d_dbgSym_cutoff_value);
    
    if(d_dbgSymPlanes.x()>0 || d_dbgSymPlanes.y()>0 || d_dbgSymPlanes.z() > 0){
      d_dbgSymmetryTest = true;
      cout << "Perform Symmetry test:  Planes of symmetry " << d_dbgSymPlanes
           << " absolute Tolerance: " << d_dbgSym_absolute_tol
           << " relative Tolerance: " << d_dbgSym_relative_tol << endl;
    }

    for (ProblemSpecP child = debug_ps->findBlock("debug"); child != 0;
        child = child->findNextBlock("debug")) {
      map<string,string> debug_attr;
      child->getAttributes(debug_attr);
      if (debug_attr["label"]      == "switchDebug_Initialize")
       switchDebug_Initialize            = true;
      else if (debug_attr["label"] == "switchDebug_equil_press")
       switchDebug_equil_press          = true;
      else if (debug_attr["label"] == "switchDebug_PressDiffRF")
       switchDebug_PressDiffRF          = true;
      else if (debug_attr["label"] == "switchDebug_vel_FC")
       switchDebug_vel_FC               = true;
      else if (debug_attr["label"] == "switchDebug_Temp_FC")
       switchDebug_Temp_FC               = true;
      else if (debug_attr["label"] == "switchDebug_Exchange_FC")
       switchDebug_Exchange_FC          = true;
      else if (debug_attr["label"] == "switchDebug_explicit_press")
       switchDebug_explicit_press       = true;
      else if (debug_attr["label"] == "switchDebug_setupMatrix")
       switchDebug_setupMatrix          = true;
      else if (debug_attr["label"] == "switchDebug_setupRHS")
       switchDebug_setupRHS             = true;
      else if (debug_attr["label"] == "switchDebug_updatePressure")
       switchDebug_updatePressure       = true;
      else if (debug_attr["label"] == "switchDebug_computeDelP")
       switchDebug_computeDelP          = true;
      else if (debug_attr["label"] == "switchDebug_PressFC")
       switchDebug_PressFC              = true;
      else if (debug_attr["label"] == "switchDebug_LagrangianValues")
       switchDebug_LagrangianValues      = true;
      else if (debug_attr["label"] == "switchDebug_LagrangianSpecificVol")
       switchDebug_LagrangianSpecificVol = true;
      else if (debug_attr["label"] == "switchDebug_LagrangianTransportedVars")
       switchDebug_LagrangianTransportedVars = true;
      else if (debug_attr["label"] == "switchDebug_MomentumExchange_CC")
       switchDebug_MomentumExchange_CC   = true;
      else if (debug_attr["label"] == "switchDebug_Source_Sink")
       switchDebug_Source_Sink           = true;
      else if (debug_attr["label"] == "switchDebug_advance_advect")
       switchDebug_advance_advect       = true;
      else if (debug_attr["label"] == "switchDebug_conserved_primitive")
       switchDebug_conserved_primitive  = true;
      else if (debug_attr["label"] == "switchDebug_AMR_refine")
       switchDebug_AMR_refine           = true;
      else if (debug_attr["label"] == "switchDebug_AMR_refineInterface")
       switchDebug_AMR_refineInterface  = true;
      else if (debug_attr["label"] == "switchDebug_AMR_coarsen")
       switchDebug_AMR_coarsen          = true;
       else if (debug_attr["label"] == "switchDebug_AMR_reflux")
       switchDebug_AMR_reflux           = true;
    }
  }
 
  d_dbgNextDumpTime = d_dbgStartTime;
  if(fabs(d_dbgStartTime - 0.0) < d_SMALL_NUM){ 
    d_dbgTime_to_printData = true;
    d_dbgNextDumpTime = d_dbgStartTime;
  }
  if(switchDebug_Initialize){ 
    d_dbgTime_to_printData = true;
  }
  
  //__________________________________
  //  default values
  if(d_dbgMatls.size() == 0 ){
    d_dbgMatls.push_back(0);
  }
  if(d_dbgLevel.size() == 0 ){
    d_dbgLevel.push_back(0);
  }
 
  cout_norm << "Pulled out the debugging switches from input file" << endl;
  cout_norm<< "  debugging starting time   "<<d_dbgStartTime<<endl;
  cout_norm<< "  debugging stopping time   "<<d_dbgStopTime<<endl;
  cout_norm<< "  debugging output interval "<<d_dbgOutputInterval<<endl;
  cout_norm<< "  debugging variable 1      "<<d_dbgVar1<<endl;
  cout_norm<< "  debugging variable 2      "<<d_dbgVar2<<endl; 
  for (int i = 0; i<(int) d_dbgMatls.size(); i++) {
    cout_norm << "  d_dbg_matls = " << d_dbgMatls[i] << endl;
  }
}

/*_______________________________________________________________________ 
 Function:  printData--  convience function
_______________________________________________________________________ */
void    ICE::printVector(int matl,
                         const Patch* patch, 
                         int include_EC,
                         const string&    message1,       
                         const string&    message2,
                          int   component,
                         const CCVariable<Vector>& q_CC)
{
  if(d_dbgSymmetryTest){  // symmetry test
    symmetryTest_Vector(matl, patch, message1, message2, q_CC);
  } else {  
    printVector_driver(matl,patch,include_EC,message1,message2,component,q_CC);
  }
}


void    ICE::printData( int matl,
                        const Patch* patch, 
                        int include_EC,
                        const string&    message1,        
                        const string&    message2,       
                        const CCVariable<int>& q_CC)
{
  printData_driver<CCVariable<int> >
        (matl, patch, include_EC, message1, message2, "CC",q_CC);
}
//__________________________________
void    ICE::printData( int matl,
                        const Patch* patch, 
                        int include_EC,
                        const string&    message1,        
                        const string&    message2,       
                        const CCVariable<double>& q_CC)
{

  if(d_dbgSymmetryTest){  // symmetry test
    IntVector cellShift(0,0,0);
    symmetryTest_driver<CCVariable<double> >
        (matl, patch, cellShift, message1, message2, q_CC);
  } else {                // 
    printData_driver<CCVariable<double> >
        (matl, patch, include_EC, message1, message2, "CC", q_CC);
  }  

}
//__________________________________
//
void    ICE::printData_FC(int matl,
                          const Patch* patch, 
                          int include_EC,
                          const string&    message1,      
                          const string&    message2,  
                          const SFCXVariable<double>& q_FC)
{
  if(d_dbgSymmetryTest){  // symmetry test
    IntVector cellShift(1,0,0);
    symmetryTest_driver< SFCXVariable<double> >
          (matl, patch, cellShift, message1, message2, q_FC);
  } else {
    printData_driver< SFCXVariable<double> >
          (matl, patch, include_EC, message1, message2, "FC", q_FC);
  }
}
//__________________________________
void    ICE::printData_FC(int matl,
                          const Patch* patch, 
                          int include_EC,
                          const string&    message1,      
                          const string&    message2,  
                          const SFCYVariable<double>& q_FC)
{
  if(d_dbgSymmetryTest){  // symmetry test
    IntVector cellShift(0,1,0);
    symmetryTest_driver< SFCYVariable<double> >
          (matl, patch, cellShift, message1, message2, q_FC);
  } else {
    printData_driver< SFCYVariable<double> >
          (matl, patch, include_EC, message1, message2, "FC", q_FC);
  }
}
//__________________________________
void    ICE::printData_FC(int matl,
                          const Patch* patch, 
                          int include_EC,
                          const string&    message1,      
                          const string&    message2,  
                          const SFCZVariable<double>& q_FC)
{
  if(d_dbgSymmetryTest){  // symmetry test
    IntVector cellShift(0,0,1);
    symmetryTest_driver< SFCZVariable<double> >
          (matl, patch, cellShift, message1, message2, q_FC);
  } else {
    printData_driver< SFCZVariable<double> >
          (matl, patch, include_EC, message1, message2, "FC", q_FC);
  }
}


/*_______________________________________________________________________
 Function:  printData_driver-- this does the actual work
_______________________________________________________________________ */
template<class T>
void    ICE::printData_driver( int matl,
                               const Patch* patch, 
                               int include_EC,
                               const string& message1,        
                               const string& message2,
                               const string& variableType,       
                               const T& q_CC)
{
  //__________________________________
  // Limit when we dump
  bool dumpThisMatl = false;
  for (int m = 0; m<(int) d_dbgMatls.size(); m++) {
    if (matl == d_dbgMatls[m]) {
      dumpThisMatl = true;
    }
  } 
  const Level* level = patch->getLevel();
  int levelIndx = level->getIndex();
    
  bool onRightLevel = false;
  int L = -1;
  for (int l = 0; l<(int) d_dbgLevel.size(); l++) {
    if(levelIndx == d_dbgLevel[l] || d_dbgLevel[l] == -9) {
      onRightLevel = true;
      L = l;
    }
  }
  
  if ( onRightLevel && dumpThisMatl == true && d_dbgTime_to_printData ) {
    IntVector low, high; 

    adjust_dbg_indices( include_EC, patch, 
                        d_dbgBeginIndx[L],d_dbgEndIndx[L], 
                        low, high); 
    
    //__________________________________
    // spew to stderr
    if ( d_dbgGnuPlot== false && 
        high.x() > low.x() && high.y() > low.y() && high.z() > low.z() ) {      
      cerr << "____________________________________________L-"<<levelIndx<<"\n";
      cerr << "$" << message1 << "\n";
      cerr << "$" << message2 << "\n";

      cerr.setf(std::ios::scientific,std::ios::floatfield);
      cerr.precision(d_dbgSigFigs);  
      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {
          for(int i = low.x(); i < high.x(); i++) {
           IntVector idx(i, j, k);
           cerr << "[" << i << "," << j << "," << k << "]~ " 
                << q_CC[idx] << "  ";

           /*  cerr << "\n"; */
          }
         cerr << "\n";
        }
        cerr << "\n";
      }
      cerr <<" ______________________________________________\n";
      cerr.setf(std::ios::scientific ,std::ios::floatfield);
    }
    
    //__________________________________
    //  spew to gnuPlot data files
    if (d_dbgGnuPlot) {
      //      FILE *fp;
      string path;
      createDirs(patch, message1, path);
        
      string filename = path + "/" + message2;
      //      fp = fopen(filename.c_str(), "w");
      std::ofstream fp;
      fp.open(filename.c_str());
      fp.precision(15);
      fp.width(16);
      fp.setf(std::ios::scientific);

      double x, dx;
      find_gnuplot_origin_And_dx(variableType, patch, low, high, &dx, &x);   

      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {

          for(int i = low.x(); i < high.x(); i++) {
            IntVector idx(i, j, k);
            //    fprintf(fp, "%16.15E %16.15E\n", x, q_CC[idx]);
            fp << x << " " <<  q_CC[idx] << endl;
            x+=dx;
          }
        }
      }
      //      fclose(fp);
      fp.close();
    } // gnuplot 
  }  // time to dump
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls",
          __FILE__, __LINE__);
  }
}

/*_______________________________________________________________________
 Function:  printVector_driver--
 Purpose:  Print to stderr a cell-centered, single material
_______________________________________________________________________ */
void    ICE::printVector_driver(int matl,
                                const Patch* patch, 
                                int include_EC,
                                const string&    message1,       
                                const string&    message2,
                                 int   /*component*/,  /*  x = 0,y = 1, z = 1  */
                                const CCVariable<Vector>& q_CC)
{

  //__________________________________
  // Limit when we dump
  bool dumpThisMatl = false;
  for (int m = 0; m<(int) d_dbgMatls.size(); m++) {
    if (matl == d_dbgMatls[m]) {
      dumpThisMatl = true;
    }
  } 
  const Level* level = patch->getLevel();
  int levelIndx = level->getIndex(); 
  
  bool onRightLevel = false;
  int L = -1;
  for (int l = 0; l<(int) d_dbgLevel.size(); l++) {
    if(levelIndx == d_dbgLevel[l] || d_dbgLevel[l] == -9) {
      onRightLevel = true;
      L = l;
    }
  }
  
  if ( onRightLevel && dumpThisMatl == true && d_dbgTime_to_printData) {        
    IntVector low, high; 

    adjust_dbg_indices( include_EC, patch, 
                        d_dbgBeginIndx[L],d_dbgEndIndx[L], 
                        low, high); 
    
    string var_name;
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

      //__________________________________
      // spew to stderr
      if ( d_dbgGnuPlot== false && 
        high.x() > low.x() && high.y() > low.y() && high.z() > low.z() ) {  
       
        cerr.setf(std::ios::scientific,std::ios::floatfield);
        cerr.precision(d_dbgSigFigs);
        cerr << "__________________________________________L-"<<levelIndx<<"\n";
        cerr << "$" << message1 << "\n";
        cerr << "$" << var_name << "\n";
        for(int k = low.z(); k < high.z(); k++)  {
          for(int j = low.y(); j < high.y(); j++) {
            for(int i = low.x(); i < high.x(); i++) {
             IntVector idx(i, j, k);
             cerr << "[" << i << "," << j << "," << k << "]~ " 
                  <<  q_CC[idx][dir] << "  ";

             /*  cerr << "\n"; */
            }
           cerr << "\n";
          }
          cerr << "\n";
        }
        cerr << " ______________________________________________\n";
        cerr.setf(std::ios::scientific, std::ios::floatfield);
      }

      //__________________________________
      //  spew to gnuPlot data files
      if (d_dbgGnuPlot) {
        FILE *fp;
        string path;
        createDirs(patch, message1, path);

        string filename = path + "/" + var_name;
        fp = fopen(filename.c_str(), "w");
        double x, dx;
       
        find_gnuplot_origin_And_dx("Vector", patch, low, high, &dx, &x);
      
        for(int k = low.z(); k < high.z(); k++)  {
          for(int j = low.y(); j < high.y(); j++) {
            for(int i = low.x(); i < high.x(); i++) {
              IntVector idx(i, j, k);
              fprintf(fp, "%16.15E %16.15E\n", x, q_CC[idx][dir]);
              x+=dx;
            }
          }
        }
        fclose(fp);
      } // gnuplot
    }  // dir loop
  } // time to dump
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls",
          __FILE__, __LINE__);
  }
}

/*_______________________________________________________________________
 Function:  symmetryTest_driver-- test for symmetry
 Note:      The comments were written presuming that the plane of symmetry
            is in the X direction.
            - CC Variables examine all cells including extra calls
            - (Y,Z)_FC variables examine only patch interior cells
_______________________________________________________________________ */
template<class T>
void    ICE::symmetryTest_driver( int matl,
                                  const Patch* patch,
                                  const IntVector& cellShift,
                                  const string& message1,        
                                  const string& message2,       
                                  const T& q_CC)
{
 
  //__________________________________
  // bulletproofing -- only works on 1 patch
  const Level* level = patch->getLevel();
  int levelIndx = level->getIndex();
  int numPatches = level->numPatches();
  if(numPatches !=1 ){
      throw ProblemSetupException("PRINT_DATA: symmetryTest_driver:  "
                                  "this only works with one patch",
                                  __FILE__, __LINE__);
  }

  // This only works with an even number of cells in the patch interior
  IntVector low, high, ncell;
  low   = patch->getCellLowIndex();
  high  = patch->getCellHighIndex();
  IntVector nCells = high - low;
 
  if((nCells.x() % 2 !=0 && d_dbgSymPlanes.x()) ||
     (nCells.y() % 2 !=0 && d_dbgSymPlanes.y()) ||
     (nCells.z() % 2 !=0 && d_dbgSymPlanes.z())){
      cout << " number of interior cells " << nCells << endl;
      throw ProblemSetupException("PRINT_DATA: symmetryTest_driver:  "
             "Only works if the number of interior cells is even ", __FILE__, __LINE__);
  }
  
  //__________________________________
  // Is this the right material
  bool dumpThisMatl = false;
  for (int m = 0; m<(int) d_dbgMatls.size(); m++) {
    if (matl == d_dbgMatls[m]) {
      dumpThisMatl = true;
    }
  }
  
  bool onRightLevel = false;
  for (int l = 0; l<(int) d_dbgLevel.size(); l++) {
    if(levelIndx == d_dbgLevel[l] || d_dbgLevel[l] == -9) {
      onRightLevel = true;
    }
  }
    
  //__________________________________
  if ( onRightLevel && dumpThisMatl == true && d_dbgTime_to_printData ) { 
    IntVector low, high, high_twk;

    if(cellShift != IntVector(0,0,0)){  // FC variables
      low   = patch->getCellLowIndex();
      high  = patch->getCellHighIndex();
    }else{                              // CC variable
      low   = patch->getExtraCellLowIndex();
      high  = patch->getExtraCellHighIndex();
    }

    bool is_FC_variable = false;
    if (cellShift != IntVector(0,0,0)){  
      is_FC_variable = true;
    }

    cerr.setf(std::ios::scientific,std::ios::floatfield);
    cerr.precision(10);
    bool printHeader = true;

    //__________________________________
    for (int dir = 0; dir <3; dir++){
      if (d_dbgSymPlanes[dir] == 1){  // examine this plane of symmetry?

        int extraCell = 1;

        // ghost cells for FC Variables     
        if (is_FC_variable){
          if (cellShift[dir] * d_dbgSymPlanes[dir] == 1){
            high += cellShift;  // X_FC variables need to shift high
          }else{
            extraCell = 0;      // no Ghost cells for (Y,Z)_FC vars
          }
        }

        // upper looping limit
        high_twk = high;
        high_twk[dir] = (low[dir] + extraCell) + (nCells[dir]/2);
        
        // loop over the lower half of the plane of symmetry
        // and compare corresponding cell on the opposite side of the plane
        for(int k = low.z(); k < high_twk.z(); k++) {
          for(int j = low.y(); j < high_twk.y(); j++) {
            for(int i = low.x(); i < high_twk.x(); i++) {

              IntVector c(i, j, k);

              // find the mirroring cell index--this is tricky for FC variables
              IntVector mirrorCell = c;  // set transverse indicies

              // paper and pencil to figure this out
              mirrorCell[dir] = low[dir] + ((high[dir]-1) - c[dir]);

              // calc. absolute and relative differences
              double abs_diff = 0;
              double rel_diff = 0;

              if( fabs(q_CC[c]) > d_dbgSym_cutoff_value || 
                  fabs (q_CC[mirrorCell]) > d_dbgSym_cutoff_value){
                abs_diff = fabs(q_CC[c] - q_CC[mirrorCell]);
                rel_diff = abs_diff/(fabs(q_CC[c]) + 1e-100);
              }
              
              // catch any asymmetries
              if (abs_diff > d_dbgSym_absolute_tol || 
                  rel_diff > d_dbgSym_relative_tol){
                if (printHeader) {  
                  cerr << "____________________________________________Symmetry Test L-"<<levelIndx<<"\n";
                  cerr << "$" << message2 <<"\t " << message1 <<endl;
                  printHeader = false; 
                } 
                cerr << "c " << c <<  " " << q_CC[c] <<" vs " 
                     << mirrorCell<< " " << q_CC[mirrorCell]
                     << " abs_diff: " << abs_diff 
                     << " relative diff: "<<rel_diff 
                     << " plane " << dir << endl;;
              }  
            }  // i loop
          }  // j loop
        }  // k loop
      }  // test this plane
    } // direction loop
   cerr.setf(std::ios::scientific ,std::ios::floatfield);
  }  // time to dump
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls",
          __FILE__, __LINE__);
  }
}
/*_______________________________________________________________________
 Function:  symmetryTest_Vector-- test for symmetry
 Notes:     This only works for CCVariables
_______________________________________________________________________ */
void    ICE::symmetryTest_Vector( int matl,
                                  const Patch* patch,
                                  const string& message1,        
                                  const string& message2,       
                                  const CCVariable<Vector>& q_CC)
{
  //__________________________________
  // bulletproofing -- only works on 1 patch
  const Level* level = patch->getLevel();
  int levelIndx = level->getIndex();
  int numPatches =level->numPatches();
  if(numPatches !=1 ){
      throw ProblemSetupException("PRINT_DATA: symmetryTest_driver:  "
                                  "this only works with one patch",
                                  __FILE__, __LINE__);
  }

  // The patch interior can only have an even number of cells
  IntVector low, high, ncell;
  low   = patch->getCellLowIndex();
  high  = patch->getCellHighIndex();
  IntVector nCells = high - low;
  
  if((nCells.x() % 2 !=0 && d_dbgSymPlanes.x()) ||
     (nCells.y() % 2 !=0 && d_dbgSymPlanes.y()) ||
     (nCells.z() % 2 !=0 && d_dbgSymPlanes.z())){
      cout << " number of interior cells " << nCells << endl;
      throw ProblemSetupException("PRINT_DATA: symmetryTest_driver:  "
             "Only works if the number of interior cells is even ",
                                  __FILE__, __LINE__);
  }
    
  //__________________________________
  // is this the right material
  bool dumpThisMatl = false;
  for (int m = 0; m<(int) d_dbgMatls.size(); m++) {
    if (matl == d_dbgMatls[m]) {
      dumpThisMatl = true;
    }
  }
  
  bool onRightLevel = false;
  for (int l = 0; l<(int) d_dbgLevel.size(); l++) {
    if(levelIndx == d_dbgLevel[l] || d_dbgLevel[l] == -9) {
      onRightLevel = true;
    }
  }
  //__________________________________
  if ( onRightLevel && dumpThisMatl == true && d_dbgTime_to_printData ) { 
    IntVector low, high, high_twk;
    low   = patch->getCellLowIndex();
    high  = patch->getCellHighIndex();

    cerr.setf(std::ios::scientific,std::ios::floatfield);
    cerr.precision(5);
    bool printHeader = true;
    
    //__________________________________
    for (int dir = 0; dir <3; dir++){
      if (d_dbgSymPlanes[dir] == 1){
      
        int extraCell = 1;
        high_twk = high;
        high_twk[dir] = (low[dir]+ extraCell) + (nCells[dir]/2);


        // loop over the lower half of the plane of symmetry
        // and compare with the on the opposite side of the plane
        for(int k = low.z(); k < high_twk.z(); k++) {
          for(int j = low.y(); j < high_twk.y(); j++) {
            for(int i = low.x(); i < high_twk.x(); i++) {

              IntVector c(i, j, k);
              
              // pencil and paper
              IntVector mirrorCell = c;
              mirrorCell[dir] = low[dir] + ((high[dir]-1) - c[dir]);
                            
              Vector rel_diff = Vector(0);
              Vector abs_diff = Vector(0);
              
              if( q_CC[c].length()          > d_dbgSym_cutoff_value || 
                  q_CC[mirrorCell].length() > d_dbgSym_cutoff_value){
                
                // absolute difference
                for(int d = 0; d < 3; d ++ ){
                  abs_diff[d] = fabs(q_CC[c][d] - q_CC[mirrorCell][d]);
                }
                // normal component is equal and opposite
                abs_diff[dir] = fabs(q_CC[c][dir] + q_CC[mirrorCell][dir]);

                // relative difference
                rel_diff.x(abs_diff.x()/(fabs(q_CC[c].x()) + 1e-100) );
                rel_diff.y(abs_diff.y()/(fabs(q_CC[c].y()) + 1e-100) );
                rel_diff.z(abs_diff.z()/(fabs(q_CC[c].z()) + 1e-100) );
              }
              
              // catch any asymmetries  
              if (abs_diff.length() > d_dbgSym_absolute_tol || 
                  rel_diff.length() > d_dbgSym_relative_tol){
                  
               if (printHeader) {  
                  cerr << "____________________________________________Symmetry Test L-"<<levelIndx<<"\n";
                  cerr << "$" << message2 <<"\t " << message1 <<endl;
                  printHeader = false; 
                } 
                cerr << c << " "  << q_CC[c]
                     << "     abs_diff: " << abs_diff 
                     << " relative diff: "<<rel_diff << endl;
                cerr << mirrorCell<< " " << q_CC[mirrorCell]  << endl;

              }  
            }  // i loop
          }  // j loop
        }  // k loop
      }  // test this plane
    } // direction loop
   cerr.setf(std::ios::scientific ,std::ios::floatfield);
  }  // time to dump
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls",
          __FILE__, __LINE__);
  }
}
/*_______________________________________________________________________
 Function:  printStencil--
_______________________________________________________________________ */
void    ICE::printStencil( int /*matl*/,
                           const Patch* patch, 
                           int include_EC,
                           const string&    message1,        
                           const string&    message2,       
                           const CCVariable<Stencil7>& q_CC)
{
  const Level* level = patch->getLevel();
  int levelIndx = level->getIndex();
    
  bool onRightLevel = false;
  int L = -1;
  for (int l = 0; l<(int) d_dbgLevel.size(); l++) {
    if(levelIndx == d_dbgLevel[l] || d_dbgLevel[l] == -9) {
      onRightLevel = true;
      L = l;
    }
  }
  
  if ( onRightLevel && d_dbgTime_to_printData) {
    IntVector low, high; 
    adjust_dbg_indices( include_EC, patch, 
                        d_dbgBeginIndx[L], 
                        d_dbgEndIndx[L], low, high); 
    //__________________________________
    // spew to stderr
    cerr << "______________________________________________L-"<<levelIndx<<"\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n";

    cerr.setf(std::ios::scientific,std::ios::floatfield);
    cerr.precision(d_dbgSigFigs);    

    for(int k = low.z(); k < high.z(); k++)  {
      for(int j = low.y(); j < high.y(); j++) {
        for(int i = low.x(); i < high.x(); i++) {
          IntVector idx(i, j, k);  
          cerr<< idx
              << " A.b "<< q_CC[idx].b
              << " A.w "<< q_CC[idx].w
              << " A.s "<< q_CC[idx].s
              << " A.p "<< q_CC[idx].p
              << " A.n "<< q_CC[idx].n
              << " A.e "<< q_CC[idx].e
              << " A.t "<< q_CC[idx].t << endl;
        }
      }
    }
    cerr << "\n";
    cerr <<" ______________________________________________\n";
    cerr.setf(std::ios::scientific ,std::ios::floatfield);
  } 
}
/*_______________________________________________________________________
 Function:  adjust_dbg_indices--
 Purpose:  tweak what the user has specified for d_dbgBegin and end 
 indices for multipatch problems
_______________________________________________________________________ */
void  ICE::adjust_dbg_indices(  const int include_EC,
                                const Patch* patch,
                                const IntVector d_dbgBeginIndx,
                                const IntVector d_dbgEndIndx,
                                IntVector& low,                 
                                IntVector& high)                
{
  //__________________________________
  // 
  IntVector lo, hi;
  if (include_EC == 1)  { 
    low   = patch->getExtraCellLowIndex();
    high  = patch->getExtraCellHighIndex();
  }
  if (include_EC == 0) {
    low   = patch->getCellLowIndex();
    high  = patch->getCellHighIndex();
  }


  IntVector beginIndx = d_dbgBeginIndx;
  IntVector endIndx   = d_dbgEndIndx;
  
  //__________________________________
  // bulletproofing
  const Level* level = patch->getLevel();
  IntVector L_lowIndex, L_highIndex;
  level->findCellIndexRange(L_lowIndex, L_highIndex);
  
  if (beginIndx.x() < L_lowIndex.x()  || 
      beginIndx.y() < L_lowIndex.y()  ||
      beginIndx.z() < L_lowIndex.z()  ||
      endIndx.x()   > L_highIndex.x() ||
      endIndx.y()   > L_highIndex.y() ||
      endIndx.z()   > L_highIndex.z()  ){
    ostringstream warn;
    warn << "WARNING:PRINT_DATA: You've specified an index range "
         << beginIndx << " " << endIndx
         << " that is outside the range of this level "
         << level->getIndex()
         << " " << L_lowIndex << " " << L_highIndex << endl;
    static SCIRun::ProgressiveWarning warning(warn.str(),2); 
    warning.invoke();
  }
  if(beginIndx.x() == endIndx.x() ||
     beginIndx.y() == endIndx.y() ||
     beginIndx.z() == endIndx.z() ){
    throw ProblemSetupException("PRINT_DATA: you've specified a beginIndex = EndIndex",
                                __FILE__, __LINE__); 
  }
  
  
  
#if 0    // turn this if you want to specify coarse level cells in the input file
  if (d_dbgGnuPlowt){                  // ignore extra cell specification
    low  = d_dbgBeginIndx;
    high = d_dbgEndIndx;
  }

  const Level* level = patch->getLevel();
  const int levelIndx = level->getIndex();
  IntVector refineRatio(level->getRefinementRatio());
  //__________________________________
  //  for multilevel problems we need
  // to adjust what the user input
  if ( levelIndx > 0 ) {  // use this if you want to specify the coarse level cells
    IntVector numLevels(levelIndx,levelIndx,levelIndx);
    beginIndx =beginIndx* refineRatio * numLevels;
    endIndx   =endIndx  * refineRatio * numLevels;
  } 
#endif  


  //__________________________________                            
  // for multipatch & multilevel problems you need                             
  // further restrict the indicies                    
  if (beginIndx!= IntVector(0,0,0)){                        
    IntVector c = beginIndx;                                 
    for (int dir = 0; dir < 3; dir ++) {  // examine each indice  
      if (c(dir) >= low(dir) &&  c(dir) <= high(dir)) {           
        low(dir) = c(dir);                                        
      } else if (c(dir) < low(dir)) {                             
        low(dir) = low(dir);                                      
      } else if (c(dir) > high(dir) ) {                           
        low(dir) = high(dir);                                     
      }                                                           
    }                                                             
  }                                                               
   if (endIndx != IntVector(0,0,0)){                         
    IntVector c = endIndx;                                   
    for (int dir = 0; dir < 3; dir ++) {  // examine each indice  
      if (c(dir) >= low(dir) &&  c(dir) <= high(dir)) {           
        high(dir) = c(dir);                                       
      } else if (c(dir) < low(dir)) {                             
        high(dir) = low(dir);                                     
      } else if (c(dir) > high(dir) ) {                           
        high(dir) = high(dir);                                    
      }                                                          
    }                                                             
  }                                                               
}
/*_______________________________________________________________________
 Function:  readData--
 Purpose:  Print to stderr a cell-centered, single material
_______________________________________________________________________ */
void    ICE::readData(const Patch* patch, int include_EC,
        const string&    filename,        /* message1                     */
        const string&    var_name,        /* var_name              */
        CCVariable<double>& q_CC)
{
  int i, j, k,xLo, yLo, zLo, xHi, yHi, zHi;
  IntVector lowIndex(0,0,0), hiIndex(0,0,0);
  string text;
  double number;
  
  ifstream fp(filename.c_str());
  if (!fp)
    throw ProblemSetupException("Couldn't open the file with hardwired variables",
                                __FILE__, __LINE__);
        
  //  fscanf(fp,"______________________________________________\n");
  fp >> text;  // scan over the "______"
  fp >> text;
  fp >> text;
  
  if (var_name != text)
    throw ProblemSetupException("You're trying to read in apples and orangs " + var_name + " " +  text,
                                __FILE__, __LINE__);
  
  if (include_EC == 1)  { 
    lowIndex = patch->getExtraCellLowIndex();
    hiIndex  = patch->getExtraCellHighIndex();
  }
  if (include_EC == 0) {
    lowIndex = patch->getCellLowIndex();
    hiIndex  = patch->getCellHighIndex();
  }
  xLo = lowIndex.x();
  yLo = lowIndex.y();
  zLo = lowIndex.z();
  
  xHi = hiIndex.x();
  yHi = hiIndex.y();
  zHi = hiIndex.z();
  
  for(k = zLo; k < zHi; k++)  {
    for(j = yLo; j < yHi; j++) {
      for(i = xLo; i < xHi; i++) {
       IntVector idx(i, j, k);
       
       char c;
       fp.get(c);
       while ( c != '~') {         
         fp.get(c);
         // cerr << c;
       }
       
       fp >> number;
       if (!fp.good())       
        throw ProblemSetupException("Having problem reading " + var_name,
                                    __FILE__, __LINE__);
              
      // cerr << number;
       q_CC[idx] = number;
      }
      char c;
      fp >> c;
    }
    char c;
    fp >> c;
  }
  fp >> text;
}

/*_______________________________________________________________________
 Function~  createDirs:
 Purpose~   generate a bunch of directories based on desc and is used by
            the gnuPlot option.  For example, if desc = 
            BOT_Lagrangian_spVolRF_Mat_0_patch_0 then the dir structure
            would be ./BOT_Lagrangian_spVolRF/patch_0/matl_0
 _______________________________________________________________________ */
void ICE::createDirs( const Patch* patch,
                      const string& desc,
                       string& path) 
{
  string::size_type pos  = desc.find ( "Mat" );
  string::size_type pos2 = desc.find ( "patch" );
  string dirName, matDir;
  string udaDir = dataArchiver->getOutputLocation();
  // bullet proofing
  DIR *check = opendir(udaDir.c_str());
  if ( check == NULL){
    ostringstream warn;
    warn << "ICE:printData:Dumping GnuPlot Data:  The main uda directory does not exist. "
         << " Make sure you're dumping out at least one timestep in the input file";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  closedir(check);
  
  if (pos2 == string::npos){
    ostringstream warn;
    warn<< "\n \nICE:PrintData:GNUPLOT the printData description isn't properly formatted"
        << " you must have _patch_ at the end of the description \n\n";
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }
  
  
  ostringstream DW, levelIndex;
  DW << dataArchiver->getCurrentTimestep();
  
  const Level* level = patch->getLevel();
  levelIndex << "L-"<<level->getIndex();
  
  //__________________________________
  // parse desc into dirName, matl and patch
  if (pos == string::npos ){  // if Mat isn't in the desc
    dirName   = desc.substr(0,pos2-1);
  } else {                    // if Mat is in the desc
    dirName   = desc.substr(0,pos-1);
    matDir    = desc.substr(pos, 5);
  }
  string patchDir  = desc.substr(pos2);
  
  if (patchDir == ""||dirName == "") {
    ostringstream warn;
    warn<< "\n \nICE:PrintData:GNUPLOT the printData description isn't properly formatted"
        << " you must have _patch_ at the end of the description \n\n";
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }
  //  cout << desc << " dirName "<< dirName << " matDir "<< matDir 
  //        << " patchDir "<< patchDir<<endl;

  //__________________________________
  // make the directories
  // code = 0 if successful
  path = udaDir + "/" + dirName;
  MKDIR( path.c_str(), 0777 );
  
  path = udaDir + "/" + dirName + "/" + DW.str();
  MKDIR( path.c_str(), 0777 );
  
  // write out the simulation time
  string filename = path + "/simTime";
  double simTime=dataArchiver->getCurrentTime();
  FILE *fp;
  fp = fopen(filename.c_str(), "w");
  fprintf(fp,"%16.15E",simTime);
  fclose(fp);
  
  // finish making the directories
  path = udaDir + "/" + dirName + "/" + DW.str() + "/" + levelIndex.str();
  MKDIR( path.c_str(), 0777 );
  
  path = udaDir + "/" + dirName + "/" + DW.str() + "/" + levelIndex.str()
         + "/" + patchDir;
  MKDIR( path.c_str(), 0777 );
  
  if (matDir != "") { 
    path = udaDir + "/" + dirName + "/" + DW.str() + "/" + levelIndex.str() 
            + "/" + patchDir + "/" + matDir ;
    MKDIR( path.c_str(), 0777 );
  }
}
/*_______________________________________________________________________
 Function~  find_gnuplot_origin_And_dx:
 Purpose~   Find principle direction the associated dx and the origin
 _______________________________________________________________________ */
void ICE::find_gnuplot_origin_And_dx(const string variableType,
                                     const Patch* patch,
                                     IntVector& low, 
                                     IntVector& high,
                                     double *dx,
                                     double *origin)
{
   //__________________________________
  //  for multilevel problems adjust the user input
  // Just a 1 to low on all non-principal dirs.2
  const Level* level = patch->getLevel();
  int levelIndx = level->getIndex();
  IntVector refineRatio(level->getRefinementRatio());
  IntVector numLevels(levelIndx,levelIndx,levelIndx);
  IntVector numCells(numLevels * refineRatio );
 
  if ( levelIndx > 0 ) {
    if (high.x() - low.x() == numCells.x() ) {
      high.x(low.x() + 1);
    }
    if (high.y() - low.y() == numCells.y()) {
      high.y(low.y() + 1);
    }
    if (high.z() - low.z() ==  numCells.z()) {
      high.z(low.z() + 1);
    }
  } 
  
  int test=0;
  int principalDir = 0;
  Vector  dx_org = patch->dCell();
  //__________________________________
  // bullet proofing
  if (high.x() - low.x() > 1) {
    test +=1;
    principalDir = 0;
  }
  if (high.y() - low.y() > 1) {
    test +=1;
     principalDir = 1;
  }
  if (high.z() - low.z() > 1) {
    test +=1;
    principalDir = 2;
  }
  
  if (test !=1) {
    ostringstream warn;
    warn << "\n PrintDebug:GNUPLOT: you have more that one principal dir. specified \n" << 
         " or you haven't specified one.  Double check dbg_BeginIndex or \n" <<
         " dbg_EndIndex\n" <<
         "low "<< low << "high " << high <<endl;
    
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
  }
  // For face centered variables subtract dx/2 off the CC position
  double offset = 0.0;
  if (variableType == "FC") {
    offset = dx_org[principalDir]/2.0;
  }
  
  //  along the principal dir find dx and the origin 
  *dx = dx_org[principalDir];
  Vector pos = patch->cellPosition(low).asVector();
  *origin = pos[principalDir] - offset;
  // cout << " dx " << *dx  << " *origin " << *origin << " offset " << offset << endl;
}

