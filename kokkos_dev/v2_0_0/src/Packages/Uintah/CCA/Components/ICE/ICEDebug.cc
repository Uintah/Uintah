#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

using std::ifstream;
using std::cerr;
using namespace SCIRun;
using namespace Uintah;

/* 
 ======================================================================*
 Function:  printData--
 Purpose:  Print to stderr a cell-centered, single material
_______________________________________________________________________ */
void    ICE::printData( int matl,
                        const Patch* patch, 
                        int include_EC,
                        const string&    message1,        
                        const string&    message2,       
                        const CCVariable<double>& q_CC)
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

    adjust_dbg_indices( include_EC, patch, d_dbgBeginIndx, d_dbgEndIndx, 
                        low, high); 
    
    //__________________________________
    // spew to stderr
    if ( d_dbgGnuPlot== false && 
        high.x() > low.x() && high.y() > low.y() && high.z() > low.z() ) {      
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
                << q_CC[idx] << "  ";

           /*  cerr << "\n"; */
          }
         cerr << "\n";
        }
        cerr << "\n";
      }
      cerr <<" ______________________________________________\n";
      cerr.setf(ios::scientific ,ios::floatfield);
    }
    //__________________________________
    //  spew to gnuPlot data files
    if (d_dbgGnuPlot) {
      FILE *fp;
      string path;
      createDirs(message1, path);
        
      string filename = path + "/" + message2;
      fp = fopen(filename.c_str(), "w");
      double x, dx;
      find_gnuplot_origin_And_dx(patch, low, high, &dx, &x);     
      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {
          for(int i = low.x(); i < high.x(); i++) {
            IntVector idx(i, j, k);
            fprintf(fp, "%16.15E %16.15E\n", x+=dx, q_CC[idx]);
          }
        }
      }
      fclose(fp);
    } // gnuplot 
  }  // time to dump
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls");
  }
}

/* 
 ======================================================================*
 Function:  printData--
 Purpose:  Print to stderr a cell-centered, single material
_______________________________________________________________________ */
void    ICE::printData(int matl,
                       const Patch* patch,                
                       int include_EC,                    
                       const string&    message1,         
                       const string&    message2,         
                       const CCVariable<int>& q_CC)       
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
    
    adjust_dbg_indices( include_EC, patch, d_dbgBeginIndx, d_dbgEndIndx, 
                        low, high);
    
    //__________________________________
    // spew to stderr
    if ( d_dbgGnuPlot== false && 
        high.x() > low.x() && high.y() > low.y() && high.z() > low.z() ) {      
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
                << q_CC[idx] << " ";

           /*  cerr << "\n"; */
          }
         cerr << "\n";
        }
        cerr << "\n";
      }
      cerr << " ______________________________________________\n";
      cerr.setf(ios::scientific ,ios::floatfield);
    }
    //__________________________________
    //  spew to gnuPlot data files
    if (d_dbgGnuPlot) {
      FILE *fp;
      string path;
      createDirs(message1, path);
        
      string filename = path + "/" + message2;
      fp = fopen(filename.c_str(), "w");
      double x, dx;
      find_gnuplot_origin_And_dx(patch, low, high, &dx, &x);
      
      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {
          for(int i = low.x(); i < high.x(); i++) {
            IntVector idx(i, j, k);
            fprintf(fp, "%16.15E %i\n", x+=dx, q_CC[idx]);
          }
        }
      }
      fclose(fp);
    } // gnuplot
  }  // time to dump
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls");
  }
}
/* 
 ======================================================================*
 Function:  printVector--
 Purpose:  Print to stderr a cell-centered, single material
_______________________________________________________________________ */
void    ICE::printVector(int matl,
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
  
  d_dbgTime= dataArchiver->getCurrentTime();  
  if ( dumpThisMatl == true        &&
       d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;      
    IntVector low, high; 

    adjust_dbg_indices( include_EC, patch, d_dbgBeginIndx, d_dbgEndIndx, 
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
       
        cerr.setf(ios::scientific,ios::floatfield);
        cerr.precision(d_dbgSigFigs);
        cerr << "______________________________________________\n";
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
        cerr.setf(ios::scientific, ios::floatfield);
      }

      //__________________________________
      //  spew to gnuPlot data files
      if (d_dbgGnuPlot) {
        FILE *fp;
        string path;
        createDirs(message1, path);

        string filename = path + "/" + var_name;
        fp = fopen(filename.c_str(), "w");
        double x, dx;
        find_gnuplot_origin_And_dx(patch, low, high, &dx, &x);

        for(int k = low.z(); k < high.z(); k++)  {
          for(int j = low.y(); j < high.y(); j++) {
            for(int i = low.x(); i < high.x(); i++) {
              IntVector idx(i, j, k);
              fprintf(fp, "%16.15E %16.15E\n", x+=dx, q_CC[idx][dir]);
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
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls");
  }
}


/* 
 ======================================================================*
 Function:  printData_FC--
 Purpose:  Print left face
_______________________________________________________________________ */
void    ICE::printData_FC(int matl,
                          const Patch* patch, 
                          int include_EC,
                          const string&    message1,      
                          const string&    message2,  
                          const SFCXVariable<double>& q_FC)
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

    adjust_dbg_indices( include_EC, patch, d_dbgBeginIndx, d_dbgEndIndx, 
                        low, high); 

    //__________________________________
    // spew to stderr
    if ( d_dbgGnuPlot== false && 
        high.x() > low.x() && high.y() > low.y() && high.z() > low.z() ) {  
    
      cerr.setf(ios::scientific,ios::floatfield);
      cerr.precision(d_dbgSigFigs); 
      cerr << "______________________________________________\n";
      cerr << "$" << message1 << "\n";
      cerr << "$" << message2 << "\n"; 
      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {
          for(int i = low.x(); i < high.x(); i++) {
           IntVector idx(i, j, k);
           cerr << "[" << i << "," << j << "," << k << "]~ " <<
             q_FC[idx] << "  ";

           /* cerr <<"\n"; */
          }
          cerr << "\n";
        }
        cerr <<"\n";
      }
      cerr << " ______________________________________________\n";
      cerr.setf(ios::scientific, ios::floatfield);
    }
    //__________________________________
    //  spew to gnuPlot data files
    if (d_dbgGnuPlot) {
      FILE *fp;
      string path;
      createDirs(message1, path);

      string filename = path + "/" + message2;
      fp = fopen(filename.c_str(), "w");
      double x, dx;
      find_gnuplot_origin_And_dx(patch, low, high, &dx, &x);

      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {
          for(int i = low.x(); i < high.x(); i++) {
            IntVector idx(i, j, k);
            fprintf(fp, "%16.15E %16.15E\n", x+=dx, q_FC[idx]);
          }
        }
      }
      fclose(fp);
    } // gnuplot
  }  // time to dump
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls");
  }
}
/* 
 ======================================================================*
 Function:  printData_FC--
 Purpose:   Prints bottom Face
_______________________________________________________________________ */
void    ICE::printData_FC(int matl,
                          const Patch* patch, 
                          int include_EC,
                          const string&    message1,        
                          const string&    message2,  
                          const SFCYVariable<double>& q_FC)
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

    adjust_dbg_indices( include_EC, patch, d_dbgBeginIndx, d_dbgEndIndx, 
                        low, high);
    //__________________________________
    // spew to stderr
    if ( d_dbgGnuPlot== false && 
        high.x() > low.x() && high.y() > low.y() && high.z() > low.z() ) {  
      cerr.setf(ios::scientific,ios::floatfield);
      cerr.precision(d_dbgSigFigs);
      cerr << "______________________________________________\n";
      cerr << "$" << message1 << "\n";
      cerr << "$" << message2 << "\n";
      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {
          for(int i = low.x(); i < high.x(); i++) {
           IntVector idx(i, j, k);
           cerr << "[" << i << "," << j << "," << k << "]~ " <<  
             q_FC[idx] << "  ";

           /*  cerr << "\n"; */
          }
          cerr << "\n";
        }
        cerr << "\n";
      }
      cerr << " ______________________________________________\n";
      cerr.setf(ios::scientific, ios::floatfield);
    }
    //__________________________________
    //  spew to gnuPlot data files
    if (d_dbgGnuPlot) {
      FILE *fp;
      string path;
      createDirs(message1, path);

      string filename = path + "/" + message2;
      fp = fopen(filename.c_str(), "w");
      double x, dx;
      find_gnuplot_origin_And_dx(patch, low, high, &dx, &x);

      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {
          for(int i = low.x(); i < high.x(); i++) {
            IntVector idx(i, j, k);
            fprintf(fp, "%16.15E %16.15E\n", x+=dx, q_FC[idx]);
          }
        }
      }
      fclose(fp);
    } // gnuplot
  } // time to dump
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls");
  }
}

/* 
 ======================================================================*
 Function:  printData_FC--
 Purpose:  Prints back face
_______________________________________________________________________ */
void    ICE::printData_FC(int matl,
                          const Patch* patch, 
                          int include_EC,
                          const string&    message1,        
                          const string&    message2,       
                          const SFCZVariable<double>& q_FC)
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
    
    adjust_dbg_indices( include_EC, patch, d_dbgBeginIndx, d_dbgEndIndx, 
                        low, high);
    //__________________________________
    // spew to stderr
    if ( d_dbgGnuPlot== false && 
        high.x() > low.x() && high.y() > low.y() && high.z() > low.z() ) {   
      cerr.setf(ios::scientific,ios::floatfield);
      cerr.precision(d_dbgSigFigs);   
      cerr << "______________________________________________\n";
      cerr << "$" << message1 << "\n";
      cerr << "$" << message2 << "\n";
      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {
          for(int i = low.x(); i < high.x(); i++) {
           IntVector idx(i, j, k);
           cerr << "[" << i << "," << j << "," << k << "]~ " << 
             q_FC[idx] << "  ";

           /*  cerr << "\n"; */
          }
         cerr << "\n";
        }
        cerr << "\n";
      }
      cerr << " ______________________________________________\n";
      cerr.setf(ios::scientific, ios::floatfield);
    }
    //__________________________________
    //  spew to gnuPlot data files
    if (d_dbgGnuPlot) {
      FILE *fp;
      string path;
      createDirs(message1, path);

      string filename = path + "/" + message2;
      fp = fopen(filename.c_str(), "w");
      double x, dx;
      find_gnuplot_origin_And_dx(patch, low, high, &dx, &x);

      for(int k = low.z(); k < high.z(); k++)  {
        for(int j = low.y(); j < high.y(); j++) {
          for(int i = low.x(); i < high.x(); i++) {
            IntVector idx(i, j, k);
            fprintf(fp, "%16.15E %16.15E\n", x+=dx, q_FC[idx]);
          }
        }
      }
      fclose(fp);
    } // gnuplot
  }  // time to dump
  //__________________________________
  //  bullet proof
  if (d_dbgMatls.size() == 0){
    throw ProblemSetupException(
          "P R I N T  D A T A: You must specify at least 1 matl in d_dbgMatls");
  }
}

/* 
 ======================================================================*
 Function:  printStencil--
_______________________________________________________________________ */
void    ICE::printStencil( int /*matl*/,
                           const Patch* patch, 
                           int include_EC,
                           const string&    message1,        
                           const string&    message2,       
                           const CCVariable<Stencil7>& q_CC)
{
  d_dbgTime= dataArchiver->getCurrentTime();  
  if ( d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;      

    IntVector low, high; 
    adjust_dbg_indices( include_EC, patch, d_dbgBeginIndx, d_dbgEndIndx, 
                        low, high); 
    //__________________________________
    // spew to stderr
    cerr << "______________________________________________\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n";

    cerr.setf(ios::scientific,ios::floatfield);
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
    cerr.setf(ios::scientific ,ios::floatfield);
  } 
}
/* 
 ======================================================================*
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
  if (include_EC == 1)  { 
    low   = patch->getCellLowIndex();
    high  = patch->getCellHighIndex();
  }
  if (include_EC == 0) {
    low   = patch->getInteriorCellLowIndex();
    high  = patch->getInteriorCellHighIndex();
  }

#if 0 
  if (d_dbgGnuPlot){                  // ignore extra cell specification
    low  = d_dbgBeginIndx;
    high = d_dbgEndIndx;
  }
#endif

  //__________________________________                            
  // for multipatch problems you need                             
  // further restrict the indicies                                
  if (d_dbgBeginIndx != IntVector(0,0,0)){                        
    IntVector c = d_dbgBeginIndx;                                 
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
   if (d_dbgEndIndx != IntVector(0,0,0)){                         
    IntVector c = d_dbgEndIndx;                                   
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
/* 
 ======================================================================*
 Function:  readData--
 Purpose:  Print to stderr a cell-centered, single material
_______________________________________________________________________ */
void    ICE::readData(const Patch* patch, int include_EC,
        const string&    filename,        /* message1                     */
        const string&    var_name,        /* var_name              */
        CCVariable<double>& q_CC)
{
  int i, j, k,xLo, yLo, zLo, xHi, yHi, zHi;
  IntVector lowIndex, hiIndex; 
  string text;
  double number;
  
  ifstream fp(filename.c_str());
  if (!fp)
    throw ProblemSetupException("Couldn't open the file with hardwired variables");
        
  //  fscanf(fp,"______________________________________________\n");
  fp >> text;  // scan over the "______"
  fp >> text;
  fp >> text;
  
  if (var_name != text)
    throw ProblemSetupException("You're trying to read in apples and orangs " + var_name + " " +  text);
  
  if (include_EC == 1)  { 
    lowIndex = patch->getCellLowIndex();
    hiIndex  = patch->getCellHighIndex();
  }
  if (include_EC == 0) {
    lowIndex = patch->getInteriorCellLowIndex();
    hiIndex  = patch->getInteriorCellHighIndex();
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
        throw ProblemSetupException("Having problem reading " + var_name);
              
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

/* 
 ======================================================================
 Function~  createDirs:
 Purpose~   generate a bunch of directories based on desc and is used by
            the gnuPlot option.  For example, if desc = 
            BOT_Lagrangian_spVolRF_Mat_0_patch_0 then the dir structure
            would be ./BOT_Lagrangian_spVolRF/patch_0/matl_0
 _______________________________________________________________________ */
void ICE::createDirs( const string& desc, string& path) 
{
  string::size_type pos  = desc.find ( "Mat" );
  string::size_type pos2 = desc.find ( "patch" );
  string dirName, matDir;
  string udaDir = dataArchiver->getOutputLocation();
  ostringstream DW;
  DW << dataArchiver->getCurrentTimestep();
  
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
    Message(1,"\nGNUPLOT the printData description isn't properly formatted",
              " you must have _patch_ at the end of the description","");
  }
  //  cout << desc << " dirName "<< dirName << " matDir "<< matDir 
  //        << " patchDir "<< patchDir<<endl;

  //__________________________________
  // make the directories
  // code = 0 if successful
  path = udaDir + "/" + dirName;
  mkdir( path.c_str(), 0777 );
  
  path = udaDir + "/" + dirName + "/" + DW.str();
  mkdir( path.c_str(), 0777 );
  
  path = udaDir + "/" + dirName + "/" + DW.str() + "/" + patchDir;
  mkdir( path.c_str(), 0777 );
  
  if (matDir != "") { 
    path = udaDir + "/" + dirName + "/" + DW.str() + "/" + 
           patchDir + "/" + matDir ;
    mkdir( path.c_str(), 0777 );
  }
}
/* 
 ======================================================================
 Function~  find_gnuplot_origin_And_dx:
 Purpose~   Find principle direction the associated dx and the origin
 _______________________________________________________________________ */
void ICE::find_gnuplot_origin_And_dx(const Patch* patch, 
                                     const IntVector low, 
                                     const IntVector high,
                                     double *dx,
                                     double *origin)
{
  int test=0;
  Vector  dx_org = patch->dCell();
  //__________________________________
  // bullet proofing
  if (high.x() - low.x() > 1) {test +=1;}
  if (high.y() - low.y() > 1) {test +=1;}
  if (high.z() - low.z() > 1) {test +=1;}
  
  if (test !=1) {
    ostringstream desc;
    desc << "\n GNUPLOT: you have more that one principal dir. specified \n" << 
         " or you haven't specified one.  Double check dbg_BeginIndex or \n" <<
         " dbg_EndIndex\n" <<
         "low "<< low << "high " << high <<endl;
    Message(1, desc.str(),"","");
  }
  //__________________________________
  //  along the principal dir find dx and
  //  the origin             
  if (high.x() - low.x() > 1) {
    *dx = dx_org.x();
    *origin = *dx * low.x();
  } 
  if (high.y() - low.y() > 1) {
    *dx = dx_org.y();
    *origin = *dx * low.y();
  }
  if (high.z() - low.z() > 1){
    *dx = dx_org.z();
    *origin = *dx * low.z();
  }
}

/* 
 ======================================================================
 Function~  ICE::Message:
 Purpose~  Output an error message and stop the program if requested. 
 _______________________________________________________________________ */
void    ICE::Message(
        int     abort,          /* =1 then abort                            */
        const string&    message1,   
        const string&    message2,   
        const string&    message3) 
{        
  cerr << "\n\n ______________________________________________\n";
  cerr << message1 << "\n";
  cerr << message2 << "\n";
  cerr << message3 << "\n";
  cerr << "\n\n ______________________________________________\n";
  char* exitMode = getenv("ICE_DEBUGGER_ON_EXIT");

  if(!exitMode)
    exitMode = "no";    //default exit mode
  //______________________________
  // Now aborting program
  string mode(exitMode);
  if(abort == 1) {
    if(mode == "yes") {
      string c;
      cerr << "\n";
      cerr << "<c> = cvd\n";
      cin >> c;
      system("date");
      if(c == "c") system("cvd -P sus");
    }
    exit(1); 
  }
}

/* 
 ======================================================================*
 Function:  printConservedQuantities--
 If the switch is turned on then print out the conserved quantities.
_______________________________________________________________________ */
void ICE::printConservedQuantities(const ProcessorGroup*,  
                                   const PatchSubset* patches,
                                   const MaterialSubset* /*matls*/,
                                   DataWarehouse* /*old_dw*/,
                                   DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++)  {
    const Patch* patch = patches->get(p);  
    Vector dx       = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    double mass;
            
    int numICEmatls = d_sharedState->getNumICEMatls();
    int numALLMatls = d_sharedState->getNumMatls();

    static double initial_total_eng;
    static Vector initial_total_mom;
    static int n_passes = 0;
    
    vector<Vector> mat_momentum(numICEmatls);
    vector<double> mat_mass(numICEmatls);
    vector<double> mat_total_eng(numICEmatls);
    vector<double> mat_int_eng(numICEmatls);
    vector<double> mat_KE(numICEmatls);
      
    constCCVariable<Vector> vel_CC, mom_L_CC, mom_L_ME_CC;
    constCCVariable<double> rho_CC, int_eng_L_CC, eng_L_ME_CC;
    constCCVariable<double> Temp_CC;
   
    Ghost::GhostType  gn  = Ghost::None;
    //__________________________________
    // Loop over all the ICE matls
    for (int m = 0; m < numICEmatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      new_dw->get(vel_CC, lb->vel_CCLabel, indx, patch,  Ghost::None, 0);
      new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch,  Ghost::None, 0);
      new_dw->get(Temp_CC,lb->temp_CCLabel,indx, patch,  Ghost::None, 0);
      double cv = ice_matl->getSpecificHeat();   
      mat_momentum[m] = Vector(0.0, 0.0, 0.0);
      mat_KE[m]       = 0.0;
      mat_int_eng[m]  = 0.0;
      mat_mass[m]     = 0.0;
      //__________________________________
      // Accumulate the momenta and energy
      for (CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        mass            = rho_CC[c] * cell_vol;
        mat_momentum[m] += vel_CC[c] * mass;
        double vel_sq = vel_CC[c].length() * vel_CC[c].length();
        mat_KE[m]      += 0.5 * mass * vel_sq;
        mat_int_eng[m] += mass * cv * Temp_CC[c];
        mat_mass[m]    += mass;        
      }
    }  // numICEmatls loop

    //__________________________________
    //  Now compute totals and the change in quantities
    Vector total_momentum(0.0, 0.0, 0.0);
    double total_energy   = 0.0;
    double total_mass     = 0.0;
    double total_KE       = 0.0;
    double total_int_eng  = 0.0;

    for (int m = 0; m < numICEmatls; m++ ) {
      mat_total_eng[m]= mat_int_eng[m] + mat_KE[m];
      total_energy   += mat_total_eng[m];
      total_KE       += mat_KE[m];
      total_int_eng  += mat_int_eng[m];
      total_mass     += mat_mass[m];
      total_momentum += mat_momentum[m];
    }
    //__________________________________
    // Dump diagnostics if only one patch
    const Level* level=patch->getLevel();
    int numPatches = level->numPatches();
    if( numPatches == 1 ){ 
      cout.setf(ios::scientific,ios::floatfield);
      cout.precision(8);
      for (int m = 0; m < numICEmatls; m++ ) {
        cout << " Mat " << m << endl;
        cout << " mass        " <<  mat_mass[m] << endl;
        cout << " momentum    " << mat_momentum[m]
             << " length: " << mat_momentum[m].length() << endl;
        cout << " Int Energy: " << mat_int_eng[m] 
             << ", Kinetic: " << mat_KE[m] 
             << " total: "    << mat_total_eng[m] << endl;
      }
   
      //__________________________________
      //  set the inital values
      if ( n_passes == 0) {
        initial_total_eng = total_energy;
        initial_total_mom = total_momentum;
        n_passes ++;
      } 

      double change_total_mom =
                  100.0 * (total_momentum.length() - initial_total_mom.length())/
                  (initial_total_mom.length() + d_SMALL_NUM);
      double change_total_eng =
                  100.0 * (total_energy - initial_total_eng)/
                  (initial_total_eng + d_SMALL_NUM);
      cout << "Totals: \t mass " << total_mass 
           << " \t\t momentum " << total_momentum 
           << " \t\t energy   " << total_energy << endl;
      cout << "Percent change in total mom.: " << change_total_mom 
           << " \t  total eng: " << change_total_eng << endl;
      
    }  // numPatrches==1

    //__________________________________
    //  Now check to see that momentum and energy
    //  are being conserved during the exchange process
    Vector sum_mom_L_CC     = Vector(0.0, 0.0, 0.0);
    Vector sum_mom_L_ME_CC  = Vector(0.0, 0.0, 0.0);
    double sum_int_eng_L_CC = 0.0;
    double sum_eng_L_ME_CC  = 0.0;
    
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();

      new_dw->get(mom_L_CC,     lb->mom_L_CCLabel,     indx, patch,gn, 0);
      new_dw->get(int_eng_L_CC, lb->int_eng_L_CCLabel, indx, patch,gn, 0);       
      new_dw->get(mom_L_ME_CC,  lb->mom_L_ME_CCLabel,  indx, patch,gn, 0);       
      new_dw->get(eng_L_ME_CC,  lb->eng_L_ME_CCLabel,  indx, patch,gn, 0); 
      
      for (CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
          sum_mom_L_CC     += mom_L_CC[c];     
          sum_mom_L_ME_CC  += mom_L_ME_CC[c];  
          sum_int_eng_L_CC += int_eng_L_CC[c]; 
          sum_eng_L_ME_CC  += eng_L_ME_CC[c];  
      }
    }
    Vector mom_exch_error = sum_mom_L_CC     - sum_mom_L_ME_CC;
    double eng_exch_error = sum_int_eng_L_CC - sum_eng_L_ME_CC;
    if( numPatches == 1 ) {
      cout << "error in momentumExchange "<< mom_exch_error<< endl;
      cout << "error in EnergyExchange   "<< eng_exch_error<< endl;
      cout.setf(ios::scientific, ios::floatfield);
    }
      
    new_dw->put(sumvec_vartype(mom_exch_error), lb->mom_exch_errorLabel);
    new_dw->put(sum_vartype(eng_exch_error),    lb->eng_exch_errorLabel);      
    new_dw->put(sum_vartype(total_mass),        lb->TotalMassLabel);
    new_dw->put(sum_vartype(total_KE),          lb->KineticEnergyLabel);
    new_dw->put(sum_vartype(total_int_eng),     lb->TotalIntEngLabel);
    new_dw->put(sumvec_vartype(total_momentum), lb->CenterOfMassVelocityLabel);
  }  // patch loop
}
