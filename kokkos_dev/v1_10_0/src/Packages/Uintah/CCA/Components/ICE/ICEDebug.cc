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
    if ( !d_dbgGnuPlot) {       
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
    if (!d_dbgGnuPlot) {   
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
      if( !d_dbgGnuPlot) {
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
    if( !d_dbgGnuPlot) {    
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
    if( !d_dbgGnuPlot) {
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
    if( !d_dbgGnuPlot) {
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
void    ICE::printStencil( int matl,
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
  int code = mkdir( path.c_str(), 0777 );
  
  path = udaDir + "/" + dirName + "/" + DW.str();
  code = mkdir( path.c_str(), 0777 );
  
  path = udaDir + "/" + dirName + "/" + DW.str() + "/" + patchDir;
  code = mkdir( path.c_str(), 0777 );
  
  if (matDir != "") { 
    path = udaDir + "/" + dirName + "/" + DW.str() + "/" + 
           patchDir + "/" + matDir ;
    code = mkdir( path.c_str(), 0777 );
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
  
  int numICEmatls = d_sharedState->getNumICEMatls();
  int flag = -9;
  double mass;
  vector<Vector> mat_mom_xyz(numICEmatls,Vector(0.,0.,0.));
  vector<double> mat_mass(numICEmatls,0.);
  vector<double> mat_total_mom(numICEmatls,0.);
  vector<double> mat_total_eng(numICEmatls,0.);
  vector<double> mat_int_eng(numICEmatls,0.);
  vector<double> mat_KE(numICEmatls,0.);
  Vector total_mom_xyz(0.0, 0.0, 0.0);
  
  double total_momentum = 0.0;
  double total_energy   = 0.0;
  double total_mass     = 0.0;
  double total_KE       = 0.0;
  double total_int_eng  = 0.0; 
  
  static double initial_total_eng = 0.0;
  static double initial_total_mom = 0.0;
  static int n_passes;
  
  //__________________________________
  //  Loop over all the patches
  for(int p=0; p<patches->size(); p++)  {
    const Patch* patch = patches->get(p);
    cout << "Doing printConservedQuantities on patch " << patch->getID()
     << "\t\t ICE" << endl;
    constCCVariable<Vector> vel_CC;
    constCCVariable<double> rho_CC;
    constCCVariable<double> Temp_CC;
    constCCVariable<double> delP_Dilatate;
    Vector dx       = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    new_dw->get(delP_Dilatate,lb->delP_DilatateLabel, 0, patch,Ghost::None, 0);
    
    //__________________________________
    // Loop over all the ICE matls
    for (int m = 0; m < numICEmatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      new_dw->get(vel_CC, lb->vel_CCLabel, indx, patch,  Ghost::None, 0);
      new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch,  Ghost::None, 0);
      new_dw->get(Temp_CC,lb->temp_CCLabel,indx, patch,  Ghost::None, 0);
      double cv = ice_matl->getSpecificHeat();   
      
      //__________________________________
      // Accumulate the momenta and energy
      for (CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
       mass            = rho_CC[*iter] * cell_vol;
       mat_mom_xyz[m] += vel_CC[*iter]*rho_CC[*iter] * mass;
       double vel_sq = vel_CC[*iter].length() * vel_CC[*iter].length();
       mat_KE[m]      += 0.5 * mass * vel_sq;
       mat_int_eng[m] += mass * cv * Temp_CC[*iter];
       mat_mass[m]    += mass;
      }
    }  // numICEmatls loop

    if (switchTestConservation) {
      //__________________________________
      // This grossness checks to see if delPress
      // near a ghost cell is > 0  
      IntVector low, hi;
      
      low = delP_Dilatate.getLowIndex();
      hi  = delP_Dilatate.getHighIndex();
      // x_plus
      for (int j = low.y(); j<hi.y(); j++) {
       for (int k = low.z(); k<hi.z(); k++) {
         if( fabs(delP_Dilatate[IntVector(hi.x()-2,j,k)]) > 0.0 )  {
           flag = 1;
         }
       }
      }
      // x_minus
      for (int j = low.y(); j<hi.y(); j++) {
       for (int k = low.z(); k<hi.z(); k++) {
         if( fabs(delP_Dilatate[IntVector(low.x()+1,j,k)]) > 0.0 )  {
           flag = 1;
         }
       }
      }
      // y_plus
      for (int i = low.x(); i<hi.x(); i++) {
       for (int k = low.z(); k<hi.z(); k++) {
         if( fabs(delP_Dilatate[IntVector(i,hi.y()-2,k)]) > 0.0 )  {
           flag = 1;
         }
       }
      }
      // y_minus
      for (int i = low.x(); i<hi.x(); i++) {
       for (int k = low.z(); k<hi.z(); k++) {
         if( fabs(delP_Dilatate[IntVector(i,low.y()+1,k)]) > 0.0 )  {
           flag = 1;
         }
       }
      }
      // z_plus
      for (int i = low.x(); i<hi.x(); i++) {
       for (int j = low.y(); j<hi.y(); j++) {
         if( fabs(delP_Dilatate[IntVector(i,j,hi.z()-2)]) > 0.0 )   {
           flag = 1;
         }
       }
      }
      // z_minus
      for (int i = low.x(); i<hi.x(); i++) {
       for (int j = low.y(); j<hi.y(); j++) {
         if( fabs(delP_Dilatate[IntVector(i,j,low.z()+1)]) > 0.0 )   {
           flag = 1;
         }
       }
      }
    } // end switchTestConservation
  }  // patch loop
  
  //__________________________________
  //  Now compute totals and the change in quantities
  for (int m = 0; m < numICEmatls; m++ ) {
    mat_total_mom[m]= mat_mom_xyz[m].x() + mat_mom_xyz[m].y() + mat_mom_xyz[m].z();
    mat_total_eng[m]= mat_int_eng[m] + mat_KE[m];
    total_momentum += mat_total_mom[m];
    total_energy   += mat_total_eng[m];
    total_KE       += mat_KE[m];
    total_int_eng  += mat_int_eng[m];
    total_mass     += mat_mass[m];
    total_mom_xyz  += mat_mom_xyz[m];
    if ( n_passes < numICEmatls) {
      initial_total_eng += mat_total_eng[m];
      initial_total_mom += mat_total_mom[m];
      n_passes ++;
    } 
    
    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(4);
    cerr << m << "Fluid mass " <<  mat_mass[m] << "\n";
    cerr.setf(ios::fixed,ios::floatfield);
    cerr << m << "Fluid momentum[ " << mat_mom_xyz[m].x() << ", " << 
      mat_mom_xyz[m].y() << ", " << mat_mom_xyz[m].z() << "]\t";
    cerr << "Components Sum: " << mat_total_mom[m] << "\n";
    cerr.setf(ios::scientific,ios::floatfield);
    cerr << m << "Fluid eng[internal " << mat_int_eng[m] <<  ", Kinetic: " 
        << mat_KE[m] << "]: " << mat_total_eng[m] << "\n";
  }
  double change_total_mom =
              100.0 * (total_momentum - initial_total_mom)/
              (initial_total_mom + d_SMALL_NUM);
  double change_total_eng =
              100.0 * (total_energy - initial_total_eng)/
              (initial_total_eng + d_SMALL_NUM);

  cerr.setf(ios::scientific, ios::floatfield);
  cerr.precision(4);
  cerr << "Totals: \t mass " << total_mass << " \t\tmomentum " << 
    total_momentum << " \t\t energy " << total_energy << "\n";
  cerr.setf(ios::fixed,ios::floatfield);
  cerr << "Percent change in total fluid mom.: " << change_total_mom <<
    " \t fluid total eng: " << change_total_eng << "\n";
  cerr.setf(ios::scientific, ios::floatfield);

  if (flag == 1)  {
    cout<< " D E L P R E S S   >   0   O N   B O U N D A R Y"<<endl;
    cout<< "******* N O   L O N G E R   C O N S E R V I N G *******\n"<<endl;
  }
  new_dw->put(sum_vartype(total_mass),      lb->TotalMassLabel);
  new_dw->put(sum_vartype(total_KE),        lb->KineticEnergyLabel);
  new_dw->put(sum_vartype(total_int_eng),   lb->TotalIntEngLabel);
  new_dw->put(sumvec_vartype(total_mom_xyz),  lb->CenterOfMassVelocityLabel);
}
