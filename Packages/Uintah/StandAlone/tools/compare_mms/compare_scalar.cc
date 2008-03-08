/*
 *  compare_scalar.cc:
 *
 *   A comparison utility for advection of passive scalar.
 *   Reads the initialization scalar profile and advects analytically 
 *   using the velocity and compares it against the last time-step of the
 *   UDA file. 
 *
 *
 *  Written by:
 *   Amjidanutpan Ramanujam (ramanuja_at_cs_dot_utah_dot_edu)
 *   C-SAFE
 *   University of Utah
 *
 *  Borrowed from:
 *   compare_mms.cc written by J. Davison de St. Germain
 *
 *  Copyright (C) 2007 U of U
 */

#include <Packages/Uintah/StandAlone/tools/compare_mms/compare_scalar.h>
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>

#include <iostream>
#include <string>
#include <sstream>
#include <math.h>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

//__________________________________
// Finds out if two doubles are equal within the given tolerance

bool
is_equal(double num1, double num2, double tol = 100 * DBL_EPSILON)
{
  return ( (num1-num2) < tol)?true:false;
}

//__________________________________
// Finds out if two SCIRun::Vectors are equal within the given tolerance

bool
is_equal(Vector num1, Vector num2, double tol = 100 * DBL_EPSILON)
{
  return ( ( is_equal(num1.x(), num2.x(), tol) ) && 
           ( is_equal(num1.y(), num2.y(), tol) ) && 
           ( is_equal(num1.z(), num2.z(), tol) ) )?true:false;
}


//__________________________________
// Rounds off the double to the nearest integer

long
iround(double num)
{
  return (long)(num+0.5);
}

//__________________________________
// is_int(double) - checks if the double can qualify as an integer

int
is_int(double a, double tol = 100*DBL_EPSILON)
{
  long b;
  b = iround(a);
  return (fabs(a-b)<=tol)? 1 : 0;
}

//////////////////////////////////////////////////////
// Arguments

string udaFileName ="";
int verbose;


static
void
usage( const std::string & message,
       const std::string& badarg,
       const std::string& progname)
{
  cerr << message << "\n";
  if(badarg != ""){
    cerr << "Error parsing argument: " << badarg << '\n';
  }
  cerr << "Usage: " << progname << " -uda <archive file>  [options] \n\n";
  cerr << "options are:\n";
  cerr << "-h[elp]    This usage information.\n";
  cerr << "-matl      material index for the velocity and scalar-f. Default is 0.\n";
  cerr << "-o         output_file_name\n";
  cerr << "-v         verbose output\n";
  
  exit(1);
}

//______________________________________________________________________
int
main( int argc, char *argv[] )
{
  // defaults
  string scalarName   = "scalar-f";     
  string velocityName = "vel_CC";
  int matl = 0;                         
  FILE *outFile = stdout;               
  
  // read in commmand line arguments
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if( (s == "-help") || (s == "-h") ) {
      usage( "", "", argv[0]);
    } else if(s == "-uda") {
      if(++i == argc){
        usage("You must provide a uda name for -uda",s, argv[0]);
      }
      udaFileName = argv[i];
    } else if(s == "-matl"){
      matl = atoi(argv[i]);
    }else if(s == "-o") {
      if(++i == argc) {
        usage("You must provide an output filename for -o", s, argv[0]);
      }
      outFile = fopen(argv[i],"w");
      
      if(!outFile) { // Checking success of file creation
        cerr << "The outputfile cannot be created\n";
        exit (1);
      }
    } else if (s == "-v"){
      verbose = 1;
    }else {
      ;
    }
  }

  if (""==udaFileName){
    usage( "", "", ""); 
  }
  
  //__________________________________
  //
  DataArchive* da1 = scinew DataArchive(udaFileName);
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(16);

  vector<int> index;
  vector<double> times;
  da1->queryTimesteps(index, times);

  double t_final   = times[times.size()-1];
  double t_initial = times[0];
  
  // bulletproofing
  if(t_initial != 0.0){ 
    cout<<"ERROR:Compare_scalar: please add <outputInitTimestep/> to the  <DataArchiver> section of the inputfile" << endl;
    exit(1);
  }
  
  //__________________________________
  // translate the initial passive scalar concentration by(t_final * velocity/dx) cells
  CCVariable<double> analytic_value;
  CCVariable<Vector> initial_vel;
  
  int timeIndex = 0;
  GridP grid = da1->queryGrid( timeIndex );
  // initialize 
  for( int levIndex = 0; levIndex < grid->numLevels(); levIndex++ ) {
    LevelP level = grid->getLevel(levIndex);

    for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      Vector  dx = patch->dCell();
      
      CCVariable<double> scalarVar;
      da1->query(analytic_value, scalarName,    matl, patch, timeIndex);
      da1->query(initial_vel,    velocityName,  matl, patch, timeIndex);
      da1->query(scalarVar,      scalarName,    matl, patch, timeIndex);
      
      IntVector patch_l = patch->getLowIndex();
      IntVector patch_h = patch->getHighIndex();
      
      // translate the initial passive scalar concentration by(t_final * velocity/dx) cells
      for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        
        Vector offset;
        offset = Vector(t_final) * (initial_vel[c]/dx);
  
        //bullet proofing
        if (!(is_int(offset.x()) && is_int(offset.y()) && is_int(offset.z()) ) ){
          cout<<"ERROR:Compare_scalar: The quantity (final timestep * velocity/dx) must be an integer:"
              << offset << "\n";
          cout << "final timestep " << t_final << " velocity " << initial_vel[c] << " dx " << dx << endl;
          exit(1);
        }
        
        int new_x, new_y, new_z;
        new_x = ( c.x() + iround(offset.x()) );
        new_y = ( c.y() + iround(offset.y()) );
        new_z = ( c.z() + iround(offset.z()) );
        
        IntVector a(new_x, new_y, new_z);
        
        // don't fall off the edge of a patch
        if(a.asVector() > patch_l.asVector() && a.asVector() < patch_h.asVector() - Vector(1,1,1)){
          analytic_value[a] = scalarVar[c];
        }
      }
    } // end patch iteration
  } // end levels iteration
  
  //__________________________________
  // Examine the passive scalar concentration
  // on the last timestep and compare it against the analytical solution
  timeIndex = index.size() -1;
  grid = da1->queryGrid( timeIndex );
  IntVector c_maxDiff, c_minDiff;
  
  for( int levIndex = 0; levIndex < grid->numLevels(); levIndex++ ) {
    LevelP level = grid->getLevel(levIndex);
    
    int i=0;
    double total_error=0.0;
    double maxDiff = -FLT_MAX;
    double minDiff = FLT_MAX;
    
    for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      
      CCVariable<double> scalarVar;
      CCVariable<Vector> velocityVar;
      da1->query(scalarVar,   scalarName,   matl, patch, timeIndex);
      da1->query(velocityVar, velocityName, matl, patch, timeIndex);
      
      for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        double diff;

        diff = scalarVar[c] - analytic_value[c];
        
        //__________________________________
        // bulletproofing
        if ( !is_equal(velocityVar[c], initial_vel[c]) ) {
          cerr<<"veloctiy fields don't match\n";
          cerr<<"current velocity"<<velocityVar[c]<<" Initial velocity:"<<initial_vel[c]<<endl;
          cerr<<"Diff:"<<(velocityVar[c]-initial_vel[c])<<endl;
          exit(1);
        }
        
        total_error+=diff*diff;
        if( diff > maxDiff ){
          maxDiff = diff;
          c_maxDiff = c;
        }
        if( diff < minDiff ){
          minDiff = diff;
          c_minDiff = c;
        }
        i=i+1;
      }
    } // end patch iteration
    if(verbose){
      cout << "\t\tTime Step: " << index[timeIndex] << " Physical Time: " << times[timeIndex] << endl;;
      cout << "\t\tMax diff: "<< maxDiff << " " << c_maxDiff << endl;
      cout << "\t\tMin_diff: "<< minDiff << " " << c_minDiff << endl; 
      cout << "\t\tNumber of cells:  " << i << endl;
      cout << "\t\tL2 norm of error: " << sqrt(total_error/i) << "\n";
    }
    fprintf(outFile, "%16.16le\n",sqrt(total_error/i) );
  } // end levels iteration
  
  return 0;
} 

