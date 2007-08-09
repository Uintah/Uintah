/*
 *  compare_scalar.cc:
 *
 *   A MMS comparison utility for advection of passive scalar.
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
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Core/OS/Dir.h>
#include <Core/Thread/Thread.h>

#include <sci_values.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <math.h>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

////////////////////////////////////////////////////
// Finds out if two doubles are equal within the given tolerance

bool
is_equal(double num1, double num2, double tol = 1e-10)
{
  return ( (num1-num2) < tol)?true:false;
}

////////////////////////////////////////////////////
// Finds out if two SCIRun::Vectors are equal within the given tolerance

bool
is_equal(Vector num1, Vector num2, double tol = 1e-10)
{
  return ( ( is_equal(num1.x(), num2.x(), tol) ) && ( is_equal(num1.y(), num2.y(), tol) ) && ( is_equal(num1.z(),num2.z(), tol) ) )?true:false;
}


////////////////////////////////////////////////////
// Rounds off the double to the nearest integer

long
iround(double num)
{
  return (long)(num+0.5);
}

/////////////////////////////////////////////////////
// is_int(double) - checks if the double can qualify as an integer

int
is_int(double a, double tol = 1e-10)
{
  long b;
  b = iround(a);
  return (fabs(a-b)<=tol)? 1 : 0;
}

//////////////////////////////////////////////////////
// Arguments

string udaFileName;

void
usage()
{
  fprintf(stderr, "\nUsage: compare_scalar <Uda Directory Name>\n\n" );
  cerr << "Valid options are:\n";
  cerr << "-h[elp]              : This usage information.\n";
  cerr << "-uda <archive file>\n";
  cerr << "-o <output_file_name>\n";
  exit(1);
}


static
void
usage( const std::string & message,
       const std::string& badarg,
       const std::string& progname)
{
   cerr << message << "\n";
   if(badarg != "")
     cerr << "Error parsing argument: " << badarg << '\n';
   cerr << "Usage: " << progname << " [options] <input_file_name>\n\n";
   cerr << "Valid options are:\n";
   cerr << "-h[elp]              : This usage information.\n";
   cerr << "-uda <archive file>\n";
   cerr << "-o <output_file_name>\n";
   exit(1);
}

int
main( int argc, char *argv[] )
{
  string varName;
  FILE *outFile = stdout;

  for(int i=1;i<argc;i++){
    string s=argv[i];
    if( (s == "-help") || (s == "-h") ) {
      usage( "", "", argv[0]);
    } else if(s == "-uda") {
      if(++i == argc){
        usage("You must provide a uda name for -uda",
              s, argv[0]);
      }
      udaFileName = argv[i];
    } else if(s == "-o") {
      if(++i == argc) {
        usage("You must provide an output filename for -o",
              s, argv[0]);
      }
      outFile = fopen(argv[i],"w");
      
      if(!outFile) { // Checking success of file creation
        cerr << "The outputfile cannot be created\n";
        exit (1);
      }
    } else {
      ;
    }
  }

  if (""==udaFileName)
  {
    usage();
  }

  varName = "scalar-f";
  
  try {
    DataArchive* da1 = scinew DataArchive(udaFileName);

    // Sample of how to read data from the DA xml file.
    ProblemSpecReader psr( udaFileName + "/input.xml" );
    ProblemSpecP docTop = psr.readInputFile();
    Vector resolution;
    Vector velocity;
     
    ProblemSpecP PS_Block = ((docTop->findBlock("Models"))->findBlock("Model")
			     ->findBlock("scalar"));
    
    
    if( PS_Block == 0 ) {                                                                                                
      printf("Failed to find Models->Model->scalar in input.xml file.\n");             
      exit(1);                                                                                                        
    }                                                                                                                   
    
    ProblemSpecP MatProp = ((docTop->findBlock("MaterialProperties"))->findBlock("ICE")->findBlock("material")
                            ->findBlock("geom_object"));

    if (0 == MatProp) {
      printf("Failed to find MaterialProperties->ICE->material->geo_object in input.xml file.\n");
      exit(1);
    }

    if( MatProp->get( string("velocity"), velocity ) == 0 ) {
     printf("Failed to find velocity in input.xml file.\n");
     exit(1);
    }

    
    ProblemSpecP GridBlock = ((docTop->findBlock("Grid"))->findBlock("Level")
                            ->findBlock("Box"));

    if( GridBlock == 0 ) {
        printf("Failed to find Grid->Level->Box in input.xml file.\n");
        exit(1);
    }
    if( GridBlock->get( string("resolution"), resolution ) == 0 ) {
     printf("Failed to find resolution in input.xml file.\n");
     exit(1);
    }

//   When done, free up problem spec:
    docTop->releaseDocument();


    vector<int> index;
    vector<double> times;
    da1->queryTimesteps(index, times);

    vector<string> vars;    
    vector<const Uintah::TypeDescription*> types;
    vector< pair<string, const Uintah::TypeDescription*> > vartypes1;

    da1->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());

    vartypes1.resize(vars.size());
//   printf( "Number of vars: %d\n", vars.size() );


    /* *************************************
     * delta_t = t_final - t_initial;
     *  double offset =  (delta_t * vel.x()/ resolution.x()  )  // checking if it is an integer
     * if ( offset != int(offset))
     *     Shout loudly and quit
     *
     * *************************************/

    double t_final;

    t_final = times[times.size()-1];
    Vector offset;
    offset.x(t_final * velocity.x() * resolution.x());  // 1/resolution = delta_x
    offset.y(t_final * velocity.y() * resolution.y());  // We need to divide by delta_x
    offset.z(t_final * velocity.z() * resolution.z());  // so we are multiplying by resolution

    cout<<"resolution"<<resolution<<endl;
    cout<<"velocity"<<velocity<<endl;
    cout<<"Offset:"<<offset<<endl;

    if (!(is_int(offset.x()) && is_int(offset.y()) && is_int(offset.z()) ) )
    {
      cerr<<"The offset is not an integer value"<<offset<<endl;
      exit(1);
    }

    unsigned int loopLowerBound = 0;
    bool initialize_analytical_values = true;


    CCVariable<double> analytic_value;
    CCVariable<Vector> init_vel;
    
    ////////////////////////////
    // Iterate over TIME
    //
    for( unsigned int timeIndex = loopLowerBound; timeIndex < index.size(); timeIndex++ ) {
      printf( "Time Step: %d Phy Time: %lf\n", index[timeIndex], times[timeIndex] );

      GridP grid = da1->queryGrid( timeIndex );

      ////////////////////////////
      // Iterate over the levels
      //
      for( int levIndex = 0; levIndex < grid->numLevels(); levIndex++ ) {
        LevelP level = grid->getLevel(levIndex);
        
        //////////////////////////////
        // Iterate over the variables
        for( unsigned int varIndex = 0; varIndex < vars.size(); varIndex++ ) {
            
          int i=0;
          double total_error=0.0;

	  int vel_var_index = -1;
	  for (int varIdx = 0; varIdx < vars.size();varIdx++) {
	    if (vars[varIdx] == "vel_CC" ) {
	      vel_var_index = varIdx;
	      break;
	    }
	  }
          if( (vars[varIndex] != varName) ) continue;
            

          ////////////////////////////
          // Iterate over the patches
          for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
            ConsecutiveRangeSet matls;
            bool first = true;

            const Patch* patch = *iter;

            if ( first ) {
                matls = da1->queryMaterials( vars[varIndex], patch, timeIndex );
            }
            else if (matls != da1->queryMaterials(vars[varIndex], patch, timeIndex)) {
                cerr << "The material set is not consistent for variable "
                     << vars[varIndex] << " across patches at time " << times[timeIndex] << endl;
                cerr << "Previously was: " << matls << endl;
                cerr << "But on patch " << patch->getID() << ": " 
                     << da1->queryMaterials(vars[varIndex], patch, timeIndex) << "\n";
                exit( 1 );
            }
            first = false;
            
            //////////////////////////////
            // Iterate over the materials
            for(ConsecutiveRangeSet::iterator matlIter = matls.begin();  matlIter != matls.end(); 
                              matlIter++) {
              int matl = *matlIter;

              CCVariable<double> scalarVar;
	      CCVariable<Vector> velocityVar;
	      //          CCVariable<Vector> vectorVar;
	      da1->query(scalarVar, vars[varIndex], matl, patch, timeIndex);
	      da1->query(velocityVar, vars[vel_var_index], matl, patch, timeIndex);
	      if (initialize_analytical_values)
	      {
		da1->query(analytic_value, vars[varIndex], matl, patch, timeIndex);
		cerr<<"vel_var_index:"<<vel_var_index<<endl;
		da1->query(init_vel, vars[vel_var_index], matl, patch, timeIndex );
	      }

	      
              IntVector low, high, size;
	      scalarVar.getSizes(low,high,size);

              double maxDiff = -FLT_MAX,  minDiff = FLT_MAX;
              
              //////////////////////////////
              // Iterate over the cells
              for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++) {
                IntVector cell = *iter;
                double diff;


		int new_x, new_y, new_z;
		
		if (initialize_analytical_values){
		   new_x = ( cell.x() + iround(offset.x()) )%iround(resolution.x());
		   new_y = ( cell.y() + iround(offset.y()) )%iround(resolution.y());
		   new_z = ( cell.z() + iround(offset.z()) )%iround(resolution.z());
		}
		else {
		   new_x = cell.x();
		   new_y = cell.y();
		   new_z = cell.z();
		}
		
		
		IntVector a(new_x, new_y, new_z);
		
		if ( initialize_analytical_values ) {
		  analytic_value[a] = scalarVar[cell];  // This is where the analytical value gets filled
		}
		else {
		  diff = scalarVar[cell] - analytic_value[a];

		  if ( !is_equal(velocityVar[cell], init_vel[cell]) ) {
		    cerr<<"veloctiy fields don't match\n";
		    cerr<<"current velocity"<<velocityVar[cell]<<" init_vel:"<<init_vel[cell]<<endl;
		    cerr<<"Diff:"<<(velocityVar[cell]-init_vel[cell])<<endl;
		    exit(1);
		  }
		  
		  total_error=total_error + (diff*diff);
		  if( diff > maxDiff ) maxDiff = diff;
		  if( diff < minDiff ) minDiff = diff;
		  
		}
//		if (new_y == 50)
//		cerr<<cell<<": " <<scalarVar[cell]<< a <<" :"<< analytic_value[a]<<endl;
                
                i=i+1;
              }
	      printf( "Max diff: %le, Min diff %le\n", maxDiff, minDiff );
            } // end materials iteration
          } // end patch iteration
          
	  cout << "i= " << i << endl << "L2norm of error: " << endl << sqrt(total_error/i) << "\n";

	  if (!initialize_analytical_values)
	    fprintf(outFile, "%d %le\n",i,sqrt(total_error/i)) ;

        } // end variable iteration
      } // end levels iteration
      if (initialize_analytical_values)
	timeIndex = index.size()-2;   // This moves the time index to the last but 2 time-step
                                      // (we need the last but one time-step,
                                      // but the loop incrementer will do that)
      initialize_analytical_values = false;

    } // end time iteration

    
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << '\n';
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }

  return 0;

} // end main()

