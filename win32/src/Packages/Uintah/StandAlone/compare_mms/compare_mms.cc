/*
 *  compare_mms.cc:
 *
 *     Compares the data in a Uintah Data Archive (UDA) with the
 *     results of an analytical solution.  
 *
 *  Written by:
 *   J. Davison de St. Germain
 *   C-SAFE
 *   University of Utah
 *   Jun 9 2005
 *
 *  Copyright (C) 2005 U of U
 */

#include <Packages/Uintah/StandAlone/compare_mms/MMS.h>
#include <Packages/Uintah/StandAlone/compare_mms/MMS1.h>

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

#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
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

//////////////////////////////////////////////////////
// Analytical solution variable

MMS * mms;

//////////////////////////////////////////////////////
// Arguments

string udaFileName;

void
usage()
{
  printf( "\nUsage: compare_mss <Uda Directory Name>\n\n" );
  exit(1);
}

void
parse_args( int argc, char *argv[] )
{
  if( argc != 2 ) {
    usage();
  }

  // From args, determine which mms to use... now just hardcoding to 1.
  mms = new MMS1();

  udaFileName = argv[1];
}

int
main( int argc, char *argv[] )
{
  parse_args( argc, argv );

  try {
    DataArchive* da1 = scinew DataArchive(udaFileName);

//     // Sample of how to read data from the DA xml file.
//     ProblemSpecReader psr( udaFileName + "/input.xml" );
//     ProblemSpecP docTop = psr.readInputFile();
//     double dyVis;
//     int    sos;

//     ProblemSpecP matBlock = ((docTop->findBlock("MaterialProperties"))->findBlock("ICE"))->findBlock("material");

//     if( matBlock == 0 )
//       {
//         printf("Failed to find MaterialProperties->ICE->material in input.xml file.\n");
//         exit(1);
//       }

//     if( matBlock->get( string("dynamic_viscosity"), dyVis ) == 0 )
//       {
//         printf("Failed to find dynamic_viscosity in input.xml file.\n");
//         exit(1);
//       }

//     if( matBlock->get( string("speed_of_sound"), sos ) == 0 )
//       {
//         printf("Failed to find speed_of_sound in input.xml file.\n");
//         exit(1);
//       }

//     printf( "read dynamic viscosity value of %lf\nSpeed of sound: %d\n", dyVis, sos );
//     // When done, free up problem spec:
//     docTop->releaseDocument();


    vector<int> index;
    vector<double> times;
    da1->queryTimesteps(index, times);

    vector<string> vars;    
    vector<const Uintah::TypeDescription*> types;
    vector< pair<string, const Uintah::TypeDescription*> > vartypes1;

    da1->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());

    vartypes1.resize(vars.size());
    printf( "Number of vars: %d\n", vars.size() );

    ////////////////////////////
    // Iterate over TIME
    //
    for( unsigned int timeIndex = 0; timeIndex < index.size(); timeIndex++ )
      {
        printf( "here: %d, %lf\n", index[timeIndex], times[timeIndex] );

        GridP grid = da1->queryGrid( times[timeIndex] );

        ////////////////////////////
        // Iterate over the levels
        //
        for( int levIndex = 0; levIndex < grid->numLevels(); levIndex++ )
          {
            printf( "Looking at level %d.\n", levIndex );
            LevelP level = grid->getLevel(levIndex);
            
            ////////////////////////////
            // Iterate over the patches
            for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) 
              {
                ConsecutiveRangeSet matls;
                bool first = true;

                const Patch* patch = *iter;

                printf( "Looking at patch:\n");
                cout << *patch << "\n";

                //////////////////////////////
                // Iterate over the variables
                for( unsigned int varIndex = 0; varIndex < vars.size(); varIndex++ )
                //for( unsigned int varIndex = 0; varIndex < 1; varIndex++ ) // JUST LOOK AT THE FIRST ONE FOR NOW FOR TESTING!!!
                  {
                    printf("variable %s is a %s\n", vars[varIndex].c_str(), types[varIndex]->getName().c_str() );

                    if( vars[varIndex] != "press_CC" ) continue;

                    if( first ) {
                      matls = da1->queryMaterials( vars[varIndex], patch, times[timeIndex] );
                    }
                    else if (matls != da1->queryMaterials(vars[varIndex], patch, times[timeIndex])) {
                      cerr << "The material set is not consistent for variable "
                           << vars[varIndex] << " across patches at time " << times[timeIndex] << endl;
                      cerr << "Previously was: " << matls << endl;
                      cerr << "But on patch " << patch->getID() << ": " << da1->queryMaterials(vars[varIndex], patch, times[timeIndex]) 
                           << "\n";
                      exit( 1 );
                    }
                    first = false;
                    
                    //////////////////////////////
                    // Iterate over the materials
                    for(ConsecutiveRangeSet::iterator matlIter = matls.begin();  matlIter != matls.end(); matlIter++)
                      {
                        int matl = *matlIter;
                        printf("working on matl: %d\n", matl);

                        // know that the first one in the test data set is pressure;
                        CCVariable<double> pressure;
                    
                        da1->query(pressure, vars[varIndex], matl, patch, times[timeIndex]);

                        IntVector low, high, size;
                        pressure.getSizes(low,high,size);
                        cout << "Low:  " << low << "\n";
                        cout << "High: " << high << "\n";
                        cout << "Size: " << size << "\n";

                        float maxDiff = -FLT_MAX, minDiff = FLT_MAX;
                        //////////////////////////////
                        // Iterate over the cells
                        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++) 
                          {
                            IntVector cell = *iter;
                            cout << cell << "\n";
                            float analytic_value = mms->pressure( cell[0], cell[1], times[timeIndex] );
                            float diff = pressure[cell] - analytic_value;

                            if( diff > maxDiff ) maxDiff = diff;
                            if( diff < minDiff ) minDiff = diff;
                            printf( "UDA value: %f, Analytic Value: %f.  Diff: %f\n", pressure[cell], analytic_value, diff );
                          }
                        printf( "Max diff: %f, Min diff %f\n", maxDiff, minDiff );

                      } // end materials iteration

                  } // end variable iteration
                
              } // end patch iteration

          } // end levels iteration

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

