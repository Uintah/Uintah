/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
 */
#include <StandAlone/tools/compare_mms/MMS.h>
#include <StandAlone/tools/compare_mms/ExpMMS.h>
#include <StandAlone/tools/compare_mms/LinearMMS.h>
#include <StandAlone/tools/compare_mms/SineMMS.h>

#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>

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
#include <cmath>
#include <cstdio>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

//////////////////////////////////////////////////////
// Analytical solution variable

MMS * mms;

//////////////////////////////////////////////////////
// Arguments

string udaFileName;


static
void
usage( const std::string& message,
       const std::string& badarg,
       const std::string& progname)
{
   cerr << message << "\n";
   if(badarg != "")
     cerr << "Error parsing argument: " << badarg << '\n';
   cerr << "Usage: " << progname << " [options] <input_file_name>\n\n";
   cerr << "Valid options are:\n";
   cerr << "-h[elp]              : This usage information.\n";
   cerr << "-ice                 : \n";
   cerr << "-arches              : \n";
   cerr << "-mms                 :<linear, sine or exp> \n";
   cerr << "-uda                 :\n";
   cerr << "-v,                  :<variable name>\n";
   cerr << "-verbose             : verbose output \n";
   cerr << "-matl                : material index. Default is 0.\n";
   cerr << "-o                   :<output_file_name>\n";
   cerr << "-L                   :Compute global error for the last time step only\n";
   exit(1);
}

//__________________________________
//
int
main( int argc, char *argv[] )
{
  string varName;
  string whichMMS;
  bool do_arches=false;
  bool do_ice   =false;
  FILE *outFile = stdout;
  bool last_time_step = false;
  int d_matl = 0;
  bool d_verbose = false;

  for(int i=1;i<argc;i++){
    string s=argv[i];
    if( (s == "-help") || (s == "-h") ) {
      usage( "", "", argv[0]);
    } else if(s == "-arches"){
      do_arches=true;
    } else if(s == "-ice"){
      do_ice=true;
    } else if(s == "-mms") {
      if(++i == argc){
        usage("You must provide a mms name for -mms",
              s, argv[0]);
      }
      whichMMS = argv[i];
    } else if(s == "-uda") {
      if(++i == argc){
        usage("You must provide a uda name for -uda",
              s, argv[0]);
      }
      udaFileName = argv[i];
    } else if(s == "-matl"){
      d_matl = atoi(argv[i]);
    }else if(s == "-verbose"){
      d_verbose = true;
    } else if(s == "-v") {
      if(++i == argc){
        usage("You must provide a variable name for -v",
              s, argv[0]);
      }
      varName = argv[i];
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
      
    } else if(s == "-L") {
      last_time_step = true;
    }
    else {
            ;
    }
  }
  // Check for valid argument combinations
  if (do_ice && do_arches) {
    usage("ICE and Arches do not work together", "", argv[0]);
  }

  if (!(do_arches || do_ice)) {
    usage("You must specify -arches or -ice", "", argv[0]);
  }
  
  DataArchive* da1 = scinew DataArchive(udaFileName);
  ProblemSpecP docTop = ProblemSpecReader().readInputFile( udaFileName + "/input.xml" );
  double A;
  double dyVis;
  double p_ref;
  double cu = -1, cv = -1, cw = -1, cp = -1;
  Vector resolution;
   
  //__________________________________
  //  ICE
  if(do_ice) {
    ProblemSpecP cfdBlock = ((docTop->findBlock("CFD"))->findBlock("ICE")
                           ->findBlock("customInitialization"))
                           ->findBlock("manufacturedSolution");

    if( cfdBlock == 0 ) {                                                                                                
      printf("Failed to find CFD->ICE->customInitialization->manufacturedSolution in input.xml file.\n");             
      exit(1);                                                                                                        
    }                                                                                                                   
    if(cfdBlock->get( string("A"), A ) == 0 ) {                                                                         
      printf("Failed to find A in input.xml file.\n");                                                                 
      exit(1);                                                                                                         
    }                                                                                                                   
                                                                                                                        
    ProblemSpecP phyConsBlock = (docTop->findBlock("PhysicalConstants"));                                               

    if(phyConsBlock == 0 ) {                                                                                            
      printf("Failed to find PhysicalConstants in input.xml file.\n");                                                 
      exit(1);                                                                                                         
    }                                                                                                                   
    if(phyConsBlock->get( string("reference_pressure"), p_ref ) == 0 ) {                                                
      printf("Failed to find pressure in input.xml file.\n");                                                          
      exit(1);                                                                                                         
    }                                                                                                                   
                                                                                                                        
    ProblemSpecP matBlock = ((docTop->findBlockWithOutAttribute("MaterialProperties"))->findBlock("ICE"))->findBlock("material");       

    if(matBlock == 0 ) {                                                                                                
      printf("Failed to find MaterialProperties->ICE->material in input.xml file.\n");                                 
      exit(1);                                                                                                         
    }                                                                                                                   

    if(matBlock->get( string("dynamic_viscosity"), dyVis ) == 0 ) {                                                     
      printf("Failed to find dynamic_viscosity in input.xml file.\n");                                                 
      exit(1);                                                                                                         
    }                                                                                                                   

    if(whichMMS=="linear") {                                                                                            
      mms = new LinearMMS(cu, cv, cw, cp, p_ref);                                                                      
    }else if(whichMMS=="sine") {                                                                                         
      mms = new SineMMS(A, dyVis, p_ref);                                                                              
    }else if(whichMMS=="exp") {                                                                                          
      mms = new ExpMMS(A, dyVis, p_ref);                                                                                              
    }else {                                                                                                              
      cout << "current MMS not supported\n";                                                                            
      exit(1);                                                                                                          
    }
  }
  //__________________________________
  //  Arches
  if(do_arches) {
    p_ref=0.0;
    ProblemSpecP mmsBlock = ((docTop->findBlock("CFD"))->findBlock("ARCHES")
                         ->findBlock("MMS"));

    if( mmsBlock == 0 ) {
      printf("Failed to find CFD->ARCHES->MMS in input.xml file.\n");
      exit(1);
    }

    if(mmsBlock->get( string("whichMMS"), whichMMS ) == 0 ) {
      printf("Failed to find A in input.xml file.\n");
      exit(1);
    }
    

    if(whichMMS=="linearMMS") {
      ProblemSpecP mmsSubBlock = mmsBlock->findBlock("linearMMS");
      if( mmsSubBlock == 0 ) {
        printf("Failed to find CFD->ARCHES->MMS->linearMMS in input.xml file.\n");
        exit(1);
      }
      
      if(mmsSubBlock->get( string("cu"), cu ) == 0 ) {
        printf("Failed to find cu in input.xml file.\n");
        exit(1);
      }
      if(mmsSubBlock->get( string("cv"), cv ) == 0 ) {
        printf("Failed to find cv in input.xml file.\n");
        exit(1);
      }
      if(mmsSubBlock->get( string("cw"), cw ) == 0 ) {
        printf("Failed to find cw in input.xml file.\n");
        exit(1);
      }
      if(mmsSubBlock->get( string("cp"), cw ) == 0 ) {
        printf("Failed to find cw in input.xml file.\n");
         exit(1);
      }
      mms = new LinearMMS(cu, cv, cw, cp, p_ref);
    }
    else if(whichMMS=="sineMMS") { 
      ProblemSpecP mmsSubBlock = mmsBlock->findBlock("sineMMS");
      if( mmsSubBlock == 0 ) {
        printf("Failed to find CFD->ARCHES->MMS->sineMMS in input.xml file.\n");
        exit(1);
      }
      
      if(mmsSubBlock->get( string("amplitude"), A ) == 0 ) {
        printf("Failed to find A in input.xml file.\n");
        exit(1);
      }
      if(mmsSubBlock->get( string("viscosity"), dyVis ) == 0 ) {
        printf("Failed to find viscosity in input.xml file.\n");
        exit(1);
      }
      p_ref=0.0;
      mms = new SineMMS(A, dyVis, p_ref);
    }else if(whichMMS=="expMMS") { 
      mms = new ExpMMS(A, dyVis, p_ref);
    }else {  
      cout << "current MMS not supported\n";
      exit(1);
    }
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

  vector<int> index;
  vector<double> times;
  da1->queryTimesteps(index, times);
  
  cout <<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<endl;
  cout << "MMS Type (whichMMS)        :" << whichMMS << endl;
  cout << "dynamic viscosity          :"<<dyVis<<endl;
  cout << "A (amplitude)              :"<<A<<endl;
  cout << "Reference Pressure (p_ref) :"<< p_ref <<endl;
  cout << "Resolution                 :" << resolution << endl;
  cout <<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<endl;

  unsigned int loopLowerBound;
  
  if (true == last_time_step) {
    loopLowerBound = index.size() -1 ;
  } else {
    loopLowerBound = 0;
  }

  //__________________________________
  // Iterate over TIME
  for( unsigned int timeIndex = loopLowerBound; timeIndex < index.size(); timeIndex++ ) {
    printf( "Timestep: %d Physical Time: %lf\n", index[timeIndex], times[timeIndex] );

    GridP grid = da1->queryGrid( timeIndex );

    //__________________________________
    // Iterate over the levels
    for( int levIndex = 0; levIndex < grid->numLevels(); levIndex++ ) {
      printf( "Level %d.\n", levIndex );
      LevelP level = grid->getLevel(levIndex);
      int i=0;
      double total_error_D = 0.0;
      Vector total_error_V = Vector(0.0);
      
      double maxDiff_D = -(1.0/FLT_MAX);
      double minDiff_D =  FLT_MAX;
      Vector maxDiff_V = Vector(-1.0/FLT_MAX);
      Vector minDiff_V = Vector(FLT_MAX);  
      
      IntVector c_maxDiff = IntVector(0,0,0);
      IntVector c_minDiff = IntVector(0,0,0);    
      
      cout<< "**************** "<<varName<<" ****************"<<endl;

      //__________________________________
      // Iterate over the patches
      for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
        const Patch* patch = *iter;

        printf( "Looking at patch:\n");
        cout << *patch << "\n";
        CCVariable<double> scalarVar;
        CCVariable<Vector> vectorVar;
        
        if (varName=="pressurePS"||varName=="press_CC" || varName=="press_equil_CC") {
          da1->query(scalarVar, varName, d_matl, patch, timeIndex);
        }
        if (varName=="vel_CC"||varName=="newCCVelocity") {
          da1->query(vectorVar, varName, d_matl, patch, timeIndex);
        }

        //__________________________________
        // Iterate over the cells
        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++) {
          IntVector c = *iter;
                                  
          Point pt = patch->cellPosition(c);
          if(d_verbose){
            cout <<"Cell:     "<< c <<"  Position: "<< pt <<endl;
          }
          double x_pos = pt.x();
          double y_pos = pt.y();
          double z_pos = pt.z();

          //__________________________________
          //
          if (varName=="pressurePS"||varName=="press_CC" || varName=="press_equil_CC") {
                
            double analytic_value;
            double diff;
            
            analytic_value = mms->pressure( x_pos, y_pos, z_pos, times[timeIndex] );
            diff = fabs(scalarVar[c]) - fabs(analytic_value);
            total_error_D +=diff*diff;
            
            if( diff > maxDiff_D ){
              maxDiff_D = diff;
              c_maxDiff = c;
            }
            if( diff < minDiff_D ){
              minDiff_D = diff;
              c_minDiff = c;
            }
            if(d_verbose){
              cout<< c << " uda " << scalarVar[c] << " Analytic Value: " << analytic_value << " Diff: " <<diff<<endl;
            }
          }
          
          //__________________________________
          //
          if (varName=="vel_CC"||varName=="newCCVelocity") {
          
            Vector analytic_value= Vector(0,0,0);
            Vector diff = Vector(0,0,0);
            
            analytic_value.x(mms->uVelocity( x_pos, y_pos, z_pos, times[timeIndex] ));
            analytic_value.y(mms->vVelocity( x_pos, y_pos, z_pos, times[timeIndex] ));
            
            diff = Abs(vectorVar[c]) - Abs(analytic_value);
            total_error_V +=diff*diff;
            
            if( diff.length() > maxDiff_V.length() ){
              maxDiff_V = diff;
              c_maxDiff = c;
            }
            if( diff.length() < minDiff_V.length() ){
              minDiff_V = diff;
              c_minDiff = c;
            }
            
            if(d_verbose){
              cout<< c << " uda " << vectorVar[c] << " Analytic Value: " << analytic_value << " Diff: " <<diff<<endl;
            } 
          } 
          i=i+1;
        }  // cell iterator
      } // end patch iteration

      if(varName=="pressurePS"||varName=="press_CC" || varName=="press_equil_CC") {
        cout << " Max. Diff: " << c_maxDiff << " "<< maxDiff_D << endl;
        cout << " Min. Diff: " << c_minDiff << " "<< minDiff_D << endl;
      }
      if(varName=="vel_CC"||varName=="newCCVelocity") {
        cout << " Max. Diff: " << c_maxDiff << " "<< maxDiff_V << endl;
        cout << " Min. Diff: " << c_minDiff << " "<< minDiff_V << endl;
      }
      
      if (varName=="pressurePS"||varName=="press_CC"|| varName=="press_equil_CC") {
        cout << "i= " << i << endl;
        cout << "L2norm of error: " << sqrt(total_error_D/i) << endl;
        fprintf(outFile, "%le\n",sqrt(total_error_D/double(i))) ;
      }
      if (varName=="vel_CC"||varName=="newCCVelocity") {
        cout << "i= " << i << ", L2norm of error= " << sqrt(total_error_V.length()/double(i)) << "\n";
        fprintf(outFile,"%le\n", sqrt(total_error_V.length()/double(i)) );
      }
    } // end levels iteration
  } // end time iteration
} // end main()

