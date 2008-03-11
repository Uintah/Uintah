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
#include <Packages/Uintah/StandAlone/tools/compare_mms/MMS.h>
#include <Packages/Uintah/StandAlone/tools/compare_mms/ExpMMS.h>
#include <Packages/Uintah/StandAlone/tools/compare_mms/LinearMMS.h>
#include <Packages/Uintah/StandAlone/tools/compare_mms/SineMMS.h>

#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
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
  ProblemSpecReader psr( udaFileName + "/input.xml" );
  ProblemSpecP docTop = psr.readInputFile();
  double A;
  double dyVis;
  double p_ref;
  double cu, cv, cw, cp;
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
                                                                                                                        
    ProblemSpecP matBlock = ((docTop->findBlock("MaterialProperties"))->findBlock("ICE"))->findBlock("material");       

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
    printf( "Time Step: %d Phy Time: %lf\n", index[timeIndex], times[timeIndex] );

    GridP grid = da1->queryGrid( timeIndex );

    //__________________________________
    // Iterate over the levels
    for( int levIndex = 0; levIndex < grid->numLevels(); levIndex++ ) {
      printf( "Looking at level %d.\n", levIndex );
      LevelP level = grid->getLevel(levIndex);
      int i=0;
      double total_error=0.0, total_errorU=0.0, total_errorV=0.0;
      cout<< "varName: " << "**************** "<<varName<<" ****************"<<endl;

      //__________________________________
      // Iterate over the patches
      for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
        const Patch* patch = *iter;

        printf( "Looking at patch:\n");
        cout << *patch << "\n";
        CCVariable<double> scalarVar;
        CCVariable<Vector> vectorVar;
        
        if (varName=="pressurePS"||varName=="press_CC"||varName=="newCCUVelocity"||varName=="newCCVVelocity") {
          da1->query(scalarVar, varName, d_matl, patch, timeIndex);
        }
        if (varName=="vel_CC"||varName=="newCCVelocity") {
          da1->query(vectorVar, varName, d_matl, patch, timeIndex);
        }

        IntVector low, high, size;
        if (varName=="pressurePS"||varName=="press_CC"||varName=="newCCUVelocity"||varName=="newCCVVelocity") {
          scalarVar.getSizes(low,high,size);
        }
        if (varName=="vel_CC"||varName=="newCCVelocity") {
          vectorVar.getSizes(low,high,size);
        }
        cout << "Low:      " << low << "\n";
        cout << "High:     " << high << "\n";
        cout << "Size:     " << size << "\n";

        double maxDiff = -FLT_MAX,  minDiff = FLT_MAX;
        double maxDiffU = -FLT_MAX, minDiffU = FLT_MAX;
        double maxDiffV = -FLT_MAX, minDiffV = FLT_MAX;
        
        //////////////////////////////
        // Iterate over the cells
        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++) {
          IntVector c = *iter;
          
          double analytic_value;
          double analytic_valueU;
          double analytic_valueV;
          double diff, diffU, diffV;
                                  
          Point pt = patch->cellPosition(c);
          cout <<"Cell:     "<< c << "\n";
          cout <<"Position: "<< pt <<endl;
          double x_pos = pt.x();
          double y_pos = pt.y();
          double z_pos = pt.z();

          //__________________________________
          //
          if (varName=="pressurePS"||varName=="press_CC") {
            analytic_value = mms->pressure( x_pos, y_pos, z_pos, times[timeIndex] );
            diff = scalarVar[c] - analytic_value;
            total_error+=diff*diff;
            if( diff > maxDiff ) maxDiff = diff;
            if( diff < minDiff ) minDiff = diff;

            printf( "UDA value: %le, Analytic Value: %le.  Diff: %le, fabs(Diff): %le\n", 
                     scalarVar[c], analytic_value, diff, fabs(diff) );
          }
          
          //__________________________________
          //
          if (varName=="newCCUVelocity") {
            analytic_value = mms->uVelocity( x_pos, y_pos, z_pos, times[timeIndex] );
              
            diff = scalarVar[c] - analytic_value;
            total_error+=diff*diff;
            if( diff > maxDiff ) maxDiff = diff;
            if( diff < minDiff ) minDiff = diff;
            
            printf( "UDA value: %f, Analytic Value: %le.  Diff: %le, fabs(Diff): %le\n", 
                      scalarVar[c], analytic_value, diff, fabs(diff) );
          }
          
          //__________________________________
          //
          if (varName=="newCCVVelocity") {
            analytic_value = mms->vVelocity( x_pos, y_pos, z_pos, times[timeIndex] );
              
            diff = scalarVar[c] - analytic_value;
            total_error+=diff*diff;
            
            if( diff > maxDiff ) maxDiff = diff;
            if( diff < minDiff ) minDiff = diff;
            
            printf( "UDA value: %le, Analytic Value: %le.  Diff: %le, fabs(Diff): %le\n", 
                      scalarVar[c], analytic_value, diff, fabs(diff) );
          }
          
          //__________________________________
          //
          if (varName=="vel_CC"||varName=="newCCVelocity") {
            analytic_valueU = mms->uVelocity( x_pos, y_pos, z_pos, times[timeIndex] );
            analytic_valueV = mms->vVelocity( x_pos, y_pos, z_pos, times[timeIndex] );

            diffU = vectorVar[c].x() - analytic_valueU;
            diffV = vectorVar[c].y() - analytic_valueV;
            total_errorU+=diffU*diffU;
            total_errorV+=diffV*diffV;
            if( diffU > maxDiffU ) maxDiffU = diffU;
            if( diffV > maxDiffV ) maxDiffV = diffV;
            if( diffU < minDiffU ) minDiffU = diffU;
            if( diffV < minDiffV ) minDiffV = diffV;
            
            printf( "UDA value: %le, Analytic Value: %le.  Diff: %le, fabs(Diff): %le\n", 
                    vectorVar[c].x(), analytic_valueU, diffU, fabs(diffU) );
            printf( "UDA value: %le, Analytic Value: %le.  Diff: %le, fabs(Diff): %le\n", 
                    vectorVar[c].y(), analytic_valueV, diffV, fabs(diffV) ); 
          } 
          i=i+1;
        }  // cell iterator
        
        if(varName=="pressurePS"||varName=="press_CC"||varName=="newCCUVelocity"||varName=="newCCVVelocity") {
          printf( "Max diff: %le, Min diff %le\n", maxDiff, minDiff );
        }
        if(varName=="vel_CC"||varName=="newCCVelocity") {
          printf( "MaxU diff: %le, MaxV diff: %le, MinU diff %le, MinV diff %le\n", maxDiffU, maxDiffV, 
                     minDiffU, minDiffV);
        }
      } // end patch iteration
      
      if (varName=="pressurePS"||varName=="press_CC"||varName=="newCCUVelocity"||varName=="newCCVVelocity") {
        cout << "i= " << i << endl << "L2norm of error: " << endl << sqrt(total_error/i) << "\n";
      }
      if (varName=="vel_CC"||varName=="newCCVelocity") {
        cout << "i= " << i << ", L2norm of error= " << sqrt(total_errorU/i) << " " << sqrt(total_errorV/i) << "\n";
      }

      if(varName=="pressurePS"||varName=="press_CC"||varName=="newCCUVelocity"||varName=="newCCVVelocity") {
        fprintf(outFile, "%le\n",sqrt(total_error/i)) ;
      }
      if(varName=="vel_CC"||varName=="newCCVelocity") {
        fprintf(outFile,"%le,%le\n", sqrt(total_errorU/i), sqrt(total_errorV/i));
      }
    } // end levels iteration
  } // end time iteration
} // end main()

