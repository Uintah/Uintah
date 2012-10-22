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

#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/GridP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <limits.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>

using namespace Uintah;
using namespace std;


/*-----------------------------------------------------------------------------------------
This prepocessing tool is used to create a separate pts file for each material and patch
after analyzing an image raw file. This was designed for raw images that contain granular
materials with binder (0) and individual grains.  To avoid issues with the contact algorithm
(grains sticking together) the individual grains will be assigned to separate matls.  
Below is a raw image slice with 4 regions or grains.  This tool indentifies all 
intensities > 0 and maps that intensity to a mpm matl.

0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  50 50 50 50
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  50 50 50
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  63 63 0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  0  63 63 63 63 0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  63 63 63 63 63 63 0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  63 63 63 63 63 63 0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  0  63 63 63 63 0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  64 64 0  0  0  0  0  0  0  0  0  0  0  63 63 0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  64 64 64 64 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  |  64 64 64 64  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  64 64 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  65 65 65 0  0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  65 65 65 65 65 0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  0  0  66 66 66 66 0  0  0  0  0  0  0
0  0  0  65 65 65 65 65 0  0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  0  0  66 66 66 66 66 66 66 66 0  0  0  0  0
0  0  0  65 65 65 65 65 65 0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  0  0  66 66 66 66 66 66 66 66 66 66 66 66 0  0  0
0  0  0  65 65 65 65 65 65 0  0  0  0  0  0  0  |  0  0  0  0  0  0  0  66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 0
0  0  0  65 65 65 65 65 65 65 0  0  0  0  0  0  |  0  0  0  0  0  0  66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66
0  0  0  0  65 65 65 65 65 65 0  0  0  0  0  0  |  0  0  0  0  0  66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66
0  0  0  0  65 65 65 65 65 65 0  0  0  0  0  0  |  0  0  0  0  66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66
0  0  0  0  0  0  65 65 65 0  0  0  0  0  0  0  |  0  0  0  66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  |  0  0  66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66


ups file specification:
  <PreprocessTools>
    <rawToUniqueGrains>
      <image>  simple_sugar_mock_blob3d_unique_grains.raw  </image>
      <ppc>      [1,1,1]       </ppc>
      <res>    [154,252,1]     </res>
      <outputBasename>   points/16bit_grains    </outputBasename>
      
      <matl index="0">
        <threshold>   [0,0]   </threshold>
      </matl>
      <matl index="2">
        <threshold>   [67,69] </threshold>
      </matl>

      <uniqueGrains>
         <matlIndex>  [1,2,3,4,5] </matlIndex>
         <threshold>    [1,80]    </threshold>
      </uniqueGrains>
    </rawToUniqueGrains>
  </PreprocessTools>

Mapping of intensity to matl:

 
Unique grain intensity levels:  min: 1 max: 79
Intensity level to mpm matl mapping
 Intensity: 0 = matl 0
 Intensity: 1 = matl 1
 Intensity: 2 = matl 2
 Intensity: 3 = matl 3
 Intensity: 4 = matl 4
 Intensity: 5 = matl 5
 Intensity: 6 = matl 1
 Intensity: 7 = matl 2
 Intensity: 8 = matl 3
 Intensity: 9 = matl 4
 Intensity: 10 = matl 5
 Intensity: 11 = matl 1
 Intensity: 12 = matl 2
 Intensity: 13 = matl 3
 Intensity: 14 = matl 4
 Intensity: 15 = matl 5
 Intensity: 16 = matl 1
 
Assumptions:
 - The particle per cell (ppc) is a constant for all matls.
-----------------------------------------------------------------------------------------*/

#if 0
  typedef unsigned char pixel;       // 8bit images
#else
  typedef unsigned short pixel;      // 16bit images
#endif


// forwared function declarations
void usage( char *prog_name );

GridP CreateGrid(ProblemSpecP ups);

bool ReadImage(const char* szfile, unsigned int nPixels, pixel* pix, const string endianness);

bool ReadAuxFile(const string auxfile, map<int,double>& data);

inline Point CreatePoint(unsigned int n, vector<int>& res, double dx, double dy, double dz, Point domain_lo, Point domain_hi)
{
  unsigned int k = n / (res[0]*res[1]); n -= k* res[0]*res[1];
  unsigned int j = n / res[0];
  unsigned int i = n % res[0];

  Point here( dx*((double)i + 0.5), dy*(((double)(res[1]-1-j)) + 0.5),dz*((double)k + 0.5));
  
  // bullet proofing
  if( here.x() < domain_lo.x() || here.x() > domain_hi.x() ||
      here.y() < domain_lo.y() || here.y() > domain_hi.y() ||
      here.z() < domain_lo.z() || here.z() > domain_hi.z() ){
    ostringstream warn;
    warn<< " ERROR: you're trying to create a point outside of the computational domain \n"
        << " point:  " << here << "\n"
        << " domain: " << domain_lo << " -> " << domain_hi << "\n"
        << " Double check your grid specification." << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  return here;
}

struct intensityMatlMapping{
  vector<int> threshold;
  unsigned int matlIndex;
};

struct dataPoint{
  Point px;
  double scalar;
  int intensity;
};
enum endian{little, big};

//______________________________________________________________________
//
int main(int argc, char *argv[])
{
  try {
    Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );
    Uintah::Parallel::initializeManager( argc, argv );

    bool binmode = false;
    string auxFile = "notUsed";                   // auxilary file name
    string endianness= "little";
    bool d_auxMap    = false;

    //__________________________________
    // parse the command arguments
    for (int i=1; i<argc; i++){
      string s   = argv[i];
      
      if (s == "-b") {
        binmode = true;
      }
      else if (s == "-B" || s == "-bigEndian") {
        endianness = "big";
      }
      else if (s == "-l" || s == "-littleEndian") {
        endianness = "little";
      }
      else if (s == "-auxScalarFile") {
        auxFile = argv[++i];
      }
      else if (s == "-h" || s == "-help") {
        usage( argv[0] );
      }
      else if ( s[0] == '-'){
        cout << "\nERROR invalid input (" << s << ")" << endl;
        usage( argv[0] );
      }
    }

    string infile = argv[argc-1];

    if( argc < 2 || argc > 11 ){ 
      usage( argv[0] );
    }

    //__________________________________
    //  Read in user specificatons
    string imgname;                   // raw image file name
    vector<int> res;                  // image resolution
    vector<int> ppc;                  // number of particles per cell
    vector<int> UG_matlIndex;         // unique grain matl index
    vector<int> UG_threshold;         // unique grain threshold
    int UG_numMatls = 0;              // number of unique grain matls
    map<int, int> intensityToMatl_map;// intensity to matl mapping
    vector<int> specifiedMatls;       // matls that have been specified in the input file

    string f_name;                    // the base name of the output file
    bool hasUniqueGrains = false;

    ProblemSpecP ups = ProblemSpecReader().readInputFile( infile );

    if( !ups ) {
      throw ProblemSetupException("Cannot read problem specification", __FILE__, __LINE__);
    }

    ProblemSpecP ppt_ps = ups->findBlockWithOutAttribute("PreprocessTools");
    if( !ppt_ps ) {
      string warn;
      warn ="\n INPUT FILE ERROR:\n <PreprocessTools>  block not found\n";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }

    ProblemSpecP raw_ps = ppt_ps->findBlockWithOutAttribute("rawToUniqueGrains");
    if( !raw_ps ) {
      string warn;
      warn ="\n INPUT FILE ERROR:\n <rawToUniqueGrains>  block not found inside of <PreprocessTools> \n";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }

    raw_ps->require("image",          imgname ); 
    raw_ps->require("ppc",            ppc);
    raw_ps->require("res",            res);
    raw_ps->require("outputBasename", f_name);


    // read in all non unique grain specs and put that in a vector
    vector<intensityMatlMapping> intMatl_Vec;

    for (ProblemSpecP child = raw_ps->findBlock("matl"); child != 0;
                      child = child->findNextBlock("matl")) {

      vector<int> threshold;
      map<string,string> matlIndex;

      child->getAttributes(matlIndex);
      child->require("threshold", threshold);

      int matl = atoi(matlIndex["index"].c_str());
      intensityMatlMapping data;
      data.matlIndex = matl;
      data.threshold = threshold;
      intMatl_Vec.push_back(data);
      
      specifiedMatls.push_back(matl);  // keep a list of all matls that have been specified

      cout << "matl index " << matl << " Threshold low: " << threshold[0] << " high: " <<threshold[1] <<endl;
    }

    // read in the unique grains
    ProblemSpecP ug_ps = raw_ps->findBlockWithOutAttribute("uniqueGrains");
    if( ug_ps ) {
      ug_ps->require("matlIndex",      UG_matlIndex);
      ug_ps->require("threshold",      UG_threshold);
      UG_numMatls = UG_matlIndex.size();
      cout << "Number of unique Grain Matls " << UG_numMatls << " {";
      hasUniqueGrains = true;
      
      for (int m = 0; m< UG_numMatls; m++){
        int matl = UG_matlIndex[m];
        specifiedMatls.push_back(matl);  // keep a list of all matls that have been specified
        cout << matl << ", ";        
      }
      cout << "}\n";
    }


    //__________________________________
    //  Read the image file
    unsigned int nPixels = res[0]*res[1]*res[2];

    pixel* pimg = scinew pixel[nPixels];

    if (ReadImage(imgname.c_str(), nPixels, pimg, endianness) == false) {
      cout << "FATAL ERROR : Failed reading image data" << endl;
      exit(0);
    }

    //__________________________________
    //  find the number of intensity levels within the unique grains threshold
    if(hasUniqueGrains){
      int maxI = 0;
      int minI =INT_MAX;
      unsigned int n = 0;
      for (int k=0; k<res[2]; k++) {
        for (int j=0; j<res[1]; j++) {
          for (int i=0; i<res[0]; i++, n++) {
            
            ASSERT(n<nPixels);
            
            int pixelValue = pimg[n];
            if ((pixelValue >= UG_threshold[0]) && (pixelValue <= UG_threshold[1])) {
              maxI = max(maxI, pixelValue);
              minI = min(minI, pixelValue);
            }
          }
        }
      }

      if( maxI == 0 && minI == INT_MAX ){
        ostringstream warn;
        warn << "\n ERROR: No unique grains found in the threshold range "<< UG_threshold[0] << " and " << UG_threshold[1] << endl;
        throw ProblemSetupException(warn.str() , __FILE__, __LINE__);
      }
      cout << "Unique grain intensity levels: "  << " min: " << minI << " max: " << maxI << endl;
      //__________________________________
      //  create the intensity level to matl index map for the unique grains   
      int m = 0;
      for (int i=minI; i<=maxI; i++) {
        intensityToMatl_map[i] = UG_matlIndex[m];
        m ++;
        if(m >= UG_numMatls){
          m = 0;
        }
      }
    }
    //__________________________________
    // Now add the mappings for the individual materials
    // These can overwrite the unique grains mapping
    for (unsigned int m=0; m<intMatl_Vec.size(); m++) {

      intensityMatlMapping data = intMatl_Vec[m];

      for (int i=data.threshold[0]; i<=data.threshold[1]; i++) {
        intensityToMatl_map[i] = data.matlIndex;
      }
    }

    cout << "Intensity level to mpm matl mapping \n";
    map<int,int>::iterator it;
     for ( it=intensityToMatl_map.begin() ; it != intensityToMatl_map.end(); it++ ){
      cout << " Intensity: " << it->first<< " = matl " << it->second << endl;
    }


    //__________________________________
    //  read in auxilary scalar file  
    //  The two column file contains
    //  intensity  scalar
    map<int,double> intensityScalar_D_map;
    if ( auxFile != "notUsed" ) {
      ReadAuxFile(auxFile,intensityScalar_D_map);
      d_auxMap = true;
    }
    //__________________________________
    // Parse the ups file for the grid specification
    // and voxel size
    GridP grid = CreateGrid(ups);
    BBox box;
    grid->getInteriorSpatialRange(box);
    Point domain_lo = box.min();
    Point domain_hi = box.max();
    
    // loop over levels
    for (int l = 0; l < grid->numLevels(); l++) {
      LevelP level = grid->getLevel(l);

      // calculate voxel size
      Vector DX = level->dCell();
      double dx = DX.x() / ppc[0];
      double dy = DX.y() / ppc[1];
      double dz = DX.z() / ppc[2];
      fprintf(stderr, "Voxel size : %g, %g, %g\n", DX.x(), DX.y(), DX.z());
      fprintf(stderr, "Voxel dimensions : %g, %g, %g\n", dx, dy, dz);


      // bulletproofing
      // must use cubic cells
      IntVector low, high;
      level->findCellIndexRange(low, high);
      IntVector diff = high-low;
      long cells = diff.x()*diff.y()*diff.z();

      if(cells != level->totalCells()){
        throw ProblemSetupException("pfs can only cubic cells", __FILE__, __LINE__);
      }

      // loop over all mpm materials
      int matl = -1;
      ProblemSpecP mp = ups->findBlockWithOutAttribute("MaterialProperties");
      ProblemSpecP mpm = mp->findBlock("MPM");

      for (ProblemSpecP child = mpm->findBlock("material"); child != 0;
                        child = child->findNextBlock("material")) {

        matl +=1;
        
        // continue if the matl has been specified in the input file
        bool foundMatl = (find(specifiedMatls.begin(), specifiedMatls.end(), matl) != specifiedMatls.end());
        if (!foundMatl)
          continue;
        
        // these points define the extremas of the grid
        Point minP(1.e30,1.e30,1.e30),maxP(-1.e30,-1.e30,-1.e30);

        map<int,vector<dataPoint*> > patch_dataPoints;
        const Patch* curPatch;
        unsigned int n = 0;
        
        //__________________________________
        // loop over image and put points and scalar
        // in the patch_dataPoints map
        for (int k=0; k<res[2]; k++) {
          for (int j=0; j<res[1]; j++) {
            for (int i=0; i<res[0]; i++, n++) {
              ASSERT(n<nPixels);
              
              int pixelValue = pimg[n];
              bool isRightMatl = ( matl == intensityToMatl_map[pixelValue]);

              if ( isRightMatl ) {

                Point pt = CreatePoint(n, res, dx, dy, dz, domain_lo, domain_hi);
                
                minP = Min(pt,minP);
                maxP = Max(pt,maxP);
                
                curPatch = level->selectPatchForCellIndex(level->getCellIndex(pt));
                int patchID = curPatch->getID();
                
                dataPoint* dp = scinew dataPoint;
                dp->px      = pt;
                
                // if auxilary variable is being used
                if(d_auxMap){
                  dp->scalar  = intensityScalar_D_map[pixelValue];
                }
                patch_dataPoints[patchID].push_back(dp);
              }
            }
          }
          fprintf(stderr, "%s : %.1f\n", "Preprocessing ", 50.0*(k+1.0)/(double)res[2]);
        }

        //__________________________________
        //  Write out the pts file for this patch and matl        
        for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){
          const Patch* patch = *iter;

          unsigned int patchID = patch->getID();
          ostringstream of_name;
          of_name << f_name.c_str() << "_mat"<< matl <<"_pts."<<patchID;

          FILE* dest = fopen(of_name.str().c_str(), "wb");
          if(dest==0){
            cout << "FATAL ERROR : Failed opening points file " << of_name.str()<<endl;
            exit(0);
          }

          cout << " Writing file: " << of_name.str() << endl;

          // write the header
          double x[6];
          x[0] = minP.x(), x[1] = minP.y(), x[2] = minP.z();
          x[3] = maxP.x(), x[4] = maxP.y(), x[5] = maxP.z();

          if(binmode) {
            fwrite(x, sizeof(double),6,dest);
          } else {
            fprintf(dest, "%g %g %g %g %g %g\n", x[0],x[1],x[2],x[3],x[4],x[5]);
          }

          //__________________________________
          // write out the datapoints for this patch
          vector<dataPoint*> dataPoints = patch_dataPoints[patchID];
          
          for ( unsigned int i= 0; i != dataPoints.size(); i++ ){ 
            dataPoint* dp = dataPoints[i];
            
            x[0] = dp->px.x();
            x[1] = dp->px.y();
            x[2] = dp->px.z();
            
            //__________________________________
            //
            if( !d_auxMap ){
              if(binmode) {
                fwrite(x, sizeof(double), 3, dest);
              } else {
                fprintf(dest, "%g %g %g\n", x[0], x[1], x[2]);
              
              }
            }
            //__________________________________
            //  if there's an auxilary mapping
            if( d_auxMap){
              x[3] = dp->scalar;
              
              if(binmode) {
                fwrite(x, sizeof(double), 4, dest);
              } else {
                fprintf(dest, "%g %g %g %g\n", x[0], x[1], x[2], x[3]);
              }
            }
          }
          
          fclose(dest);
        } // loop over patches
      } // loop over materials
    } // loop over levels

    // delete image data
    delete [] pimg;
  } catch (Exception& e) {
    cerr << "\nCaught exception: " << e.message() << '\n';
    if(e.stackTrace())
      cerr << "Stack trace: " << e.stackTrace() << '\n';
  } catch(...){
    cerr << "Caught unknown exception\n";
  }
}

//--------------------------------------------------------------------------------------
// function CreateGrid : creates a grid object from the ProblemSpec
//
GridP CreateGrid(ProblemSpecP ups)
{
    // Setup the initial grid
    GridP grid=scinew Grid();
    IntVector extraCells(0,0,0);

    // save and remove the extra cells before the problem setup
    ProblemSpecP g = ups->findBlock("Grid");
    for( ProblemSpecP levelspec = g->findBlock("Level"); levelspec != 0;
         levelspec = levelspec->findNextBlock("Level")) {
      for (ProblemSpecP box = levelspec->findBlock("Box"); box != 0 ; 
           box = box->findNextBlock("Box")) {
        
        ProblemSpecP cells = box->findBlock("extraCells");
        if (cells != 0) {
          box->get("extraCells", extraCells);
          box->removeChild(cells);
        }
      }
    }
    const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();

    grid->problemSetup(ups, world, false);  

    return grid;
}

//------------------------------------------------------------------------------------------------
// function usage : prints a message on how to use the program
//
void usage( char *prog_name )
{
  cout << "Usage: " << prog_name << " [options]  <ups file> \n";
  cout << "options:" << endl;
  cout << "-b, -binary:            binary output \n";
  cout << "-l, -littleEndian:      input file contains little endian bytes  [default]\n";
  cout << "-B, -bigEndian:         input file contains big endian bytes\n";
  cout << "-auxScalarFile:         name of file that contains two columns of data (intensity scalar) \n";
  cout << "       # case:  grains \n";
  cout << "       # Number of blobs  75 \n";
  cout << "       # max diameter (cm) 0.0365792455209 \n";
  cout << "       # min diameter (cm) 0.00202786936452 \n";
  cout << "       # average diameter (cm) 0.00990896874237 \n";
  cout << "       # color    equivalent spherical diameter (cm) \n";
  cout << "       1.0 0.0365792455209 \n"; 
  cout << "       2.0 0.0031813408649 \n";
  cout << "       3.0 0.0113385664577" << endl;
  
  
  exit( 1 );
}

//-----------------------------------------------------------------------------------------------
// function ReadImage : Reads the image data from the file and stores it in a buffer
//
bool ReadImage(const char* szfile, unsigned int nPixels, pixel* pb, const string endianness)
{
  FILE* fp = fopen(szfile, "rb");
  if (fp == 0){ 
    return false;
  }
   
  unsigned int nread = fread(pb, sizeof(pixel), nPixels, fp);
  fclose(fp);
  cout <<"Reading: " << szfile << ", Bytes per pixel " << sizeof(pixel) << ", number of pixels read " << nread << endl;
  
  //__________________________________
  //  Display what the max intensity
  // is using big & little endianness bytes
  pixel minI = 0;
  pixel maxI = 0;

  // if the user specifies bigEndian then return
  if( endianness == "big" ){
    for(unsigned int i = 0; i< nread;  i++ ){
      swapbytes(pb[i]);
      maxI = max(pb[i], maxI);
      minI = min(pb[i], minI);
    }
    cout << "Big endian intensity: max (" << maxI << "), min(" << minI << " )"<< endl;
  } 
  else{
    for(unsigned int i = 0; i< nread;  i++ ){
      maxI = max(pb[i], maxI);
      minI = min(pb[i], minI);
    }
    cout << "Little endian intensity: max (" << maxI << "), min(" << minI << " )"<< endl;
  }
  return (nread == nPixels);  
}

//-----------------------------------------------------------------------------------------------
//
bool ReadAuxFile(const string auxFileName, map<int,double>& data)
{
  ifstream auxFile(auxFileName.c_str());
  
  if (!auxFile){ 
    throw ProblemSetupException("\nERROR:Couldn't open the auxilary file: " + auxFileName + "\n", __FILE__, __LINE__);
  }
  
  string line;
  int  intensity;
  double scalar;
  double tmp;
  
  cout << "\nIntensity -> auxilary scalar mapping \n" << endl;
  
  while (getline(auxFile, line)) {
    
    // throw away any line that contains #
    if (line.find("#") == 0 ){
      continue;
    }
     
    // read in the data
    stringstream(line) >> tmp >> scalar;
    intensity = (int) tmp;
    
    cout.setf(ios::scientific,ios::floatfield);
    cout.precision(17);
    
    // bullet proofing
    if(isnormal(intensity) ==0 || isnormal(scalar == 0 ) ){
      ostringstream warn;
      warn << "ERROR: auxFile: either the intensity (" << intensity << ") or scalar (" << scalar << ") is not a number\n";
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
    
    data[intensity] = scalar;
    cout << "Intensity: "<<intensity << " = scalar " << scalar << endl;
  }
  
  cout << " done reading auxfile " << endl;
  return true; 
}
