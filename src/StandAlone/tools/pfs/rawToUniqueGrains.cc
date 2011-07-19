/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Core/Exceptions/ProblemSetupException.h>
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
  int scale = 1;
#else
  typedef unsigned short pixel;      // 16bit images
  int scale = 256;
#endif


// forwared function declarations
void usage( char *prog_name );

GridP CreateGrid(ProblemSpecP ups);

bool ReadImage(const char* szfile, unsigned int nPixels, pixel* pix);

inline Point CreatePoint(unsigned int n, vector<int>& res, double dx, double dy, double dz)
{
  unsigned int k = n / (res[0]*res[1]); n -= k* res[0]*res[1];
  unsigned int j = n / res[0];
  unsigned int i = n % res[0];

  return Point( dx*((double)i + 0.5), dy*(((double)(res[1]-1-j)) + 0.5),dz*((double)k + 0.5));
}

struct intensityMapping{
  vector<int> threshold;
  unsigned int matlIndex;
};

//______________________________________________________________________
//
int main(int argc, char *argv[])
{
  Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );
  Uintah::Parallel::initializeManager( argc, argv );

  bool binmode = false;

  //__________________________________
  // parse the command arguments
  for (int i=1; i<argc; i++){
    string s=argv[i];
    if (s == "-b" || s == "-B") {
      binmode = true;
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
  vector<intensityMapping> intMatl_Vec;
  
  for (ProblemSpecP child = raw_ps->findBlock("matl"); child != 0;
                    child = child->findNextBlock("matl")) {
    
    vector<int> threshold;
    map<string,string> matlIndex;
    
    child->getAttributes(matlIndex);
    child->require("threshold", threshold);
    
    intensityMapping data;
    data.matlIndex = atoi(matlIndex["index"].c_str());
    data.threshold = threshold;
    intMatl_Vec.push_back(data);
    
    cout << "matl index " << matlIndex["index"] << " Threshold low: " << threshold[0] << " high: " <<threshold[1] <<endl;
  }
  
  // read in the unique grains
  ProblemSpecP ug_ps = raw_ps->findBlockWithOutAttribute("uniqueGrains");
  if( ug_ps ) {
    ug_ps->require("matlIndex",      UG_matlIndex);
    ug_ps->require("threshold",      UG_threshold);
    UG_numMatls = UG_matlIndex.size();
    cout << "Number of unique Grain Matls " << UG_numMatls << endl;
    hasUniqueGrains = true;
  }


  //__________________________________
  //  Read the image file
  unsigned int nPixels = res[0]*res[1]*res[2];
  cout << "Reading " << nPixels << " nPixels\n";
  
  pixel* pimg = scinew pixel[nPixels];

  if (ReadImage(imgname.c_str(), nPixels, pimg) == false) {
    cout << "FATAL ERROR : Failed reading image data" << endl;
    exit(0);
  }
  cout << "Done reading " << nPixels << " pixels\n";

  //__________________________________
  //  find the number of intensity levels within the unique grains threshold
  pixel* pb = pimg;
  
  if(hasUniqueGrains){
    int maxI = 0;
    int minI =INT_MAX;

    for (int k=0; k<=res[2]; k++) {
      for (int j=0; j<=res[1]; j++) {
        for (int i=0; i<=res[0]; i++, pb++) {

          int pixelValue = *pb/scale;
          if ((pixelValue >= UG_threshold[0]) && (pixelValue <= UG_threshold[1])) {
            maxI = max(maxI, pixelValue);
            minI = min(minI, pixelValue);
          }
        }
      }
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
  for (int m=0; m<intMatl_Vec.size(); m++) {
    
    intensityMapping data = intMatl_Vec[m];
    
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
  // Parse the ups file for the grid specification
  // and voxel size
  GridP grid = CreateGrid(ups);

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
      // these points define the extremas of the grid
      Point minP(1.e30,1.e30,1.e30),maxP(-1.e30,-1.e30,-1.e30);

      int nPatches = level->numPatches();
      vector< vector<int> > points(nPatches);
      vector<int> numPoints(nPatches);
      
      Point pt;
      pixel* pb = pimg;

      // first determine the number of points for each patch for this matl
      for (int p=0; p<nPatches; p++){
        numPoints[p] = 0;
      }

      const Patch* curPatch;
      int n = 0;
      
      for (int k=0; k<res[2]; k++) {
        for (int j=0; j<res[1]; j++) {
          for (int i=0; i<res[0]; i++, pb++, n++) {

            int pixelValue = *pb/scale;
            bool isRightMatl      = ( matl == intensityToMatl_map[pixelValue]);
            //bool withinThreshold  = ( (pixelValue >= L[0]) && (pixelValue <= L[1]) );

            if ( isRightMatl ) {

              pt = CreatePoint(n, res, dx, dy, dz);
              curPatch = level->selectPatchForCellIndex(level->getCellIndex(pt));
              int patchID = curPatch->getID();
              numPoints[patchID]++;
            }
          }
        }
        fprintf(stderr, "%s : %.1f\n", "Preprocessing ", 50.0*(k+1.0)/(double)res[2]);
      }

      // allocate storage for the patches
      for (int p=0; p<nPatches; p++){
        points[p].resize(numPoints[p]);
        numPoints[p] = 0;
      }

      // put the points in the correct patches
      // keep track of the min/max point locations
      pb = pimg;

      n = 0;
      for (int k=0; k<res[2]; k++) {
        for (int j=0; j<res[1]; j++) {
          for (int i=0; i<res[0]; i++, pb++, n++) {

            int pixelValue = *pb/scale;
            bool isRightMatl      = ( matl == intensityToMatl_map[pixelValue]);
            //bool withinThreshold  = ( (pixelValue >= L[0]) && (pixelValue <= L[1]) );

            if ( isRightMatl) {

              pt = CreatePoint(n, res, dx, dy, dz);

              const Patch* patch =   level->selectPatchForCellIndex(level->getCellIndex(pt));
              unsigned int patchID = patch->getID();
              minP = Min(pt,minP);
              maxP = Max(pt,maxP);
              points[patchID][ numPoints[patchID] ] = n;
              numPoints[patchID]++;
            }
          }
        }
        fprintf(stderr, "%s : %.1f\r", "Preprocessing ", 50+50.0*(k+1.0)/(double)res[2]);
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

        // output individual points
        for (int I = 0; I < numPoints[patchID]; I++) {
          pt = CreatePoint(points[patchID][I], res, dx, dy, dz);
          x[0] = pt.x();
          x[1] = pt.y();
          x[2] = pt.z();

          if(binmode) {
            fwrite(x, sizeof(double), 3, dest);
          } else {
            fprintf(dest, "%g %g %g\n", x[0], x[1], x[2]);
          }
        }
        fclose(dest);
      } // loop over patches
    } // loop over materials
  } // loop over levels
  
  // delete image data
  delete [] pimg;
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
  cout << "Usage: " << prog_name << " [-b] [-B] <ups file> \n";
  cout << "-b,B: binary output \n";
  exit( 1 );
}

//-----------------------------------------------------------------------------------------------
// function ReadImage : Reads the image data from file and stores it in a buffer
//
bool ReadImage(const char* szfile, unsigned int nPixels, pixel* pb)
{
  FILE* fp = fopen(szfile, "rb");
  if (fp == 0){ 
    return false;
  }
  
  unsigned int nread = fread(pb, sizeof(pixel), nPixels, fp);
  fclose(fp);

  cout << szfile << " Bytes per pixel " << sizeof(pixel) << " nPixels " << nPixels << " Nread " << nread << endl;
  return (nread == nPixels);  
}
