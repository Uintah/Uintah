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

#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Core/Grid/GridP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>

using namespace Uintah;
using namespace std;
#define IS_8_BIT

#ifdef IS_8_BIT
  typedef unsigned char pixel;       // 8bit images
#else
  typedef unsigned short pixel;      // 16bit images
#endif


// forwared function declarations
void usage( char *prog_name );

GridP CreateGrid(ProblemSpecP ups);

bool ReadImage(const char* szfile, unsigned int nPixels, pixel* pix, const string endianness);

inline Point CreatePoint(unsigned int n, vector<int>& res, double dx, double dy, double dz)
{
  unsigned int k = n / (res[0]*res[1]); n -= k* res[0]*res[1];
  unsigned int j = n / res[0];
  unsigned int i = n % res[0];

  return Point( dx*((double)i + 0.5), dy*(((double)(res[1]-1-j)) + 0.5),dz*((double)k + 0.5));
//  return Point( dx*((double)i + 0.5), dy*(((double)(j)) + 0.5),dz*((double)k + 0.5));
}

enum endian{little, big};



//-----------------------------------------------------------------------------------------
/*
pfs2 is used in conjunction with particle geometries derived by thresholding image
data. (pfs=Particle File Splitter) Given an raw file pfs2 reads an input file which contains
an "image" geometry piece description, and it also reads in the data in the raw file and the
intensity range associated with that geometry.  pfs2 then creates a separate file for each patch,
and places in that file those points which lie within that patch.  These individual files may
be ASCII or binary.  All of this is done to prevent numerous procs from trying to access the
same file, which is hard on file systems.

Can create a cylinder out of the image data given the coordinates of the bottom and top, and
the radius of the cylinder. The cylinder must be inside the image data. Note that the bounding
box is not changed.

*/
// function main : main entry point of application
//
int main(int argc, char *argv[])
{
  try {
    // Do some Uintah initialization
    Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );
    Uintah::Parallel::initializeManager( argc, argv );

    string infile;
    bool binmode = false;
    bool do_cylinder = false;
    string endianness= "little";
    
    CylinderGeometryPiece* cylinder;
    cylinder = 0;

    //default cylinder
    double xb = 0.0, yb = 0.0, zb = 0.0; //coordinates of the bottom
    double xt = 1.0, yt = 1.0, zt = 1.0; //coordinates of the top
    double radius = 1.0;                 //radius of the cylinder
    
    //__________________________________
    // parse the command arguments
    for (int i=1; i<argc; i++){
      string s=argv[i];
      if (s == "-b" || s == "-B") {
        binmode = true;
      } 
      else if (s == "-B" || s == "-bigEndian") {
        endianness = "big";
      }
      else if (s == "-l" || s == "-littleEndian") {
        endianness = "little";
      }
      else if (s == "-cyl") {
        do_cylinder = true;        
        xb = atof(argv[++i]);      
        yb = atof(argv[++i]);      
        zb = atof(argv[++i]);      
        xt = atof(argv[++i]);      
        yt = atof(argv[++i]);      
        zt = atof(argv[++i]);      
        radius = atof(argv[++i]);  
      }
      else if (s == "-h" || s == "-help") {
        usage( argv[0] );
      }
      else if ( s[0] == '-'){
        cout << "\nERROR invalid input (" << s << ")" << endl;
        usage( argv[0] );
      }
    }
    
    infile = argv[argc-1];

    if( argc < 2 || argc > 11 ){ 
      usage( argv[0] );
    }
    
    if (do_cylinder) {
      try {                                                                                                
        Point bottom(xb,yb,zb);                                                                          
        Point top(xt,yt,zt);                                                                             
        cylinder = scinew CylinderGeometryPiece(top, bottom, radius);                                    
        fprintf(stderr, "Cylinder height, volume: %g, %g\n", cylinder->height(), cylinder->volume());    
      } catch (Exception& e) {                                                                             
        cerr << "Caught exception: " << e.message() << endl;                                             
        abort();                                                                                         
      } catch(...){                                                                                        
        cerr << "Caught unknown exception\n";                                                            
        abort();                                                                                         
      }                                                                                                    
    }

    //__________________________________
    // Get the problem specification
    ProblemSpecP ups = ProblemSpecReader().readInputFile( infile );

    if( !ups ) {
      throw ProblemSetupException("Cannot read problem specification", __FILE__, __LINE__);
    }
    
    if( ups->getNodeName() != "Uintah_specification" ) {
      throw ProblemSetupException("Input file is not a Uintah specification", __FILE__, __LINE__);
    }

    // Create the grid
    GridP grid = CreateGrid(ups);

    // repeat for all grid levels
    for (int l = 0; l < grid->numLevels(); l++) {
      LevelP level = grid->getLevel(l);

      // calculate voxel size
      Vector DX = level->dCell();

      fprintf(stderr, "Voxel size : %g, %g, %g\n", DX.x(), DX.y(), DX.z());      

      // bulletproofing
      // make sure the grid level is one that we can handle
      IntVector low, high;
      level->findCellIndexRange(low, high);
      IntVector diff = high-low;
      long cells = diff.x()*diff.y()*diff.z();
      
      if(cells != level->totalCells()){
        throw ProblemSetupException("pfs can only handle square grids", __FILE__, __LINE__);
      }
      
      
      // Parse the geometry from the UPS
      ProblemSpecP mp = ups->findBlockWithOutAttribute("MaterialProperties");
      ProblemSpecP mpm = mp->findBlock("MPM");
      
      for (ProblemSpecP child = mpm->findBlock("material"); child != 0;
                        child = child->findNextBlock("material")) {
      
        for (ProblemSpecP geom_obj_ps = child->findBlock("geom_object");
          geom_obj_ps != 0;
          geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

          fprintf(stderr, "\n--- Reading geometry object --- \n");

          string imgname;  // raw image file name
          vector<int> res; // image resolution
          vector<int> ppc; // nr particles per cell
          vector<int> L;   // lower and upper threshold
          string f_name;   // the base name of the output file
          string of_name;  // actual output file name
          int ncols = 0;   // nr. of additional data columns
          int ncheck = 0;  // check to make sure we have a "image" and a "file" section

          // read the particles per cell
          geom_obj_ps->require("res", ppc);

          fprintf(stderr, "ppc = %d, %d, %d\n", ppc[0], ppc[1], ppc[2]);
          double dx = DX.x() / ppc[0];
          double dy = DX.y() / ppc[1];
          double dz = DX.z() / ppc[2];
          fprintf(stderr, "Voxel dimensions : %g, %g, %g\n", dx, dy, dz);

          for(ProblemSpecP child = geom_obj_ps->findBlock(); child != 0;
                           child = child->findNextBlock()){
            std::string go_type = child->getNodeName();

            // get the image data
            if (go_type == "image") {
                child->require("name", imgname);
                child->require("res", res);
                child->require("threshold", L);
                
                cout << "Image name : " << imgname << endl;
                cout << "Resolution : " << res[0] << ", " << res[1] << ", " << res[2] << endl;
                cout << "Min threshold : " << L[0] << endl;
                cout << "Max threshold : " << L[1] << endl;
                ncheck++;
            }
            
            // Read the output file data
            if (go_type == "file"){
              child->require("name",f_name);
              
              // count number of vars, and their sizes
              for(ProblemSpecP varblock = child->findBlock("var");
                  varblock;varblock=varblock->findNextBlock("var")) {
                string next_var_name("");
                varblock->get(next_var_name);
                if      (next_var_name=="p.volume")        ncols += 1;
                else if (next_var_name=="p.temperature")   ncols += 1;
                else if (next_var_name=="p.color")         ncols += 1;
                else if (next_var_name=="p.externalforce") ncols += 3;
                else if (next_var_name=="p.fiberdir")      ncols += 3;
                else 
                  throw ProblemSetupException("Unexpected field variable of '"+next_var_name+"'", __FILE__, __LINE__);
              }
              cout << "Output file name : " << f_name << endl;
              cout << "Nr of additional columns :" << ncols << endl;
              ncheck++;
            }
          }

          // only do the following if we found an image section and a file section
          if (ncheck != 2) {
            fprintf(stderr, "  ...Skipping\n");
            continue;
          }
          
          //__________________________________
          // read the image data
          unsigned int nPixels = res[0]*res[1]*res[2];
          cout << "Reading " << nPixels << " nPixels\n";
          pixel* pimg = scinew pixel[nPixels];
          
          if (ReadImage(imgname.c_str(), nPixels, pimg, endianness) == false) {
            cout << "FATAL ERROR : Failed reading image data" << endl;
            exit(0);
          }
          cout << "Done reading " << nPixels << " pixels\n";


          // these points define the extremas of the grid
          Point minP(1.e30,1.e30,1.e30),maxP(-1.e30,-1.e30,-1.e30);

          // create the points
          // It was noted that the original algorithm was using
          // a lot of memory. To reduce memory we don't store the
          // actual points anymore but an index that can be used
          // to recreate the points
          
          int npatches = level->numPatches();
          vector< vector<int> > points(npatches);
          vector<int> sizes(npatches);

          int i, j, k;
          unsigned int n;
          Point pt;
          pixel* pb = pimg;

          // first determine the nr of points for each patch
          for (i=0; i<npatches; i++){
            sizes[i] = 0;
          }
          
          const Patch* currentpatch;
          n = 0;
          
          for (k=0; k<res[2]; k++) {
            for (j=0; j<res[1]; j++) {
              for (i=0; i<res[0]; i++, pb++, n++) {
              
                int pixelValue = *pb;
                
                if ((pixelValue >= L[0]) && (pixelValue <= L[1])) {
                  
                  pt = CreatePoint(n, res, dx, dy, dz);
                  currentpatch = level->selectPatchForCellIndex(level->getCellIndex(pt));
                  int pid = currentpatch->getID();
                  sizes[pid]++;
                }
              }
            }
            fprintf(stderr, "%s : %.1f\n", "Preprocessing ", 50.0*(k+1.0)/(double)res[2]);
          }

          // allocate storage for the patches
          for (i=0; i<npatches; i++){
             points[i].resize(sizes[i]);
             sizes[i] = 0;
          }

          // put the points in the correct patches
          pb = pimg;

          n = 0;
          for (k=0; k<res[2]; k++) {
            for (j=0; j<res[1]; j++) {
              for (i=0; i<res[0]; i++, pb++, n++) {
              
                int pixelValue = *pb;
                if ((pixelValue >= L[0])  && (pixelValue <= L[1])) {

                  pt = CreatePoint(n, res, dx, dy, dz);

                 const Patch* currentpatch =
                                      level->selectPatchForCellIndex(level->getCellIndex(pt));
                  unsigned int pid = currentpatch->getID();
                  minP = Min(pt,minP);
                  maxP = Max(pt,maxP);
                  points[pid][ sizes[pid] ] = n;
                  sizes[pid]++;
                }
              }
            }
            fprintf(stderr, "%s : %.1f\r", "Preprocessing ", 50+50.0*(k+1.0)/(double)res[2]);
          }

          // clean up image data
          delete [] pimg;

          // loop over all patches
          for(Level::const_patchIterator iter = level->patchesBegin();
              iter != level->patchesEnd(); iter++){
            const Patch* patch = *iter;
            unsigned int pid = patch->getID();

            char fnum[5];
            sprintf(fnum,".%d",pid);
            of_name = f_name+fnum;
            fprintf(stderr, "Writing %s   \r", of_name.c_str());

            // ADB: change this to always be 128 bytes, so that we can 
            // cleanly read the header off a binary file
            FILE* dest = fopen(of_name.c_str(), "wb");
            if(dest==0){
              cout << "FATAL ERROR : Failed opening points file" << endl;
              exit(0);
            }
            
            double x[6];
            x[0] = minP.x(), x[1] = minP.y(), x[2] = minP.z();
            x[3] = maxP.x(), x[4] = maxP.y(), x[5] = maxP.z();
            
            if(binmode) {
              fwrite(x, sizeof(double),6,dest);
            } else {
              fprintf(dest, "%g %g %g %g %g %g\n", x[0],x[1],x[2],x[3],x[4],x[5]);
            }
            
            for (int I = 0; I < sizes[pid]; I++) {
              pt = CreatePoint(points[pid][I], res, dx, dy, dz);
              x[0] = pt.x();
              x[1] = pt.y();
              x[2] = pt.z();

              //points outside the cylinder are ignored.
              if (do_cylinder && !cylinder->inside(pt)) continue;

              // FIXME: should have way of specifying endiness
              if(binmode) {
                fwrite(x, sizeof(double), 3, dest);
              } else {
                fprintf(dest, "%g %g %g\n", x[0], x[1], x[2]);
              }
            }
            fclose(dest);
          } // loop over patches
        } // loop over geometry objects
      } // loop over materials
    } // loop over grid levels
    
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << '\n';
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
  cout << "-cyl <args> :           defines a cylinder within the geometry args = xbot ybot zbot xtop ytop ztop radius \n";         
  cout << "-l, -littleEndian:      input file contains little endian bytes  [default]\n";
  cout << "-B, -bigEndian:         input file contains big endian bytes\n";
  exit( 1 );
}

//-----------------------------------------------------------------------------------------------
// function ReadImage : Reads the image data from file and stores it in a buffer
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
