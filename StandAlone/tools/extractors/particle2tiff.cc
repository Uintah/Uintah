/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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

/*______________________________________________________________________
 *  particle2tiff.cc: 
 *
 *  A post processing utility for udas containing particle data.  
 *  The utility, tools/extractors/particle2tiff, averages particle data to the cell center and writes 
 *  this data to a 32-bit tiff file.  The tiff file contains 'slices' where each slice corresponds 
 *  to a plane in the z-direction.  The grayscale value of each pixel represents the 
 *  averaged value for that computational cell.  You can use the 
 *  image processing tool 'imageJ' to further analyze the tiffs.  Note, not all 
 *  tools can handle 32-bit images.  Currently, particle variables of type 
 *  double, Vector and Matrix3 are supported.  The equations for averaging 
 *  these data are:
 * 
 * cc_ave = sum( double[p].         ) /( # particles in cell )
 * cc_ave = sum( Vector[p].length() ) /( # particles in cell )
 * cc_ave = sum( Matrix3[p].Norm()  ) /( # particles in cell )
 *
 * The users can select what material to analyze, the temporal and physical range to examine, and
 * clamp the averaged data.
 *
 *
 * This utility depends on libtiff4, libtiff4-dev, & libtiffxx0c2, please
 * verify that they are installed on your system before configuring and compiling.
 *
 *  Written by:
 *   Todd Harman
 *   Department of Mechancial Engineering 
 *   by stealing lineextract from:
 *   University of Utah
 *   May 2012
 *
 *  Copyright (C) 2012 U of U
*______________________________________________________________________*/
 

#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/ShareAssignParticleVariable.h>

#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>

#include <fstream>
#include <string>
#include <vector>

#include <iomanip>
#include <tiffio.h>
#include <cstdio>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

// format for saving slices
enum saveSlices{ asVolume, asIndividualSlices, None};

struct clampVals{
  double minVal;
  double maxVal;
  ~clampVals() {}
};

// contains all tiff flags
struct tiffFlags{
  uint16 orientation;
};


//______________________________________________________________________
//   Base class
class Bits{
  public:
    virtual void fillSlice(CCVariable<double>& ave,
                           bool doVerification,
                           const IntVector& lo,
                           uint32 page ) = 0;

    virtual void verifySlice(TIFF* out, 
                             tsize_t size,
                             tsize_t imageSize) = 0;
                           
    virtual tsize_t writeRawStrip(TIFF* out, tsize_t imageSize ) = 0;
};

//______________________________________________________________________
//      8 - bit class & methods
class eightBit  : public Bits
{
  public:
    eightBit(uint32 height, uint32 width){
      d_height = height;
      d_width = width;
      d_slice = (uint8*) malloc(height * width * sizeof(uint8));
      
    }
    
    ~eightBit(){}
    
    //__________________________________
    //
    void fillSlice(CCVariable<double>& ave,
                   bool doVerification,
                   const IntVector& lo,
                   uint32 page ){
      
      if( doVerification ) {      // create test image
        for (uint32 j = 0; j < d_height; j++){
          for(uint32 i = 0; i < d_width; i++){
            d_slice[j * d_width + i] = (uint8) j + (uint8) i;
          }
        }
      } else {                    // actual data
        for (uint32 j = 0; j < d_height; j++){
          for(uint32 i = 0; i < d_width; i++){
            IntVector c = IntVector(i,j, page) + lo;
            d_slice[j * d_width + i] = ave[c];;
          }
        }
      } 
    }

    //__________________________________
    // read the image back in and verify that 
    // it's correct
    void verifySlice(TIFF* out, 
                     tsize_t size,
                     tsize_t imageSize){
 
      tsize_t nStrips   = TIFFNumberOfStrips (out);
      cout  << "    Bytes Written: "<< size << " number of strips " << nStrips << endl; 
      cout  << "    Now reading in slice and verify data: ";

      uint8 buffer[d_height * d_width];
      
      long result;
      if((result = TIFFReadRawStrip (out, (tstrip_t) 0, buffer, imageSize)) == -1){
        fprintf(stderr, "Read error on input strip number %d\n", 0);
        exit(42);
      }

      for (uint32 j = 0; j < d_height; j++){
        for(uint32 i = 0; i < d_width; i++){
          if (buffer[j * d_width + i] != d_slice[j * d_width + i]  ){
            cout << " ERROR:  ["<<i << "," << j << "] "
                 << " input pixel " <<   buffer[j * d_width + i] 
                 << " output pixel " << d_slice[j * d_width + i] << endl;
          }
        }
      }
      cout << " PASSED " << endl;    
    }
    
    //__________________________________
    //
    tsize_t writeRawStrip(TIFF* out, tsize_t imageSize ){
      tsize_t size = TIFFWriteRawStrip(out, (tstrip_t) 0, d_slice, imageSize);
      return size;
    }

    private:
      uint32 d_height;
      uint32 d_width;

      uint8* d_slice;
      uint8* d_buffer;
};

//______________________________________________________________________
//      16 - bit class & methods
class sixteenBit  : public Bits
{
  public:
    sixteenBit(uint32 height, uint32 width){
      d_height = height;
      d_width = width;
      d_slice = (uint16*) malloc(height * width * sizeof(uint16));
    }
    
    ~sixteenBit(){}
    
    //__________________________________
    //
    void fillSlice(CCVariable<double>& ave,
                   bool doVerification,
                   const IntVector& lo,
                   uint32 page ){
      
      if( doVerification ) {      // create test image
        for (uint32 j = 0; j < d_height; j++){
          for(uint32 i = 0; i < d_width; i++){
            d_slice[j * d_width + i] = (uint16) j * (uint16) i;
          }
        }
      } else {                    // actual data
        for (uint32 j = 0; j < d_height; j++){
          for(uint32 i = 0; i < d_width; i++){
            IntVector c = IntVector(i,j,page) + lo;
            d_slice[j * d_width + i] = ave[c];;
          }
        }
      }
    }
    
    //__________________________________
    // read the image back in and verify that 
    // it's correct
    void verifySlice(TIFF* out, 
                     tsize_t size,
                     tsize_t imageSize){
    
      tsize_t nStrips   = TIFFNumberOfStrips (out);
      cout  << "    Bytes Written: "<< size << " number of strips " << nStrips << endl; 
      cout  << "    Now reading in slice and verify data: ";

      uint16  buffer[d_height * d_width];
      
      long result;
      if((result = TIFFReadRawStrip (out, (tstrip_t) 0, buffer, imageSize)) == -1){
        fprintf(stderr, "Read error on input strip number %d\n", 0);
        exit(42);
      }

      for (uint32 j = 0; j < d_height; j++){
        for(uint32 i = 0; i < d_width; i++){
          if (buffer[j * d_width + i] != d_slice[j * d_width + i]  ){
            cout << " ERROR:  ["<<i << "," << j << "] "
                 << " input pixel " <<   buffer[j * d_width + i] 
                 << " output pixel " << d_slice[j * d_width + i] << endl;
          }
        }
      }
      cout << " PASSED " << endl;    
    }    
    //__________________________________
    //
    tsize_t writeRawStrip(TIFF* out, tsize_t imageSize ){
      tsize_t size = TIFFWriteRawStrip(out, (tstrip_t) 0, d_slice, imageSize);
      return size;
    }

    private:
      uint32 d_height;
      uint32 d_width;

      uint16* d_slice;
      uint16* d_buffer;
};

//______________________________________________________________________
//      32 - bit class & methods
class thirtytwoBit  : public Bits
{
  public:
    thirtytwoBit(uint32 height, uint32 width){
      d_height = height;
      d_width = width;
      d_slice = (float*) malloc(height * width * sizeof(float));
    }
    
    ~thirtytwoBit(){}
    
    //__________________________________
    //
    void fillSlice(CCVariable<double>& ave,
                   bool doVerification,
                   const IntVector& lo,
                   uint32 page ){
      
      if( doVerification ) {      // create test image
        for (uint32 j = 0; j < d_height; j++){
          for(uint32 i = 0; i < d_width; i++){
            d_slice[j * d_width + i] = (float) j * (float) i;
          }
        }
      } else {                    // actual data
        for (uint32 j = 0; j < d_height; j++){
          for(uint32 i = 0; i < d_width; i++){
            IntVector c = IntVector(i,j,page) + lo;
            d_slice[j * d_width + i] = ave[c];
          }
        }
      }      
    }

    //__________________________________
    // read the image back in and verify that 
    // it's correct
    void verifySlice(TIFF* out, 
                     tsize_t size,
                     tsize_t imageSize){
    
      tsize_t nStrips   = TIFFNumberOfStrips (out);
      cout  << "    Bytes Written: "<< size << " number of strips " << nStrips << endl; 
      cout  << "    Now reading in slice and verify data: ";

      float buffer[d_height * d_width];
      
      long result;
      if((result = TIFFReadRawStrip (out, (tstrip_t) 0, buffer, imageSize)) == -1){
        fprintf(stderr, "Read error on input strip number %d\n", 0);
        exit(42);
      }

      for (uint32 j = 0; j < d_height; j++){
        for(uint32 i = 0; i < d_width; i++){
          if (buffer[j * d_width + i] != d_slice[j * d_width + i]  ){
            cout << " ERROR:  ["<<i << "," << j << "] "
                 << " input pixel " <<   buffer[j * d_width + i] 
                 << " output pixel " << d_slice[j * d_width + i] << endl;
          }
        }
      }
      cout << " PASSED " << endl;    
    }
    
    //__________________________________
    //
    tsize_t writeRawStrip(TIFF* out, tsize_t imageSize ){
    
      tsize_t size = TIFFWriteRawStrip(out, (tstrip_t) 0, d_slice, imageSize);
      return size;
    }

    private:
      uint32 d_height;
      uint32 d_width;

      float* d_slice;
};                        


//______________________________________________________________________
//  
void
usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
        cerr << "Error parsing argument: " << badarg << endl;
    cerr << "Usage: " << progname << " [options] "
         << "-uda <archive file>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h,        --help\n";
    cerr << "  -v,        --variable:      [string] variable name\n";
    cerr << "  -m,        --material:      [int or string 'a, all'] material index [defaults to 0]\n\n";
    cerr << "  -max                        [double] (maximum clamp value)\n";
    cerr << "  -min                        [double] (minimum clamp value)\n";    
    cerr << "  -verify                     [none]   (output test image 256x256x5) \n";
    cerr << "  -nBits                      [int]    (number of bits/pixel in output images) [defaults to 8] \n";
    cerr << "  -asVolume                   [none]   (save all slices in 1 tiff file \n";
    cerr << "  -asSlices                   [none]   (save slices as separate files \n";
    cerr << "  -orientation                [string] (The orientation of the image with respect to the rows and columns.)\n";
    cerr << "                                      Options: \n";
    cerr << "                                        topleft  0th row represents the visual top of the image, and the 0th column represents the visual left-hand side.\n";
    cerr << "                                        topright 0th row represents the visual top of the image, and the 0th column represents the visual right-hand side\n";
    cerr << "                                        botright 0th row represents the visual bottom of the image, and the 0th column represents the visual right-hand side\n";
    cerr << "                             default->  botleft  0th row represents the visual bottom of the image, and the 0th column represents the visual left-hand side.\n";
    cerr << "                                        lefttop  0th row represents the visual left-hand side of the image, and the 0th column represents the visual top.\n";
    cerr << "                                        righttop 0th row represents the visual right-hand side of the image, and the 0th column represents the visual top.\n";
    cerr << "                                        rightbot 0th row represents the visual right-hand side of the image, and the 0th column represents the visual bottom.\n";
    cerr << "                                        leftbot  0th row represents the visual left-hand side of the image, and the 0th column represents the visual bottom. \n";
    cerr << "                                                    Many readers ignore this tag \n\n";
    
    cerr << "  -tlow,     --timesteplow:   [int] (start output timestep) [defaults to 0]\n";
    cerr << "  -thigh,    --timestephigh:  [int] (sets end output timestep) [defaults to last timestep]\n";
    cerr << "  -timestep, --timestep:      [int] (only outputs timestep)  [defaults to 0]\n\n";
    
    cerr << "  -istart,   --indexs:        <i> <j> <k> [ints] starting point cell index  [defaults to 0 0 0]\n";
    cerr << "  -iend,     --indexe:        <i> <j> <k> [ints] end-point cell index [defaults to 0 0 0]\n";
    cerr << "  -startPt                    <x> <y> <z> [doubles] starting point in physical coordinates\n";
    cerr << "  -endPt                      <x> <y> <z> [doubles] end-point in physical coordinates\n\n"; 
     
    cerr << "  -l,        --level:         [int] (level index to query range from) [defaults to 0]\n";
    cerr << "  -d,        --dir:           output directory name [none]\n"; 
    cerr << "  --cellIndexFile:            <filename> (file that contains a list of cell indices)\n";
    cerr << "                                   [int 100, 43, 0]\n";
    cerr << "                                   [int 101, 43, 0]\n";
    cerr << "                                   [int 102, 44, 0]\n";
    cerr << "----------------------------------------------------------------------------------------\n";
    cerr << " For particle variables the average over all particles in a cell is returned.\n";
    exit(1);
}

//______________________________________________________________________
//  return the tiff flag option for the orientation
uint16
whichOrientation(const string& in){
  uint16 ans= ORIENTATION_BOTLEFT;
  if( in == "topleft" ){
    ans = ORIENTATION_TOPLEFT;
  } else if( in == "topright" ){
    ans = ORIENTATION_TOPRIGHT;
  } else if( in == "botright" ){
    ans = ORIENTATION_BOTRIGHT;
  } else if( in == "botleft" ){
    ans = ORIENTATION_BOTLEFT;
  } else if( in == "lefttop" ){
    ans = ORIENTATION_LEFTTOP;
  } else if( in == "righttop" ){
    ans = ORIENTATION_RIGHTTOP;
  } else if( in == "rightbot" ){
    ans = ORIENTATION_RIGHTBOT;
  } else if( in == "leftbot" ){
    ans = ORIENTATION_LEFTBOT;
  } else{
    cerr << "\n\nThe orientation [" << in << "] is not valid. \n";
    usage("","");
  }
  return ans;
};

//______________________________________________________________________
//

void set_tiff_options(TIFF* out,
                      const uint32 imageWidth,
                      const uint32 imageHeight,
                      const uint16 depth,
                      const tiffFlags* flags){
    float xres = 150;
    float yres = 150;
    uint16 spp = 1;                       // samples per pixel 1 for black & white or gray and 3 for color
    uint16 photo =  PHOTOMETRIC_MINISBLACK;
      
    // We need to set some values for basic tags before we can add any data
    TIFFSetField(out, TIFFTAG_IMAGEWIDTH,       imageWidth*spp );         // set the width of the image
    TIFFSetField(out, TIFFTAG_IMAGELENGTH,      imageHeight );            // set the height of the image
    TIFFSetField(out, TIFFTAG_ROWSPERSTRIP,     imageHeight);
    TIFFSetField(out, TIFFTAG_BITSPERSAMPLE,    depth*8 );                // bits per channel
    TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL,  spp );                    // number of channels per pixel, 1 for B&W or gray

    TIFFSetField(out, TIFFTAG_SAMPLEFORMAT,     SAMPLEFORMAT_IEEEFP);     // IEEE floating point data
    TIFFSetField(out, TIFFTAG_ORIENTATION,      flags->orientation);      // set the origin of the image.

  //=  TIFFSetField(out, TIFFTAG_COMPRESSION,      COMPRESSION_DEFLATE);
    TIFFSetField(out, TIFFTAG_PHOTOMETRIC,      photo);  
        
    TIFFSetField(out, TIFFTAG_FILLORDER,        FILLORDER_MSB2LSB);
    TIFFSetField(out, TIFFTAG_PLANARCONFIG,     PLANARCONFIG_CONTIG);

    TIFFSetField(out, TIFFTAG_XRESOLUTION,      xres);
    TIFFSetField(out, TIFFTAG_YRESOLUTION,      yres);
    TIFFSetField(out, TIFFTAG_RESOLUTIONUNIT,   RESUNIT_INCH);
}


//______________________________________________________________________
//
void write_tiff_volume(const tiffFlags* flags,
                       const ostringstream& tname,
                       const IntVector& lo,
                       const IntVector& hi,
                       CCVariable<double>& ave,
                       const bool doVerification,
                       const int nBits){

  uint32 imageWidth = hi.x() - lo.x();
  uint32 imageHeight= hi.y() - lo.y();
  uint32 imageDepth = hi.z() - lo.z();
  
  if(doVerification){
    imageWidth =256;
    imageHeight=256;
    imageDepth = 5;
  }                       
  uint16 depth = -9;                         // bytes per pixel;
  
  //__________________________________
  //  Create the objects that create the slices
  //  and write the data
  Bits* whichBit;
  
  if(nBits == 8){
    depth = TIFFDataWidth(TIFF_BYTE); 
    whichBit = new eightBit(imageHeight, imageWidth);
    cout << "  Writing 8-bit images " << endl;
  } else if (nBits == 16){
    depth = TIFFDataWidth(TIFF_SHORT);
    whichBit = new sixteenBit(imageHeight, imageWidth);
    cout << "  Writing 16-bit images " << endl;
  } else if (nBits == 32){
    depth = TIFFDataWidth(TIFF_FLOAT);
    whichBit = new thirtytwoBit(imageHeight, imageWidth);
    cout << "  Writing 32-bit images " << endl;
  }
  
  tsize_t imageSize = imageHeight * imageWidth * depth;
  
  // Open the TIFF file
  TIFF *out;
  if((out = TIFFOpen(tname.str().c_str(), "w")) == NULL){
    cout << "Could not open " << tname << " for writing\n";
    exit(1);
  }

  //__________________________________
  //  loop over slices in z direction
  for (uint32 page = 0; page < imageDepth; page++) {
  
    // fill the slice with data
    whichBit->fillSlice( ave, doVerification, lo, page);
    
    
    set_tiff_options(out, imageWidth, imageHeight, depth, flags);

    // We are writing a page of the multi page file
    TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);

    // slice number 
    TIFFSetField(out, TIFFTAG_PAGENUMBER, page, imageDepth);

    tsize_t size = whichBit->writeRawStrip(out, imageSize );
    
    cout << "  writing slice: [" << page << "/"<< imageDepth << "]" << " width: " << imageWidth << " height " << imageHeight << endl;

    if( doVerification ){
      whichBit->verifySlice( out, size, imageSize);
    }
       
    TIFFWriteDirectory(out);
   
  }  // page loop
  TIFFClose(out);
}


//______________________________________________________________________
//
void write_tiff_slices(const tiffFlags* flags,
                       const ostringstream& sliceBaseName,
                       const IntVector& lo,
                       const IntVector& hi,
                       CCVariable<double>& ave,
                       const bool doVerification,
                       const int nBits){

  uint32 imageWidth = hi.x() - lo.x();
  uint32 imageHeight= hi.y() - lo.y();
  uint32 imageDepth = hi.z() - lo.z();
  
  if(doVerification){
    imageWidth =256;
    imageHeight=256;
    imageDepth = 5;
  }                       
  uint16 depth = -9;                         // bytes per pixel;
  
  //__________________________________
  //  Create the objects that create the slices
  //  and write the data
  Bits* whichBit;
  
  if(nBits == 8){
    depth = TIFFDataWidth(TIFF_BYTE); 
    whichBit = new eightBit(imageHeight, imageWidth);
    cout << "  Writing 8-bit images " << endl;
  } else if (nBits == 16){
    depth = TIFFDataWidth(TIFF_SHORT);
    whichBit = new sixteenBit(imageHeight, imageWidth);
    cout << "  Writing 16-bit images " << endl;
  } else if (nBits == 32){
    depth = TIFFDataWidth(TIFF_FLOAT);
    whichBit = new thirtytwoBit(imageHeight, imageWidth);
    cout << "  Writing 32-bit images " << endl;
  }
  
  tsize_t imageSize = imageHeight * imageWidth * depth;
  

  //__________________________________
  //  loop over slices in z direction
  for (uint32 page = 0; page < imageDepth; page++) {
    ostringstream sliceName;
    sliceName << sliceBaseName.str()<<page<<".tif";

    // Open the TIFF file
    TIFF *out;
    if((out = TIFFOpen(sliceName.str().c_str(), "w")) == NULL){
      cout << "Could not open " << sliceName << " for writing\n";
      exit(1);
    }
  
    // fill the slice with data
    whichBit->fillSlice( ave, doVerification, lo, page);
    
    set_tiff_options(out, imageWidth, imageHeight, depth, flags);

    // slice number 
    TIFFSetField(out, TIFFTAG_PAGENUMBER, page, imageDepth);

    tsize_t size = whichBit->writeRawStrip(out, imageSize );
    
    cout << "  writing slice: [" << page << "/"<< imageDepth << "]" << " width: " << imageWidth << " height " << imageHeight << endl;

    if( doVerification ){
      whichBit->verifySlice( out, size, imageSize);
    }
       
    TIFFClose(out);
   
  }  // page loop
}


//______________________________________________________________________
//  compute the cell centered average of the particles in a cell for 1 patch
//       D O U B L E   V E R S I O N
void
compute_ave( vector<int>                         & matls,
             const clampVals                     * clamp,
             vector< ParticleVariable<double>* > & var,
             CCVariable<double>                  & ave,
             vector< ParticleVariable<Point>* >  & pos,
             const Patch                         * patch ) 
{
  IntVector lo = patch->getExtraCellLowIndex();
  IntVector hi = patch->getExtraCellHighIndex();
  
  ave.allocate(lo,hi);
  ave.initialize(0.0);
  
  CCVariable<double> count;
  count.allocate(lo,hi);
  count.initialize(0.0);

  vector<int>::iterator iter;
  for (iter = matls.begin(); iter < matls.end(); iter++) {
    int m = *iter;
    
    ParticleSubset* pset = var[m]->getParticleSubset();
    
    //cout << " m: " << m << " *pset " << *pset << endl;
    
    if(pset->numParticles() > 0){
      ParticleSubset::iterator iter = pset->begin();
      for( ;iter != pset->end(); iter++ ){
        IntVector c;
        patch->findCell((*pos[m])[*iter], c);
        ave[c]    = ave[c] + (*var[m])[*iter];
        count[c] += 1;  
      }
    }
  }

  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    ave[c] = ave[c]/(count[c] + 1e-100);
    
    // apply clamps to data only in cells where there are particles
    if(count[c] > 0.0){
      ave[c] = min(ave[c], clamp->maxVal);
      ave[c] = max(ave[c], clamp->minVal);
    } 
  }
}

//__________________________________
//      V E C T O R   V E R S I O N  
void compute_ave( vector<int>                         & matls,
                  const clampVals                     * clamp,
                  vector< ParticleVariable<Vector>* > & var,
                  CCVariable<double>                  & ave,
                  vector< ParticleVariable<Point>* >  & pos,
                  const Patch                         * patch)
{
  IntVector lo = patch->getExtraCellLowIndex();
  IntVector hi = patch->getExtraCellHighIndex();
  
  ave.allocate(lo,hi);
  ave.initialize(0.0);
  
  CCVariable<double> count;
  count.allocate(lo,hi);
  count.initialize(0.0);
  
  vector<int>::iterator iter;
  for (iter = matls.begin(); iter < matls.end(); iter++) {
    int m = *iter;
  
    ParticleSubset* pset = var[m]->getParticleSubset();
    
    if(pset->numParticles() > 0){
      ParticleSubset::iterator iter = pset->begin();

      for( ;iter != pset->end(); iter++ ){
        IntVector c;
        patch->findCell((*pos[m])[*iter], c);
        ave[c]    = ave[c] + (*var[m])[*iter].length();
        count[c] += 1; 
      }
    }
  }

  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    ave[c] = ave[c]/(count[c] + 1e-100);
    
    // apply clamps to data only in cells where there are particles
    if(count[c] > 0.0){
      ave[c] = min(ave[c], clamp->maxVal);
      ave[c] = max(ave[c], clamp->minVal);
    } 
  }
}

//__________________________________
//       M A T R I X 3   V E R S I O N  
void compute_ave( vector<int>                          & matls,
                  const clampVals                      * clamp,
                  vector< ParticleVariable<Matrix3>* > & var,
                  CCVariable<double>                   & ave,
                  vector< ParticleVariable<Point>* >   & pos,
                  const Patch                          * patch)
{
  IntVector lo = patch->getExtraCellLowIndex();
  IntVector hi = patch->getExtraCellHighIndex();
  
  ave.allocate(lo,hi);
  ave.initialize(0.0);
  
  CCVariable<double> count;
  count.allocate(lo,hi);
  count.initialize(0.0);
  
  vector<int>::iterator iter;
  for (iter = matls.begin(); iter < matls.end(); iter++) {
    int m = *iter;    
  
    ParticleSubset* pset = var[m]->getParticleSubset();
    
    if(pset->numParticles() > 0){
      ParticleSubset::iterator iter = pset->begin();

      for( ;iter != pset->end(); iter++ ){
        IntVector c;
        patch->findCell((*pos[m])[*iter], c);
        ave[c]    = ave[c] + (*var[m])[*iter].Norm();
        count[c] += 1;      
      }
    }
  }

  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    ave[c] = ave[c]/(count[c] + 1e-100);
    
    // apply clamps to data only in cells where there are particles
    if(count[c] > 0.0){
      ave[c] = min(ave[c], clamp->maxVal);
      ave[c] = max(ave[c], clamp->minVal);
    } 
  }
}


//__________________________________
//  scale the pixel to 0->255
//  CURRENTLY NOT USED
void scaleImage( const IntVector& lo,
                 const IntVector& hi,
                 CCVariable<double>& ave ) {
  double maxVal = -DBL_MAX;
  double minVal = DBL_MAX;
  
  for (CellIterator iter(lo, hi ); !iter.done(); iter++) {
    IntVector c = *iter;
    maxVal = Max( maxVal, ave[c] );
    minVal = Min( minVal, ave[c] );
  }
  
  double range = fabs(maxVal - minVal);
  for (CellIterator iter(lo, hi ); !iter.done(); iter++) {
    IntVector c = *iter;
    ave[c] = 255 * fabs(ave[c] - minVal)/range;
    //cout << c << " aveScaled: " << ave[c] << endl;
  }
}

//______________________________________________________________________
// Compute the average over all cell & patches in a level
template<class T>
void find_CC_ave( DataArchive                   * archive, 
                  string                        & variable_name, 
                  const Uintah::TypeDescription * subtype,
                  vector<int>                   & matls, 
                  const bool                      use_cellIndex_file,
                  const clampVals               * clampVals,
                  int                             levelIndex,
                  IntVector                     & var_start, 
                  IntVector                     & var_end, 
                  vector<IntVector>               cells,
                  unsigned long                   time_step,
                  CCVariable<double>            & aveLevel )
{
  //__________________________________
  //  does the requested level exist
  bool levelExists = false;
  GridP grid = archive->queryGrid(time_step); 
  int numLevels = grid->numLevels();

  for (int L = 0;L < numLevels; L++) {
    const LevelP level = grid->getLevel(L);
    if (level->getIndex() == levelIndex){
      levelExists = true;
    }
  }

  if (!levelExists){
    cerr<< " Level " << levelIndex << " does not exist at this timestep " << time_step << endl;
  }

  if(levelExists){   // only extract data if the level exists
    const LevelP level = grid->getLevel(levelIndex);

    // find the corresponding patches
    Level::selectType patches;
    level->selectPatches(var_start, var_end + IntVector(1,1,1), patches,true);
    if( patches.size() == 0){
      cerr << " Could not find any patches on Level " << level->getIndex()
           << " that contain cells: " << var_start << " and " << var_end 
           << " Double check the starting and ending indices "<< endl;
      exit(1);
    }
    
    //__________________________________
    // query all the data and compute the average over all the patches
    vector<CCVariable<double>*> ave(patches.size());

    for (int p = 0; p < patches.size(); p++) {
      const Patch* patch = patches[p];

      vector<ParticleVariable<T>*>     pVar(matls.size());
      vector<ParticleVariable<Point>*> pos(matls.size());
      
      vector<int>::iterator iter;
      for (iter = matls.begin(); iter < matls.end(); iter++) {
        int m = *iter;
      
        pVar[m] = scinew ParticleVariable<T>;
        pos[m]  = scinew ParticleVariable<Point>;

        archive->query( *(ParticleVariable<T>*)pVar[m], variable_name, 
                        m, patch, time_step);
        
        archive->query( *(ParticleVariable<Point>*)pos[m], "p.x", 
                        m, patch, time_step);
                        
          
      }
      ave[p] = scinew CCVariable<double>;

      compute_ave( matls, clampVals, pVar, *ave[p], pos, patch );
      
     // cleanup
      for (iter = matls.begin(); iter < matls.end(); iter++) {
        int m = *iter;
        delete pVar[m];
        delete pos[m];
      }
                  
    }  // patches loop
    
    
    //__________________________________
    //  copy the computed average into the level array
    // User input starting and ending indicies    
    if(!use_cellIndex_file) {

      for (CellIterator iter(var_start, var_end ); !iter.done(); iter++) {
        IntVector c = *iter;

        // find out which patch it's on (to keep the printing in sorted order.
        // alternatively, we could just iterate through the patches)
        int p = 0;
        for (; p < patches.size(); p++) {
          IntVector low  = patches[p]->getExtraCellLowIndex();
          IntVector high = patches[p]->getExtraCellHighIndex();

          if (c.x() >= low.x() && c.y() >= low.y() && c.z() >= low.z() && 
              c.x() < high.x() && c.y() < high.y() && c.z() < high.z())
            break;
        }
        if (p == patches.size()) {
          continue;
        }
        
       aveLevel[c] = (*dynamic_cast<CCVariable<double>*>(ave[p]))[c]; 
      }
    }

    //__________________________________
    // If the cell indicies were read from a file. 
    if(use_cellIndex_file) {
      
      for (int i = 0; i<(int) cells.size(); i++) {
        IntVector c = cells[i];
        int p = 0;

        for (; p < patches.size(); p++) {
          IntVector low  = patches[p]->getExtraCellLowIndex();
          IntVector high = patches[p]->getExtraCellHighIndex();

          if (c.x() >= low.x() && c.y() >= low.y() && c.z() >= low.z() && 
              c.x() < high.x() && c.y() < high.y() && c.z() < high.z())
            break;
        }

        if (p == patches.size()) {
          continue;
        }
        aveLevel[c] = (*dynamic_cast<CCVariable<double>*>(ave[p]))[c];

      }
    } // if cell index file
  } // if level exists
}

/*_______________________________________________________________________
 Function:  readCellIndicies--
 Purpose: reads in a list of cell indicies
_______________________________________________________________________ */
void readCellIndicies(const string& filename, vector<IntVector>& cells)
{ 
  // open the file
  ifstream fp(filename.c_str());
  if (!fp){
    cerr << "Couldn't open the file that contains the cell indicies " << filename<< endl;
  }
  char c;
  int i,j,k;
  string text, comma;  
  
  while (fp >> c) {
    fp >> text>>i >> comma >> j >> comma >> k;
    IntVector indx(i,j,k);
    cells.push_back(indx);
    fp.get(c);
  }
  // We should do some bullet proofing here
  //for (int i = 0; i<(int) cells.size(); i++) {
  //  cout << cells[i] << endl;
  //}
}
//______________________________________________________________________
//______________________________________________________________________

int main(int argc, char** argv)
{

  //__________________________________
  //  Default Values
  bool use_cellIndex_file = false;
  bool findCellIndices = true;

  unsigned long time_start = 0;
  unsigned long time_end = (unsigned long)-1;
  
  string input_uda_name;  
  string input_file_cellIndices;

  string base_dir_name("-");  
  Dir base_dir;                      // base output directory
  
  IntVector var_start(0,0,0);
  IntVector var_end(0,0,0);
  
  Point     start_pt(-9,-9,-9);
  Point     end_pt(-9,-9,-9);
  
  int levelIndex = 0;
  vector<IntVector> cells;
  string variable_name;
  
  clampVals* clamps = scinew clampVals();
  clamps->minVal = -DBL_MAX;
  clamps->maxVal = DBL_MAX;
  
  tiffFlags* flags = scinew tiffFlags();
  flags->orientation = ORIENTATION_BOTLEFT;

  int matl = 0;
  unsigned int nBits     = 8;
  bool doVerification    = false;
  saveSlices sliceFormat = asVolume;

  //__________________________________
  // Parse arguments

  for(int i=1;i<argc;i++){
    string s  =argv[i];
    
    if(s == "-v" || s == "--variable") {
      variable_name = string(argv[++i]);
    } else if ( s == "-m" || s == "--material") {
      string me = string(argv[++i]);
      if( me == "a" || me == "all" ){
        matl = 999;
      } else {
        matl = atoi(argv[++i]);
      }
    } else if ( s == "-tlow" || s == "--timesteplow") {
      time_start = strtoul(argv[++i],NULL,10);
    } else if ( s == "-thigh" || s == "--timestephigh") {
      time_end = strtoul(argv[++i], NULL,10);
    } else if ( s == "-timestep" || s == "--timestep") {
      int me = strtoul(argv[++i], NULL,10);
      time_start = me;
      time_end   = me;
    } else if ( s == "-istart" || s == "--indexs") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      var_start = IntVector(x,y,z);
    } else if ( s == "-iend" || s == "--indexe") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      
      findCellIndices = false;
      var_end = IntVector(x,y,z);
    } else if ( s == "-startPt" ) {
      double x = atof(argv[++i]);
      double y = atof(argv[++i]);
      double z = atof(argv[++i]);
      start_pt = Point(x,y,z);
    } else if ( s == "-endPt" ) {
      double x = atof(argv[++i]);
      double y = atof(argv[++i]);
      double z = atof(argv[++i]);
      end_pt = Point(x,y,z);
      findCellIndices = true;
    } else if ( s == "-l" || s == "--level" ) {
      levelIndex = atoi(argv[++i]);
    } else if ( s == "-h" || s == "--help" ) {
      usage( "", argv[0] );
    } else if ( s == "-uda" ) {
      input_uda_name = string(argv[++i]);
    } else if ( s == "-d" || s == "--dir" ) {
      base_dir_name = string(argv[++i]);
    } else if ( s == "--cellIndexFile" ) {
      use_cellIndex_file = true;
      input_file_cellIndices = string(argv[++i]);
    } else if ( s == "-max" ) {
      clamps->maxVal = atof(argv[++i]);
    } else if ( s == "-min" ) {
      clamps->minVal = atof(argv[++i]);
    } else if ( s == "-orientation" ) {
      flags->orientation = whichOrientation( string(argv[++i]) );
    } else if ( s == "-verify" ) {
      doVerification = true;
    } else if ( s == "-asVolume" ) {
      sliceFormat = asVolume;
    } else if ( s == "-asSlices" ) {
      sliceFormat = asIndividualSlices;
    } else if ( s == "-nBits" ) {
      nBits = atoi(argv[++i]);
    }else {
      usage(s, argv[0]);
    }
  }
  //__________________________________
  //  bulletproofing inputs
  if(input_uda_name == ""){
    cerr << "\n\nNo archive file specified, now exiting....\n";
    usage("", argv[0]);
  }
  if( base_dir_name == "-" ){
    cerr << "\n\nUnspecified output directory name, now exiting....\n";
    usage("", argv[0]);
  }
  
  

  try {
    DataArchive* archive = scinew DataArchive(input_uda_name);
    
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;

    archive->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());

    bool var_found = false;
    unsigned int var_index = 0;
    for (;var_index < vars.size(); var_index++) {
      if (variable_name == vars[var_index]) {
        var_found = true;
        break;
      }
    }
    
    //__________________________________
    // bulletproofing
    if (!var_found) {
      cerr << "Variable \"" << variable_name << "\" was not found.\n";
      cerr << "If a variable name was not specified try -var [name].\n";
      cerr << "Possible variable names are:\n";
      var_index = 0;
      for (;var_index < vars.size(); var_index++) {
        cout << "vars[" << var_index << "] = " << vars[var_index] << endl;
      }
      cerr << "Aborting!!\n";
      exit(-1);
    }

    //__________________________________
    // get type and subtype of data
    const Uintah::TypeDescription* td = types[var_index];
    const Uintah::TypeDescription* subtype = td->getSubType();

    //______________________________________________________________________
    //query time info from data archive
    vector<int> index;
    vector<double> times;

    archive->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());

    // set default max time value
    if (time_end == (unsigned long)-1) {
      cout <<"There are " << index.size() << " timesteps\n Initializing time_step_upper to "<<times.size()-1<<"\n";
      time_end = times.size() - 1;
    }

    //__________________________________
    // bullet proofing 
    if (time_end >= times.size() || time_end < time_start) {
      cout << "timestephigh("<<time_end<<") must be greater than " << time_start 
           << " and less than " << times.size()-1 << endl;
      exit(1);
    }
    if (time_start >= times.size() || time_end > times.size()) {
      cout << "timestep must be between 0 and " << times.size()-1 << endl;
      exit(1);
    }
    
    // create the base output directory
    if( Dir::removeDir(base_dir_name.c_str() ) ){
      cout << "Removed directory: "<<base_dir_name<<"\n";
    }
    base_dir = Dir::create(base_dir_name);

    if(base_dir.exists() ) {
      cout << "Created directory: "<<base_dir_name<<"\n";
    }else{
      cout << "Failed creating  base output directory: "<<base_dir_name<<"\n";
      exit(1);
    }

    //__________________________________
    // loop over timesteps
    for (unsigned long time_step = time_start; time_step <= time_end; time_step++) {

      cout << "Timestep["<<time_step<<"] = " << times[time_step]<< endl;
      GridP grid = archive->queryGrid(time_step);
      const LevelP level = grid->getLevel(levelIndex);

      //__________________________________
      //  find indices to extract for
      if(findCellIndices) {
        if( level  ){ 
          if (start_pt != Point(-9,-9,-9) ) {
            var_start=level->getCellIndex(start_pt);
            var_end  =level->getCellIndex(end_pt); 
          } else{
            level->findInteriorCellIndexRange(var_start, var_end);
          }
        }
      }
      
      //__________________________________
      //  find the number of matls at this timestep
      vector<int> matls;
      if(matl == 999){       // all matls
        const Patch* patch = *(level->patchesBegin());
        int numMatls = archive->queryNumMaterials( patch, time_step);
        for (int m = 0; m< numMatls; m++ ){
          matls.push_back(m);
        }
      } else {
        matls.push_back(matl);
      }

      cout << "  " << vars[var_index] << ": " << types[var_index]->getName() 
           << " being extracted and averaged for material(s): ";
           
      vector<int>::iterator m;
      for (m = matls.begin(); m < matls.end(); m++) {
        cout << *m << ", "; 
      }
      cout  <<" between cells "<<var_start << " and " << var_end <<endl;
 
      //__________________________________
      // read in cell indices from a file
      if ( use_cellIndex_file) {
        readCellIndicies(input_file_cellIndices, cells);
      }
      
      //__________________________________
      //  Array containing the average over all patches
      CCVariable<double> aveLevel;
      IntVector lo, hi;
      level->findInteriorCellIndexRange(lo, hi);
      aveLevel.allocate(lo,hi);
      aveLevel.initialize(0.0);
         
      //__________________________________
      //  P A R T I C L E   V A R I A B L E  
      if(td->getType() == Uintah::TypeDescription::ParticleVariable){
        switch (subtype->getType()) {
        case Uintah::TypeDescription::double_type:
          find_CC_ave<double>( archive, variable_name, subtype, matls, use_cellIndex_file,
                               clamps, levelIndex, var_start, var_end, cells, time_step, aveLevel);
          break;
        case Uintah::TypeDescription::Vector:
          find_CC_ave<Vector>( archive, variable_name, subtype, matls, use_cellIndex_file,
                               clamps, levelIndex, var_start, var_end, cells, time_step, aveLevel);
          break;
        case Uintah::TypeDescription::Matrix3:
          find_CC_ave<Matrix3>( archive, variable_name, subtype, matls, use_cellIndex_file,
                               clamps, levelIndex, var_start, var_end, cells, time_step, aveLevel);
          break;
        case Uintah::TypeDescription::Other:
          // don't break on else - flow to the error statement
        case Uintah::TypeDescription::bool_type:
        case Uintah::TypeDescription::short_int_type:
        case Uintah::TypeDescription::long_type:
        case Uintah::TypeDescription::long64_type:
          cerr << "Subtype is not implemented\n";
          exit(1);
          break;
        default:
          cerr << "Unknown subtype\n";
          exit(1);
        }
      }

      //__________________________________
      //  write out a tiff volume
      if (sliceFormat == asVolume) {
        ostringstream tname;
        tname << base_dir_name<<"/t" << setw(5) << setfill('0') << time_step << ".tif";
        
        write_tiff_volume(flags,tname, var_start, var_end, aveLevel, doVerification, nBits);            // write the tiff out
      }
      
      //__________________________________
      //
      // create the base output directory
      if (sliceFormat == asIndividualSlices) {
        
        ostringstream tname;
        tname << base_dir_name<<"/t" << setw(5) << setfill('0') << time_step;
        
        Dir timestep_dir = Dir::create(tname.str());

        if(timestep_dir.exists() ) {
          cout << "  Created directory: "<<tname.str()<<"\n";
        }else{
          cout << "Failed creating  base output directory: "<<tname<<"\n";
          exit(1);
        }
        
        ostringstream sliceName;
        sliceName << tname.str() << "/";
        write_tiff_slices(flags,sliceName, var_start, var_end, aveLevel, doVerification, nBits);            // write the tiff out
        
      }
      
      
    }  // timestep loop     
    
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    exit(1);
  } catch(...){
    cerr << "Caught unknown exception\n";
    exit(1);
  }
  
  // cleanup
  delete clamps;
  delete flags;
  
}
