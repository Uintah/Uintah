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

#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>

using namespace Uintah;
using namespace std;

typedef unsigned char byte;

// forwared function declarations
void usage( char *prog_name );
bool isPointInsideSphere(double xv, double yv, double zv,
                         double xcen, double ycen, double zcen, double rad);

//----------------------------------------------------------------------------------
/*
Generate a raw file containing a 3D image based on a text file containing sphere
descriptions, namely, x, y and z of the centers, and radii.  This can then be used
with the file geometry piece type, after processing with pfs2, to generate geometry for Uintah simulations.
*/
int main(int argc, char *argv[])
{
  // Establish physical size of the image
  vector<double> X(3);
  X[0]=1.;
  X[1]=1.;
  X[2]=1.;

  // image resolution
  vector<int> res(3);
  res[0]=256;
  res[1]=256;
  res[2]=256;

  // calculate voxel size
  double dx=X[0]/((double) res[0]);
  double dy=X[1]/((double) res[1]);
  double dz=X[2]/((double) res[2]);
  if(dx!=dy || dx!=dz || dy !=dz){
    cerr << "WARNING:  Subsequent code assumes that voxel dimensions are equal\n";
  }

  // Open file containing sphere center locations and radii
  string spherefile_name = "spheres.txt";
  ifstream fp(spherefile_name.c_str());
  if(fp==0){
     cout << "FATAL ERROR : Failed opening spheres.txt file" << endl;
     exit(0);
  }

  // Read data from file
  double xc, yc, zc, r;
  vector<double> xcen,ycen,zcen,rad;
  while(fp >> xc >> yc >> zc >> r){
   xcen.push_back(xc);
   ycen.push_back(yc);
   zcen.push_back(zc);
   rad.push_back(r);
  } 

  // make room to store the image
  int nsize = res[0]*res[1]*res[2];
  byte* pimg = scinew byte[nsize];

  // Initialize pimg to zero
  for(int n=0;n<nsize;n++){
    pimg[n] = 0;
  }

  // Loop over spheres
  for(unsigned int ns = 0; ns<xcen.size(); ns++){
    // find the cell that contains the sphere center
    int ic = static_cast<int>(xcen[ns]/dx);
    int jc = static_cast<int>(ycen[ns]/dy);
    int kc = static_cast<int>(zcen[ns]/dz);
    // find a conservative estimate of how many cells make up the sphere radius
    int rc = static_cast<int>(rad[ns]/dx)+2;
    // determine the bounding box that contains the sphere
    int bbx_min = max(0,ic-rc);
    int bbx_max = min(res[0],ic+rc);
    int bby_min = max(0,jc-rc);
    int bby_max = min(res[1],jc+rc);
    int bbz_min = max(0,kc-rc);
    int bbz_max = min(res[2],kc+rc);
  
   // Loop over all voxels of the image to determine which are "inside" the sphere
   for(int i=bbx_min;i<bbx_max;i++){
     for(int j=bby_min;j<bby_max;j++){
       for(int k=bbz_min;k<bbz_max;k++){
         double xv=((double) i + 0.5)*dx;
         double yv=((double) j + 0.5)*dy;
         double zv=((double) k + 0.5)*dz;
         if(isPointInsideSphere(xv,yv,zv,xcen[ns],ycen[ns],zcen[ns],rad[ns])){
           //find the index into pimg and increment pimg
           unsigned int p=k*(res[0]*res[1])+j*res[0]+i;
           pimg[p]++;
         }
       }
     }
   }
  }

  // Write image data to a file
  string f_name = "spheres.raw";
  FILE* dest = fopen(f_name.c_str(), "wb");
  if(dest==0){
    cout << "FATAL ERROR : Failed opening points file" << endl;
    exit(0);
  }
  fwrite(pimg, sizeof(byte), nsize, dest);

  // clean up image data
  delete [] pimg;
}

// function usage : prints a message on how to use the program
//
void usage( char *prog_name )
{
  cout << "Usage: " << prog_name << " [-b] [-B] [-cyl <args>] infile \n";
  cout << "-b,B: binary output \n";
  cout << "-cyl: defines a cylinder within the geometry \n";
  cout << "args = xbot ybot zbot xtop ytop ztop radius \n";
  exit( 1 );
}

//--------------------------------------------------------------------------------
bool isPointInsideSphere(double xv, double yv, double zv,
                         double xcen, double ycen, double zcen, double rad)
{
  Vector diff(xv-xcen,yv-ycen,zv-zcen);

  if(diff.length2() > rad*rad){
    return false;
  } else{
    return true;
  } 
}
