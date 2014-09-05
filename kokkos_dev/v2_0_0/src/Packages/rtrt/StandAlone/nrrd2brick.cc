

#include <Packages/rtrt/Core/BrickArray3.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <iostream>
#include <fstream>
#include <limits.h>
#include <teem/nrrd.h>
//#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

using namespace std;
using namespace rtrt;
using namespace SCIRun;

void compute_hist(char *filebase, int nhist, int nx, int ny, int nz,
		  BrickArray3<float> &blockdata,
		  float datamin, float datamax)
{
  long *hist = new long[nhist];
  bool recompute_hist = true;
  char buf[200];
  if (filebase != NULL) {
    sprintf(buf, "%s.hist_%d", filebase, nhist);
    cerr << "Looking for histogram in " << buf << "\n";
    ifstream in(buf);
    if(in){
      for(int i=0;i<nhist;i++){
	in >> hist[i];
      }
      recompute_hist = false;
    }
  }
  if (recompute_hist) {
    cerr << "Recomputing Histgram\n";
    float scale=(nhist-1)/(datamax-datamin);
    int nx1=nx-1;
    int ny1=ny-1;
    int nz1=nz-1;
    int nynz=ny*nz;
    //cerr << "scale = " << scale;
    //cerr << "\tnx1 = " << nx1;
    //cerr << "\tny1 = " << ny1;
    //cerr << "\tnz1 = " << nz1;
    //cerr << "\tnynz = " << nynz << endl;
    for(int ix=0;ix<nx1;ix++){
      for(int iy=0;iy<ny1;iy++){
	int idx=ix*nynz+iy*nz;
	for(int iz=0;iz<nz1;iz++){
	  float p000=blockdata(ix,iy,iz);
	  float p001=blockdata(ix,iy,iz+1);
	  float p010=blockdata(ix,iy+1,iz);
	  float p011=blockdata(ix,iy+1,iz+1);
	  float p100=blockdata(ix+1,iy,iz);
	  float p101=blockdata(ix+1,iy,iz+1);
	  float p110=blockdata(ix+1,iy+1,iz);
	  float p111=blockdata(ix+1,iy+1,iz+1);
	  float min=Min(Min(Min(p000, p001), Min(p010, p011)), Min(Min(p100, p101), Min(p110, p111)));
	  float max=Max(Max(Max(p000, p001), Max(p010, p011)), Max(Max(p100, p101), Max(p110, p111)));
	  int nmin=(int)((min-datamin)*scale);
	  int nmax=(int)((max-datamin)*scale+.999999);
	  if(nmax>=nhist)
	    nmax=nhist-1;
	  if(nmin<0)
	    nmin=0;
	  if(nmax>nhist)
	    nmax=nhist;
	  //if ((nmin != 0) || (nmax != 0))
	  //  cerr << "nmin = " << nmin << "\tnmax = " << nmax << endl;
	  for(int i=nmin;i<nmax;i++){
	    hist[i]++;
	  }
	  idx++;
	}
      }
    }
    if (filebase != NULL) {
      ofstream out(buf);
      for(int i=0;i<nhist;i++){
	out << hist[i] << '\n';
      }
    }
  }
  cerr << "Done building histogram\n";
}    


// OK, so this takes a raw file that should have a nhdr associated with it.
// It converts the data to float if need be
// Then shoves it into a brick, while computing the min and max.
// The histogram is created, and written to disk.
// The brick is written to disk.
// The header is written to disk with the min and max stuff.

int main(int argc, char *argv[]) {
  char *in_file = 0;
  char rawbase[1000];
  char nrrdname[1000];
  char headername[1000];
  char brickname[1000];
  int nhist = 400;
  
  if (argc < 2) {
    cout << "nrrd2brick.cc <nrrd base name>"<<endl;
    return 0;
  }

  in_file = argv[1];
  
  // Create file names
  sprintf(rawbase, "%s.raw", in_file);
  sprintf(nrrdname, "%s.nhdr", in_file);
  sprintf(brickname, "%s.brick", rawbase);
  sprintf(headername, "%s.hdr", rawbase);
  
  BrickArray3<float> data;
  float data_min = FLT_MAX;
  float data_max = -FLT_MAX;
  Point minP, maxP;
  // Do the nrrd stuff
  Nrrd *n = nrrdNew();
  // load the nrrd in
  if (nrrdLoad(n,nrrdname,NULL)) {
    char *err = biffGet(NRRD);
    cerr << "Error reading nrrd "<< nrrdname <<": "<<err<<"\n";
    free(err);
    biffDone(NRRD);
    return 0;
  }
  // check to make sure the dimensions are good
  if (n->dim != 3) {
    cerr << "VolumeVisMod error: nrrd->dim="<<n->dim<<"\n";
    cerr << "  Can only deal with 3-dimensional scalar fields... sorry.\n";
    return 0;
  }
  // convert the type to floats if you need to
  size_t num_elements = nrrdElementNumber(n);
  cerr << "Number of data members = " << num_elements << endl;
  if (n->type != nrrdTypeFloat) {
    cerr << "Converting type from ";
    switch(n->type) {
    case nrrdTypeUnknown: cerr << "nrrdTypeUnknown"; break;
    case nrrdTypeChar: cerr << "nrrdTypeChar"; break;
    case nrrdTypeUChar: cerr << "nrrdTypeUChar"; break;
    case nrrdTypeShort: cerr << "nrrdTypeShort"; break;
    case nrrdTypeUShort: cerr << "nrrdTypeUShort"; break;
    case nrrdTypeInt: cerr << "nrrdTypeInt"; break;
    case nrrdTypeUInt: cerr << "nrrdTypeUInt"; break;
    case nrrdTypeLLong: cerr << "nrrdTypeLLong"; break;
    case nrrdTypeULLong: cerr << "nrrdTypeULLong"; break;
    case nrrdTypeDouble: cerr << "nrrdTypeDouble"; break;
    default: cerr << "Unknown!!";
    }
    cerr << " to nrrdTypeFloat\n";
    Nrrd *new_n = nrrdNew();
    nrrdConvert(new_n, n, nrrdTypeFloat);
    // since the data was copied blow away the memory for the old nrrd
    nrrdNuke(n);
    n = new_n;
    cerr << "Number of data members = " << num_elements << endl;
  }
  // get the dimensions
  int nx, ny, nz;
  nx = n->axis[0].size;
  ny = n->axis[1].size;
  nz = n->axis[2].size;
  cout << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
  cout << "total = " << nz * ny * nz << endl;
  cout << "spacing = " << n->axis[0].spacing << " x "<<n->axis[1].spacing<< " x "<<n->axis[2].spacing<< endl;
  data.resize(nx,ny,nz); // resize the bricked data
  // get the physical bounds
  minP = Point(0,0,0);
  maxP = Point((nx - 1) * n->axis[0].spacing,
	       (ny - 1) * n->axis[1].spacing,
	       (nz - 1) * n->axis[2].spacing);
  // lets normalize the dimensions to 1
  Vector size = maxP - minP;
  // find the biggest dimension
  double max_dim = Max(Max(size.x(),size.y()),size.z());
  maxP = ((maxP-minP)/max_dim).asPoint();
  minP = Point(0,0,0);
  // copy the data into the brickArray
  cerr << "Number of data members = " << num_elements << endl;
  float *p = (float*)n->data; // get the pointer to the raw data
  for (int z = 0; z < nz; z++)
    for (int y = 0; y < ny; y++)
      for (int x = 0; x < nx; x++) {
	float val = *p++;
	data(x,y,z) = val;
	// also find the min and max
	if (val < data_min)
	  data_min = val;
	else if (val > data_max)
	  data_max = val;
      }
#if 0
  // compute the min and max of the data
  double dmin,dmax;
  nrrdMinMaxFind(&dmin,&dmax,n);
  data_min = (float)dmin;
  data_max = (float)dmax;
#endif
  // delete the memory that is no longer in use
  nrrdNuke(n);


  ///////////////////////////////////////////////////////////////
  // write the bricked data to a file, so that we don't have to rebrick it
  //    ofstream bout(buf);
  //    if (!bout) {
  int bout_fd = open (brickname, O_WRONLY | O_CREAT | O_TRUNC,
		      S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  
  if (bout_fd == -1 ) {
    cerr << "Error in opening file " << brickname << " for writing.\n";
    exit(1);
  }
  cerr << "Writing " << brickname << "...";
  write(bout_fd, data.get_dataptr(), data.get_datasize());
  
  //////////////////////////////////////////////////////////////////
  // write the header
#if 0
  // override the min and max if it was passed in
  if (override_data_min)
    data_min = data_min_in;
  if (override_data_max)
    data_max = data_max_in;
#endif
  
  cout << "minP = "<<minP<<", maxP = "<<maxP<<endl;
  FILE *header = fopen(headername, "wc");
  if (header == 0) {
    cerr << "Error opening header named "<<headername<<endl;
    return 1;
  }
  minP = Point(-1,-1,-1);
  maxP = Point(1,1,1);
  fprintf(header, "%d %d %d\n", nx, ny, nz);
  fprintf(header, "%g %g %g\n", minP.x(), minP.y(), minP.z());
  fprintf(header, "%g %g %g\n", maxP.x(), maxP.y(), maxP.z());
  fprintf(header, "%g %g\n", data_min, data_max);

  ////////////////////////////////////////////////////////////////////
  // compute the histogram
  compute_hist(rawbase, nhist, nx, ny, nz, data, data_min, data_max);
  
  return 0;
}
