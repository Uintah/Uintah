#include <Packages/rtrt/Core/GeoProbeReader.h>
#include <Core/Geometry/Vector.h>
#include <stdio.h>
#include <iostream>

using namespace SCIRun;
using std::cerr;
using std::cout;

int read_geoprobe(const char *fname, int &nx, int &ny, int &nz,
		  Point &min, Point &max, unsigned char &datamin,
		  unsigned char &datamax, Array3<unsigned char> &data) {
  FILE *data_file = fopen(fname, "rb");
  if (!data_file) {
    cerr << "Error -- unable to open geovoxel file: "<<fname<<"\n";
    return 0;
  }
  GP_hdr hdr;
  if (fread(&hdr, sizeof(GP_hdr), 1, data_file) != 1) {
    cerr << "Error reading geovoxel header in file: "<<fname<<"\n";
    return 0;
  }
  cerr << "hdr.nbits="<<hdr.nbits<<"\n";
  if (hdr.nbits != 8) {
    cerr << "Error -- we only know how to read 8-bit data right now.\n";
    return 0;
  }
  nx = hdr.zsize;
  ny = hdr.ysize;
  nz = hdr.xsize;
  cout << "Data size = ("<<hdr.xsize<<", "<<hdr.ysize<<", "<<hdr.zsize<<")\n";
  cout << "Return size = ("<<nx<<", "<<ny<<", "<<nz<<")\n";

  hdr.xstep = hdr.ystep = hdr.zstep = 1;
  hdr.xoffset = hdr.yoffset = hdr.zoffset = 0;

  // This is the data structure that we are passing back, because of
  // the internal workings of this array, we need to make sure that
  // the order of dimension sizes is consistent with what we return.
  // It should be nx, ny, and nz.
  data.resize( nx , ny , nz);

#if 1
  // Because the data is ordered backwards (z,y,z instead of x,y,z),
  // we simply can't just stuff all the memory into a single pointer.
  // We must iterate over the data and then store it one piece at a
  // time.
  int buffer_size = nz*ny;
  unsigned char *buffer = new unsigned char[buffer_size];
  size_t total_size = nx*ny*nz;
  size_t total_read = 0;
 
  for(int x = 0; x < nx; x++) {
    if (fread(buffer, sizeof(unsigned char), buffer_size, data_file)
	!= buffer_size) {
      cerr << "Error -- did not find "<<total_size<<" in input file!\n";
      return 0;
    }
    total_read+=buffer_size;
    // Copy the data over to our array, we could also compute min and
    // max here if we wanted.
    unsigned char *buffp = buffer;
    for(int y = 0; y < ny; y++)
      for(int z = 0; z < nz; z++)
	{
	  data(x,y,z) = *buffp;
	  buffp++;
	}
  }
  if (total_read != total_size) {
    cerr << "Error -- got "<<total_size<<" bytes instead of "<<total_read<<" bytes\n";
    return 0;
  }
#endif

  int i,j,k;
  unsigned char swap;

#if 0
  unsigned char *datap = scinew unsigned char[nx*ny*nz];
  if (!datap) {
    cerr << "Could not allocate datap memory\n";
    return 0;
  }
  if (fread(datap, sizeof(unsigned char), nx*ny*nz, data_file) != nx*ny*nz) {
    cerr << "Error -- did not find "<<nx*ny*nz<<" in input file!\n";
    return 0;
  }

  // The data coming in is ordered fastest to slowest z,y,x.  This
  // needs to be reordered to be x,y,z.
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++, datap++) {
	swap=data(i,j,k);
	data(i,j,k)=*datap;
	*datap=swap;
      }
  delete datap;
#endif
  
  bool flipx = false;
  bool flipy = false;
  bool flipz = false;

  cerr << "hdr.offset="<<Vector(hdr.xoffset,hdr.yoffset,hdr.zoffset)<<"\n";
  cerr << "hdr.step="<<Vector(hdr.xstep,hdr.ystep,hdr.zstep)<<"\n";

  if (hdr.xstep < 0) {
    flipx = true;
    hdr.xoffset=hdr.xoffset+(hdr.xsize-1)*hdr.xstep;
    hdr.xstep=-hdr.xstep;
  }
  if (hdr.ystep < 0) {
    flipy = true;
    hdr.yoffset=hdr.yoffset+(hdr.ysize-1)*hdr.ystep;
    hdr.ystep=-hdr.ystep;
  }
  if (hdr.zstep < 0) {
    flipz = true;
    hdr.zoffset=hdr.zoffset+(hdr.zsize-1)*hdr.zstep;
    hdr.zstep=-hdr.zstep;
  }
  min=Point(hdr.zoffset, hdr.yoffset, hdr.xoffset);
  max=min+Vector(hdr.zstep*(hdr.zsize-1),
		 hdr.ystep*(hdr.ysize-1),
		 hdr.xstep*(hdr.xsize-1));


#if 0
  // Compute the min and max of the data
  datamin=datamax=data(0,0,0);
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++)
	if (data(i,j,k) < datamin)
	  datamin=data(i,j,k);
	else if (data(i,j,k) > datamax)
	  datamax=data(i,j,k);
#else
  datamin = 0;
  datamax = 255;
#endif

#if 0
  if (flipx)
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for (i=0; i<nx; i++) {
	  swap=data(i,j,k);
	  data(i,j,k)=data(i,j,nz-k-1);
	  data(j,i,nz-k-1)=swap;
	}

  if (flipy)
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for (i=0; i<nx; i++) {
	  swap=data(i,j,k);
	  data(i,j,k)=data(i,ny-j-1,k);
	  data(i,ny-j-1,k)=swap;
	}

  if (flipz)
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for (i=0; i<nx; i++) {
	  swap=data(i,j,k);
	  data(i,j,k)=data(nx-i-1,j,k);
	  data(nx-i-1,j,k)=swap;
	}
#endif
  
  return 1;
}
