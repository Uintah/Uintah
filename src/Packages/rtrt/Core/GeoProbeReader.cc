#include <Packages/rtrt/Core/GeoProbeReader.h>
#include <Core/Geometry/Vector.h>
#include <stdio.h>
#include <iostream>

using namespace SCIRun;
using std::cerr;

int read_geoprobe(char *fname, int &nx, int &ny, int &nz,
		  Point &min, Point &max, unsigned char &datamin,
		  unsigned char &datamax, Array3<unsigned char> &data) {
  FILE *f = fopen(fname, "rb");
  if (!f) {
    cerr << "Error -- unable to open geovoxel file: "<<fname<<"\n";
    return 0;
  }
  GP_hdr hdr;
  if (fread(&hdr, sizeof(GP_hdr), 1, f) != 1) {
    cerr << "Error reading geovoxel header in file: "<<fname<<"\n";
    return 0;
  }
  if (hdr.nbits != 8) {
    cerr << "Error -- we only know how to read 8-bit data right now.\n";
    return 0;
  }
  data.resize(hdr.zsize, hdr.ysize, hdr.xsize);
  nx=hdr.xsize;
  ny=hdr.ysize;
  nz=hdr.zsize;

  hdr.xstep = hdr.ystep = hdr.zstep = 1;
  hdr.xoffset = hdr.yoffset = hdr.zoffset = 0;

  unsigned char *d = &(data(0,0,0));

  if (fread(d, sizeof(unsigned char), nx*ny*nz, f) != nx*ny*nz) {
    cerr << "Error -- did not find "<<nx*ny*nz<<" in input file!\n";
    return 0;
  }
  int flipx=0;
  int flipy=0;
  int flipz=0;

  cerr << "hdr.offset="<<Vector(hdr.xoffset,hdr.yoffset,hdr.zoffset)<<"\n";
  cerr << "hdr.step="<<Vector(hdr.xstep,hdr.ystep,hdr.zstep)<<"\n";

  if (hdr.xstep<0) {
    flipx=1;
    hdr.xoffset=hdr.xoffset+(hdr.xsize-1)*hdr.xstep;
    hdr.xstep=-hdr.xstep;
  }
  if (hdr.ystep<0) {
    flipy=1;
    hdr.yoffset=hdr.yoffset+(hdr.ysize-1)*hdr.ystep;
    hdr.ystep=-hdr.ystep;
  }
  if (hdr.zstep<0) {
    flipz=1;
    hdr.zoffset=hdr.zoffset+(hdr.zsize-1)*hdr.zstep;
    hdr.zstep=-hdr.zstep;
  }
  min=Point(hdr.xoffset, hdr.yoffset, hdr.zoffset);
  max=min+Vector(hdr.xstep*(hdr.xsize-1),
		 hdr.ystep*(hdr.ysize-1),
		 hdr.zstep*(hdr.zsize-1));

  int i,j,k;
  unsigned char swap;
  unsigned char *pts = d;

  for (i=0; i<hdr.xsize; i++)
    for (j=0; j<hdr.ysize; j++)
      for (k=0; k<hdr.zsize; k++, d++) {
	swap=data(k,j,i);
	data(k,j,i)=*d;
	*d=swap;
      }

  datamin=datamax=data(0,0,0);
  for (k=0; k<hdr.zsize; k++)
    for (j=0; j<hdr.ysize; j++)
      for (i=0; i<hdr.xsize; i++)
	if (data(k,j,i)<datamin) datamin=data(k,j,i);
	else if (data(k,j,i)>datamax) datamax=data(k,j,i);

  if (flipx)
    for (k=0; k<hdr.zsize; k++)
      for (j=0; j<hdr.ysize; j++)
	for (i=0; i<hdr.xsize; i++) {
	  swap=data(k,j,i);
	  data(k,j,i)=data(hdr.zsize-k-1,j,i);
	  data(hdr.zsize-k-1,j,i)=swap;
	}

  if (flipy)
    for (k=0; k<hdr.zsize; k++)
      for (j=0; j<hdr.ysize; j++)
	for (i=0; i<hdr.xsize; i++) {
	  swap=data(k,j,i);
	  data(k,j,i)=data(k,hdr.ysize-j-1,i);
	  data(k,hdr.ysize-j-1,i)=swap;
	}

  if (flipz)
    for (k=0; k<hdr.zsize; k++)
      for (j=0; j<hdr.ysize; j++)
	for (i=0; i<hdr.xsize; i++) {
	  swap=data(k,j,i);
	  data(k,j,i)=data(k,j,hdr.xsize-i-1);
	  data(k,j,hdr.xsize-i-1)=swap;
	}

  return 1;
}
