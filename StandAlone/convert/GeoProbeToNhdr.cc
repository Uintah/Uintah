#include <StandAlone/convert/GpVolHdr.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <stdio.h>
#include <iostream>

using std::cerr;

int main(int argc, char **argv) {
  if (argc < 3) {
    cerr << "Usage: "<<argv[0]<<" inputFile.vol outputFile.nhdr \n";
    return 2;
  }

  FILE *vol_file = fopen(argv[1], "rb");
  if (!vol_file) {
    cerr << "Error -- unable to open geovoxel file: "<<argv[1]<<"\n";
    return 0;
  }
  GP_hdr vol_hdr;
  if (fread(&vol_hdr, sizeof(GP_hdr), 1, vol_file) != 1) {
    cerr << "Error reading geovoxel header in file: "<<argv[1]<<"\n";
    return 0;
  }
  fclose(vol_file);

  if (vol_hdr.nbits != 8 && vol_hdr.nbits != 16 && vol_hdr.nbits != 32) {
    cerr << "Error -- we only know how to read 8/16/32-bit data right now.\n";
    return 0;
  }
  FILE *nhdr_file = fopen(argv[2], "wt");
  if (!nhdr_file) {
    cerr << "Error -- unable to open NRRD header file: "<<argv[2]<<"\n";
    return 0;
  }
  fprintf(nhdr_file, "NRRD0001\n");
  fprintf(nhdr_file, "content: %s %s %s\n", argv[0], argv[1], argv[2]);
  fprintf(nhdr_file, "type: ");
  if (vol_hdr.nbits == 8)
    fprintf(nhdr_file, "uchar\n");
  else if (vol_hdr.nbits == 16)
    fprintf(nhdr_file, "ushort\n");
  else //if (vol_hdr.nbits == 32)
    fprintf(nhdr_file, "uint\n");
  fprintf(nhdr_file, "dimension: 3\n");
  fprintf(nhdr_file, "sizes: %d %d %d\n", 
	  vol_hdr.xsize, vol_hdr.ysize, vol_hdr.zsize);
  fprintf(nhdr_file, "spacings: %lf %lf %lf\n", 
	  vol_hdr.xstep, vol_hdr.ystep, vol_hdr.zstep);
  fprintf(nhdr_file, "axis mins: %lf %lf %lf\n",
	  vol_hdr.xoffset, vol_hdr.yoffset, vol_hdr.zoffset);
  fprintf(nhdr_file, "axis maxs: %lf %lf %lf\n", 
	  vol_hdr.xoffset+vol_hdr.xstep*(vol_hdr.xsize-1), 
	  vol_hdr.yoffset+vol_hdr.ystep*(vol_hdr.ysize-1), 
	  vol_hdr.zoffset+vol_hdr.zstep*(vol_hdr.zsize-1));
  fprintf(nhdr_file, "data file: %s\n", argv[1]);
  fprintf(nhdr_file, "encoding: raw\n");
  fprintf(nhdr_file, "byte skip: %d\n", sizeof(GP_hdr)); 
  fclose(nhdr_file);
  return 1;
}
