/*

cd /home/bigler/SCIRun/src/Packages/rtrt/Core/PathTracer/
g++ -I/home/bigler/SCIRun/Thirdparty/1.20/Linux/gcc-3.2.2-32bit/include -g dilate.cc -o ~/SCIRun/rtrt/Packages/rtrt/StandAlone/dilate -Wl,-rpath -Wl,/home/bigler/SCIRun/Thirdparty/1.20/Linux/gcc-3.2.2-32bit/lib -L/home/bigler/SCIRun/Thirdparty/1.20/Linux/gcc-3.2.2-32bit/lib -lteem -lm

*/

#include <teem/nrrd.h>
#include <teem/biff.h>

int main(int argc, char *argv[]) {
  char *me = argv[0];
  char *err = 0;
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <inside> <texture> <outfile> [threshold]\n", me);
    return 1;
  }

  char *input = argv[1];
  char *texfile = argv[2];
  char *outfile = argv[3];
  float threshold = 0.30;

  if (argc > 4) {
    threshold = atof(argv[4]);
  }


  Nrrd* nin = nrrdNew();
  Nrrd* texture = nrrdNew();
  if (nrrdLoad(nin, input, 0) || nrrdLoad(texture, texfile, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: Error loading input nrrds:\n %s", me, err);
    return 2;
  }

  if (nin->dim != 3) {
    fprintf(stderr, "%s: inside must be of dimension 3.\n", me);
    return 2;
  }

  if (texture->dim != 4) {
    fprintf(stderr, "%s: texture must be of dimension 4.\n", me);
    return 2;
  }

  printf("Computing the min and max\n");
  // Compute the min and max
  float *data = (float*)(nin->data);
  size_t num_data = nrrdElementNumber(nin);
  printf("num_data = %lu\n", num_data);
  float min = data[0];
  float max = data[0];
  for(size_t i = 1; i < num_data; i++) {
    float val = data[i];
    if (val < min)
      min = val;
    if (val > max)
      max = val;
  }

  printf("min and max = %g and %g\n", min, max);
  // Now we need to normalize our nrrd
  float inv_maxmin = 1/(max-min);
  for(size_t i = 0; i < num_data; i++) {
    data[i] = (data[i]-min)*inv_maxmin;
  }  


  printf("Done normalizing the data\n");
  
  Nrrd *nout = nrrdNew();
  if (nrrdCopy(nout, texture)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: Error copying to out:\n %s", me, err);
    return 2;
  }

  int width = nin->axis[0].size; 
  int height = nin->axis[1].size; 
  int depth = nin->axis[2].size;
  float *tex = (float*)(texture->data);
  float *outdata = (float*)(nout->data);
  
  // For each slice
  for(int z = 0; z < depth; z++) {
    for(int y = 0; y < height; y++)
      for(int x = 0; x < width; x++)
	{
	  // determine if a given pixel should be dilated
	  float val = data[z*width*height + y*width + x];
	  if (val <= 0)
	    // Value is not occluded, so go to the next one
	    continue;

	  float ave[3] = {0,0,0};
	  float contribution_total = 0;
	  // Loop over each neighbor
	  for(int j = y-1; j <= y+1; j++)
	    for(int i = x-1; i <= x+1; i++)
	      {
		// Fix boundary conditions
		int newi = i;
		if (newi == width)
		  newi = 0;
		else if (newi == -1)
		  newi = width-1;

		int newj = j;
		if (newj == height)
		  newj = height - 1;
		else if (newj == -1)
		  newj = 0;

		float contributer = data[z*width*height + newj*width + newi];
		if (contributer < threshold) {
		  // Comment out this line if you want weighted averages
		  contributer = 0;
		  float *con = tex + (z*width*height + newj*width + newi)*3;
		  ave[0] = ave[0] + con[0]*(1-contributer);
		  ave[1] = ave[1] + con[1]*(1-contributer);
		  ave[2] = ave[2] + con[2]*(1-contributer);
		  contribution_total += (1-contributer);
		}
	      }

	  // dilate the pixel
	  float *out = outdata + (z*width*height + y*width + x)*3;
	  if (contribution_total > 0) {
	    out[0] = ave[0]/contribution_total;
	    out[1] = ave[1]/contribution_total;
	    out[2] = ave[2]/contribution_total;
	  }
	}
    printf("Done with %d\n", z);
  }

  // Write out the data
  if (nrrdSave(outfile, nout, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: Error writing out:\n %s", me, err);
    return 2;
  }
  
  return 0;
}
