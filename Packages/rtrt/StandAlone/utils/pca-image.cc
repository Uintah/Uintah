#include <teem/nrrd.h>
#include <teem/ell.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace std;

void usage(char *me, const char *unknown = 0) {
  if (unknown) {
    fprintf(stderr, "%s: unknown argument %s\n", me, unknown);
  }
  // Print out the usage
  printf("-input  <filename>\n");
  printf("-output <filename>\n");
  printf("-numbases <int>\n");

  if (unknown)
    exit(1);
}

int main(int argc, char *argv[]) {
  char *me = argv[0];
  char *err;
  char *infilename = 0;
  char *outfilename_base = "trasimage";
  // -1 defaults to all
  int num_bases = 0;
  int num_channels;

  if (argc < 2) {
    usage(me);
    return 0;
  }

  for(int i = 1; i < argc; i++) {
    string arg(argv[i]);
    if (arg == "-input" || arg == "-i") {
      infilename = argv[++i];
    } else if (arg == "-output" || arg == "-o") {
      outfilename_base = argv[++i];
    } else if (arg == "-numbases" ) {
      num_bases = atoi(argv[++i]);
    } else {
      usage(me, arg.c_str());
    }
  }

  // load input nrrd
  Nrrd *nin = nrrdNew();
  if (nrrdLoad(nin, infilename, 0)) {
    err = biffGet(NRRD);
    cerr << me << ": error loading NRRD: " << err << endl;
    free(err);
    biffDone(NRRD);
    exit(2);
  }

  // verify number of dimensions
  if (nin->dim != 3 ) {
    cerr << me << ":  number of dimesions " << nin->dim << " is not equal to 4." << endl;
    exit(2);
  }
  
  num_channels = nin->axis[0].size;
  
  // verify size of axis[0]
  if (num_channels != 3) {
    cerr << me << ":  size of axis[0] is not equal to 3" << endl;
    exit(2);
  }

  // Determine the number of eigen thingies to use
  if (num_bases < 1)
    // Use all of them
    num_bases = num_channels;
  // Check to make sure we haven't overstepped the max
  if (num_bases > num_channels) {
    cerr << "You have specified too many basis for the number of channels ("<<num_bases<<").  Clamping to "<< num_channels<<".\n";
    num_bases = num_channels;
  }

  // convert the type to floats if you need to
  if (nin->type != nrrdTypeFloat) {
    cerr << "Converting type from ";
    switch(nin->type) {
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
    if (nrrdConvert(new_n, nin, nrrdTypeFloat)) {
      err = biffGet(NRRD);
      cerr << me << ": unable to convert nrrd: " << err << endl;
      biffDone(NRRD);
      exit(2);
    }
    // since the data was copied blow away the memory for the old nrrd
    nrrdNuke(nin);
    nin = new_n;
  }

  // compute the mean value
  Nrrd *meanY = nrrdNew();
  Nrrd *mean = nrrdNew();
  int E = 0;
  if (!E) E |= nrrdProject(meanY, nin, 2, nrrdMeasureMean, nrrdTypeDefault);
  if (!E) E |= nrrdProject(mean, meanY, 1, nrrdMeasureMean, nrrdTypeDefault);
  if (E) {
    err = biffGet(NRRD);
    cerr << me << ": computing the mean value failed: " << err << endl;
    biffDone(NRRD);
    exit(2);
  }

  // Allocate our covariance matrix
  Nrrd *cov = nrrdNew();
  if (nrrdAlloc(cov, nrrdTypeDouble,2, num_channels, num_channels))
    {
      err = biffGetDone(NRRD);
      fprintf(stderr, "%s: error allocating covariance matrix:\n%s", me, err);
      exit(2);
    }
    
  // loop over each pixel to compute cov(x,y)
  int height = nin->axis[2].size;
  int width = nin->axis[1].size;
  Nrrd *slab = nrrdNew();
  Nrrd *pixel = nrrdNew();
  for (int y=0; y < height; y++) {
    if (nrrdSlice(slab, nin, 2, y)) {
      err = biffGetDone(NRRD);
      fprintf(stderr, "%s: error slicing nrrd:\n%s", me, err);
      exit(2);
    }

    for (int x = 0; x < width; x++ ) {
      if (nrrdSlice(pixel, slab, 1, x)) {
	err = biffGetDone(NRRD);
	fprintf(stderr, "%s: error slicing nrrd:\n%s", me, err);
	exit(2);
      }

      // compute cov(x,y) for current pixel
      double *cov_data = (double*)(cov->data);
      float *pixel_data = (float*)(pixel->data);
      for(int c = 0; c < num_channels; c++)
	for(int r = 0; r < num_channels; r++)
	  {
	    *cov_data = *cov_data + pixel_data[c] * pixel_data[r];
	    cov_data++;
	  }
    }
  }
  nrrdEmpty(slab);
  nrrdEmpty(pixel);

  {
    double *cov_data = (double*)(cov->data);
    for(int c = 0; c < cov->axis[1].size; c++)
      {
	cout << "[";
	for(int r = 0; r < cov->axis[0].size; r++)
	  {
	    cout << cov_data[c*cov->axis[0].size + r] << ", ";
	  }
	cout << "]\n";
      }
  }

  {
    double *cov_data = (double*)(cov->data);
    float *mean_data = (float*)(mean->data);
    float inv_num_pixels = 1.0f/(width*height);
    for(int c = 0; c < num_channels; c++)
      for(int r = 0; r < num_channels; r++)
	{
	  *cov_data = *cov_data * inv_num_pixels - mean_data[c] * mean_data[r];
	  cov_data++;
	}
  }

  {
    cout << "After minux mean\n";
    double *cov_data = (double*)(cov->data);
    for(int r = 0; r < cov->axis[0].size; r++)
      {
	cout << "[";
	for(int c = 0; c < cov->axis[1].size; c++)
	  {
	    cout << cov_data[c*cov->axis[0].size + r] << ", ";
	  }
	cout << "]\n";
      }
  }
  
  // Convariance matrix computed

  // Compute eigen values/vectors

  // Here's where the general solution diverges to the RGB case.
  double eval[3], evec[9];
  int roots = ell_3m_eigensolve_d(eval, evec, (double*)(cov->data), 1);
  if (roots != ell_cubic_root_three) {
    cerr << me << "Something with the eighen solve went haywire.  Did not get three roots but "<<roots<<" roots.\n";
    exit(2);
  }

  cout << "Eigen values are ["<<eval[0]<<", "<<eval[1]<<", "<<eval[2]<<"]\n";


  // Cull our eigen vectors
  Nrrd *transform = nrrdNew();
  if (nrrdAlloc(transform, nrrdTypeFloat, 2, num_channels, num_bases)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: error allocating transform matrix :\n%s", me, err);
    exit(0);
  }

  {
    float *tdata = (float*)(transform->data);
    double *edata = evec;
    for(int channel = 0; channel < num_channels; channel++) {
      for (int basis = 0; basis < num_bases; basis++) {
	*tdata = *edata;
	tdata++;
	edata++;
      }
    }

    cout << "\ntransform matrix\n";
    tdata = (float*)(transform->data);
    for(int c = 0; c < transform->axis[1].size; c++)
      {
	cout << "[";
	for(int r = 0; r < transform->axis[0].size; r++)
	  {
	    cout << tdata[c*transform->axis[0].size + r] << ", ";
	  }
	cout << "]\n";
      }
  }
  
  // Compute our basis textures
  Nrrd *basesTextures = nrrdNew();
  if (nrrdAlloc(basesTextures, nrrdTypeFloat, 3, num_bases, width, height)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: error allocating bases textures :\n%s", me, err);
    exit(0);
  }

  {
    float *btdata = (float*)(basesTextures->data);
    
    for (int y=0; y < height; y++) {
      if (nrrdSlice(slab, nin, 2, y)) {
	err = biffGetDone(NRRD);
	fprintf(stderr, "%s: error slicing nrrd:\n%s", me, err);
	exit(2);
      }
      
      for (int x = 0; x < width; x++ ) {
	if (nrrdSlice(pixel, slab, 1, x)) {
	  err = biffGetDone(NRRD);
	  fprintf(stderr, "%s: error slicing nrrd:\n%s", me, err);
	  exit(2);
	}
	
	// Subtract the mean
	float *pixel_data = (float*)(pixel->data);
	float *mdata = (float*)(mean->data);
	for(int i = 0; i < mean->axis[0].size; i++) {
	  pixel_data[i] -= mdata[i];
	}
	
	// Now do transform * pixel_data
	float *tdata = (float*)(transform->data);

	for(int c = 0; c < transform->axis[1].size; c++)
	  for(int r = 0; r < transform->axis[0].size; r++)
	    {
	      btdata[c] += tdata[c*transform->axis[0].size+r]*pixel_data[r];
	    }
	btdata += num_bases;
      }
    }
    nrrdEmpty(slab);
    nrrdEmpty(pixel);
  }

  // Now generate the image back again
  Nrrd *nout = nrrdNew();
  if (nrrdAlloc(nout, nrrdTypeFloat, 3, num_channels, width, height)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: error allocating result image:\n%s", me, err);
    exit(0);
  }

  {
    float *outdata = (float*)(nout->data);
    float *btdata = (float*)(basesTextures->data);
    float *mdata = (float*)(mean->data);
    // Produce one chanel at a time
    // Loop over each pixel
    for (int pixel = 0; pixel < (width*height); pixel++) {
      // Now do transform_transpose * btdata
      float *tdata = (float*)(transform->data);
	  
      for(int r = 0; r < transform->axis[0].size; r++)
	for(int c = 0; c < transform->axis[1].size; c++)
	  {
	    outdata[r] += tdata[c*transform->axis[0].size+r] * btdata[c];
	  }

      // Add the mean
      for(int i = 0; i < num_channels; i++) {
	outdata[i] += mdata[i];
      }
      outdata += num_channels;
      btdata += num_bases;
    }
  }

  // Fix the outfilename_base
  string outfilename(string(outfilename_base) + ".nrrd");
  if (nrrdSave(outfilename.c_str(), nout, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: trouble saving to %s: %s\n", me, outfilename.c_str(), err);
    exit(2);
  }

  return 0;
}
