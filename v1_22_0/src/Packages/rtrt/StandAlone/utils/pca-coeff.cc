#include <teem/nrrd.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace std;

void usage(char *me, const char *unknown = 0) {
  if (unknown)
    fprintf(stderr, "%s: unknown argument %s\n", me, unknown);

  // Print out the usage
  printf("usage:  %s [options]\n", me);
  cout << "options:" << endl;
  cout << "  -i <filename>    basename of input nrrds (null)" << endl;
  cout << "  -t <filename>    load input textures from file,"
       << " channels as the fastest axis (null)" << endl;
  cout << "  -b <filename>    load basis textures from file (null)" << endl;
  cout << "  -o <filename>    basename of output nrrds (null)" << endl;
  cout << "  -nrrd            use .nrrd extension (false)" << endl;
  cout << "  -verbose         print verbose status messages (false)" << endl;
  
  if (unknown)
    exit(1);
}

Nrrd* convertNrrdToFloat(Nrrd* nin) {
  char *me = "convertNrrd";
  char *err;
  
  if (nin->type != nrrdTypeFloat) {
    cout << "Converting NRRD from type ";
    switch(nin->type) {
    case nrrdTypeUnknown: cout << "nrrdTypeUnknown"; break;
    case nrrdTypeChar: cout << "nrrdTypeChar"; break;
    case nrrdTypeUChar: cout << "nrrdTypeUChar"; break;
    case nrrdTypeShort: cout << "nrrdTypeShort"; break;
    case nrrdTypeUShort: cout << "nrrdTypeUShort"; break;
    case nrrdTypeInt: cout << "nrrdTypeInt"; break;
    case nrrdTypeUInt: cout << "nrrdTypeUInt"; break;
    case nrrdTypeLLong: cout << "nrrdTypeLLong"; break;
    case nrrdTypeULLong: cout << "nrrdTypeULLong"; break;
    case nrrdTypeDouble: cout << "nrrdTypeDouble"; break;
    default: cout << "Unknown NRRD type"<<endl;
    }
    cout << " to nrrdTypeFloat\n";
    
    Nrrd *new_nin = nrrdNew();
    if (nrrdConvert(new_nin, nin, nrrdTypeFloat)) {
      err = biffGet(NRRD);
      cerr << me << ": unable to convert NRRD: " << err << endl;
      biffDone(NRRD);
      return 0;
    }
    
    // Data was copied, so nuke the original version and reassign
    nrrdNuke(nin);
    return new_nin;
  }

  // No need to convert
  return nin;
}

int main(int argc, char *argv[]) {
  char *me = argv[0];
  char *err;
  char *infilename_base=0;
  char *input_filename=0;
  char *bases_filename=0;
  char *outfilename_base=0;
  char *nrrd_ext = ".nhdr";
  int verbose = 0;

  // Parse input arguments
  for(int i = 1; i < argc; i++) {
    string arg(argv[i]);
    if (arg == "-input" || arg == "-i") {
      infilename_base = argv[++i];
    } else if (arg == "-tex" || arg == "-t") {
      input_filename = argv[++i];
    } else if (arg == "-bases" || arg == "-b") {
      bases_filename = argv[++i];
    } else if (arg == "-output" || arg == "-o") {
      outfilename_base = argv[++i];
    } else if (arg == "-nrrd") {
      nrrd_ext = ".nrrd";
    } else if (arg == "-v" || arg == "-verbose") {
      verbose = 1;
    } else {
      usage(me, arg.c_str());
    }
  }

  // Check for errors
  if (infilename_base) {
    size_t inname_len = strlen(infilename_base);
    if (!input_filename) {
      input_filename = new char[inname_len+20];
      sprintf(input_filename, "%s%s", infilename_base, nrrd_ext);
    }
    if (!bases_filename) {
      bases_filename = new char[inname_len+20];
      sprintf(bases_filename, "%s-basis%s", infilename_base, nrrd_ext);
    }
  } else {
    bool error=false;
    if (!input_filename) {
      cerr << "filename of input textures not specified"<<endl;
      error=true;
    }
    if (!bases_filename) {
      cerr << "filename of basis textures not specified"<<endl;
      error=true;
    }
    if (error) {
      cerr <<"did you mean to specify a base input filename?"<<endl;
      usage(me);
      exit(1);
    }
  }

  if (!outfilename_base) {
    cerr << "output base name not specified" << endl;
    usage(me);
    exit(1);
  }
  size_t outname_len = strlen(outfilename_base);

  // Load input nrrds
  Nrrd *nin = nrrdNew();
  Nrrd *bases = nrrdNew();
  int E = 0;
  if (!E) E |= nrrdLoad(nin, input_filename, 0);
  if (!E) E |= nrrdLoad(bases, bases_filename, 0);
  if (E) {
    err = biffGet(NRRD);
    cerr << me << ": error loading NRRD: " << err << endl;
    free(err);
    biffDone(NRRD);
    exit(2);
  }

  // Verify dimensionality of input nrrds
  if (nin->dim != 3) {
    cerr << me << ":  number of dimesions " << nin->dim
	 << " of input textures is not equal to 3" << endl;
    exit(2);
  }
  if (bases->dim != 3) {
    cerr << me << ":  number of dimesions " << bases->dim
	 << " of bases textures is not equal to 3" << endl;
    exit(2);
  }

  // Convert to float, if necessary
  nin = convertNrrdToFloat(nin);
  if (!nin) {
    cerr << me << ": input textures are null" << endl;
    exit(2);
  }

  bases = convertNrrdToFloat(bases);
  if (!bases) {
    cerr << me << ": bases textures are null" << endl;
    exit(2);
  }
  
  // Compute the mean value of each texture
  if (verbose)
    cout << "Computing mean values" << endl;

  // Allocate the mean nrrds
  Nrrd *meanY = nrrdNew();
  Nrrd *mean = nrrdNew();
  if (!E) E |= nrrdProject(meanY, nin, 2, nrrdMeasureMean, nrrdTypeDefault);
  if (!E) E |= nrrdProject(mean, meanY, 1, nrrdMeasureMean, nrrdTypeDefault);
  if (E) {
    err = biffGet(NRRD);
    cerr << me << ": computing the mean values failed: " << err << endl;
    biffDone(NRRD);
    exit(2);
  }

  nrrdAxisInfoSet(mean, nrrdAxisInfoLabel, "mean");

  // Free meanY
  meanY = nrrdNuke(meanY);
  
  if (verbose)
    cout << "Mean values calculation complete" << endl;
  
  // Write the mean file
  char *mean_filename = new char[outname_len+20];
  sprintf(mean_filename, "%s-mean%s", outfilename_base, nrrd_ext);
  if (nrrdSave(mean_filename, mean, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: trouble saving to %s: %s\n", me, mean_filename, err);
    exit(2);
  }
  
  if (verbose)
    cout << "Wrote mean values to " << mean_filename << endl;

  // Set loop control variables
  int num_channels = nin->axis[0].size;
  int width = nin->axis[1].size;
  int height = nin->axis[2].size;
  int num_bases = bases->axis[0].size;

  // Calculate the magnitude of the basis textures
  if (verbose)
    cout << "Calculating magnitude of basis textures" << endl;

  float *mag = new float[num_bases];
  for (int basis = 0; basis < num_bases; basis++)
    mag[basis] = 0;
    
  float *bases_data = (float*)(bases->data);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int basis = 0; basis < num_bases; basis++)
	mag[basis] += bases_data[basis]*bases_data[basis];
      
      bases_data += num_bases;
    }
  }

  for (int basis = 0; basis < num_bases; basis++)
    mag[basis] = sqrt(mag[basis]);
    
  // Normalize the basis textures
  if (verbose)
    cout << "Normalizing basis textures" << endl;
  
  bases_data = (float*)(bases->data);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int basis = 0; basis < num_bases; basis++)
	bases_data[basis] /= mag[basis];
    
      bases_data += num_bases;
    }
  }

  if (verbose)
    cout << "Basis textures normalized" << endl;
  
  // Write the normalized basis textures file
  char *nbases_filename = new char[outname_len+20];
  sprintf(nbases_filename, "%s-basis%s", outfilename_base, nrrd_ext);
  if (nrrdSave(nbases_filename, bases, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: trouble saving to %s: %s\n", me, nbases_filename, err);
    exit(2);
  }

  if (verbose)
    cout << "Wrote normalized basis textures to " << nbases_filename << endl;

  // Allocate the transform nrrd
  Nrrd *nout = nrrdNew();
  if (nrrdAlloc(nout, nrrdTypeFloat, 2, num_bases, num_channels)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: error allocating transform nrrd:\n%s", me, err);
    exit(2);
  }
  
  nrrdAxisInfoSet(nout, nrrdAxisInfoLabel, "bases", "channels");

  // Calculate new PCA coefficients
  if (verbose)
    cout << "Computing new PCA coefficients" << endl;

  // Subtract the mean from the textures
  float *tex_data = (float*)(nin->data);
  float *mean_data = (float*)(mean->data);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++ ) {
      for (int channel = 0; channel < num_channels; channel++)
        tex_data[channel] -= mean_data[channel];
      
      tex_data += num_channels;
    }
  }
 
  // Compute the dot product of bases and textures
  tex_data = (float*)(nin->data);
  bases_data = (float*)(bases->data);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float* out_data = (float*)(nout->data);
      for (int channel = 0; channel < num_channels; channel++) {
	for (int basis = 0; basis < num_bases; basis++)
	  out_data[basis] += bases_data[basis]*tex_data[channel];
	
	out_data += num_bases;
      }

      bases_data += num_bases;
      tex_data += num_channels;
    }
  }
  
  if (verbose)
    cout << "New PCA coefficient calculation complete" << endl;

  // Write the transform file
  char *transform_filename = new char[outname_len+20];
  sprintf(transform_filename, "%s-coeff%s", outfilename_base, nrrd_ext);
  if (nrrdSave(transform_filename, nout, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: trouble saving to %s: %s\n", me, transform_filename, err);
    exit(2);
  }

  if (verbose)
    cout << "Wrote new PCA coefficients to "<<transform_filename<<endl;
  
  // Clean-up memory
  mean = nrrdNuke(mean);
  bases = nrrdNuke(bases);
  nin = nrrdNuke(nin);
  nout = nrrdNuke(nout);
  
  return 0;
}
