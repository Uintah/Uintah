#include <teem/nrrd.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <sci_values.h>

using namespace std;

typedef struct {
  int idx;
  float value;
} MeasureUnit;

int compare_asc(const void* elem1, const void* elem2) {
  MeasureUnit *m1 = (MeasureUnit*)(elem1);
  MeasureUnit *m2 = (MeasureUnit*)(elem2);

  // Sort in ascending order
  return m1->value - m2->value;
}

int compare_dsc(const void* elem1, const void* elem2) {
  MeasureUnit *m1 = (MeasureUnit*)(elem1);
  MeasureUnit *m2 = (MeasureUnit*)(elem2);

  // Sort in descending order
  return m2->value - m1->value;
}

void usage(char *me, const char *unknown = 0) {
  if (unknown) {
    fprintf(stderr, "%s: unknown argument %s\n", me, unknown);
  }
  
  // Print out the usage
  printf("usage:  %s [options]\n", me);
  printf("options:\n");
  printf("  -i <filename>   input nrrd, axis[x,y,channel] (null)\n");
  printf("  -o <filename>   filename of texture subset (null)\n");
  printf("  -numtex <int>   number of textures to place in subset (0)\n");
  printf("  -mean           sort according to mean value (variance)\n");
  printf("  -ascending      sort textures in ascending order (descending)\n");
  printf("  -p <float>      perturb texture ordering (false)\n");
  printf("  -seed <int>     seed the random number generator (12)\n");
  printf("  -verbose        print verbose status messages (false)\n");

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
  char *infilename = 0;
  char *outfilename = 0;
  int num_subset = 0;
  bool variance = false;
  bool descending = true;
  float perturb_thresh = 1.0;
  int seed = 12;
  int verbose = 0;

  for(int i = 1; i < argc; i++) {
    string arg(argv[i]);
    if (arg == "-input" || arg == "-i") {
      infilename = argv[++i];
    } else if (arg == "-output" || arg == "-o") {
      outfilename = argv[++i];
    } else if (arg == "-numtex") {
      num_subset = atoi(argv[++i]);
    } else if (arg == "-mean") {
      variance = false;
    } else if (arg == "-ascending") {
      descending = false;
    } else if (arg == "-p") {
      perturb_thresh = atof(argv[++i]);;
    } else if (arg == "-seed") {
      seed = atoi(argv[++i]);
    } else if (arg == "-v" ) {
      verbose++;
    } else {
      usage(me, arg.c_str());
    }
  }

  if (!infilename) {
    cerr << "no input filename specified" << endl;
    usage(me);
    exit(1);
  }

  if (!outfilename) {
    cerr << "no output filename specified" << endl;
    usage(me);
    exit(1);
  }

  if (num_subset <= 0) {
    cerr << "invlid number of textures to select for subset ("
	 << num_subset << ")" << endl;
    usage(me);
    exit(1);
  }
  
  if (perturb_thresh > 1.0 || perturb_thresh < 0.0) {
    cerr << "invalid perturbation threshold " << perturb_thresh << endl;
    cerr << "not perturbing texture ordering" << endl;
    perturb_thresh = 1.0;
  }

  // Load the textures
  if (verbose)
    cout << "Loading " << infilename << endl;
  
  Nrrd *nin = nrrdNew();
  if (nrrdLoad(nin, infilename, 0)) {
    err = biffGet(NRRD);
    cerr << me << ": error loading NRRD: " << err << endl;
    free(err);
    biffDone(NRRD);
    exit(2);
  }

  // Verify number of dimensions
  if (nin->dim != 3 ) {
    cerr << me << ":  number of dimesions " << nin->dim
	 << " is not equal to 3." << endl;
    exit(2);
  }
  
  // Convert the type to floats, if necessary
  nin = convertNrrdToFloat(nin);
  if (!nin) {
    cerr << me << ": converting textures to float failed" << endl;
    exit(2);
  }
  
  // Determine texture size and number
  int width = nin->axis[0].size;
  int height = nin->axis[1].size;
  int num_channels = nin->axis[2].size;
  
  // Create the sorting data structure
  MeasureUnit *sort = (MeasureUnit*)malloc(sizeof(MeasureUnit)*num_channels);

  // Compute the measure on the textures
  float* tex_data = (float*)(nin->data);
  int num_pixels = width*height;
  float max_value = -FLT_MAX;
  for (int channel = 0; channel < num_channels; channel++) {
    float min = FLT_MAX;
    float max = -FLT_MAX;
    float sum = 0;
    for (int pixel = 0; pixel < num_pixels; pixel++) {
      float val = tex_data[pixel];
      if (val < min)
	min = val;
      if (val > max)
	max = val;

      sum += val;
    }
    
    tex_data += num_pixels;
    
    sort[channel].idx = channel;
    if (variance)
      sort[channel].value = max - min;
    else
      sort[channel].value = sum/num_pixels;
    
    if (sort[channel].value > max_value)
      max_value = sort[channel].value;
  }
  
  // Use quick sort to order the textures
  if (verbose) {
    cout << "Sorting textures (";
    if (descending)
      cout << "descending";
    else
      cout << "ascending";
    cout << " order)" << endl;
  }

  if (descending)
    qsort(sort, num_channels, sizeof(MeasureUnit), compare_dsc);
  else
    qsort(sort, num_channels, sizeof(MeasureUnit), compare_asc);

  if (verbose)
    cout << "Sort complete" << endl;

  // Perturb the sorted textures slightly
  if (perturb_thresh < 1.0) {
    if (verbose)
      cout << "Perturbing sorted textures slightly" << endl;

    // Seed the RNG
    srand48(seed);

    for (int tex = num_subset; tex < num_channels; tex++) {
      double p = drand48()*sort[tex].value/max_value;
      if (p > perturb_thresh) {
	int r_idx = (int)(random()%num_subset);
	sort[r_idx].idx = sort[tex].idx;
	sort[r_idx].value = sort[tex].value;
      }
    }
    
    if (verbose)
      cout << "Perturbation complete" << endl;
  }
  
  // Allocate output nrrd
  Nrrd *nout = nrrdNew();
  if (nrrdAlloc(nout, nrrdTypeUChar, 3, num_subset, width, height)) {
    err = biffGet(NRRD);
    cerr << me << ": error allocating texture subset: " << err << endl;
    free(err);
    biffDone(NRRD);
    exit(2);
  }

  nrrdAxisInfoSet(nout, nrrdAxisInfoLabel, "channel", "width", "height");

  // Populate the texture subset
  float* texture = (float*)(nin->data);
  unsigned char* subset = (unsigned char*)(nout->data);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int channel = 0; channel < num_subset; channel++) {
	int idx = (sort[channel].idx*height + y)*width + x;
	subset[channel] = texture[idx];
      }

      subset += num_subset;
    }
  }
  
  // Write the texture subset
  if (verbose)
    cout << "Writing texture subset" << endl;
  
  if (nrrdSave(outfilename, nout, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: trouble saving to %s: %s\n", me, outfilename, err);
    exit(2);
  }

  if (verbose)
    cout << "Wrote texture subset to " << outfilename << endl;

  // Free allocated memory
  nrrdNuke(nin);
  nrrdNuke(nout);
  
  return 0;
}
