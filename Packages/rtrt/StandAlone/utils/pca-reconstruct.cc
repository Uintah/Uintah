#include <teem/nrrd.h>

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
  printf("usage:  %s [options]\n", me);
  printf("options:\n");
  printf("  -i <filename>   basename of input nrrds (null)\n");
  printf("  -o <filename>   filename of output nrrd (null)\n");
  printf("  -b <filename>   load basis textures from file (null)\n");
  printf("  -c <filename>   load PCA coefficients from file (null)\n");
  printf("  -m <filename>   load mean vector from file (null)\n");
  printf("  -nrrd           use .nrrd extension (false)\n");
  
  if (unknown)
    exit(1);
}

int main(int argc, char *argv[]) {
  char *me = argv[0];
  char *err;
  char *infilename_base=0;
  char *outfilename=0;
  char *bases_filename=0;
  char *coeff_filename=0;
  char *mean_filename=0;
  char *nrrd_ext = ".nhdr";
  
  for(int i = 1; i < argc; i++) {
    string arg(argv[i]);
    if (arg == "-input" || arg == "-i") {
      infilename_base = argv[++i];
    } else if (arg == "-output" || arg == "-o") {
      outfilename = argv[++i];
    } else if (arg == "-basis" || arg == "-b") {
      bases_filename = argv[++i];
    } else if (arg == "-coeff" || arg == "-c") {
      coeff_filename = argv[++i];
    } else if (arg == "-mean" || arg == "-m") {
      mean_filename = argv[++i];            
    } else if (arg == "-nrrd") {
      nrrd_ext = ".nrrd";
    } else {
      usage(me, arg.c_str());
    }
  }

  if (!infilename_base) {
    bool error=false;
    if (!bases_filename) {
      cerr << "filename of basis textures not specified"<<endl;
      error=true;
    }
    if (!coeff_filename) {
      cerr << "filename of PCA coefficients not specified"<<endl;
      error=true;
    }
    if (!mean_filename) {
      cerr << "filename of mean vector not specified"<<endl;
      error=true;
    }
    if (error) {
      cerr <<"did you mean to specify a base input filename?"<<endl;
      usage(me);
      exit(1);
    }
  } else {
    size_t len=strlen(infilename_base);
    if (!bases_filename) {
      bases_filename=new char[len+15];
      sprintf(bases_filename, "%s-basis%s", infilename_base, nrrd_ext);
    }
    if (!coeff_filename) {
      coeff_filename=new char[len+20];
      sprintf(coeff_filename, "%s-coeff%s", infilename_base, nrrd_ext);
    }
    if (!mean_filename) {
      mean_filename=new char[len+15];
      sprintf(mean_filename, "%s-mean%s", infilename_base, nrrd_ext);
    }
  }

  if (!outfilename) {
    cerr<<"output filename not specified"<<endl;
    usage(me);
    exit(1);
  }

  // Load the input nrrds
  Nrrd *bases = nrrdNew();
  Nrrd *coeff = nrrdNew();
  Nrrd *mean = nrrdNew();
  int E = 0;
  cerr<<"attempting to load "<<bases_filename<<endl;
  if (!E) E |= nrrdLoad(bases, bases_filename, 0);
  cerr<<"attempting to load "<<coeff_filename<<endl;
  if (!E) E |= nrrdLoad(coeff, coeff_filename, 0);
  cerr<<"attempting to load "<<mean_filename<<endl;
  if (!E) E |= nrrdLoad(mean, mean_filename, 0);
  if (E) {
    err = biffGet(NRRD);
    cerr << me << ": error loading NRRD: " << err << endl;
    free(err);
    biffDone(NRRD);
    exit(2);
  }
    
  int num_bases = coeff->axis[0].size;
  int num_channels = coeff->axis[1].size;
  int width = bases->axis[1].size;
  int height = bases->axis[2].size;

  if (mean->axis[0].size != num_channels) {
    cerr<<"number of channels in mean vector ("<<mean->axis[0].size
	<<") is not correct"<<endl;
    exit(2);
  }

  if (bases->axis[0].size != num_bases) {
    cerr<<"number of basis textures ("<<bases->axis[0].size
	<<") is not correct"<<endl;
    exit(2);
  }

  // Now generate the image back again
  Nrrd *nout = nrrdNew();
  if (nrrdAlloc(nout, nrrdTypeFloat, 3, num_channels, width, height)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: error allocating result image:\n%s", me, err);
    exit(0);
  }

  float *outdata = (float*)(nout->data);
  float *btdata = (float*)(bases->data);
  float *tdata = (float*)(coeff->data);
  float *mdata = (float*)(mean->data);
  // Loop over each pixel
  for (int pixel = 0; pixel < (width*height); pixel++) {
    // Now do coeff_transpose * btdata
    for(int r = 0; r < num_channels; r++)
      for(int c = 0; c < num_bases; c++)
	outdata[r] += tdata[r*num_bases+c] * btdata[c];
    
    // Add the mean
    for(int i = 0; i < num_channels; i++)
      outdata[i] += mdata[i];

    outdata += num_channels;
    btdata += num_bases;
  }
  
  // Write the output file
  if (nrrdSave(outfilename, nout, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: trouble saving to %s: %s\n", me, outfilename, err);
    exit(2);
  }

  return 0;
}
