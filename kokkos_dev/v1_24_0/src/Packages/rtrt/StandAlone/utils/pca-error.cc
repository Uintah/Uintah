#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <teem/nrrd.h>

using namespace std;

// Forward declare necessary functions
void usage(char* me, const char* unknown=0);
Nrrd* permuteNrrd(Nrrd* nin, int* axes);
Nrrd* convertNrrdToFloat(Nrrd* nin);
Nrrd* computeMean(Nrrd* tex);
Nrrd* normalizeBasis(Nrrd* basis);
Nrrd* computeCoefficients(Nrrd* tex, Nrrd* basis);
float* computeError(Nrrd* basis, Nrrd* coeff, Nrrd* mean, Nrrd* tex);

// Declare global variables
int width=0;
int height=0;
int ntextures=0;
int nbases=0;
int npixels=0;
bool inf_norm=true;
int verbose=0;

int main(int argc, char *argv[]) {
  char* me=argv[0];
  char* err;
  char* inbasename=0;
  char* outfilename=0;
  char* basisfilename=0;
  char* coefffilename=0;
  char* meanfilename=0;
  char* texfilename=0;
  char* nrrd_ext=".nhdr";

  // Parse arguments
  for(int i=1; i<argc; i++) {
    string arg(argv[i]);
    if (arg=="-i") {
      inbasename=argv[++i];
    } else if (arg=="-o") {
      outfilename=argv[++i];
    } else if (arg=="-b") {
      basisfilename=argv[++i];
    } else if (arg=="-c") {
      coefffilename=argv[++i];
    } else if (arg=="-m") {
      meanfilename=argv[++i];
    } else if (arg=="-t") {
      texfilename=argv[++i];
    } else if (arg=="-mse") {
      inf_norm=false;
    } else if (arg=="-nrrd") {
      nrrd_ext=".nrrd";
    } else if (arg=="-v") {
      verbose=atoi(argv[++i]);
    } else if (arg=="--help") {
      usage(me);
      exit(0);
    } else {
      usage(me, arg.c_str());
    }
  }

  // Verify the arguments
  if (!inbasename) {
    bool error=false;
    if (!basisfilename) {
      cerr<<me<<":  filename of basis textures not specified"<<endl;
      error=true;
    }

    if (!texfilename) {
      cerr<<me<<":  filename of original textures not specified"<<endl;
      error=true;
    }
    
    if (error) {
      usage(me);
      exit(1);
    }
  } else {
    size_t in_len=strlen(inbasename);
    size_t ext_len=strlen(nrrd_ext);
    
    if (!basisfilename) {
      basisfilename=new char[in_len+6+ext_len];
      sprintf(basisfilename, "%s-basis%s", inbasename, nrrd_ext);
    }

    if (!texfilename) {
      texfilename=new char[in_len+4+ext_len];
      sprintf(texfilename, "%s-tex%s", inbasename, nrrd_ext);
    }
  }
  
  if (!outfilename) {
    cerr<<me<<":  output filename not specified"<<endl;
    usage(me);
    exit(1);
  }

  // Load the input files
  Nrrd* basis=nrrdNew();
  Nrrd* tex=nrrdNew();
  int E=0;
  if (!E) E|=nrrdLoad(basis, basisfilename, 0);
  if (!E) E|=nrrdLoad(tex, texfilename, 0);
  if (E) {
    err=biffGet(NRRD);
    cerr<<me<<":  error loading input files:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  Nrrd* coeff;
  if (coefffilename) {
    coeff=nrrdNew();
    if (nrrdLoad(coeff, coefffilename, 0)) {
      err=biffGet(NRRD);
      cerr<<me<<":  error loading PCA coefficients:  "<<err<<endl;
      free(err);
      biffDone(NRRD);
      exit(1);
    }
  }
  
  Nrrd* mean;
  if (meanfilename) {
    mean=nrrdNew();
    if (nrrdLoad(mean, meanfilename, 0)) {
      err=biffGet(NRRD);
      cerr<<me<<":  error loading mean values:  "<<err<<endl;
      free(err);
      biffDone(NRRD);
      exit(1);
    }
  }

  // Sanity check
  // XXX - insert sanity checks here

  // Permute tex[x,y,channel] to tex[channel,x,y]
  int tex_axes[3]={2,0,1};
  tex=permuteNrrd(tex, tex_axes);
  if (!tex) {
    cerr<<me<<":  error permuting textures"<<endl;
    exit(1);
  }
  
  // Convert textures to nrrdTypeFloat, if necessary
  basis=convertNrrdToFloat(basis);
  if (!basis) {
    cerr<<me<<":  error converting basis textures to nrrdTypeFloat"<<endl;
    exit(1);
  }

  tex=convertNrrdToFloat(tex);
  if (!tex) {
    cerr<<me<<":  error converting original textures to nrrdTypeFloat"<<endl;
    exit(1);
  }
  
  if (coefffilename) {
    coeff=convertNrrdToFloat(coeff);
    if (!coeff) {
      cerr<<me<<":  error converting PCA coefficient to nrrdTypeFloat"<<endl;
      exit(1);
    }
  }

  if (meanfilename) {
    mean=convertNrrdToFloat(mean);
    if (!mean) {
      cerr<<me<<":  error converting mean values to nrrdTypeFloat"<<endl;
      exit(1);
    }
  }

  // Set up useful variables
  ntextures=tex->axis[0].size;
  width=tex->axis[1].size;
  height=tex->axis[2].size;
  nbases=basis->axis[0].size;
  npixels=width*height;

  // Compute the mean value of each texture
  if (!meanfilename) {
    mean=computeMean(tex);
    if (!mean) {
      cerr<<me<<":  error computing mean value of textures"<<endl;
      exit(1);
    }
  }
    
  // Subtract the mean value from each texture
  if (verbose)
    cout<<"Subtracting mean values from textures"<<endl;
  
  float *tex_data=(float*)(tex->data);
  float *mean_data=(float*)(mean->data);
  for (int y=0; y<height; y++) {
    for (int x=0; x<height; x++) {
      for(int tex=0; tex<ntextures; tex++)
	tex_data[tex]-=mean_data[tex];
      
      tex_data+=ntextures;
    }
  }
  
  // Determine new PCA coefficients for input textures
  basis=normalizeBasis(basis);
  if (!coefffilename) {
    coeff=computeCoefficients(tex, basis);
    if (!coeff) {
      cerr<<me<<":  error computing PCA coefficients"<<endl;
      exit(1);
    }
  }
  
  // Compute error between reconstructed and original textures
  float* error=computeError(basis, coeff, mean, tex);
  if (!error) {
    cerr<<me<<":  error computing quantization error"<<endl;
    exit(1);
  }

  // Write the output file
  Nrrd* nout=nrrdNew();
  if (nrrdWrap(nout, error, nrrdTypeFloat, 1, ntextures)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error creating error nrrd:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }
  
  if (nrrdSave(outfilename, nout, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error saving to "<<outfilename<<":  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  if (verbose)
    cout<<"Wrote data to "<<outfilename<<endl;
  
  // Clean up memory
  basis=nrrdNuke(basis);
  coeff=nrrdNuke(coeff);
  mean=nrrdNuke(mean);
  tex=nrrdNuke(tex);
  delete [] error;
  // XXX - don't know why, but this causes a seg fault
  // nout=nrrdNuke(nout);
  
  return 0;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<":  unrecognized option \""<<unknown<<"\""<<endl;
  
  // Print out the usage
  cerr<<"usage:  "<<me<<" [options] -o <filename>"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -i <basename>   basename of input files (null)"<<endl;
  cerr<<"  -b <filename>   filename of basis textures (null)"<<endl;
  cerr<<"  -c <filename>   filename of PCA coefficients (null)"<<endl;
  cerr<<"  -m <filename>   filename of mean values (null)"<<endl;
  cerr<<"  -t <filename>   filename of original textures (null)"<<endl;
  cerr<<"  -inf_norm       use infinity norm to calculate error (false)"<<endl;
  cerr<<"  -nrrd           use .nrrd extension (false)"<<endl;
  cerr<<"  -v <int>        set verbosity level (0)"<<endl;
  cerr<<"  --help          print this message and exit"<<endl;

  if (unknown)
    exit(1);
}

Nrrd* permuteNrrd(Nrrd* nin, int* axes) {
  char* me="permuteNrrd";
  char* err;
  
  Nrrd *tmp=nrrdNew();
  if (nrrdAxesPermute(tmp, nin, axes)) {
    err=biffGetDone(NRRD);
    cerr<<me<<":  error permuting nrrd:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    return 0;
  }
  
  nrrdNuke(nin);

  return tmp;
}

Nrrd* convertNrrdToFloat(Nrrd* nin) {
  char* me="convertNrrdToFloat";
  char* err;
  
  if (nin->type!=nrrdTypeFloat) {
    if (verbose) {
      cout<<"Converting nrrd from type ";
      switch(nin->type) {
      case nrrdTypeUnknown: cout<<"nrrdTypeUnknown"; break;
      case nrrdTypeChar: cout<<"nrrdTypeChar"; break;
      case nrrdTypeUChar: cout<<"nrrdTypeUChar"; break;
      case nrrdTypeShort: cout<<"nrrdTypeShort"; break;
      case nrrdTypeUShort: cout<<"nrrdTypeUShort"; break;
      case nrrdTypeInt: cout<<"nrrdTypeInt"; break;
      case nrrdTypeUInt: cout<<"nrrdTypeUInt"; break;
      case nrrdTypeLLong: cout<<"nrrdTypeLLong"; break;
      case nrrdTypeULLong: cout<<"nrrdTypeULLong"; break;
      case nrrdTypeDouble: cout<<"nrrdTypeDouble"; break;
      default: cout<<"unknown type"<<endl;
      }
      cout<<" to nrrdTypeFloat\n";
    }
    
    Nrrd *tmp=nrrdNew();
    if (nrrdConvert(tmp, nin, nrrdTypeFloat)) {
      err=biffGet(NRRD);
      cerr<<me<<":  unable to convert:  "<<err<<endl;
      biffDone(NRRD);
      return 0;
    }
    
    // Data was copied, so nuke the original version and reassign
    nrrdNuke(nin);
    return tmp;
  }

  // No need to convert
  return nin;
}

Nrrd *computeMean(Nrrd* tex) {
  char* me="computeMean";
  char* err;

  // Compute the mean values
  if (verbose)
    cout<<"Computing mean values"<<endl;

  // Allocate the mean values
  Nrrd* meanY=nrrdNew();
  Nrrd* mean=nrrdNew();
  int E=0;
  if (!E) E|=nrrdProject(meanY, tex, 2, nrrdMeasureMean, nrrdTypeDefault);
  if (!E) E|=nrrdProject(mean, meanY, 1, nrrdMeasureMean, nrrdTypeDefault);
  if (E) {
    err=biffGet(NRRD);
    cerr<<me<<":  error computing the mean values:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    return 0;
  }

  nrrdAxisInfoSet(mean, nrrdAxisInfoLabel, "mean");

  // Nuke meanY
  meanY=nrrdNuke(meanY);
  
  return mean;
}

Nrrd* normalizeBasis(Nrrd* basis) {
  char* me="normalizeBasis";
  
  // Normalize the basis textures
  if (verbose)
    cout<<"Normalizing basis textures"<<endl;
  
  float* mag=new float[nbases];
  if (!mag) {
    cerr<<me<<":  couldn't allocate the memory for mag"<<endl;
    return 0;
  }
  
  for (int b=0; b<nbases; b++)
    mag[b]=0;
    
  float* bt_data=(float*)(basis->data);
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      for (int b=0; b<nbases; b++)
	mag[b]+=bt_data[b]*bt_data[b];
      
      bt_data+=nbases;
    }
  }

  for (int b=0; b<nbases; b++)
    mag[b]=sqrt(mag[b]);
    
  bt_data=(float*)(basis->data);
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      for (int b=0; b<nbases; b++)
	bt_data[b]/=mag[b];
    
      bt_data+=nbases;
    }
  }

  delete [] mag;
  
  return basis;
}

Nrrd* computeCoefficients(Nrrd* tex, Nrrd* basis) {
  char* me="computeCoefficients";
  char* err;
  
  // Allocate the PCA coefficient
  Nrrd *coeff=nrrdNew();
  if (nrrdAlloc(coeff, nrrdTypeFloat, 2, nbases, ntextures)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error allocating PCA coefficients:  "<<err<<endl;
    return 0;
  }
  
  nrrdAxisInfoSet(coeff, nrrdAxisInfoLabel, "basis", "channel");

  // Calculate new PCA coefficients
  if (verbose)
    cout << "Computing new PCA coefficients" << endl;

  // Mean values has already been subtracted, so compute the
  // dot product of bases and textures
  float *tex_data=(float*)(tex->data);
  float* bt_data=(float*)(basis->data);
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      float* c_data=(float*)(coeff->data);
      for (int tex=0; tex<ntextures; tex++) {
	for (int b=0; b<nbases; b++)
	  c_data[b]+=bt_data[b]*tex_data[tex];
	
	c_data+=nbases;
      }

      bt_data+=nbases;
      tex_data+=ntextures;
    }
  }
  
  return coeff;
}

float* computeError(Nrrd* basis, Nrrd* coeff, Nrrd* mean, Nrrd* tex) {
  char* me="computeError";

  // Compute reconstruction error
  if (verbose)
    cout<<"Computing reconstruction error"<<endl;

  // Allocate error array
  float* error=new float[ntextures];
  if (!error) {
    cerr<<me<<":  error allocating memory for error array"<<endl;
    exit(1);
  }

  // Allocate memory for the reconstructed texture
  float* recon=new float[npixels];
  if (!recon) {
    cerr<<me<<":  error allocating memory for reconstructed texture"<<endl;
    return 0;
  }

  float* tex_data=(float*)(tex->data);
  float* c_data=(float*)(coeff->data);
  float* m_data=(float*)(mean->data);
  for (int tex=0; tex<ntextures; tex++) {
    float* b_data=(float*)(basis->data);
    float* r_data=recon;
    for (int i=0; i<npixels; i++)
      r_data[i]=0;
    
    // Reconstruct texture in current basis
    for (int y=0; y<height; y++) {
      for (int x=0; x<width; x++) {
	// Multiply coefficients and basis textures
	for(int b=0; b<nbases; b++)
	  r_data[x]+=c_data[b]*b_data[b];

	// Add the mean value to reconstructed texture
	r_data[x]+=m_data[tex];

	b_data+=nbases;
      }
      
      r_data+=width;
    }

    // Compute error between reconstructed and original
    r_data=recon;
    if (inf_norm) {
      // Using infinity norm
      error[tex]=-FLT_MAX;
      for (int y=0; y<height; y++) {
	for (int x=0; x<width; x++) {
	  float residual=fabsf(r_data[x] -
			       tex_data[(y*width + x)*ntextures + tex]);
	  if (residual>error[tex])
	    error[tex]=residual;
	}
	
	r_data+=width;
      }
    } else {
      // Using mean squared error
      for (int y=0; y<height; y++) {
	for (int x=0; x<width; x++) {
	  float residual=r_data[x] - tex_data[(y*width + x)*ntextures + tex];
	  error[tex]+=residual*residual;
	}
	
	r_data+=width;
      }
      
      error[tex]/=npixels;
    }
    
    c_data+=nbases;
  }

  delete [] recon;

  float max_error=-FLT_MAX;
  float total_error=0;
  for (int tex=0; tex<ntextures; tex++) {
    if (error[tex]>max_error)
      max_error=error[tex];
    
    total_error+=error[tex];
  }
  
  if (verbose) {
    cout<<"Maximum error:  "<<max_error<<endl;
    cout<<"Mean error:  "<<total_error/ntextures<<endl;
  }

  return error;
}
