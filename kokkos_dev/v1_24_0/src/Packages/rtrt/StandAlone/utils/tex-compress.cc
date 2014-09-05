#include <teem/nrrd.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <sgi_stl_warnings_on.h>

// #define SAVE_ERROR 1

using namespace std;

// Declare external FORTRAN function
extern "C" {
  void ssyevx_(const char& jobz, const char& range, const char& uplo,
	       const int& n, float data_array[], const int& lda,
	       const float& vl, const float& vu,
	       const int& il, const int& iu,
	       const float& tolerance,
	       int& eval_cnt, float eigen_val[],
	       float eigen_vec[], const int& ldz,
	       float work[], const int& lwork, int iwork[],
	       int ifail[], int& info);
}

class IndexValuePair {
public:
  int idx;
  float value;

  // Defined for sorting in descending order
  inline bool operator<(const IndexValuePair& other) const {
    return (other.value<value);
  };
};


// Global variables
int num_textures=0;
int num_subset=100;
int num_add=10;
int num_bases=10;
int width=0;
int height=0;
int num_pixels=0;
float max_error=0;
float mean_error=0;
float prev_error=0;
bool use_mean=true;
char* outbasename=0;
bool inf_norm=true;
char* nrrd_ext=".nhdr";
int verbose=0;

void usage(char* me, const char* unknown=0) {
  if (unknown)
    cerr<<me<<":  unrecognized option \""<<unknown<<"\""<<endl;
  
  // Print out the usage
  cerr<<"usage:  "<<me<<" [options] -i <filename> -o <basename>"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -nsubset <int>    maximum number of textures to include in PCA subset (100)"<<endl;
  cerr<<"  -nadd <int>       number of textures to add per iteration (10)"<<endl;
  cerr<<"  -nbases <int>     maximum number of basis textures for PCA (10)"<<endl;
  cerr<<"  -mse              compute the mean squared error (false)"<<endl;
  cerr<<"  -thresh <float>   set error threshold for early termination (0)"<<endl;
  cerr<<"  -maxerr           use maximum error when evaluating early termination (false)"<<endl;
  cerr<<"  -nrrd             use .nrrd extension (false)"<<endl;
  cerr<<"  -v <int>          set verbosity level (0)"<<endl;
  cerr<<"  --help            print this message and exit"<<endl;

  if (unknown)
    exit(1);
}

Nrrd* convertNrrdToFloat(Nrrd* nin) {
  char *me="convertNrrdToFloat";
  char *err;
  
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

Nrrd *computeMean(Nrrd* nin) {
  char* me="computeMean";
  char* err;

  // Compute the mean values
  if (verbose)
    cout<<"Computing mean values"<<endl;

  // Allocate the mean values
  Nrrd* meanY=nrrdNew();
  Nrrd* mean=nrrdNew();
  int E=0;
  if (!E) E|=nrrdProject(meanY, nin, 2, nrrdMeasureMean, nrrdTypeDefault);
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

IndexValuePair* computeVariance(Nrrd* nin) {
  char* me="computeVariance";
  
  if (verbose)
    cout<<"Computing texture variance"<<endl;
  
  IndexValuePair* variance=new IndexValuePair[num_textures];
  if (!variance) {
    cerr<<me<<":  error allocating memory for variance"<<endl;
    return 0;
  }
  
  float* in_data=(float*)(nin->data);
  for (int tex=0; tex<num_textures; tex++) {
    float min = FLT_MAX;
    float max = -FLT_MAX;
    for (int y=0; y<height; y++) {
      for (int x=0; x<width; x++) {
	float value = in_data[(y*width + x)*num_textures + tex];
	if (value < min)
	  min = value;
	if (value > max)
	  max = value;
      }
    }
    
    variance[tex].idx = tex;
    variance[tex].value = max - min;
  }

  return variance;
}

Nrrd* computeCovariance(Nrrd* nin, Nrrd* mean, int* pca_idx) {
  char* me="computeCovariance";
  char* err;

  // Compute the covariance matrix
  if (verbose)
    cout<<"Computing covariance matrix ("
	<<num_subset<<"x"<<num_subset<<")"<<endl;
  
  // Allocate the covariance matrix
  Nrrd* cov=nrrdNew();
  if (nrrdAlloc(cov, nrrdTypeFloat, 2, num_subset, num_subset)) {
    err=biffGetDone(NRRD);
    cerr<<me<<":  error allocating covariance matrix:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    return 0;
  }
    
  // Loop over each pixel to compute cov(x,y)
  float* in_data=(float*)(nin->data);
  float* mean_data=(float*)(mean->data);
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      float* cov_data=(float*)(cov->data);
      for(int column=0; column<num_subset; column++) {
	for(int row=column; row<num_subset; row++) {
	  int pixel=(y*width + x)*num_textures;
	  int idx1=pixel + pca_idx[column];
	  int idx2=pixel + pca_idx[row];

	  cov_data[row]+=(in_data[idx1] + mean_data[pca_idx[column]])*
			 (in_data[idx2] + mean_data[pca_idx[row]]);
	}
	
	cov_data+=num_subset;
      }
    }
  }

  // Subtract the mean values
  if (verbose)
    cout<<"Subtracting mean values from covariance matrix"<<endl;
  
  float* cov_data=(float*)(cov->data);
  mean_data=(float*)(mean->data);
  float inv_num_pixels=1.0f/(width*height);
  for(int column=0; column<num_subset; column++) {
    for(int row=column; row<num_subset; row++)
      cov_data[row]=cov_data[row]*inv_num_pixels -
	mean_data[pca_idx[column]]*mean_data[pca_idx[row]];


    cov_data+=num_subset;
  }

  return cov;
}

float* computeEigenvectors(Nrrd* cov) {
  char* me="computeEigenvectors";
  
  // Compute eigenvectors
  if (verbose)
    cout<<"Computing eigenvectors"<<endl;
  
  float* eval=new float[num_subset];
  float* cov_data=(float*)(cov->data);
  float* evec_data=0;
  
  // Use LAPACK's expert driver
  if (verbose)
    cout<<"Using LAPACK's expert driver (ssyevx)"<<endl;
  
  const char jobz='V';
  const char uplo='L';
  const int N=num_subset;
  const int lda=N;
  const char range='I';
  const float vl=0;
  const float vu=0;
  const int il=N - num_bases + 1;
  const int iu=N;
  const float tolerance=0;
  const int ldz=N;
  int eval_cnt;
  int info;
  
  evec_data=new float[ldz*N];
  if (!evec_data) {
    cerr<<me<<":  couldn't allocate the memory for evec_data"<<endl;
    return 0;
  }
    
  const int lwork=8*N;
  float* work=new float[lwork];
  if (!work) {
    cerr<<me<<":  couldn't allocate the memory for work"<<endl;
    return 0;
  }
    
  int* iwork=new int[5*N];
  if (!iwork) {
    cerr<<me<<":  couldn't allocate the memory for iwork"<<endl;
    return 0;
  }
    
  int* ifail=new int[N];;
      
  if (verbose>10)
    cout<<"Solving for eigenvectors in the range ["
	<<il<<", "<<iu<<"]"<<endl;
    
  ssyevx_(jobz, range, uplo, N, cov_data, lda, vl, vu, il, iu,
	  tolerance, eval_cnt, eval, evec_data, ldz, work, lwork,
	  iwork, ifail, info);
      
  delete [] work;
  delete [] iwork;
  delete [] ifail;
      
  if (info!=0) {
    if (info<0) {
      cerr<<me<<":  ssyevx_ error:  "<<(-info)
	  <<"th argument has an illegal value"<<endl;
      return 0;
    } else if (info>0) {
      cerr<<me<<":  ssyevx_ error:  "<<info
	  <<" failed to converge"<<endl;
      return 0;
    }
  }
  
  if (verbose) {
    float recovered_var=0;
    for (int i=0; i<num_bases; i++)
      recovered_var+=eval[i];
    
    cout <<"Recovered "<<recovered_var<<" units of the total variance"<<endl;
  }
  
  delete[] eval;
  
  return evec_data;
}

Nrrd* computeTransform(float* evec_data) {
  char* me="computeTransform";
  char* err;

  // Compute the transformation matrix
  if (verbose)
    cout<<"Computing the transformation matrix ("
	<<num_bases<<"x"<<num_subset<<")"<<endl;

  // Allocate the transformation matrix
  Nrrd* transform=nrrdNew();
  if (nrrdAlloc(transform, nrrdTypeFloat, 2, num_subset, num_bases)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error allocating transformation matrix:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    return 0;
  }
  
  nrrdAxisInfoSet(transform, nrrdAxisInfoLabel, "channel", "basis");
  
  // Copy the eigenvectors
  float* tdata=(float*)(transform->data);
  float* edata=evec_data;
  for (int b=0; b<num_bases; b++) {
    for(int c=0; c<num_subset; c++) {
      *tdata=edata[b*num_subset+c];
      tdata++;
    }
  }

  return transform;
}

Nrrd* computeBasis(Nrrd* nin, Nrrd* transform, int* pca_idx) {
  char* me="computeBasis";
  char* err;
  
  // Compute the basis textures
  if (verbose)
    cout<<"Computing "<<num_bases<<" basis textures"<<endl;
  
  Nrrd *basis=nrrdNew();
  if (nrrdAlloc(basis, nrrdTypeFloat, 3, num_bases, width, height)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error allocating basis textures:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    return 0;
  }
  
  nrrdAxisInfoSet(basis, nrrdAxisInfoLabel, "basis", "width", "height");

  float* in_data=(float*)(nin->data);
  float* bt_data=(float*)(basis->data);
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      // Mean value has already been subtracted, so multiply
      // transformation matrix and textures
      float* t_data=(float*)(transform->data);
      for (int c=0; c<num_bases; c++) {
	for (int r=0; r<num_subset; r++) {
	  int idx=(y*width + x)*num_textures + pca_idx[r];
	  bt_data[c]+=t_data[c*num_subset+r]*in_data[idx];
	}
      }

      bt_data+=num_bases;
    }
  }

  return basis;
}

Nrrd *normalizeBasis(Nrrd* basis) {
  char* me="normalizeBasis";
  
  // Normalize the basis textures
  if (verbose)
    cout<<"Normalizing basis textures"<<endl;
  
  float* mag=new float[num_bases];
  if (!mag) {
    cerr<<me<<":  couldn't allocate the memory for mag"<<endl;
    return 0;
  }
  
  for (int b=0; b<num_bases; b++)
    mag[b]=0;
    
  float* bt_data=(float*)(basis->data);
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      for (int b=0; b<num_bases; b++)
	mag[b]+=bt_data[b]*bt_data[b];
      
      bt_data+=num_bases;
    }
  }

  for (int b=0; b<num_bases; b++)
    mag[b]=sqrt(mag[b]);
    
  bt_data=(float*)(basis->data);
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      for (int b=0; b<num_bases; b++)
	bt_data[b]/=mag[b];
    
      bt_data+=num_bases;
    }
  }

  delete [] mag;
  
  return basis;
}

Nrrd* computeCoefficients(Nrrd* nin, Nrrd* basis) {
  char* me="computeCoefficients";
  char* err;
  
  // Allocate the PCA coefficient
  Nrrd *coeff=nrrdNew();
  if (nrrdAlloc(coeff, nrrdTypeFloat, 2, num_bases, num_textures)) {
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
  float *in_data=(float*)(nin->data);
  float* bt_data=(float*)(basis->data);
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      float* c_data=(float*)(coeff->data);
      for (int tex=0; tex<num_textures; tex++) {
	for (int b=0; b<num_bases; b++)
	  c_data[b]+=bt_data[b]*in_data[tex];
	
	c_data+=num_bases;
      }

      bt_data+=num_bases;
      in_data+=num_textures;
    }
  }
  
  return coeff;
}

IndexValuePair* computeError(Nrrd* nin, Nrrd* basis, Nrrd* coeff,
			     Nrrd* mean, IndexValuePair* error) {
  char* me="computeError";

  // Compute texture with maximum error in current basis
  if (verbose)
    cout<<"Computing error in current basis"<<endl;
  
  // Allocate memory for the reconstructed texture
  float* recon=new float[num_pixels];
  if (!recon) {
    cerr<<me<<":  error allocating memory for reconstructed texture"<<endl;
    return 0;
  }

  float* in_data=(float*)(nin->data);
  float* c_data=(float*)(coeff->data);
  float* mean_data=(float*)(mean->data);
  for (int tex=0; tex<num_textures; tex++) {
    float* bt_data=(float*)(basis->data);
    float* r_data=recon;
    for (int i=0; i<num_pixels; i++)
      r_data[i]=0;
    
    // Reconstruct texture in current basis
    for (int y=0; y<height; y++) {
      for (int x=0; x<width; x++) {
	// Multiply coefficients and basis textures
	for(int b=0; b<num_bases; b++)
	  r_data[x]+=c_data[b]*bt_data[b];

	// Add the mean value to reconstructed texture
	r_data[x]+=mean_data[tex];

	bt_data+=num_bases;
      }
      
      r_data+=width;
    }

    // Compute error between reconstructed and original
    r_data=recon;
    error[tex].idx=tex;
    if (inf_norm) {
      // Using infinity norm
      error[tex].value=-FLT_MAX;
      for (int y=0; y<height; y++) {
	for (int x=0; x<width; x++) {
	  float residual=fabsf(r_data[x] -
			       in_data[(y*width + x)*num_textures + tex]);
	  if (residual>error[tex].value)
	    error[tex].value=residual;
	}
	
	r_data+=width;
      }
    } else {
      // Using mean squared error
      for (int y=0; y<height; y++) {
	for (int x=0; x<width; x++) {
	  float residual=r_data[x] - in_data[(y*width + x)*num_textures + tex];
	  error[tex].value+=residual*residual;
	}
	
	r_data+=width;
      }
      
      error[tex].value/=num_pixels;
    }
    
    c_data+=num_bases;
  }

  delete [] recon;

  if (use_mean)
    prev_error=mean_error;
  else
    prev_error=max_error;
  
  float total_error=0;
  for (int tex=0; tex<num_textures; tex++) {
    if (error[tex].value>max_error)
      max_error=error[tex].value;
    
    total_error+=error[tex].value;
  }

  mean_error=total_error/num_textures;
  
  if (verbose) {
    cout<<"Maximum error:  "<<max_error<<endl;
    cout<<"Mean error:  "<<mean_error<<endl;
  }

  return error;
}

int* updatePCASubset(IndexValuePair* error, int* pca_idx) {
  char* me="updatePCASubset";

  if (verbose)
    cout<<"Updating PCA subset"<<endl;
  
  // Ignore already included textures
  for (int subset=0; subset<num_subset; subset++) {
    for (int tex=0; tex<num_textures; tex++) {
      if (pca_idx[subset]==error[tex].idx) {
	error[tex].value=0;
	break;
      }
    }
  }
  
#ifdef SAVE_ERROR
  for (int add=0; add<num_add; add++) {
    // Find texture with highest error
    float max_error=-FLT_MAX;
    int tex_idx=-1;
    for (int tex=0; tex<num_textures; tex++) {
      if (error[tex].value>max_error) {
	max_error=error[tex].value;
	tex_idx=error[tex].idx;
      }
    }
    
    if (tex_idx<0) {
      cerr<<me<<":  error updating PCA subset:  invalid texture index"<<endl;
      return 0;
    }

    // Update PCA subset
    if (verbose>4)
      cout<<"Adding texture["<<tex_idx<<"] to PCA subset"<<endl;

    for (int tex=0; tex<num_textures; tex++) {
      if (error[tex].idx==tex_idx) {
	error[tex].value=0;
	break;
      }
    }
    
    pca_idx[num_subset]=tex_idx;
    num_subset++;
  }
#else
#if 0
  // XXX - Why doesn't nth element give the same result as
  //       actually sorting the array?
  // Put num_add elements with highest error at top
  nth_element(error, error+num_add, error+num_textures);
#else
  // Sort the error array
  sort(error, error+num_textures);
#endif

  // Update PCA subset
  for (int add=0; add<num_add; add++) {
    if (verbose>4)
      cout<<"Adding texture["<<error[add].idx<<"] to PCA subset"<<endl;
    
    pca_idx[num_subset]=error[add].idx;
    num_subset++;
  }
#endif

  return pca_idx;
}

int saveNrrd(Nrrd* nin, char* type, char* ext) {
  char* me="saveNrrd";
  char* err;
  
  size_t outbasename_len=strlen(outbasename);
  size_t type_len=strlen(type);
  size_t ext_len=strlen(ext);
  size_t length=outbasename_len+type_len+ext_len;
  
  char* fname=new char[length];
  sprintf(fname, "%s%s%s", outbasename, type, ext);
  if (nrrdSave(fname, nin, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error saving to "<<fname<<":  "<<err<<endl;
    return 1;
  }

  if (verbose)
    cout<<"Wrote data to "<<fname<<endl;
  
  return 0;
}

#ifdef SAVE_ERROR
int saveError(IndexValuePair* error, Nrrd* errorNrrd) {
  char* me="saveError";
  
  // Sort the reconstruction error array
  sort(error, error+num_textures);
  
  if (num_subset%10==0) {
    // Save the error nrrd
    float* error_data=(float*)(errorNrrd->data);
    for (int tex=0; tex<num_textures; tex++)
      error_data[tex]=error[tex].value;
    
    char type[24];
    sprintf(type, "-error%05d", num_subset);
    if (saveNrrd(errorNrrd, type, ".nrrd")) {
      cerr<<me<<":  error saving reconstruction error nrrd"<<endl;
      return 1;
    }
  }

  return 0;
}
#endif

int main(int argc, char *argv[]) {
  char* me=argv[0];
  char* err;
  char* infilename=0;
  int max_num_subset=0;
  int max_num_bases=0;
  float error_thresh=0;
  
  // Parse arguments
  for(int i=1; i<argc; i++) {
    string arg(argv[i]);
    if (arg=="-input" || arg=="-i") {
      infilename=argv[++i];
    } else if (arg=="-output" || arg=="-o") {
      outbasename=argv[++i];
    } else if (arg=="-nsubset") {
      max_num_subset=atoi(argv[++i]);
    } else if (arg=="-nadd") {
      num_add=atoi(argv[++i]);
    } else if (arg=="-nbases") {
      max_num_bases=atoi(argv[++i]);
    } else if (arg=="-mse") {
      inf_norm=false;
    } else if (arg=="-thresh") {
      error_thresh=atof(argv[++i]);
    } else if (arg=="-maxerr") {
      use_mean=false;
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
  if (!infilename) {
    cerr<<me<<":  input filename not specified"<<endl;
    usage(me);
    exit(1);
  }
  
  if (!outbasename) {
    cerr<<me<<":  output basename not specified"<<endl;
    usage(me);
    exit(1);
  }

  if (max_num_bases<=0) {
    cerr<<me<<":  invalid number of basis textures ("<<num_bases<<")"<<endl;
    exit(1);
  }

  if (num_add<=0) {
    cerr<<me<<":  invalid number of textures to add to PCA subset per iteration ("
        <<num_add<<")"<<endl;
    exit(1);
  }
  
  // Load the input textures
  Nrrd* nin=nrrdNew();
  if (nrrdLoad(nin, infilename, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error loading textures:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Verify the texture dimensions
  if (nin->dim!=3) {
    cerr<<me<<":  number of dimesions "<<nin->dim
	<<" of textures is not equal to 3,"
	<<" expecting (width, height, texture)"<<endl;
    exit(1);
  }

  // Set up useful variables
  width=nin->axis[0].size;
  height=nin->axis[1].size;
  num_textures=nin->axis[2].size;
  num_pixels=width*height;

  // Sanity check
  if (max_num_bases>num_textures) {
    cerr<<"number of basis textures ("<<num_bases<<") is greater"
	<<" than the number input textures ("<<num_textures<<")"<<endl;
    exit(1);
  } else if (max_num_subset<=0) {
    max_num_subset=num_textures;
    cerr<<"setting maximum number of textures in PCA subset to "
	<<max_num_subset<<endl;
  }
  
  // Permute nin[x,y,channel] to nin[channel,x,y]
  int nin_axes[3]={2,0,1};
  nin=permuteNrrd(nin, nin_axes);
  if (!nin) {
    cerr<<me<<":  error permuting textures"<<endl;
    exit(1);
  }
  
  // Convert textures to nrrdTypeFloat, if necessary
  nin=convertNrrdToFloat(nin);
  if (!nin) {
    cerr<<me<<":  error converting textures to nrrdTypeFloat"<<endl;
    exit(1);
  }

  // Compute the mean value of each texture
  Nrrd* mean=computeMean(nin);
  if (!mean) {
    cerr<<me<<":  error computing mean value of textures"<<endl;
    exit(1);
  }

  // Subtract the mean value from each texture
  if (verbose)
    cout<<"Subtracting mean values from textures"<<endl;
  
  float *in_data=(float*)(nin->data);
  float *mean_data=(float*)(mean->data);
  for (int y=0; y<height; y++) {
    for (int x=0; x<height; x++) {
      for(int tex=0; tex<num_textures; tex++)
	in_data[tex]-=mean_data[tex];

      in_data+=num_textures;
    }
  }
  
  // Setup the variables 
  Nrrd* basis=0;
  Nrrd* coeff=0;

  int* pca_idx=new int[max_num_subset];
  if (!pca_idx) {
    cerr<<me<<":  error allocating memory for PCA index array"<<endl;
    exit(1);
  }

  IndexValuePair* error=new IndexValuePair[num_textures];
  if (!error) {
    cerr<<me<<":  error allocating memory for error array"<<endl;
    exit(1);
  }

#ifdef SAVE_ERROR
  Nrrd* errorNrrd=nrrdNew();
  if (nrrdAlloc(errorNrrd, nrrdTypeFloat, 1, num_textures)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error allocating error nrrd:  "<<err<<endl;
    exit(1);
  }
#endif

  // Prime the PCA subset with highest variance textures
  if (verbose)
    cout<<"Priming the PCA subset"<<endl;
  
  IndexValuePair *variance=computeVariance(nin);
  sort(variance, variance+num_textures);
  for (num_subset=0; num_subset<num_add; num_subset++) {
    if (verbose>4)
      cout<<"Adding texture["<<variance[num_subset].idx<<"] to PCA subset"<<endl;
    
    pca_idx[num_subset]=variance[num_subset].idx;
  }

  delete [] variance;
  
  if (verbose)
    cout << endl;

  // Main loop
  do {
    if (verbose)
      cout<<"-------------------------------------------------------"<<endl;
    
    // Determine new basis textures via PCA
    num_bases=(num_subset>max_num_bases) ? max_num_bases : num_subset;

    Nrrd* cov=computeCovariance(nin, mean, pca_idx);
    if (!cov) {
      cerr<<me<<":  error computing covariance matrix"<<endl;
      exit(1);
    }

    float* evec=computeEigenvectors(cov);
    if (!evec) {
      cerr<<me<<":  error solving for eigenvectors"<<endl;
      exit(1);
    }

    nrrdNuke(cov);

    Nrrd* transform=computeTransform(evec);
    if (!transform) {
      cerr<<me<<":  error computing transformation matrix"<<endl;
      exit(1);
    }

    delete [] evec;

    basis=computeBasis(nin, transform, pca_idx);
    if (!basis) {
      cerr<<me<<":  error computing basis textures"<<endl;
      exit(1);
    }

    nrrdNuke(transform);
    
    // Determine new PCA coefficients for input textures
    basis=normalizeBasis(basis);
    coeff=computeCoefficients(nin, basis);
    if (!coeff) {
      cerr<<me<<":  error computing PCA coefficients"<<endl;
      exit(1);
    }

    // Compute error between reconstructed and original textures
    error=computeError(nin, basis, coeff, mean, error);
    if (!error) {
      cerr<<me<<":  error computing reconstruction error"<<endl;
      exit(1);
    }

#ifdef SAVE_ERROR
    if (saveError(error, errorNrrd)) {
      cerr<<me<<":  error saving reconstruction error nrrd"<<endl;
      exit(1);
    }
#endif

    // Check termination criteria
    float delta;
    if (use_mean)
      delta=fabsf(mean_error-prev_error);
    else
      delta=fabsf(max_error-prev_error);

    if (verbose)
      cout<<"Error delta:  "<<delta<<endl;

    if (delta<error_thresh) {
      if (verbose) {
	cout<<"-------------------------------------------------------"<<endl;
	cout<<endl;
	cout<<"Error between iterations ("<<delta<<") is below threshold ("
	    <<error_thresh<<")"<<endl;
	cout<<endl;
      }
      
      break;
    } else if (num_subset>=max_num_subset) {
      if (verbose) {
	cout<<"-------------------------------------------------------"<<endl;
	cout<<endl;
	cout<<"Maximum number of PCA subset textures has been reached"<<endl;
	cout<<endl;
      }

      break;
    }

    // Update PCA subset
    pca_idx=updatePCASubset(error, pca_idx);
    if (!pca_idx) {
      cerr<<me<<":  error updating PCA subset"<<endl;
      exit(1);
    }
    
    basis=nrrdNuke(basis);
    coeff=nrrdNuke(coeff);
  } while (true);

  if (verbose) {
  }
  
  // Save the mean values, basis textures, and PCA coefficients
  int E=0;
  if (!E) E|=saveNrrd(mean, "-mean", nrrd_ext);
  if (!E) E|=saveNrrd(basis, "-basis", nrrd_ext);
  if (!E) E|=saveNrrd(coeff, "-coeff", nrrd_ext);
  if (E) {
    cerr<<me<<":  error saving output nrrds"<<endl;
    exit(1);
  }

  // Clean up memory
  nin=nrrdNuke(nin);
  mean=nrrdNuke(mean);
  basis=nrrdNuke(basis);
  coeff=nrrdNuke(coeff);
  delete [] pca_idx;
  delete [] error;
#ifdef SAVE_ERROR
  errorNrrd=nrrdNuke(errorNrrd);
#endif
  
  return 0;
}
