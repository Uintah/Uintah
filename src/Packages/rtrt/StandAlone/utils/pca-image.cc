/* Some useful commands

unu permute -i ../sphere00000.nrrd -p 0 3 1 2 | unu reshape -s 24 64 64 -o group.nrrd

./pca-image -i group.nrrd -numbases 1 -o munged1

unu reshape -i munged1.nrrd -s 3 8 64 64 | unu permute -p 0 2 3 1 | unu reshape -s 3 64 64 2 4 | unu permute -p 0 1 3 2 4 | unu reshape -s 3 128 256 | XV &

icpc -O3 -o pca-image ../pca-image.cc -I/home/sci/bigler/SCIRun/src/include -I/home/sci/bigler/pub/src/teem/linux.64/include -Wl,-rpath -Wl,/home/sci/bigler/pub/src/teem/linux.64/lib -L/home/sci/bigler/pub/src/teem/linux.64/lib -lteem -lpng -lbz2 -llapack -L/usr/lib/gcc-lib/ia64-redhat-linux/3.0.4 -lg2c -lm

*/

#include <teem/nrrd.h>
#include <teem/ell.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

extern "C" {
  void ssyev_(const char& jobz, const char& uplo,
	      const int& n, float data_array[],
	      const int& lda, float eigen_val[],
	      float dwork[], const int& ldwork,
	      int& info);

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

using namespace std;

void usage(char *me, const char *unknown = 0) {
  if (unknown) {
    fprintf(stderr, "%s: unknown argument %s\n", me, unknown);
  }
  
  // Print out the usage
  printf("usage:  %s [options]\n", me);
  printf("options:\n");
  printf("  -i <filename>   input nrrd, channels as fastest axis (null)\n");
  printf("  -o <filename>   basename of output nrrds (\"pca\")\n");
  printf("  -nbases <int>   number of basis textures to use (0)\n");
  printf("  -saveall        write mean values and PCA coefficients to files (false)\n");
  printf("  -simple         use the LAPACK simple driver (false)\n");
  printf("  -nrrd           use .nrrd extension (false)\n");
  printf("  -v <int>        set verbosity level (0)\n");

  if (unknown)
    exit(1);
}

int main(int argc, char *argv[]) {
  char *me = argv[0];
  char *err;
  char *infilename = 0;
  char *outfilename_base = "pca";
  int num_bases = 0;
  bool saveall = false;
  bool use_simple = false;
  char *nrrd_ext = ".nhdr";
  int verbose = 0;

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
    } else if (arg == "-nbases") {
      num_bases = atoi(argv[++i]);
    } else if (arg == "-saveall") {
      saveall = true;
    } else if (arg == "-simple") {
      use_simple = true;
    } else if (arg == "-nrrd") {
      nrrd_ext = ".nrrd";
    } else if (arg == "-v" ) {
      verbose = atoi(argv[++i]);
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
    cerr << me << ":  number of dimesions " << nin->dim << " is not equal to 3." << endl;
    exit(2);
  }
  
  int num_channels = nin->axis[0].size;

  // Determine the number of eigen thingies to use
  if (num_bases < 1)
    // Use all of them
    num_bases = num_channels;
  // Check to make sure we haven't overstepped the max
  if (num_bases > num_channels) {
    cerr << "You have specified too many basis for the number of channels ("
	 <<num_bases<<").  Clamping to "<< num_channels<<".\n";
    num_bases = num_channels;
  }

  // convert the type to floats if you need to
  if (nin->type != nrrdTypeFloat) {
    cout << "Converting type from ";
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
    default: cout << "Unknown!!";
    }
    cout << " to nrrdTypeFloat\n";
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

  nrrdAxisInfoSet(mean, nrrdAxisInfoLabel, "mean");

  // Free meanY
  meanY=nrrdNuke(meanY);
  
  // Write out the mean value, if necessary
  if (saveall) {
    string meanname(string(outfilename_base) + "-mean" + string(nrrd_ext));
    if (nrrdSave(meanname.c_str(), mean, 0)) {
      err = biffGet(NRRD);
      fprintf(stderr, "%s: trouble saving to %s: %s\n", me, meanname.c_str(), err);
      exit(2);
    }
    
    if (verbose)
      cout << "Wrote out mean to "<<meanname<<"\n";
  }
  
  // Allocate our covariance matrix
  Nrrd *cov = nrrdNew();
  if (nrrdAlloc(cov, nrrdTypeFloat,2, num_channels, num_channels))
    {
      err = biffGetDone(NRRD);
      fprintf(stderr, "%s: error allocating covariance matrix:\n%s", me, err);
      exit(2);
    }
    
  if (verbose)
    cout << "Computing covariance matrix" << endl;
  
  // loop over each pixel to compute cov(x,y)
  int height = nin->axis[2].size;
  int width = nin->axis[1].size;
  float *data = (float*)nin->data;
  for (int y=0; y < height; y++) {
    for (int x = 0; x < width; x++ ) {
      // compute cov(x,y) for current pixel
      float *cov_data = (float*)(cov->data);
      for(int c = 0; c < num_channels; c++) {
	for(int r = c; r < num_channels; r++) {
	  cov_data[r] += data[c] * data[r];
	}
	cov_data += num_channels;
      }
      data += num_channels;
    }
  }

  if (verbose)
    cout << "Done computing covariance matrix" << endl;
  
  if (verbose > 10) {
    float *cov_data = (float*)(cov->data);
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
    float *cov_data = (float*)(cov->data);
    float *mean_data = (float*)(mean->data);
    float inv_num_pixels = 1.0f/(width*height);
    for(int c = 0; c < num_channels; c++) {
      for(int r = c; r < num_channels; r++) {
	cov_data[r] = cov_data[r]*inv_num_pixels - mean_data[c] * mean_data[r];
      }
      cov_data += num_channels;
    }
  }

  if (verbose)
    cout << "Subtracted mean value" << endl;

  if (verbose > 10)
  {
    float *cov_data = (float*)(cov->data);
    for(int r = 0; r < cov->axis[0].size; r++)
      {
	cout << "[";
	for(int c = 0; c < cov->axis[1].size; c++)
	  cout << cov_data[c*cov->axis[0].size + r] << ", ";
	cout << "]\n";
      }
  }

  if (verbose)
    cout<<"Beginning eigensolve"<<endl;
  
  // Compute eigen values/vectors
  float *eval = new float[num_channels];
  float *cov_data = (float*)(cov->data);
  float *evec_data = 0;
  
  {
    // Variables common to both LAPACK drivers
    const char jobz = 'V';
    const char uplo = 'L';
    const int N = num_channels;
    const int lda = N;
    int info;
    
    if (use_simple) {
      // Use LAPACK's simple driver
      if (verbose)
	cout<<"Using LAPACK's simple driver (ssyev)"<<endl;
      
      const int lwork = (3*N-1)*2;
      cout << "Wanting to allocate "<<lwork<<" floats.\n";
      float *work = new float[lwork];
      if (!work) {
	cerr << "Could not allocate the memory for work\n";
	exit(2);
      }
      
      ssyev_(jobz, uplo, N, cov_data, lda, eval, work, lwork, info);
      
      delete[] work;
      
      if (info != 0) {
	cout << "Eigensolver did not converge.\n";
	cout << "info  = "<<info<<"\n";
	exit(2);
      }
      
      evec_data = cov_data;
    } else {
      // Use LAPACK's expert driver
      if (verbose)
	cout<<"Using LAPACK's expert driver (ssyevx)"<<endl;
      
      const char range = 'I';
      const float vl = 0;
      const float vu = 0;
      const int il = N - num_bases + 1;
      const int iu = N;
      if (verbose > 10)
	cout<<"Solving for eigenvalues/vectors in the range = ["
	    <<il<<", "<<iu<<"]"<<endl;
      const float tolerance = 0;
      int eval_cnt;
      const int ldz = N;
      evec_data=new float[ldz*N];
      if (!evec_data) {
	cerr<<"Couldn't allocate the memory for evec_data"<<endl;
	exit(2);
      }
      const int lwork = 8*N;
      float *work = new float[lwork];
      if (!work) {
	cerr<<"Couldn't allocate the memory for work"<<endl;
	exit(2);
      }
      int *iwork=new int[5*N];
      if (!iwork) {
	cerr<<"Couldn't allocate the memory for iwork"<<endl;
	exit(2);
      }
      int *ifail=new int[N];;
      
      ssyevx_(jobz, range, uplo, N, cov_data, lda, vl, vu, il, iu,
	      tolerance, eval_cnt, eval, evec_data, ldz, work, lwork,
	      iwork, ifail, info);
      
      delete [] work;
      delete [] iwork;
      delete [] ifail;
      
      if (info != 0) {
	if (info < 0) {
	  cerr<<"ssyevx_ error:  "<<(-info)
	      <<"th argument has an illegal value"<<endl;
	  exit(2);
	} else if (info > 0) {
	  cerr<<"ssyevx_ error:  "<<info
	      <<" eigenvalues failed to converge"<<endl;
	  exit(2);
	}
      }
    }
  }

  if (verbose)
    cout<<"Eigensolve complete"<<endl;
  
  if (verbose > 10) {
    cout << "Eigenvalues are [";
    for (int i = 0; i < num_channels; i++)
      cout << eval[i]<<", ";
    cout << "]\n";
  }

  float recovered_var = 0;
  if (use_simple) {
    float total_var = 0;
    for (int i=0; i<num_channels; i++) {
      total_var += eval[i];
      if(i>=(num_channels-num_bases))
	recovered_var+=eval[i];
    }

    cout<<"Total variance equals "<<total_var<<endl;
    cout<<"Recovered "<<(recovered_var/total_var)*100.0<<"% of the "
	<<"variance with "<<num_bases<<" basis textures"<<endl;
  } else {
    // XXX - how to account for the total variance, now that we only solve
    //       for num_bases of the eigenvalues/vectors?
    for (int i=0; i<num_bases; i++) {
      recovered_var+=eval[i];
    }
    
    cout <<"Recovered "<<recovered_var<<" units of the total "
	 <<"variance with "<<num_bases<<" basis textures"<<endl;
  }
  
  delete[] eval;

  Nrrd *transform = nrrdNew();
  if (nrrdAlloc(transform, nrrdTypeFloat, 2, num_channels, num_bases)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: error allocating transform matrix :\n%s", me, err);
    exit(0);
  }
  
  nrrdAxisInfoSet(transform, nrrdAxisInfoLabel, "channel", "basis");
  
  float *tdata = (float*)(transform->data);
  float *edata = evec_data;
  if (use_simple) {
    // Cull the eigenvectors
    int basis_start = num_channels - 1;
    int basis_end = basis_start - num_bases;
    for (int basis = basis_start; basis > basis_end; basis--) {
      for(int channel = 0; channel < num_channels; channel++) {
	*tdata = edata[basis * num_channels + channel];
	tdata++;
      }
    }
  } else {
    // Copy the eigenvectors
    for (int basis=0;basis<num_bases;basis++) {
      for(int channel=0;channel<num_channels;channel++) {
	*tdata = edata[basis*num_channels+channel];
	tdata++;
      }
    }
  }

  // Free covariance matrix, eigenvectors
  cov = nrrdNuke(cov);
  cov_data = evec_data = 0;
  
  if (verbose > 10) {
    cout << "\ntransform matrix\n";
    float* tdata = (float*)(transform->data);
    for(int c = 0; c < num_bases; c++)
      {
	cout << "[";
	for(int r = 0; r < num_channels; r++)
	  {
	    cout << tdata[c*num_channels + r] << ", ";
	  }
	cout << "]\n";
      }
    cout << "\n\n";
  }
  
  if (verbose)
    cout << "Done filling the transformation matrix" << endl;
  
  // Compute our basis textures
  if ( verbose)
    cout << "Computing " << num_bases << " basis textures" << endl;
  
  Nrrd *bases = nrrdNew();
  if (nrrdAlloc(bases, nrrdTypeFloat, 3, num_bases, width, height)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: error allocating bases textures :\n%s", me, err);
    exit(0);
  }
  
  nrrdAxisInfoSet(bases, nrrdAxisInfoLabel, "basis", "width", "height");
  
  {
    float *btdata = (float*)(bases->data);
    float *data = (float*)(nin->data);
    for (int y=0; y < height; y++) {
      for (int x = 0; x < width; x++ ) {
	// Subtract the mean
	float *mdata = (float*)(mean->data);
	for(int i = 0; i < mean->axis[0].size; i++) {
	  data[i] -= mdata[i];
	}
	
	// Now do transform * data
	float *tdata = (float*)(transform->data);
	for(int c = 0; c < num_bases; c++)
	  for(int r = 0; r < num_channels; r++)
	    {
	      btdata[c] += tdata[c*num_channels+r]*data[r];
	    }
	btdata += num_bases;
	data += num_channels;
      }
    }
  }

  // Free input nrrd and mean
  nin = nrrdNuke(nin);
  mean = nrrdNuke(mean);

  if (verbose)
    cout << "Basis textures complete" << endl;
  
  // Write out the basis textures
  string basesname(string(outfilename_base) + "-basis" + string(nrrd_ext));
  if (nrrdSave(basesname.c_str(), bases, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: trouble saving to %s: %s\n", me, basesname.c_str(), err);
    exit(2);
  }

  if (verbose)
    cout << "Wrote basis textures to " << basesname << endl;

  // Free basis textures
  bases = nrrdNuke(bases);

  // Write out the transformation matrix, if necessary
  if (saveall) {
    // Permute tranformation matrix axes
    Nrrd *newTrans=nrrdNew();
    int axes[2] = {1,0};
    if (nrrdAxesPermute(newTrans, transform, axes)) {
      err = biffGetDone(NRRD);
      fprintf(stderr, "%s: error permuting the transformation matrix:\n%s", me, err);
      exit(2);
    }
    
    // Free old transform
    transform = nrrdNuke(transform);

    // Write out the new tranformation matrix
    string transname(string(outfilename_base) + "-coeff" + string(nrrd_ext));
    if (nrrdSave(transname.c_str(), newTrans, 0)) {
      err = biffGet(NRRD);
      fprintf(stderr, "%s: trouble saving to %s: %s\n", me, transname.c_str(), err);
      exit(2);
    }
    
    if (verbose)
      cout << "Wrote transform matrix to " << transname << endl;

    // Free new transform
    newTrans = nrrdNuke(newTrans);
  }
  
  return 0;
}
