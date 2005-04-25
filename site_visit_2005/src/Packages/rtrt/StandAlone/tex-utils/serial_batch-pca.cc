#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <stdio.h>
#include <teem/nrrd.h>

using namespace std;

// Declare necessary functions
extern "C" {
  void dsyevx_(const char& jobz, const char& range, const char& uplo,
	       const int& n, double array[], const int& lda,
	       const double& vl, const double& vu,
	       const int& il, const int& iu,
	       const double& tolerance,
	       int& nevals, double eval[],
	       double evec[], const int& ldz,
	       double work[], const int& lwork, int iwork[],
	       int ifail[], int& info);
}

void usage(char* me, const char* unknown=0);
int computeCovariance(FILE* v_rawfile, unsigned char* in_vec,
		      double* vec, double* mean, double* cov);
double* computeEigenvectors(double* cov);
void fillBasisVectors(double* evec, double* basis);
inline double dot(double* v1, double* v2);
inline double mag(double* vec);
inline void normalize(double* vec);
int computeCoefficients(FILE* v_rawfile, unsigned char* in_vec, double* vec,
			double* mean, double* basis, char* c_rawfname);
int saveNrrd(Nrrd* nin, char* type, char* ext);

// Declare global variables
int nbases=0;
int nvectors=0;
double inv_nvectors=0.0;
int ndims=0;
int nt=1;
char* out_bname="pca";
char* nrrd_ext=".nhdr";

int main(int argc, char* argv[]) {
  char* me=argv[0];
  char* err;
  char* in_fname=0;

  if (argc<3) {
    usage(me);
    exit(1);
  }

  for (int i=1; i<argc; i++) {
    string arg(argv[i]);
    if (arg=="-i") {
      in_fname=argv[++i];
    } else if (arg=="-o") {
      out_bname=argv[++i];
    } else if (arg=="-nbases") {
      nbases=atoi(argv[++i]);
    } else if (arg=="-nrrd") {
      nrrd_ext=".nrrd";
    } else if (arg=="--help") {
      usage(me);
      exit(0);
    } else {
      usage(me, arg.c_str());
    }
  }

  // Verify arguments
  if (!in_fname) {
    cerr<<me<<":  input filename not specified"<<endl;
    usage(me);
    exit(1);
  }

  // Read header of input vectors
  Nrrd* vecNrrd=nrrdNew();
  NrrdIoState* nio=nrrdIoStateNew();
  nio->skipData=1;
  if (nrrdLoad(vecNrrd, in_fname, nio)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error reading header:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Initialize useful variables
  int width=0;
  int height=0;
  if (vecNrrd->dim==3 ) {
    width=vecNrrd->axis[0].size;
    height=vecNrrd->axis[1].size;
    ndims=width*height;
    nvectors=vecNrrd->axis[2].size;
    
    cout<<"Found "<<nvectors<<" "<<width<<"x"<<height<<" vectors in \""
	<<in_fname<<"\""<<endl;
  } else if (vecNrrd->dim==2) {
    ndims=vecNrrd->axis[0].size;
    nvectors=vecNrrd->axis[1].size;
    
    cout<<"Found "<<nvectors<<" "<<ndims<<"-dimensional vectors in \""
	<<in_fname<<"\""<<endl;
  } else {
    cerr<<me<<":  invalid vector nrrd:  expecting a 2- or 3-dimensional nrrd,"
	<<" but \""<<in_fname<<"\" has "<<vecNrrd->dim<<" dimensions"<<endl;
    exit(1);
  }
  
  inv_nvectors=1.0/nvectors;
  
  cout<<"Sample-to-dimension ratio:  "<<nvectors/(double)ndims<<endl;

  // Nix vecNrrd and nio
  vecNrrd=nrrdNix(vecNrrd);
  nio=nrrdIoStateNix(nio);

  // Verify requested number of basis vectors
  if (nbases<1 || nbases>ndims) {
    cerr<<me<<":  invalid number of basis vectors ("<<nbases<<"):  "
	<<"resetting to "<<ndims<<endl;
    nbases=ndims;
  }

  cout<<endl;
  cout<<"-----------------------"<<endl;
  
  // Open raw vector file
  size_t len=strlen(in_fname);
  char* v_rawfname=new char[len-1];
  if (!v_rawfname) {
    cerr<<me<<":  error allocating "<<(len-1)*sizeof(char)
	<<" bytes for raw vector filename"<<endl;
    exit(1);
  }
  
  strncpy(v_rawfname, in_fname, len-4);
  sprintf(v_rawfname+(len-4), "raw");
  FILE* v_rawfile=fopen(v_rawfname, "r");
  if (!v_rawfile) {
    cerr<<me<<":  error opening raw vector file \""<<v_rawfname<<"\""<<endl;
    exit(1);
  }

  delete [] v_rawfname;

  // Allocate memory for input, working, and mean vectors
  unsigned char* in_vec=new unsigned char[ndims];
  if (!in_vec) {
    cerr<<me<<":  error allocating "<<ndims*sizeof(unsigned char)
	<<" for input vector"<<endl;
    exit(1);
  }
  
  double* vec=new double[ndims];
  if (!vec) {
    cerr<<me<<":  error allocating "<<ndims*sizeof(double)
	<<" bytes for vector"<<endl;
    exit(1);
  }

  double* mean=new double[ndims];
  if (!mean) {
    cerr<<me<<":  error allocating "<<ndims*sizeof(double)<<
      " bytes for mean vector"<<endl;
    exit(1);
  }
  
  // Allocate memory for covariance matrix
  double* cov=new double[ndims*ndims];
  if (!cov) {
    cerr<<me<<":  error allocating "<<ndims*ndims*sizeof(double)
	<<" bytes for covariance matrix"<<endl;
    exit(1);
  }

  if (computeCovariance(v_rawfile, in_vec, vec, mean, cov)) {
    cerr<<me<<":  error computing mean vector or covariance matrix"<<endl;
    exit(1);
  }

  // Save mean vector
  Nrrd* meanNrrd=nrrdNew();
  int E=0;
  if (!E) E|=nrrdWrap(meanNrrd, mean, nrrdTypeDouble, 1, ndims);
  if (!E) E|=saveNrrd(meanNrrd, "-mean", nrrd_ext);
  if (E)
    cerr<<me<<":  error saving mean vector"<<endl;
  
  meanNrrd=nrrdNix(meanNrrd);

  cout<<endl;
  
  // Compute eigenvectors
  double* evec=computeEigenvectors(cov);
  if (!evec) {
    cerr<<me<<":  error computing eigenvalues/vectors"<<endl;
    exit(1);
  }

  cout<<endl;
  
  // Allocate memory for basis vectors
  double* basis=new double[nbases*ndims];
  if (!basis) {
    cerr<<me<<":  error allocating "<<nbases*ndims*sizeof(double)
	<<" bytes for basis vectors"<<endl;
    exit(1);
  }

  // Fill basis vectors
  fillBasisVectors(evec, basis);

  // Normalize basis vectors
  for (int b=0; b<nbases; b++)
    normalize(&basis[b]);

  // Save basis vectors
  Nrrd* basisNrrd=nrrdNew();
  E=0;
  if (!E) E|=nrrdWrap(basisNrrd, basis, nrrdTypeDouble, 2, ndims, nbases);
  if (E)
    cerr<<me<<":  error wrapping basis vectors"<<endl;

  nrrdAxisInfoSet(basisNrrd, nrrdAxisInfoLabel, "width*height", "basis");
  
  if (!E) E|=saveNrrd(basisNrrd, "-basis", nrrd_ext);
  if (E)
    cerr<<me<<":  error saving basis vectors"<<endl;
  
  basisNrrd=nrrdNix(basisNrrd);

  // Clean up unnecessary memory
  delete [] cov;
  delete [] evec;
  
  cout<<endl;

  // Create raw coefficients filename
  len=strlen(out_bname);
  char* c_rawfname=new char[len+10];
  if (!c_rawfname) {
    cerr<<me<<":  error allocating "<<(len+10)*sizeof(char)
	<<" bytes for raw coefficients filename"<<endl;
    exit(1);
  }
  
  strncpy(c_rawfname, out_bname, len);
  sprintf(c_rawfname+len, "-coeff.raw");

  if (computeCoefficients(v_rawfile, in_vec, vec, mean, basis, c_rawfname)) {
    cerr<<me<<":  error computing coefficient matrix"<<endl;
    exit(1);
  }

  // Write nrrd header for coefficient matrix
  if (strcmp(nrrd_ext, ".nhdr")!=0)
    cerr<<me<<":  warning:  must save coefficient matrix as a detatched nrrd"<<endl;

  // Create coefficients .nhdr filename
  len=strlen(c_rawfname);
  char* nhdr_fname=new char[len+1];
  if (!nhdr_fname) {
    cerr<<me<<":  error allocating "<<(len+1)*sizeof(char)
	<<" bytes for nrrd header filename"<<endl;
    return 1;
  }
  
  strncpy(nhdr_fname, c_rawfname, len-3);
  sprintf(nhdr_fname+(len-3), "nhdr");

  // Write nrrd header
  Nrrd* coeffNrrd=nrrdNew();
  nio=nrrdIoStateNew();
  nio->skipData=1;
  
  if (nrrdWrap(coeffNrrd, (void*)1, nrrdTypeDouble, 2, nbases, nvectors)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error creating header:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }
  
  nrrdAxisInfoSet(coeffNrrd, nrrdAxisInfoLabel, "basis", "vector");

  if (nrrdSave(nhdr_fname, coeffNrrd, nio)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error writing header:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }
  
  cout<<"-----------------------"<<endl;
  cout<<endl;

  // Clean up remaining memory
  delete [] in_vec;
  delete [] mean;
  delete [] basis;
  delete [] vec;
  delete [] c_rawfname;
  coeffNrrd=nrrdNix(coeffNrrd);
  nio=nrrdIoStateNix(nio);
  delete [] nhdr_fname;
  

  return 0;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<":  unknown argument \""<<unknown<<"\""<<endl;

  cerr<<"usage:  "<<me<<" [options] -i <filename>"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -o <basename>   basename of output files (\"pca\")"<<endl;
  cerr<<"  -nbases <int>   number of basis vectors to use (0)"<<endl;
  cerr<<"  -nrrd           use \".nrrd\" extension for nrrd files (false)"<<endl;
  cerr<<"  --help          print this message and exit"<<endl;
  
  if (unknown)
    exit(1);
}

int computeCovariance(FILE* v_rawfile, unsigned char* in_vec,
		      double* vec, double* mean, double* cov) {
  char* me="computeCovariance";
  
  // Compute mean vector and covariance matrix
  cout<<"Computing the mean vector and covariance matrix"<<endl;

  for (int d=0; d<ndims; d++)
    mean[d]=0.0;

  for (int d=0; d<ndims*ndims; d++)
    cov[d]=0.0;
  
  fseek(v_rawfile, 0, SEEK_SET);
  for (int v=0; v<nvectors; v++) {
    // Read vector from raw file
    size_t nread=fread(in_vec, sizeof(unsigned char), ndims, v_rawfile);
    if (nread<ndims) {
      cerr<<me<<":  error reading vector["<<v<<"]:  expected "
	  <<ndims<<" bytes but only read "<<nread<<" bytes"
	  <<endl;
      return 1;
    }

    // Cast input vector to double
    for (int d=0; d<ndims; d++)
      vec[d]=(double)in_vec[d];

    // Add vector to mean
    for (int d=0; d<ndims; d++)
      mean[d]+=vec[d];

    // Compute covariance for vector
    double* cov_ptr=cov;
    for(int r=0; r<ndims; r++) {
      for(int c=r; c<ndims; c++)
	cov_ptr[c]+=vec[r]*vec[c];
      
      cov_ptr+=ndims;
    }
  }

  // Divide by number of vectors
  for (int d=0; d<ndims; d++)
    mean[d]*=inv_nvectors;

  // Average covariance values and subtract mean values
  double* cov_ptr=cov;
  for(int r=0; r<ndims; r++) {
    for(int c=r; c<ndims; c++)
      cov_ptr[c]=inv_nvectors*cov_ptr[c]-mean[r]*mean[c];

    cov_ptr+=ndims;
  }
  
  cout<<"Done computing the mean vector and covariance matrix"<<endl;

  return 0;
}

double* computeEigenvectors(double* cov) {
  char* me="computeEigenvectors";
  
  // Compute eigenvalues/vectors
  cout<<"Computing the eigenvalues/vectors"<<endl;
  cout<<"Using LAPACK's expert driver (dsyevx)"<<endl;
  
  const char jobz='V';
  const char range='I';
  const char uplo='L';
  const int N=ndims;
  const int lda=N;
  const double vl=0;
  const double vu=0;
  const int il=N-nbases+1;
  const int iu=N;
  const double tolerance=0;
  int nevals;
  const int ldz=N;
  int info;

  // Allocate memory for eigenvalues
  double* eval=new double[N];
  if (!eval) {
    cerr<<me<<":  error allocating "<<N*sizeof(double)
	<<" bytes for eigenvalues"<<endl;
    return 0;
  }

  // Allocate memory for eigenvectors
  double* evec=new double[ldz*N];
  if (!evec) {
    cerr<<me<<":  error allocating "<<ldz*N*sizeof(double)
	<<" bytes for eigenvectors"<<endl;
    return 0;
  }

  // Allocate memory for doubleing-point work space
  const int lwork=8*N;
  double* work=new double[lwork];
  if (!work) {
    cerr<<me<<":  error allocating "<<lwork*sizeof(double)
	<<" bytes for doubleing-point work space"<<endl;
    return 0;
  }

  // Allocate memory for integer work space
  int* iwork=new int[5*N];
  if (!iwork) {
    cerr<<me<<":  error allocating "<<5*N*sizeof(int)
	<<" bytes for integer work space"<<endl;
    return 0;
  }

  // Allocate memory for failure codes
  int* ifail=new int[N];
  if (!ifail) {
    cerr<<me<<":  error allocating "<<N*sizeof(int)
	<<" bytes for failure codes"<<endl;
    return 0;
  }
      
  cout<<"Solving for the eigenvalues/vectors in the range ["
      <<il<<", "<<iu<<"]"<<endl;
    
  dsyevx_(jobz, range, uplo, N, cov, lda, vl, vu, il, iu,
	  tolerance, nevals, eval, evec, ldz, work, lwork,
	  iwork, ifail, info);
      
  delete [] work;
  delete [] iwork;
  delete [] ifail;

  if (info!=0) {
    cerr<<me<<":  dsyevx_ error:  ";
    if (info<0) {
      cerr<<(-info)<<"th argument has an illegal value"<<endl;
      return 0;
    } else if (info>0) {
      cerr<<info<<" eigenvalues failed to converge"<<endl;
      return 0;
    }
  }
  
  double recovered=0.0;
  for (int b=0; b<nbases; b++)
    recovered+=eval[b];
  
  cout <<"Recovered "<<recovered<<" units of the total "
       <<"variance with "<<nbases<<" basis vectors"<<endl;
  
  delete[] eval;

  cout<<"Done computing the eigenvalues/vectors"<<endl;

  return evec;
}

void fillBasisVectors(double* evec, double* basis) {
  // Fill basis vectors
  cout<<"Filling the basis vectors"<<endl;

  double* b_ptr=basis;
  double* e_ptr=evec;
  for (int b=0; b<nbases; b++) {
    for (int d=0; d<ndims; d++) {
      *b_ptr=*e_ptr;
      b_ptr++;
      e_ptr++;
    }
  }

  cout<<"Done filling the basis vectors"<<endl;
}

double dot(double* v1, double* v2) {
  double result=0;
  for (int d=0; d<ndims; d++)
    result+=v1[d]*v2[d];

  return result;
}
 
double mag(double* vec) {
  return sqrt(dot(vec, vec));
}

void normalize(double* vec) {
  double inv_mag=1.0/mag(vec);
  for (int d=0; d<ndims; d++)
    vec[d]*=inv_mag;
}

int computeCoefficients(FILE* v_rawfile, unsigned char* in_vec, double* vec,
			double* mean, double* basis, char* c_rawfname) {
  char* me="computeCoefficients";
  
  // Allocate memory for coefficients
  double* coeff=new double[nbases];
  if (!coeff) {
    cerr<<me<<":  error allocating "<<nbases*sizeof(double)
	<<" bytes for coefficients"<<endl;
    return 1;
  }

  // Open raw coefficient file
  FILE* c_rawfile=fopen(c_rawfname, "w");
  if (!c_rawfile) {
    cerr<<me<<":  error opening raw coefficient file \""
	<<c_rawfname<<"\""<<endl;
    return 1;
  }

  // Compute coefficient matrix
  cout<<"Computing "<<nvectors<<"x"<<nbases<<" coefficient matrix"<<endl;

  fseek(v_rawfile, 0, SEEK_SET);
  for (int v=0; v<nvectors; v++) {
    // Reset coefficients
    for (int b=0; b<nbases; b++)
      coeff[b]=0.0;

    // Read input vector from raw file
    size_t nread=fread(in_vec, sizeof(unsigned char), ndims, v_rawfile);
    if (nread<ndims) {
      cerr<<me<<":  error reading vector["<<v<<"]:  expected "
	  <<ndims<<" bytes but only read "<<nread<<" bytes"
	  <<endl;
      return 1;
    }

    // Subtract mean vector
    for (int d=0; d<ndims; d++)
      vec[d]=(double)in_vec[d]-mean[d];

    // Multiply input vector by basis vector
    double* b_ptr=basis;
    for (int b=0; b<nbases; b++) {
      for (int d=0; d<ndims; d++)
	coeff[b]+=b_ptr[d]*vec[d];
      
      b_ptr+=ndims;
    }

    // Write coefficients
    size_t nwritten=fwrite(coeff, sizeof(double), nbases, c_rawfile);
    if (nwritten!=nbases) {
      cerr<<me<<":  error writing coefficients for vector["<<v<<"]"<<endl;
      return 1;
    }
  }

  // Close raw coefficients file
  fclose(c_rawfile);
  
  delete [] coeff;
  
  cout<<"Done computing the coefficient matrix"<<endl;

  return 0;
}

int saveNrrd(Nrrd* nin, char* type, char* ext) {
  char* me="saveNrrd";
  char* err;
  
  size_t outbname_len=strlen(out_bname);
  size_t type_len=strlen(type);
  size_t ext_len=strlen(ext);
  size_t len=outbname_len+type_len+ext_len;
  
  char* fname=new char[len];
  if (!fname) {
    cerr<<me<<":  error allocating "<<len*sizeof(char)
	<<" bytes for output filename"<<endl;
    return 1;
  }
  
  sprintf(fname, "%s%s%s", out_bname, type, ext);
  if (nrrdSave(fname, nin, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error saving to "<<fname<<":  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    return 1;
  }

  cout<<"Wrote data to \""<<fname<<"\""<<endl;

  delete [] fname;
  
  return 0;
}
