#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <stdio.h>

#include <Core/Thread/Barrier.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>

#include <teem/nrrd.h>

#define MAX_NELTS (100000*256)
// #define VERBOSE_THREADS 1

using namespace std;
using namespace SCIRun;

// Declare some useful classes
class FileIO : public Runnable {
public:
  enum FileIOMode {
    Read,
    Write
  };
  
  FileIO(FileIOMode mode, int nwait, char* fname);
  
  void read(void);
  void write(void);
  void run(void);

private:
  FileIOMode mode;
  int nwait;
  int buf_idx;
  char* fname;
  FILE* file;
};

class Worker : public Runnable {
public:
  enum ComputeMode {
    CovarianceMatrix,
    PCACoefficients
  };
  
  Worker(ComputeMode mode, int id, int npeers, float* mean, float* other);
  ~Worker(void);
  
  void computeCovariance(void);
  void computeCoefficients(void);
  void run(void);

private:
  ComputeMode mode;
  int id;
  int npeers;
  int rbuf_idx;
  int wbuf_idx;
  float* vec;
  float* mean;
  float* other;
};

// Declare necessary functions
extern "C" {
  void ssyevx_(const char& jobz, const char& range, const char& uplo,
	       const int& n, float array[], const int& lda,
	       const float& vl, const float& vu,
	       const int& il, const int& iu,
	       const float& tolerance,
	       int& nevals, float eval[],
	       float evec[], const int& ldz,
	       float work[], const int& lwork, int iwork[],
	       int ifail[], int& info);
}

void usage(char* me, const char* unknown=0);
char* createFName(char* in_fname, char* ext);
void range(int id, int nprocs, size_t total, size_t* start, size_t* end);
int computeBasis(float* cov, float* basis);
inline float dot(float* v1, float* v2);
inline float mag(float* vec);
inline void normalize(float* vec);
int saveNrrd(Nrrd* nin, char* type, char* ext);

// Declare global variables
int nbases=0;
int nvectors=0;
float inv_nvectors=0.0;
int ndims=0;
float inv_ndims=0.0;
int nworkers=1;
// float in0[MAX_NELTS];
// float in1[MAX_NELTS];
// float* in_ptr[2]={in0, in1};
unsigned char in0[MAX_NELTS];
unsigned char in1[MAX_NELTS];
unsigned char* in_ptr[2]={in0, in1};
size_t nin[2]={0, 0};
float out0[MAX_NELTS];
float out1[MAX_NELTS];
float* out_ptr[2]={out0, out1};
size_t nout[2]={0, 0};
Barrier read_barrier("Read barrier");
Barrier write_barrier("Write barrier");
#ifdef VERBOSE_THREADS
Mutex cout_mutex("cout mutex");
#endif
char* out_bname="pca";
char* nrrd_ext=".nrrd";

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
    } else if (arg=="-nworkers") {
      nworkers=atoi(argv[++i]);
    } else if (arg=="-nbases") {
      nbases=atoi(argv[++i]);
    } else if (arg=="-nhdr") {
      nrrd_ext=".nhdr";
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

  if (nworkers<1) {
    cerr<<me<<":  invalid number of worker threads ("<<nworkers
	<<"):  resetting to one"<<endl;
    nworkers=1;
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
    if (ndims<=0) {
      cerr<<me<<":  invalid number of dimensions ("<<ndims<<")"<<endl;
      exit(1);
    }
    
    nvectors=vecNrrd->axis[2].size;
    
    cout<<"Found "<<nvectors<<" "<<width<<"x"<<height<<" vectors in \""
	<<in_fname<<"\""<<endl;
  } else if (vecNrrd->dim==2) {
    ndims=vecNrrd->axis[0].size;
    if (ndims<=0) {
      cerr<<me<<":  invalid number of dimensions ("<<ndims<<")"<<endl;
      exit(1);
    }
    
    nvectors=vecNrrd->axis[1].size;
    
    cout<<"Found "<<nvectors<<" "<<ndims<<"-dimensional vectors in \""
	<<in_fname<<"\""<<endl;
  } else {
    cerr<<me<<":  invalid vector nrrd:  expecting a 2- or 3-dimensional nrrd,"
	<<" but \""<<in_fname<<"\" has "<<vecNrrd->dim<<" dimensions"<<endl;
    exit(1);
  }
  
  inv_nvectors=1.0/nvectors;
  inv_ndims=1.0/ndims;
  
  cout<<"Sample-to-dimension ratio:  "<<nvectors*inv_ndims<<endl;

  // Nix vecNrrd
  vecNrrd=nrrdNix(vecNrrd);

  // Verify requested number of basis vectors
  if (nbases<1 || nbases>ndims) {
    cerr<<me<<":  invalid number of basis vectors ("<<nbases<<"):  "
	<<"resetting to "<<ndims<<endl;
    nbases=ndims;
  }

  cout<<endl;
  cout<<"-----------------------"<<endl;
  
  // Create raw vector filename
  size_t len=strlen(in_fname)-4;
  char* in_bname=new char[len];
  strncpy(in_bname, in_fname, len);
  sprintf(in_bname+len-1, "%c", '\0');
  char* v_rawfname=createFName(in_bname, ".raw");
  if (!v_rawfname) {
    cerr<<me<<":  error creating filename of raw vector data"<<endl;
    exit(1);
  }

  // XXX - causing a segfault
  // delete [] in_bname;

  // Create threads
  ThreadGroup* tgroup=new ThreadGroup("Parallel Covariance Computation");
  Thread* r_thread=new Thread(new FileIO(FileIO::Read, nworkers+1, v_rawfname),
			      "FileIO", tgroup);

  // Allocate memory for mean vector workspace
  float* m_workspace=new float[nworkers*ndims];
  if (!m_workspace) {
    cerr<<me<<":  error allocating "<<nworkers*ndims*sizeof(float)
	<<" bytes for mean vector workspace"<<endl;
    exit(1);
  }
    
  // Allocate memory for covariance matrix workspace
  float* c_workspace=new float[nworkers*ndims*ndims];
  if (!c_workspace) {
      cerr<<me<<":  error allocating "<<nworkers*ndims*ndims*sizeof(float)
	<<" bytes for covariance matrix workspace"<<endl;
    exit(1);
  }

  cout<<"Computing the mean vector and covariance matrix"<<endl;

  for (int w=0; w<nworkers; w++) {
    Runnable* runner=new Worker(Worker::CovarianceMatrix, w, nworkers,
				m_workspace+w*ndims,
				c_workspace+w*ndims*ndims);
    Thread* t=new Thread(runner, "Worker", tgroup);
  }

  // Wait for reader/worker threads to complete
  tgroup->join();
  delete tgroup;

  // Allocate memory for mean vector
  float* mean=new float[ndims];
  if (!mean) {
    cerr<<me<<":  error allocating "<<ndims*sizeof(float)
	<<" bytes for mean vector"<<endl;
    exit(1);
  }
    
  for (int d=0; d<ndims; d++)
    mean[d]=0.0;
    
  // Allocate memory for covariance matrix
  float* cov=new float[ndims*ndims];
  if (!cov) {
    cerr<<me<<":  error allocating "<<ndims*ndims*sizeof(float)
	<<" bytes for covariance matrix"<<endl;
    exit(1);
  }

  for (int d=0; d<ndims*ndims; d++)
    cov[d]=0.0;

  // Gather results of worker threads
  float* mw_ptr=m_workspace;
  float* cw_ptr=c_workspace;
  for (int w=0; w<nworkers; w++) {
    for (int d=0; d<ndims; d++)
      mean[d]+=mw_ptr[d];
    
    mw_ptr+=ndims;

    float* cov_ptr=cov;
    for (int r=0; r<ndims; r++) {
      for (int c=r; c<ndims; c++)
	cov_ptr[c]+=cw_ptr[c];

      cw_ptr+=ndims;
      cov_ptr+=ndims;
    }
  }

  delete [] m_workspace;
  delete [] c_workspace;

  // Divide by total number of vectors
  for (int d=0; d<ndims; d++)
    mean[d]*=inv_nvectors;

  // Average covariance values and subtract mean values
  float* cov_ptr=cov;
  for (int r=0; r<ndims; r++) {
    for (int c=r; c<ndims; c++)
      cov_ptr[c]=inv_nvectors*cov_ptr[c]-mean[r]*mean[c];
      
    cov_ptr+=ndims;
  }

  cout<<"Done computing the mean vector and covariance matrix"<<endl;

  // Save mean vector
  Nrrd* meanNrrd=nrrdNew();
  int E=0;
  if (!E) E|=nrrdWrap(meanNrrd, mean, nrrdTypeFloat, 1, ndims);
  nrrdAxisInfoSet(meanNrrd, nrrdAxisInfoLabel, "mean");
  if (!E) E|=saveNrrd(meanNrrd, "-mean", nrrd_ext);
  if (E)
    cerr<<me<<":  error saving mean vector"<<endl;
  
  meanNrrd=nrrdNix(meanNrrd);

  cout<<endl;

  // Allocate memory for basis vectors
  float* basis=new float[nbases*ndims];
  if (!basis) {
    cerr<<me<<":  error allocating "<<nbases*ndims*sizeof(float)
	<<" bytes for basis vectors"<<endl;
    exit(1);
  }

  // Compute basis vectors
  if (computeBasis(cov, basis)) {
    cerr<<me<<":  error computing basis vectors"<<endl;
    exit(1);
  }

  for (int b=0; b<nbases; b++)
    normalize(&basis[b]);

  // Reclaim unnecessary memory
  delete [] cov;

  // Save basis vectors
  Nrrd* basisNrrd=nrrdNew();
  E=0;
  if (!E) E|=nrrdWrap(basisNrrd, basis, nrrdTypeFloat, 2, ndims, nbases);
  nrrdAxisInfoSet(basisNrrd, nrrdAxisInfoLabel, "width*height", "basis");
  if (!E) E|=saveNrrd(basisNrrd, "-basis", nrrd_ext);
  if (E)
    cerr<<me<<":  error saving basis vectors"<<endl;
  
  basisNrrd=nrrdNix(basisNrrd);

  cout<<endl;

  // Create raw coefficient filename
  char* c_rawfname=createFName(out_bname, "-coeff.raw");
  if (!c_rawfname) {
    cerr<<me<<":  error creating raw filename for coefficient matrix data"<<endl;
    exit(1);
  }

  // Create threads
  tgroup=new ThreadGroup("Parallel Coefficient Computation");
  r_thread=new Thread(new FileIO(FileIO::Read, nworkers+1, v_rawfname),
		      "FileIO", tgroup);
  Thread* w_thread=new Thread(new FileIO(FileIO::Write, nworkers+1, c_rawfname),
			      "FileIO", tgroup);
  
  cout<<"Computing "<<nvectors<<"x"<<nbases<<" coefficient matrix"<<endl;
  
  for (int w=0; w<nworkers; w++) {
    Runnable* runner=new Worker(Worker::PCACoefficients, w, nworkers,
				mean, basis);
    Thread* t=new Thread(runner, "Worker", tgroup);
  }

  // Wait for reader/writer/worker threads to complete
  tgroup->join();
  delete tgroup;

  cout<<"Done computing the coefficient matrix"<<endl;

  // Reclaim unnecessary memory
  // XXX - this delete is causing problems...
  // delete [] v_rawfname;
  delete [] c_rawfname;
  delete [] mean;
  delete [] basis;
  
  // Write nrrd header for coefficient matrix
  if (strcmp(nrrd_ext, ".nhdr")!=0)
    cerr<<"warning:  must save coefficient matrix as a detatched nrrd"
        <<endl;
  
  char* nhdr_fname=createFName(out_bname, "-coeff.nhdr");
  if (!nhdr_fname) {
    cerr<<me<<":  error creating header filename for coefficient matrix"<<endl;
    exit(1);
  }
  
  Nrrd* coeffNrrd=nrrdNew();
  if (nrrdWrap(coeffNrrd, (void*)1, nrrdTypeFloat, 2, nbases, nvectors)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error creating header:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  nrrdAxisInfoSet(coeffNrrd, nrrdAxisInfoLabel, "basis", "vector");

  nio->skipData=1;
  if (nrrdSave(nhdr_fname, coeffNrrd, nio)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error writing header:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // XXX - causing a segfault
  // delete [] nhdr_fname;
  coeffNrrd=nrrdNix(coeffNrrd);
  nio=nrrdIoStateNix(nio);

  cout<<"Wrote data to \""<<nhdr_fname<<"\""<<endl;
  cout<<"-----------------------"<<endl;
  cout<<endl;

  return 0;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<":  unknown argument \""<<unknown<<"\""<<endl;

  cerr<<"usage:  "<<me<<" [options] -i <filename>"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -o <basename>     basename of output files (\"pca\")"<<endl;
  cerr<<"  -nworkers <int>   number of worker threads to use (1)"<<endl;
  cerr<<"  -nbases <int>     number of basis vectors to use (0)"<<endl;
  cerr<<"  -nhdr             use \".nhdr\" extension for nrrd files (false)"<<endl;
  cerr<<"  --help            print this message and exit"<<endl;
  
  if (unknown)
    exit(1);
}

char* createFName(char* bname, char* ext) {
  char* me="createFName";
  
  size_t len=strlen(bname);
  size_t ext_len=strlen(ext);
  
  char* fname=new char[len+ext_len];
  if (!fname) {
    cerr<<me<<":  error allocating "<<(len+ext_len)*sizeof(char)
	<<" bytes for filename"<<endl;
    return 0;
  }
  
  strncpy(fname, bname, len);
  sprintf(fname+len, ext);
  
  return fname;
}

void range(int id, int nprocs, size_t total, size_t* start, size_t* end) {
  size_t size=total/nprocs;
  size_t r=total%nprocs;
  *start=0;
  if (r!=0) {
    if (id<r)
      size++;
    else
      *start=r;
  }
    
  *start+=id*size;
  *end=*start+size;
}

int computeBasis(float* cov, float* basis) {
  char* me="computeBasis";
  
  // Compute eigenvalues/vectors
  cout<<"Computing the eigenvalues/vectors"<<endl;
  cout<<"Using LAPACK's expert driver (ssyevx)"<<endl;
  
  const char jobz='V';
  const char range='I';
  const char uplo='L';
  const int N=ndims;
  const int lda=N;
  const float vl=0;
  const float vu=0;
  const int il=N-nbases+1;
  const int iu=N;
  const float tolerance=0;
  int nevals;
  const int ldz=N;
  int info;

  // Allocate memory for eigenvalues
  float* eval=new float[N];
  if (!eval) {
    cerr<<me<<":  error allocating "<<N*sizeof(float)
	<<" bytes for eigenvalues"<<endl;
    return 1;
  }

  // Allocate memory for eigenvectors
  float* evec=new float[ldz*N];
  if (!evec) {
    cerr<<me<<":  error allocating "<<ldz*N*sizeof(float)
	<<" bytes for eigenvectors"<<endl;
    return 1;
  }

  // Allocate memory for floating-point work space
  const int lwork=8*N;
  float* work=new float[lwork];
  if (!work) {
    cerr<<me<<":  error allocating "<<lwork*sizeof(float)
	<<" bytes for floating-point work space"<<endl;
    return 1;
  }

  // Allocate memory for integer work space
  int* iwork=new int[5*N];
  if (!iwork) {
    cerr<<me<<":  error allocating "<<5*N*sizeof(int)
	<<" bytes for integer work space"<<endl;
    return 1;
  }

  // Allocate memory for failure codes
  int* ifail=new int[N];
  if (!ifail) {
    cerr<<me<<":  error allocating "<<N*sizeof(int)
	<<" bytes for failure codes"<<endl;
    return 1;
  }
      
  cout<<"Solving for the eigenvalues/vectors in the range ["
      <<il<<", "<<iu<<"]"<<endl;
    
  ssyevx_(jobz, range, uplo, N, cov, lda, vl, vu, il, iu,
	  tolerance, nevals, eval, evec, ldz, work, lwork,
	  iwork, ifail, info);
      
  delete [] work;
  delete [] iwork;
  delete [] ifail;

  if (info!=0) {
    cerr<<me<<":  ssyevx_ error:  ";
    if (info<0) {
      cerr<<(-info)<<"th argument has an illegal value"<<endl;
      return 1;
    } else if (info>0) {
      cerr<<info<<" eigenvalues failed to converge"<<endl;
      return 1;
    }
  }
  
  delete[] eval;
  
  cout<<"Done computing the eigenvalues/vectors"<<endl;
  cout<<endl;

  // Fill basis vectors
  cout<<"Filling the basis vectors"<<endl;

  float* b_ptr=basis;
  float* e_ptr=evec;
  for (int b=0; b<nbases; b++) {
    for (int d=0; d<ndims; d++) {
      *b_ptr=*e_ptr;
      b_ptr++;
      e_ptr++;
    }
  }

  delete [] evec;
  
  cout<<"Done filling the basis vectors"<<endl;

  return 0;
}

float dot(float* v1, float* v2) {
  float result=0;
  for (int d=0; d<ndims; d++)
    result+=v1[d]*v2[d];

  return result;
}
 
float mag(float* vec) {
  return sqrt(dot(vec, vec));
}

void normalize(float* vec) {
  float inv_mag=1.0/mag(vec);
  for (int d=0; d<ndims; d++)
    vec[d]*=inv_mag;
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

  // XXX - causing a segfault
  // delete [] fname;
  
  return 0;
}

FileIO::FileIO(FileIOMode mode, int nwait, char* fname) :
  mode(mode), nwait(nwait), file(file), buf_idx(0), Runnable(true) {
  char* me="FileIO::FileIO";
  
  if (mode!=FileIO::Read && mode!=FileIO::Write) {
    cerr<<me<<": invalid I/O mode ("<<mode<<"):  expecting "
	<<"FileIO::Read ("<<FileIO::Read<<") or FileIO::Write ("
	<<FileIO::Write<<")"<<endl;
    exit(1);
  }

  if (mode==FileIO::Read)
    file=fopen(fname, "r");
  else
    file=fopen(fname, "w");
  
  if (!file) {
    cerr<<me<<":  error opening file \""<<fname<<"\""<<endl;
    exit(1);
  }
}

void FileIO::read(void) {
  fseek(file, 0, SEEK_SET);
  for (;;) {
    // Switch buffers
    buf_idx=!buf_idx;

    // Read data from raw file
    // nin[buf_idx]=fread(in_ptr[buf_idx], sizeof(float),
    nin[buf_idx]=fread(in_ptr[buf_idx], sizeof(unsigned char),
		       MAX_NELTS, file);
    nin[buf_idx]*=inv_ndims;
#if VERBOSE_THREADS
    cout_mutex.lock();
    cout<<"FileIO::read - Read "<<nin[buf_idx]<<" vectors into buffer "
	<<buf_idx<<endl;
    cout_mutex.unlock();
#endif

    read_barrier.wait(nwait);

    if (nin[buf_idx]==0) {
#if VERBOSE_THREADS
      cout_mutex.lock();
      cout<<"FileIO::read - Complete"<<endl;
      cout_mutex.unlock();
#endif
          
      break;
    }
  }
}

void FileIO::write(void) {
  char* me="FileIO::write";

  fseek(file, 0, SEEK_SET);
  for (;;) {
    write_barrier.wait(nwait);
    
    // Switch buffers
    buf_idx=!buf_idx;

    unsigned long ocnt=nout[buf_idx];
    if (ocnt<=0) {
#if VERBOSE_THREADS
      cout_mutex.lock();
      cout<<"FileIO::write - Complete"<<endl;
      cout_mutex.unlock();
#endif
      
      break;
    }
    
    // Write data to raw file
    size_t nelts=ocnt*nbases;
    size_t nwritten=fwrite(out_ptr[buf_idx], sizeof(float), nelts, file);
    if (nwritten!=nelts) {
      cerr<<me<<":  error writing data"<<endl;
      exit(1);
    }
      
#if VERBOSE_THREADS
    cout_mutex.lock();
    cout<<"FileIO::write - Wrote "<<nwritten/nbases<<" vectors from buffer "
	<<buf_idx<<endl;
    cout_mutex.unlock();
#endif

    float* outvec_ptr=out_ptr[buf_idx];
    for (size_t e=0; e<MAX_NELTS; e++)
      outvec_ptr[e]=0.0;
  }
}

void FileIO::run(void) {
  if (mode==FileIO::Read)
    read();
  else
    write();

  fclose(file);
}

Worker::Worker(ComputeMode mode, int id, int npeers, float* mean, float* other) :
  mode(mode), id(id), npeers(npeers), mean(mean), other(other),
  rbuf_idx(0), wbuf_idx(0), Runnable(true) {
  char* me="Worker::Worker";

  if (mode!=Worker::CovarianceMatrix && mode!=Worker::PCACoefficients) {
    cerr<<me<<":  invalid computation mode ("<<mode<<"):  expecting "
	<<"Worker::CovarianceMatrix ("<<Worker::CovarianceMatrix
	<<") or Worker::PCACoefficients ("
	<<Worker::PCACoefficients<<")"<<endl;
    exit(1);
  }
  
  // Allocate memory for working vector
  vec=new float[ndims];
  if (!vec) {
    cerr<<me<<":  error allocating "<<ndims*sizeof(float)
	<<" bytes for working vector"<<endl;
    exit(1);
  }

  if (mode==Worker::CovarianceMatrix) {
    // Initialize mean vector workspace
    for (int d=0; d<ndims; d++)
      mean[d]=0.0;
    
    // Initialize covariance matrix workspace
    for (int d=0; d<ndims*ndims; d++)
      other[d]=0.0;
  }
}

Worker::~Worker(void) {
  delete [] vec;
}

void Worker::computeCovariance(void) {
  for (;;) {
    read_barrier.wait(npeers+1);

    // Switch buffers
    rbuf_idx=!rbuf_idx;
    
    unsigned long icnt=nin[rbuf_idx];
    if (icnt<=0) {
#if VERBOSE_THREADS
      cout_mutex.lock();
      cout<<"Worker::computeCovariance - Worker["<<id<<"] complete"<<endl;
      cout_mutex.unlock();
#endif
      
      break;
    }
    
    // Determine range of vectors
    size_t start=0;
    size_t end=0;
    range(id, npeers, icnt, &start, &end);

#if VERBOSE_THREADS
    cout_mutex.lock();
    cout<<"Worker::computeCovariance - Worker["<<id
	<<"] operating on buffer "<<rbuf_idx<<" ["<<start
	<<":"<<end<<")"<<endl;
    cout_mutex.unlock();
#endif

    // float* invec_ptr=in_ptr[rbuf_idx];
    unsigned char* invec_ptr=in_ptr[rbuf_idx];
    invec_ptr+=start*ndims;
    for (size_t v=start; v<end; v++) {
      // Cast input vector to float
      for (int d=0; d<ndims; d++)
        // vec[d]=invec_ptr[d];
        vec[d]=(float)invec_ptr[d];

      // Add vector to mean
      for (int d=0; d<ndims; d++)
	mean[d]+=vec[d];

      // Compute covariance for vector
      float* cov_ptr=other;
      for(int r=0; r<ndims; r++) {
	for(int c=r; c<ndims; c++)
	  cov_ptr[c]+=vec[r]*vec[c]; 
      
	cov_ptr+=ndims;
      }
      // Increment vector pointer
      invec_ptr+=ndims;
    }
  }
}

void Worker::computeCoefficients(void) {
  for (;;) {
    read_barrier.wait(npeers+1);

    // Switch buffers
    rbuf_idx=!rbuf_idx;
    wbuf_idx=!wbuf_idx;

    unsigned long icnt=nin[rbuf_idx];
    if (id==0)
      nout[wbuf_idx]=icnt;
    if (icnt<=0) {
      write_barrier.wait(npeers+1);

#if VERBOSE_THREADS
      cout_mutex.lock();
      cout<<"Worker::computeCoefficients - Worker["<<id<<"] complete"<<endl;
      cout_mutex.unlock();
#endif
    
      break;
    }

    // Determine range of vectors
    size_t start=0;
    size_t end=0;
    range(id, npeers, icnt, &start, &end);

#if VERBOSE_THREADS
    cout_mutex.lock();
    cout<<"Worker::computeCovariance - Worker["<<id
	<<"] operating on buffer "<<rbuf_idx<<" ["<<start
	<<":"<<end<<")"<<endl;
    cout_mutex.unlock();
#endif

    // float* invec_ptr=in_ptr[rbuf_idx];
    unsigned char* invec_ptr=in_ptr[rbuf_idx];
    float* outvec_ptr=out_ptr[wbuf_idx];
    invec_ptr+=start*ndims;
    outvec_ptr+=start*nbases;
    for (size_t v=start; v<end; v++) {
      // Subtract mean vector
      for (int d=0; d<ndims; d++)
        // vec[d]=invec_ptr[d]-mean[d];
	vec[d]=(float)invec_ptr[d]-mean[d];
      
      // Multiply input vector by basis vector
      float* b_ptr=other;
      for (int b=0; b<nbases; b++) {
	for (int d=0; d<ndims; d++)
	  outvec_ptr[b]+=b_ptr[d]*vec[d];
      
	b_ptr+=ndims;
      }

      // Increment pointers
      invec_ptr+=ndims;
      outvec_ptr+=nbases;
    }

    write_barrier.wait(npeers+1);
  }
}

void Worker::run(void) {
  if (mode==Worker::CovarianceMatrix)
    computeCovariance();
  else
    computeCoefficients();
}
