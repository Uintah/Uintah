// #ifdef PEDANTIC
// #  undef PEDANTIC
// #endif
// #define PEDANTIC 1

#ifdef CHECK_VERBOSITY
#  undef CHECK_VERBOSITY
#endif
#define CHECK_VERBOSITY 1

#include <Core/Thread/Thread.h>
#include <Core/Thread/WorkQueue.h>

#include <teem/nrrd.h>

#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sci_values.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace SCIRun;

char *search_type="standard test (with partial distortion)";

WorkQueue work("Work Queue");

int num_dims=-1;
int num_vecs=10;
int num_cwords=2;
// forward declare data structures for vectors
int* ir_idx = 0;
float* vec = 0;
int* cluster_idx = 0;

// Code word stuff
float* prev_cw = 0;
float* current_cw = 0;

void assign_cw(int vector_index) {
  float min=FLT_MAX;
  int my_cluster=-1;
  for (int j=0;j<num_cwords;j++) {
    float distance=0;
    // compute the real distortion
    distance=0;
    for (int k=0;k<num_dims;k++) {
      float tmp=vec[vector_index*num_dims+k]-current_cw[j*num_dims+k];
      distance+=tmp*tmp;
      if (distance>min) {
	break;
      }
    }
    if (distance>min)
      continue;
	
    // set the cluster for the vector
    min=distance;
    my_cluster=j;
  }
	
  cluster_idx[vector_index]=my_cluster;
}


class ClusterParallel {
public:
  void cluster_it(int /*proc*/) {
    int start, end;
    while(work.nextAssignment(start, end))
      for(int vector_index = start; vector_index < end; vector_index++)
	assign_cw(vector_index);
  }
};

ClusterParallel cp;

int main(int argc, char *argv[]) {
  char *me = argv[0];
  char *err = 0;
  char *infilename=0;
  char *cw_outfilename="cw.nhdr";
  char *idx_outfilename="cbidx.nhdr";
  float thresh=1e-5;
  int max_iteration = 100;
  int seed=12;
  int np = 1;

  for (int i=1;i<argc;i++) {
    if (strcmp(argv[i], "-i")==0) {
      infilename=argv[++i];
    } else if (strcmp(argv[i], "-np")==0) {
      np=atoi(argv[++i]);
    } else if (strcmp(argv[i], "-cw_out")==0) {
      cw_outfilename=argv[++i];
    } else if (strcmp(argv[i], "-idx_out")==0) {
      idx_outfilename=argv[++i];
#if 0
    } else if (strcmp(argv[i], "-ndims")==0) {
      num_dims=atoi(argv[++i]);
    } else if (strcmp(argv[i], "-nvecs")==0) {
      num_vecs=atoi(argv[++i]);
#endif
    } else if (strcmp(argv[i], "-ncwords")==0) {
      num_cwords=atoi(argv[++i]);
    } else if (strcmp(argv[i], "-thresh")==0) {
      thresh=atof(argv[++i]);
    } else if (strcmp(argv[i], "-seed")==0) {
      seed=atoi(argv[++i]);
    } else if (strcmp(argv[i], "-niters")==0) {
      max_iteration=atoi(argv[++i]);
    } else if (strcmp(argv[i], "--search")==0) {
      cout<<"using "<<search_type<<" for codebook search"<<endl;
    } else {
      if (strcmp(argv[i], "--help")!=0)
	cerr<<"unrecognized option \""<<argv[i]<<"\""<<endl;
      cerr<<"usage:  vq [options]"<<endl;
      cerr<<"  options:"<<endl;
      cerr<<"    -i <filename>         input filename (null)"<<endl;
      cerr<<"    -np <int>             number of threads to use (1)"<<endl;
      cerr<<"    -cw_out <filename>    write codewords to file (null)"<<endl;
      cerr<<"    -idx_out <filename>   write vector indices to file (null)"<<endl;
#if 0
      cerr<<"    -ndims <int>          dimensionality of the vectors (2)"<<endl;
      cerr<<"    -nvecs <int>          number of vectors to quantize (10)"<<endl;
#endif
      cerr<<"    -ncwords <int>        number of codewords (2)"<<endl;
      cerr<<"    -thresh <float>       termination threshold (1e-5)"<<endl;
      cerr<<"    -niters <int>         maximum number of iterations (100)"<<endl;
      cerr<<"    -seed <int>           random number seed (12)"<<endl;
      cerr<<"    --search              print codebook search method"<<endl;
      cerr<<"    --help                print this message and exit"<<endl;
      exit(1);
    }
  }

  if (!infilename) {
    cerr << "Need to specify an input filename\n";
    exit(1);
  }

  // seed the RNGs
  srand48(seed);
  srand(seed);

  // Load in the vectors
  Nrrd *nvec = nrrdNew();
  if (nrrdLoad(nvec, infilename, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: Error loading nrrd:\n%s", me, err);
    exit(2);
  }
  // Convert to float if need be
  if (nvec->type != nrrdTypeFloat) {
    Nrrd *new_n = nrrdNew();
    if (nrrdConvert(new_n, nvec, nrrdTypeFloat)) {
      err = biffGet(NRRD);
      cerr << me << ": unable to convert nrrd: " << err << endl;
      biffDone(NRRD);
      exit(2);
    }
    // since the data was copied blow away the memory for the old nrrd
    nrrdNuke(nvec);
    nvec = new_n;
  }
  // Compute the dimensionality of the vectors.  Assume the last
  // dimension is the number of vectors.
  num_vecs = nvec->axis[nvec->dim-1].size;
  num_dims = 1;
  for(int i = 0; i < nvec->dim-1; i++)
    num_dims *= nvec->axis[i].size;
  cout << "num_vecs = "<<num_vecs<<", num_dims = "<<num_dims<<"\n";

  size_t num_elements = nrrdElementNumber(nvec);
  size_t num_elements_maybe = num_vecs * num_dims;
  if (num_elements_maybe != num_elements) {
    cerr << "num_vecs*num_dims ("<<num_elements_maybe<<") != num_elements ("<<num_elements<<")\n";
    exit(2);
  }
  

  if (num_cwords>num_vecs) {
    cerr<<"number of codewords ("<<num_cwords<<") is greater than the "
	<<"number of vectors ("<<num_vecs<<")"<<endl;
    exit(1);
  }
  
  // allocate memory for codewords
  prev_cw=new float[num_cwords*num_dims];
  current_cw=new float[num_cwords*num_dims];

  // allocate the memory for vector structures
  vec=(float*)nvec->data;
  cluster_idx=new int[num_vecs];
  ir_idx=new int[num_vecs];

  // randomly choose the initial codewords
  for (int i=0;i<num_vecs;i++)
    ir_idx[i] = i;
  
  for (int i=0;i<num_vecs;i++) {
    // pick an index to swap
    int r_idx = rand()%num_vecs;
    int tmp = ir_idx[i];
    ir_idx[i] = ir_idx[r_idx];
    ir_idx[r_idx] = tmp;
  }
  
  for (int i=0;i<num_cwords;i++)
    for (int j=0;j<num_dims;j++)
      current_cw[i*num_dims+j] = vec[ir_idx[i]*num_dims+j];

  // cluster the vectors around the codewords
  bool done=false;
  bool first=true;
  unsigned int iteration=0;
  float* tmp_mean=new float[num_dims];

  do {
    // output a status message
    cout<<"Beginning iteration "<<iteration<<endl;
    
    if (np > 1) {
      work.refill(num_vecs, np);

      Parallel<ClusterParallel> phelper(&cp, &ClusterParallel::cluster_it);
      Thread::parallel(phelper, np, true);
    } else {
      for (int i=0;i<num_vecs;i++)
	assign_cw(i);
    }
    
    // compute the mean vector of each cluster
    for (int i=0;i<num_cwords;i++) {
      for (int k=0;k<num_dims;k++) {
	tmp_mean[k] = 0;
      }
      
      unsigned long count=0;
      for (int j=0;j<num_vecs;j++) {
	if (cluster_idx[j]==i) {
	  count++;
	  for (int k=0;k<num_dims;k++) {
	    tmp_mean[k]+=vec[j*num_dims+k];
	  }
	}
      }

      // update codewords
      done=true;
      for (int k=0;k<num_dims;k++) {
	prev_cw[i*num_dims+k]=current_cw[i*num_dims+k];
	current_cw[i*num_dims+k]=tmp_mean[k]/(float)count;
	
	// compute the difference between iterations
	if (!first) {
	  done&=((current_cw[i*num_dims+k]-prev_cw[i*num_dims+k])<thresh);
	} else {
	  first=false;
	  done=false;
	}
      }
    }

    iteration++;
  } while (!done && iteration < max_iteration);

  // Write out codebook
  Nrrd *nout = nrrdNew();
  if (nrrdWrap(nout, current_cw, nrrdTypeFloat, 2, num_dims, num_cwords) ||
      nrrdSave(cw_outfilename, nout, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: Error wrapping or saving codebook:\n%s", me, err);
    exit(2);
  }

  // Write out the index file
  if (nrrdWrap(nout, cluster_idx, nrrdTypeInt, 1, num_vecs) ||
      nrrdSave(idx_outfilename, nout, 0)) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: Error wrapping or saving index file:\n%s", me, err);
    exit(2);
  }
  
  // delete allocated memory
  nrrdNuke(nvec);
  delete cluster_idx;
  delete ir_idx;
  delete prev_cw;
  delete current_cw;
  delete tmp_mean;
  
  return 0;
}
