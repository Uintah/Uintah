#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <Core/Thread/Barrier.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/WorkQueue.h>
#include <teem/nrrd.h>

using namespace std;
using namespace SCIRun;

// Declare some useful classes
class NearestNeighbor {
public:
  int nearest;
  float distance;
  bool update;

  NearestNeighbor(void);

  NearestNeighbor& operator=(const NearestNeighbor& nn);
};

class Cluster {
public:
  static int ndims;
  float* v;
  int nmap;

  Cluster(void);
  Cluster(float* in_v);

  Cluster& operator=(const Cluster c);
  void vec(float* in_v);
};

class ParallelVQ {
public:
  ParallelVQ(void);
  ~ParallelVQ(void);
  
  void initNearestNeighbors(int proc);
  void initMerge(int i1, int i2);
  void mergeClusters(int proc);

private:
  int idx1;
  int idx2;
  int last;
  Barrier* barrier;

  void range(int rank, int total, int* start, int* end);
};

// Declare necessary functions
void printUsage(char* me, const char* unknown=0);
Nrrd* convertNrrdToFloat(Nrrd* nin);
int saveNrrd(Nrrd* nin, char* type, char* ext);
NearestNeighbor findNearestNeighbor(int idx);
int findMinDistance(void);
float computeDistortion(Cluster c1, Cluster c2, float min);
float* computeCentroid(Cluster c1, Cluster c2);

// Declare global variables
int Cluster::ndims=0;
float* vec=0;
int* vc_map=0;
NearestNeighbor* nnt=0;
Cluster* cluster=0;
int cb_size=0;
int nvecs=0;
char* outbasename=0;
int nt=1;
char* nrrd_ext=".nhdr";
int verbose=0;

int main(int argc, char* argv[]) {
  char* me=argv[0];
  char* err=0;
  char* infilename=0;
  int ncwords=0;
  bool stats=false;
  
  for (int i=1; i<argc; i++) {
    string arg(argv[i]);
    if (arg=="-i") {
      infilename=argv[++i];
    } else if (arg=="-o") {
      outbasename=argv[++i];
    } else if (arg=="-ncwords") {
      ncwords=atoi(argv[++i]);
    } else if (arg=="-nt") {
      nt=atoi(argv[++i]);
    } else if (arg=="-stats") {
      stats=true;
    } else if (arg=="-nrrd") {
      nrrd_ext=".nrrd";
    } else if (arg=="-v") {
      verbose=atoi(argv[++i]);
    } else if (arg=="--help") {
      printUsage(me);
      exit(0);
    } else {
      printUsage(me, argv[i]);
    }
  }

  // Validate arguments
  if (!infilename) {
    cerr<<me<<":  no input filename specified"<<endl;
    printUsage(me);
    exit(1);
  }

  if (!outbasename) {
    cerr<<me<<":  no output basename specified"<<endl;
    printUsage(me);
    exit(1);
  }

  if (ncwords<=0) {
    cerr<<me<<":  invalid number of target codewords ("<<ncwords<<")"<<endl;
    exit(1);
  }
  
  // Load input vectors
  Nrrd *vec=nrrdNew();
  if (nrrdLoad(vec, infilename, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error loading input vectors: "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Determine dimensionality of input vectors
  nvecs=vec->axis[vec->dim-1].size;
  Cluster::ndims=1;
  for (int i=0; i<vec->dim-1; i++)
    Cluster::ndims*=vec->axis[i].size;

  if (verbose)
    cout<<"Loaded "<<nvecs<<" "<<Cluster::ndims<<"-dimensional vectors from "
	<<infilename<<endl;

  // Sanity check
  size_t nelts_calc=nvecs*Cluster::ndims;
  size_t nelts=nrrdElementNumber(vec);
  if (nelts_calc!=nelts) {
    cerr<<"Number of calculated elements ("<<nelts_calc<<") is not equal"
	<<" to number of reported elements ("<<nelts<<")"<<endl;
    exit(1);
  }

  if (ncwords>nvecs) {
    cerr<<"Number of desired codewords ("<<ncwords<<") is greater than the "
	<<"number of input vectors ("<<nvecs<<")"<<endl;
    exit(1);
  }

  // Convert input vectors to nrrdTypeFloat, if necessary
  vec=convertNrrdToFloat(vec);
  if (!vec) {
    cerr<<me<<":  error converting input vectors to nrrdTypeFloat"<<endl;
    exit(1);
  }
  
  // Allocate necessary memory
  vc_map=new int[nvecs];
  if (!vc_map) {
    cerr<<me<<":  error allocating memory for texture indices"<<endl;
    exit(1);
  }
  
  nnt=new NearestNeighbor[nvecs];
  if (!nnt) {
    cerr<<me<<":  error allocating memory for nearest neighbor table"<<endl;
    exit(1);
  }

  cluster=new Cluster[nvecs];
  if (!cluster) {
    cerr<<me<<":  error allocating memory for codebook"<<endl;
    exit(1);
  }

  // Begin timer
  double stime=Time::currentSeconds();
  
  // Initialize codebook and vc_map
  if (verbose)
    cout<<"Initializing codebook and texture indices"<<endl;
  
  float* vec_data=(float*)(vec->data);
  for (int i=0; i<nvecs; i++) {
    cluster[i]=Cluster(vec_data);
    vc_map[i]=i;
    
    vec_data+=Cluster::ndims;
  }
  cb_size=nvecs;

  if (verbose>4) {
    cout<<endl;
    cout<<"------------------------------------"<<endl;
  }
  
  // Compute nearest neighbors
  if (verbose)
    cout<<"Finding nearest neighbors"<<endl;

  ParallelVQ pvq;
  if (nt>1) {
    Parallel<ParallelVQ> nearest(&pvq, &ParallelVQ::initNearestNeighbors);
    Thread::parallel(nearest, nt, true);
  } else {
    pvq.initNearestNeighbors(0);
  }

  if (verbose>4) {
    cout<<"Completed "<<nvecs<<" of "<<nvecs<<endl;
    cout<<"------------------------------------"<<endl;
  }
  
  // Merge clusters
  if (verbose)
    cout<<"Merging clusters"<<endl;

  if (verbose>4)
    cout<<cb_size-ncwords<<" clusters remain"<<endl;
  
  while(cb_size>ncwords) {
    int idx1=findMinDistance();
    if (idx1<0) {
      cerr<<me<<":  error finding cluster with minimum distance"<<endl;
      exit(1);
    }

    pvq.initMerge(idx1, nnt[idx1].nearest);
    if (nt>1) {
      Parallel<ParallelVQ> merge(&pvq, &ParallelVQ::mergeClusters);
      Thread::parallel(merge, nt, true);
    } else {
      pvq.mergeClusters(0);
    }
    
    if (verbose>4) {
      int diff=cb_size-ncwords;
      if (diff%100==0 && diff!=0)
	cout<<diff<<" clusters remain"<<endl;
    }
  }

  if (verbose>4) {
    cout<<"Merge complete"<<endl;
    cout<<"------------------------------------"<<endl;
    cout<<endl;
  }

  // Stop timer
  double etime=Time::currentSeconds()-stime;
  
  // Report statistics
  if (stats) {
    size_t len=strlen(outbasename);
    char* sfilename=new char[len+6];
    sprintf(sfilename, "%s.stats", outbasename);
    ofstream sfile(sfilename);
    if (!sfile.is_open())
      cerr<<me<<":  error opening "<<sfilename<<" for writing"<<endl;
    
    float cnt=0;
    for (int i=0; i<ncwords; i++) {
      if (sfile) {
	sfile<<"cluster["<<i<<"] contains "<<cluster[i].nmap
	     <<" vectors"<<endl;
      }
      
      cnt+=cluster[i].nmap;
    }
    
    if (sfile) {
      sfile<<endl;
      sfile<<"Quantized "<<cnt<<" vectors to "<<ncwords<<" codewords in "
	   <<etime<<" seconds ("<<nvecs/etime<<" vectors/second)"<<endl;
      sfile<<endl;
    }
    
    cout<<"Quantized "<<cnt<<" vectors to "<<ncwords<<" codewords in "
	<<etime<<" seconds ("<<nvecs/etime<<" vectors/second)"<<endl;
    cout<<endl;
    
    if (verbose && sfile)
      cout<<"Wrote detailed statistics to "<<sfilename<<endl;
  }
    
  // Create codebook nrrd
  Nrrd* cb=nrrdNew();
  if (nrrdAlloc(cb, nrrdTypeFloat, 2, Cluster::ndims, ncwords)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error allocating codebook nrrd:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  float* cb_data=(float*)(cb->data);
  for (int i=0; i<ncwords; i++) {
    for (int dim=0; dim<Cluster::ndims; dim++)
      cb_data[dim]=cluster[i].v[dim];
    
    cb_data+=Cluster::ndims;
  }
  
  // Create index_loc nrrd
  Nrrd* idx=nrrdNew();
  if (nrrdWrap(idx, vc_map, nrrdTypeInt, 1, nvecs)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error creating index_loc nrrd:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Save codewords and vector-cluster mappings
  int E=0;
  if (!E) E|=saveNrrd(cb, "-cb", nrrd_ext);
  if (!E) E|=saveNrrd(idx, "-idx", nrrd_ext);
  if (E) {
    cerr<<me<<":  error saving output nrrds"<<endl;
    exit(1);
  }

  // Free allocated memory
  delete [] vec;
  delete [] vc_map;
  delete [] nnt;
  delete [] cluster;
  cb=nrrdNuke(cb);
  idx=nrrdNix(idx);
  
  return 0;
}

void printUsage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<":  unrecongnized option \""<<unknown<<"\""<<endl;

  cerr<<"usage:  "<<me<<" [options] -i <filename> -o <basename>"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -ncwords <int>   number of target codewords (0)"<<endl;
  cerr<<"  -nt <int>        set the number of threads (1)"<<endl;
  cerr<<"  -stats           report cluster statistics (false)"<<endl;
  cerr<<"  -nrrd            use .nrrd extenstion (false)"<<endl;
  cerr<<"  -v <int>         set verbosity level (0)"<<endl;
  cerr<<"  --help           print this message and exit"<<endl;

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

NearestNeighbor findNearestNeighbor(int idx) {
  NearestNeighbor nn;

  for (int c=0; c<cb_size; c++) {
    if (c!=idx) {
      float d=computeDistortion(cluster[idx], cluster[c], nn.distance);
      if (d<nn.distance) {
	nn.nearest=c;
	nn.distance=d;
      }
    }
  }

  return nn;
}

int findMinDistance(void) {
  float min_dist=FLT_MAX;
  int min_idx=-1;

  for (int c=0; c<cb_size; c++) {
    if (nnt[c].distance<min_dist) {
      min_dist=nnt[c].distance;
      min_idx=c;
    }
  }

  return min_idx;
}

float computeDistortion(Cluster c1, Cluster c2, float min) {
  float weight=(c1.nmap*c2.nmap)/(float)(c1.nmap + c2.nmap);
  float distance=0;
  for (int dim=0; dim<Cluster::ndims; dim++) {
    float tmp=c1.v[dim]-c2.v[dim];
    distance+=tmp*tmp;

    if (weight*distance>min)
      break;
  }
  
  return weight*distance;
}

float* computeCentroid(Cluster c1, Cluster c2) {
  char* me="centroid";
  
  float* centroid=new float[Cluster::ndims];
  if (!centroid) {
    cerr<<me<<":  error allocating memory for centroid"<<endl;
    return 0;
  }

  float weight=1.0/(c1.nmap + c2.nmap);
  for (int dim=0; dim<Cluster::ndims; dim++)
    centroid[dim]=weight*(c1.nmap*c1.v[dim] + c2.nmap*c2.v[dim]);

  return centroid;
}

NearestNeighbor::NearestNeighbor(void) {
  nearest=-1;
  distance=FLT_MAX;
  update=false;
}

NearestNeighbor& NearestNeighbor::operator=(const NearestNeighbor& nn) {
  nearest=nn.nearest;
  distance=nn.distance;
  update=nn.update;

  return *this;
}

Cluster::Cluster(void) {
  v=new float[ndims];
  if (!v) {
    cerr<<"Cluster::Cluster - error allocating memory for vector"<<endl;
    return;
  }
  
  nmap=0;
}
  
Cluster::Cluster(float* in_v) {
  if (!in_v) {
    cerr<<"Cluster::Cluster - input vector is null"<<endl;
    return;
  }
  
  v=new float[ndims];
  if (!v) {
    cerr<<"Cluster::Cluster - error allocating memory for vector"<<endl;
    return;
  }
  
  for (int i=0; i<ndims; i++)
    v[i]=in_v[i];
  
  nmap=1;
}

Cluster& Cluster::operator=(const Cluster c) {
  for (int i=0; i<ndims; i++)
    v[i]=c.v[i];
  
  nmap=c.nmap;
  
  return *this;
}

void Cluster::vec(float* in_v) {
  if (!in_v) {
    cerr<<"Cluster::vec - new vector is null"<<endl;
    return;
  }
  
  for (int i=0; i<ndims; i++)
    v[i]=in_v[i];
}

ParallelVQ::ParallelVQ(void) {
  idx1=idx2=last=-1;
  barrier=new Barrier("ParallelVQ barrier");
}

ParallelVQ::~ParallelVQ(void) {
  if (barrier)
    delete barrier;
}

void ParallelVQ::initNearestNeighbors(int proc) {
  // Determine ranges of parallel calculations
  int start=0;
  int end=0;
  range(proc, nvecs, &start, &end);
    
  for(int i=start; i<end; i++) {
    nnt[i]=findNearestNeighbor(i);
    if (nnt[i].nearest<0) {
      cerr<<"ParallelVQ::initNearestNeighbor:  error finding nearest neighbor"
	  <<" for cluster["<<i<<"]"<<endl;
      exit(1);
    }
      
    if (verbose>4 && i%100==0 && i!=0)
      cout<<"Completed "<<i<<" of "<<nvecs<<endl;
  }
}

void ParallelVQ::initMerge(int i1, int i2) {
  idx1=i1;
  idx2=i2;
    
  // Ensure idx1 is smaller than idx2
  if (idx1>idx2) {
    int tmp=idx1;
    idx1=idx2;
    idx2=tmp;
  }

  last=cb_size-1;
    
  if (verbose>9)
    cout<<"Merging cluster["<<idx2<<"] into cluster["<<idx1<<"]"<<endl;
}
  
void ParallelVQ::mergeClusters(int proc) {
  // Determine ranges of parallel calculations
  int cb_start=0;
  int cb_end=0;
  range(proc, cb_size, &cb_start, &cb_end);
    
  int vec_start=0;
  int vec_end=0;
  range(proc, nvecs, &vec_start, &vec_end);
    
  // Mark clusters for update
  if (proc==0)
    nnt[idx1].update=true;

  for (int i=cb_start; i<cb_end; i++) {
    if (nnt[i].nearest==idx1 || nnt[i].nearest==idx2)
      nnt[i].update=true;
    else
      nnt[i].update=false;
  }

  // Join nearest clusters
  if (proc==0) {
    float *centroid=computeCentroid(cluster[idx1], cluster[idx2]);
    cluster[idx1].vec(centroid);
    cluster[idx1].nmap+=cluster[idx2].nmap;
      
    delete [] centroid;
  }

  // Adjust vector-cluster mappings
  for (int i=vec_start; i<vec_end; i++) {
    if (vc_map[i]==idx2)
      vc_map[i]=idx1;
  }

  if (verbose>9) {
    for (int i=vec_start; i<vec_end; i++)
      cout<<"vector["<<i<<"] maps to cluster["<<vc_map[i]<<"]"<<endl;
    cout<<endl;
  }
  
  // Fill empty position in codebook
  if (idx2!=last) {
    if (proc==0) {
      if (verbose>9) {
	cout<<"Moving cluster["<<last<<"] to cluster["<<idx2<<"]"<<endl;
	cout<<"-------------------"<<endl;
	cout<<endl;
      }
	
      cluster[idx2]=cluster[last];
      nnt[idx2]=nnt[last];
    }

    // Update nearest neighbor table
    for (int i=cb_start; i<cb_end; i++) {
      if (nnt[i].nearest==last)
	nnt[i].nearest=idx2;
    }

    // Update vector-cluster mappings
    for (int i=vec_start; i<vec_end; i++) {
      if (vc_map[i]==last)
	vc_map[i]=idx2;
    }
  }

  // Decrement codebook size
  if (proc==0)
    cb_size--;

  barrier->wait(nt);

  // Find new nearest neighbors, if necessary
  range(proc, cb_size, &cb_start, &cb_end);
  for (int i=cb_start; i<cb_end; i++) {
    if (nnt[i].update) {
      nnt[i]=findNearestNeighbor(i);
      if (nnt[i].nearest<0) {
	cerr<<"ParallelVQ::mergeClusters:  error finding nearest"
	    <<" neighbor for cluster["
	    <<i<<"]"<<endl;
	return;
      }
  
      nnt[i].update=false;
    }
  }
}

void ParallelVQ::range(int rank, int total, int* start, int* end) {
  // Determine start and end of range
  int size=total/nt;
  int r=total%nt;
  *start=0;
  if (r!=0) {
    if (rank<r)
      size++;
    else
      *start=r;
  }
    
  *start+=rank*size;
  *end=*start + size;
}
