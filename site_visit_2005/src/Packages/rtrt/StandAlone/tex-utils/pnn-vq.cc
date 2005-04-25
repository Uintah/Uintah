#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <Core/Thread/Barrier.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Time.h>

#include <teem/nrrd.h>

#define MAX_NELTS (25e6*256)

using namespace std;
using namespace SCIRun;

// Declare some useful classes
class NearestNeighbor {
public:
  NearestNeighbor(void);

  NearestNeighbor& operator=(const NearestNeighbor& nn);

  int nearest;
  float distance;
  bool update;
};

class Cluster {
public:
  Cluster(void);
  Cluster(float* in_v);

  Cluster& operator=(const Cluster c);
  void copyVector(float* in_v);

  static int ndims;
  float* v;
  int nmap;
};

class Worker : public Runnable {
public:
  Worker(int id, int npeers);
  ~Worker(void);
  
  void initNNT(void);
  void mergeClusters(void);
  void run(void);

private:
  NearestNeighbor findNearestNeighbor(size_t idx);
  int findMinDistance(void);
  float computeDistortion(Cluster c1, Cluster c2, float min);
  void computeCentroid(Cluster c1, Cluster c2);
  void range(int id, int nprocs, size_t total,
	     size_t* start, size_t* end);

  int id;
  int npeers;
  float* centroid;
};

// Declare necessary functions
void usage(char* me, const char* unknown=0);
Nrrd* convertNrrdToFloat(Nrrd* nin);
void reportStats(double etime);
int saveNrrd(Nrrd* nin, char* type, char* ext);

// Declare global variables
int Cluster::ndims=0;
float* vec=0;
int* vc_map=0;
NearestNeighbor* nnt=0;
Cluster* cluster=0;
int cb_size=0;
int nvecs=0;
int nworkers=1;
int ncwords=0;
Nrrd* vecNrrd=0;
int idx1=0;
int idx2=0;
int last=0;
Barrier barrier("Worker barrier");
char* out_bname="vq";
char* nrrd_ext=".nrrd";
int verbose=0;

int main(int argc, char* argv[]) {
  char* me=argv[0];
  char* err=0;
  char* in_fname=0;
  char* idx_fname=0;
  bool stats=false;

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
    } else if (arg=="-ncwords") {
      ncwords=atoi(argv[++i]);
    } else if (arg=="-idx") {
      idx_fname=argv[++i];
    } else if (arg=="-stats") {
      stats=true;
    } else if (arg=="-nhdr") {
      nrrd_ext=".nhdr";
    } else if (arg=="-v") {
      verbose=atoi(argv[++i]);
    } else if (arg=="--help") {
      usage(me);
      exit(0);
    } else {
      usage(me, argv[i]);
    }
  }

  // Verify arguments
  if (!in_fname) {
    cerr<<me<<":  no input filename specified"<<endl;
    usage(me);
    exit(1);
  }

  if (nworkers<1) {
    cerr<<me<<":  invalid number of worker threads ("<<nworkers
	<<"):  resetting to one"<<endl;
    nworkers=1;
  }
  
  if (ncwords<=0) {
    cerr<<me<<":  invalid number of target codewords ("<<ncwords<<")"<<endl;
    exit(1);
  }
  
  // Load input vectors
  vecNrrd=nrrdNew();
  if (nrrdLoad(vecNrrd, in_fname, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error loading input vectors:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Initialize useful variables
  nvecs=vecNrrd->axis[vecNrrd->dim-1].size;
  Cluster::ndims=1;
  for (int i=0; i<vecNrrd->dim-1; i++)
    Cluster::ndims*=vecNrrd->axis[i].size;

  cout<<"Found "<<nvecs<<" "<<Cluster::ndims<<"-dimensional vectors in \""
      <<in_fname<<"\""<<endl;

  // Sanity check
  if (nvecs*Cluster::ndims>MAX_NELTS) {
    cerr<<me<<":  number of vectors ("<<nvecs
	<<") is too large:  must be less than "
	<<MAX_NELTS/Cluster::ndims<<endl;
    exit(1);
  }
  
  size_t nelts_calc=nvecs*Cluster::ndims;
  size_t nelts=nrrdElementNumber(vecNrrd);
  if (nelts_calc!=nelts) {
    cerr<<me<<":  number of calculated elements ("<<nelts_calc<<") is not equal"
	<<" to number of reported elements ("<<nelts<<")"<<endl;
    exit(1);
  }

  if (ncwords>nvecs) {
    cerr<<me<<":  number of desired codewords ("<<ncwords<<") is greater than the "
	<<"number of input vectors ("<<nvecs<<")"<<endl;
    exit(1);
  }

  // Convert input vectors to nrrdTypeFloat, if necessary
  vecNrrd=convertNrrdToFloat(vecNrrd);
  if (!vecNrrd) {
    cerr<<me<<":  error converting input vectors to nrrdTypeFloat"<<endl;
    exit(1);
  }

  cout<<endl;
  cout<<"------------------------------------"<<endl;

  // Allocate memory for vector indices
  vc_map=new int[nvecs];
  if (!vc_map) {
    cerr<<me<<":  error allocating memory for vector indices"<<endl;
    exit(1);
  }

  // Allocate memory for nearest neighbor table
  nnt=new NearestNeighbor[nvecs];
  if (!nnt) {
    cerr<<me<<":  error allocating memory for nearest neighbor table"<<endl;
    exit(1);
  }

  // Allocate memory for codebook
  cluster=new Cluster[nvecs];
  if (!cluster) {
    cerr<<me<<":  error allocating memory for codebook"<<endl;
    exit(1);
  }

  // Begin timer
  double stime=Time::currentSeconds();
  
  // Create threads
  ThreadGroup* tgroup=new ThreadGroup("Parallel PNN-VQ");

  for (int w=0; w<nworkers; w++) {
    Runnable* runner=new Worker(w, nworkers);
    Thread* t=new Thread(runner, "Worker", tgroup);
  }

  // Wait for worker threads to complete
  tgroup->join();
  delete tgroup;

  // Stop timer
  double etime=Time::currentSeconds()-stime;
  
  // Report statistics
  if (stats)
    reportStats(etime);
  
  // Create codebook nrrd
  Nrrd* cbNrrd=nrrdNew();
  if (nrrdAlloc(cbNrrd, nrrdTypeFloat, 2, Cluster::ndims, ncwords)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error allocating codebook nrrd:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  float* cb_ptr=(float*)(cbNrrd->data);
  for (int i=0; i<ncwords; i++) {
    for (int dim=0; dim<Cluster::ndims; dim++)
      cb_ptr[dim]=cluster[i].v[dim];
    
    cb_ptr+=Cluster::ndims;
  }

  // Save codebook
  if (saveNrrd(cbNrrd, "-cb", nrrd_ext)) {
    cerr<<me<<":  error saving codebook"<<endl;
    exit(1);
  }
  
  // Reclaim unnecessary memory
  delete [] nnt;
  delete [] cluster;
  vecNrrd=nrrdNuke(vecNrrd);
  cbNrrd=nrrdNuke(cbNrrd);

  // Create index nrrd
  Nrrd* idxNrrd=nrrdNew();
  if (!idx_fname) {
    if (nrrdWrap(idxNrrd, vc_map, nrrdTypeInt, 1, nvecs)) {
      err=biffGet(NRRD);
      cerr<<me<<":  error creating index nrrd:  "<<err<<endl;
      free(err);
      biffDone(NRRD);
      exit(1);
    }
  } else {
    // Load vector indices
    if (nrrdLoad(idxNrrd, idx_fname, 0)) {
      err=biffGet(NRRD);
      cerr<<me<<":  error loading vector indices:  "<<err<<endl;
      free(err);
      biffDone(NRRD);
      exit(1);
    }

    // Remap vector indices
    cout<<"Remapping "<<idxNrrd->axis[0].size<<" vector indices"<<endl;

    int mcnt=0;
    int* idx=(int*)(idxNrrd->data);
    for (int v=0; v<idxNrrd->axis[0].size; v++) {
      if (idx[v]!=-1) {
	idx[v]=vc_map[mcnt];
	mcnt++;
      }
    }
  }

  // Save vector-cluster mappings
  if (saveNrrd(idxNrrd, "-idx", nrrd_ext)) {
    cerr<<me<<":  error saving vector-cluster mappings"<<endl;
    exit(1);
  }

  // XXX - causing segfaults
  // delete [] vc_map;
  // idxNrrd=nrrdNuke(idxNrrd);
  // XXX - end causing segfaults

  return 0;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<":  unrecongnized option \""<<unknown<<"\""<<endl;

  cerr<<"usage:  "<<me<<" -i <filename> [options]"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -o <basename>     basename of output files (\"vq\")"<<endl;
  cerr<<"  -nworkers <int>   number of worker threads to use (1)"<<endl;
  cerr<<"  -ncwords <int>    number of target codewords (0)"<<endl;
  cerr<<"  -idx <filename>   filename of existing index file (null)"<<endl;
  cerr<<"  -stats            report cluster statistics (false)"<<endl;
  cerr<<"  -nhdr             use .nrrd extenstion (false)"<<endl;
  cerr<<"  -v <int>          set verbosity level (0)"<<endl;
  cerr<<"  --help            print this message and exit"<<endl;

  if (unknown)
    exit(1);
}

Nrrd* convertNrrdToFloat(Nrrd* nin) {
  char *me="convertNrrdToFloat";
  char *err;

  if (nin->type!=nrrdTypeFloat) {
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

void reportStats(double etime) {
  char* me="reportStats";
  
  size_t len=strlen(out_bname);
  char* sfilename=new char[len+6];
  sprintf(sfilename, "%s.stats", out_bname);
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
    
  if (sfile)
    cout<<"Wrote detailed statistics to \""<<sfilename<<"\""<<endl;
}

int saveNrrd(Nrrd* nin, char* type, char* ext) {
  char* me="saveNrrd";
  char* err;
  
  size_t outbname_len=strlen(out_bname);
  size_t type_len=strlen(type);
  size_t ext_len=strlen(ext);
  size_t length=outbname_len+type_len+ext_len;
  
  char* fname=new char[length];
  sprintf(fname, "%s%s%s", out_bname, type, ext);
  if (nrrdSave(fname, nin, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error saving to "<<fname<<":  "<<err<<endl;
    return 1;
  }

  cout<<"Wrote data to \""<<fname<<"\""<<endl;
  
  return 0;
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
  v=0;
  nmap=0;
}
  
Cluster::Cluster(float* in_v) {
  if (!in_v) {
    cerr<<"Cluster::Cluster:  input vector is null"<<endl;
    return;
  }

  v=in_v;
  nmap=1;
}

Cluster& Cluster::operator=(const Cluster c) {
  v=c.v;
  nmap=c.nmap;
  
  return *this;
}

void Cluster::copyVector(float* in_v) {
  char* me="Cluster::copyVector";
  
  if (!in_v) {
    cerr<<me<<":  input vector is null"<<endl;
    return;
  }

  if (!v) {
    cerr<<me<<":  local vector is null"<<endl;
    return;
  }

  for (int d=0; d<ndims; d++)
    v[d]=in_v[d];
}

Worker::Worker(int id, int npeers) :
  id(id), npeers(npeers), centroid(0), Runnable(true) {
  if (id==0) {
    // Allocate memory for centroid
    centroid=new float[Cluster::ndims];
    if (!centroid) {
      cerr<<"Worker::Worker:  error allocating memory for centroid"<<endl;
      exit(1);
    }
    
    // Initialize centroid
    for (int d=0; d<Cluster::ndims; d++)
      centroid[d]=0.0;
  }
}

Worker::~Worker(void) {
  if (id==0)
    delete [] centroid;
}

void Worker::initNNT(void) {
  // Determine range of vectors
  size_t start=0;
  size_t end=0;
  range(id, npeers, nvecs, &start, &end);

  for(size_t c=start; c<end; c++) {
    nnt[c]=findNearestNeighbor(c);
    if (nnt[c].nearest<0) {
      cerr<<"Worker::initNearestNeighbor:  error finding nearest neighbor"
	  <<" for cluster["<<c<<"]"<<endl;
    }

    
    if (verbose && c%10000==0 && c!=0)
      cout<<"Completed "<<c<<" of "<<nvecs<<endl;
  }
}

void Worker::mergeClusters(void) {
  char* me="Worker::mergeClusters";
  
  if (id==0) {
    idx1=findMinDistance();
    if (idx1<0) {
      cerr<<me<<":  error finding cluster with minimum distance"<<endl;
      return;
    }

    idx2=nnt[idx1].nearest;
    
    // Ensure idx1 is smaller than idx2
    if (idx1>idx2) {
      int tmp=idx1;
      idx1=idx2;
      idx2=tmp;
    }
      
    last=cb_size-1;

    if (verbose>4)
      cout<<"Merging cluster["<<idx2<<"] into cluster["<<idx1<<"]"<<endl;
  }

  barrier.wait(npeers);
    
  // Determine range of codewords
  size_t cb_start=0;
  size_t cb_end=0;
  range(id, npeers, cb_size, &cb_start, &cb_end);

  // Determine range of vectors
  size_t vec_start=0;
  size_t vec_end=0;
  range(id, npeers, nvecs, &vec_start, &vec_end);
    
  // Mark clusters for update
  if (id==0)
    nnt[idx1].update=true;

  for (size_t i=cb_start; i<cb_end; i++) {
    if (nnt[i].nearest==idx1 || nnt[i].nearest==idx2)
      nnt[i].update=true;
    else
      nnt[i].update=false;
  }

  // Join nearest clusters
  if (id==0) {
    computeCentroid(cluster[idx1], cluster[idx2]);
    cluster[idx1].copyVector(centroid);
    cluster[idx1].nmap+=cluster[idx2].nmap;
  }

  // Adjust vector-cluster mappings
  for (size_t i=vec_start; i<vec_end; i++) {
    if (vc_map[i]==idx2)
      vc_map[i]=idx1;
  }

  // Fill empty position in codebook
  if (idx2!=last) {
    if (id==0) {
      cluster[idx2]=cluster[last];
      nnt[idx2]=nnt[last];
    }

    // Update nearest neighbor table
    for (size_t i=cb_start; i<cb_end; i++) {
      if (nnt[i].nearest==last)
	nnt[i].nearest=idx2;
    }

    // Update vector-cluster mappings
    for (size_t i=vec_start; i<vec_end; i++) {
      if (vc_map[i]==last)
	vc_map[i]=idx2;
    }
  }

  // Decrement codebook size
  if (id==0)
    cb_size--;

  barrier.wait(nworkers);

  if (cb_size<=ncwords)
    return;
  
  // Find new nearest neighbors, if necessary
  range(id, npeers, cb_size, &cb_start, &cb_end);
  for (size_t c=cb_start; c<cb_end; c++) {
    if (nnt[c].update) {
      nnt[c]=findNearestNeighbor(c);
      if (nnt[c].nearest<0) {
	cerr<<me<<":  error finding nearest neighbor for cluster["
	    <<c<<"]"<<endl;
      }
  
      nnt[c].update=false;
    }
  }
}

void Worker::run(void) {
  if (id==0)
    cout<<"Initializing codebook and vector indices"<<endl;
  
  // Determine range of work
  size_t start=0;
  size_t end=0;
  range(id, npeers, nvecs, &start, &end);
  
  // Initialize codebook and vc_map
  float* vec_ptr=(float*)(vecNrrd->data);
  vec_ptr+=start*Cluster::ndims;
  for (size_t i=start; i<end; i++) {
    cluster[i]=Cluster(vec_ptr);
    vc_map[i]=(int)i;
    
    vec_ptr+=Cluster::ndims;
  }
  
  if (id==0) {
    cb_size=nvecs;

    cout<<"Done initializing codebook and vector indices"<<endl;
    cout<<endl;
  }

  barrier.wait(npeers);
  
  // Initialize nearest neighbor table
  if (id==0)
    cout<<"Intializing nearest neighbor table"<<endl;
  
  initNNT();

  barrier.wait(npeers);
  
  if (id==0) {
    cout<<"Done initializing nearest neighbor table"<<endl;
    cout<<endl;
    cout<<"Quantizing "<<nvecs<<" vectors to "<<ncwords
	<<" codewords"<<endl;
    if (verbose)
      cout<<cb_size-ncwords<<" clusters remain"<<endl;
  }
  
  // Merge clusters
  while (cb_size>ncwords) {
    mergeClusters();

    if (verbose>4 && id==0) {
      int delta=cb_size-ncwords;
      if (delta%10000==0 && delta!=0)
	cout<<delta<<" clusters remain"<<endl;
    }
  }

  barrier.wait(npeers);
  
  if (id==0) {
    cout<<"Done quantizing vectors"<<endl;
    cout<<"------------------------------------"<<endl;
    cout<<endl;
  }
}

NearestNeighbor Worker::findNearestNeighbor(size_t idx) {
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

int Worker::findMinDistance(void) {
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

float Worker::computeDistortion(Cluster c1, Cluster c2, float min) {
  float weight=(c1.nmap*c2.nmap)/(float)(c1.nmap+c2.nmap);
  float distance=0;
  for (int dim=0; dim<Cluster::ndims; dim++) {
    float tmp=c1.v[dim]-c2.v[dim];
    distance+=tmp*tmp;

    if (weight*distance>min)
      break;
  }
  
  return weight*distance;
}

void Worker::computeCentroid(Cluster c1, Cluster c2) {
  float weight=1.0/(c1.nmap+c2.nmap);
  for (int dim=0; dim<Cluster::ndims; dim++)
    centroid[dim]=weight*(c1.nmap*c1.v[dim]+c2.nmap*c2.v[dim]);
}

void Worker::range(int id, int nprocs, size_t total,
		   size_t* start, size_t* end) {
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
