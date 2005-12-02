#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <teem/nrrd.h>

using namespace std;

// Forward declare necessary functions
void usage(char* me, const char* unknown=0);

// Declare global variables
unsigned int width=0;
unsigned int height=0;
unsigned int ndims=0;
unsigned int nvectors=0;

int main(int argc, char *argv[]) {
  char* me=argv[0];
  char* err;
  char* v_fname=0;
  char* vq_bname="vq";
  char* out_fname="vq-error.nrrd";
  char* cb_fname=0;
  char* idx_fname=0;
  char* nrrd_ext=".nrrd";

  // Parse arguments
  for(int i=1; i<argc; i++) {
    string arg(argv[i]);
    if (arg=="-v") {
      v_fname=argv[++i];
    } else if (arg=="-vq") {
      vq_bname=argv[++i];
    } else if (arg=="-cb") {
      cb_fname=argv[++i];
    } else if (arg=="-idx") {
      idx_fname=argv[++i];
    } else if (arg=="-o") {
      out_fname=argv[++i];
    } else if (arg=="-nhdr") {
      nrrd_ext=".nhdr";
    } else if (arg=="--help") {
      usage(me);
      exit(0);
    } else {
      usage(me, arg.c_str());
    }
  }
  
  // Verify the arguments
  if (!v_fname) {
    cerr<<me<<":  filename of original vectors not specified"<<endl;
    usage(me);
    exit(1);
  }
  
  size_t name_len=strlen(vq_bname);
  size_t ext_len=strlen(nrrd_ext);
  if (!cb_fname) {
    cb_fname=new char[name_len+5+ext_len];
    sprintf(cb_fname, "%s-cb-q%s", vq_bname, nrrd_ext);
  }
  
  if (!idx_fname) {
    idx_fname=new char[name_len+4+ext_len];
    sprintf(idx_fname, "%s-idx%s", vq_bname, nrrd_ext);
  }

  // Read header of input vectors
  Nrrd* vecNrrd=nrrdNew();
  NrrdIoState* vecNios=nrrdIoStateNew();
  vecNios->skipData=1;
  if (nrrdLoad(vecNrrd, v_fname, vecNios)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error reading header:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Load the codebook and index files
  Nrrd* cbNrrd=nrrdNew();
  Nrrd* idxNrrd=nrrdNew();
  int E=0;
  if (!E) E|=nrrdLoad(cbNrrd, cb_fname, 0);
  if (!E) E|=nrrdLoad(idxNrrd, idx_fname, 0);
  if (E) {
    err=biffGet(NRRD);
    cerr<<me<<":  error loading input files:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Initialize useful variables
  width=vecNrrd->axis[0].size;
  height=vecNrrd->axis[1].size;
  ndims=width*height;
  nvectors=vecNrrd->axis[2].size;

  // Sanity check
  if (vecNrrd->dim!=3) {
    cerr<<me<<":  original vectors must have three dimensions"<<endl;
    exit(1);
  }

  if (cbNrrd->dim!=3) {
    cerr<<me<<":  codebook vectors must have three dimensions"<<endl;
    exit(1);
  }
  
  if (idxNrrd->dim!=1) {
    cerr<<me<<":  index array must have one dimension"<<endl;
    exit(1);
  }

  if (cbNrrd->axis[0].size*cbNrrd->axis[1].size!=ndims) {
    cerr<<me<<":  dimensionality of codebook vectors ("
        <<cbNrrd->axis[0].size*cbNrrd->axis[1].size
        <<") is not equal to the dimensionality of original vectors ("
	<<ndims<<")"<<endl;
    exit(1);
  }

  if (idxNrrd->axis[0].size!=nvectors) {
    cerr<<me<<":  number of vector indices ("<<idxNrrd->axis[0].size
	<<") is not equal to the number of original vectors ("
        <<nvectors<<")"<<endl;
    exit(1);
  }

  cout<<"Found "<<nvectors<<" "<<ndims<<"-dimensional vectors in \""
      <<v_fname<<"\""<<endl;
  cout<<"Found "<<cbNrrd->axis[2].size<<" codebook vectors in \""
      <<cb_fname<<"\""<<endl;
  cout<<"Found "<<nvectors<<" vector indices in \""
      <<idx_fname<<"\""<<endl;
  
  // Clean up unnecessary memory
  vecNrrd=nrrdNix(vecNrrd);
  vecNios=nrrdIoStateNix(vecNios);
  // XXX - Only free if not passed in on the cmdln, but
  //       how to tell?
  if (vq_bname) {
    delete [] cb_fname;
    delete [] idx_fname;
  }
  // XXX - end

  // Open raw vector file
  size_t len=strlen(v_fname);
  char* v_rawfname=new char[len+4];
  if (!v_rawfname) {
    cerr<<me<<":  error allocating "<<(len+4)*sizeof(char)
        <<" bytes for raw vector filename"<<endl;
    exit(1);
  }
  
  strncpy(v_rawfname, v_fname, len-4);
  sprintf(v_rawfname+(len-4), "raw");
  FILE* v_rawfile=fopen(v_rawfname, "r");
  if (!v_rawfile) {
    cerr<<me<<":  error opening raw vector file \""<<v_rawfname<<"\""<<endl;
    exit(1);
  }

  delete [] v_rawfname;

  // Allocate memory for input vector
  unsigned char* in_vec=new unsigned char[ndims];
  if (!in_vec) {
    cerr<<me<<":  error allocating "<<ndims*sizeof(float)
        <<" bytes for input vector"<<endl;
    exit(1);
  }

  cout<<endl;
  cout<<"-----------------------"<<endl;
  cout<<"Computing error for "<<nvectors<<" vectors"<<endl;
  
  // Allocate error array
  float* error=new float[nvectors];
  if (!error) {
    cerr<<me<<":  error allocating memory for error array"<<endl;
    exit(1);
  }
  
  float max_error=-FLT_MAX;
  float min_error=FLT_MAX;
  float total_error=0;

  // float* cb_data=(float*)(cbNrrd->data);
  unsigned char* cb_data=(unsigned char*)(cbNrrd->data);
  int* idx_data=(int*)(idxNrrd->data);
  for (int v=0; v<nvectors; v++) {
    // Read original vector from raw file
    size_t nread=fread(in_vec, sizeof(unsigned char), ndims, v_rawfile);
    if (nread<ndims) {
      cerr<<me<<":  error reading elements of vector["<<v<<"]:  expected "
          <<ndims*sizeof(unsigned char)<<" bytes but only read "
          <<nread<<" bytes"
          <<endl;
      exit(1);
    }

    unsigned char* cb_vec=cb_data+ndims*idx_data[v];

#if 0
    if (v<10) {
      cout<<"in_vec["<<v<<"]=( ";
      for (int i=0; i<10; i++)
        cout<<(float)in_vec[i]<<" ";
      cout<<"... )"<<endl;
      
      cout<<"cb_vec["<<idx_data[v]<<"]=( ";
      for (int i=0; i<10; i++)
        cout<<(float)cb_vec[i]<<" ";
      cout<<"... )"<<endl;
      cout<<endl;
    }
#endif

    // Compute distance between the two vectors
    for (int d=0; d<ndims; d++) {
      float residual=(float)cb_vec[d] - (float)in_vec[d];
      error[v]+=residual*residual;
    }

    error[v]=sqrt(error[v]);
    
    if (error[v]>max_error)
      max_error=error[v];
    if (error[v]<min_error)
      min_error=error[v];
    
    total_error+=error[v];
  }

  cout<<endl;
  cout<<"Maximum error:  "<<max_error<<endl;
  cout<<"Minimum error:  "<<min_error<<endl;
  cout<<"Mean error:     "<<total_error/nvectors<<endl;
  cout<<"-----------------------"<<endl;

  // Close raw files
  fclose(v_rawfile);

  // Write the output file
  Nrrd* nout=nrrdNew();
  if (nrrdWrap(nout, error, nrrdTypeFloat, 1, nvectors)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error creating error nrrd:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }
  
  if (nrrdSave(out_fname, nout, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error saving to "<<out_fname<<":  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }
  
  cout<<endl;
  cout<<"Wrote data to "<<out_fname<<endl;
  
  // Clean up memory
  cbNrrd=nrrdNuke(cbNrrd);
  idxNrrd=nrrdNuke(idxNrrd);
  nout=nrrdNuke(nout);
  
  return 0;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<":  unrecognized option \""<<unknown<<"\""<<endl;
  
  // Print out the usage
  cerr<<"usage:  "<<me<<" [options] -v <filename>"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -vq <basename>    basename of VQ input files (\"vq\")"<<endl;
  cerr<<"  -cb <filename>    filename of codebook vectors (null)"<<endl;
  cerr<<"  -idx <filename>   filename of vector indices (null)"<<endl;
  cerr<<"  -o <filename>     filename of output file (\"vq-error.nrrd\")"
      <<endl;
  cerr<<"  -nhdr             use \".nhdr\" extension (false)"<<endl;
  cerr<<"  --help            print this message and exit"<<endl;

  if (unknown)
    exit(1);
}
