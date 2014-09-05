#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>
#include <stdio.h>

#include <teem/nrrd.h>

using namespace std;

#define MAX_VALUE 255
#define MAX_VALUE2 MAX_VALUE*MAX_VALUE

// Declare necessary functions
void usage(char* me, const char* unknown=0);
char* createFName(char* in_fname, char* ext);
int saveNrrd(Nrrd* nin, char* type, char* ext);

// Declare global variables
char* out_bname="rm";

int main(int argc, char* argv[]) {
  char* me=argv[0];
  char* err;
  char* in_fname=0;
  float threshold=0.0;

  if (argc<3) {
    usage(me);
    exit(1);
  }

  for(int i=1; i<argc; i++) {
    string arg(argv[i]);
    if (arg=="-i") {
      in_fname=argv[++i];
    } else if (arg=="-o") {
      out_bname=argv[++i];
    } else if (arg=="-t") {
      threshold=atof(argv[++i]);
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
  
  if (threshold<0.0 || threshold>100.0) {
    cerr<<me<<":  invalid threshold ("<<threshold
	<<"):  resetting to zero"<<endl;
    threshold=0.0;
  }

  // Read input header
  Nrrd* texNrrd=nrrdNew();
  NrrdIoState* nios=nrrdIoStateNew();
  nios->skipData=1;
  if (nrrdLoad(texNrrd, in_fname, nios)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error reading header:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Sanity check
  if (texNrrd->type!=nrrdTypeUChar) {
    cerr<<me<<":  expecting nrrdTypeUChar:  textures are ";
    switch(texNrrd->type) {
    case nrrdTypeUnknown: cerr<<"nrrdTypeUnknown"; break;
    case nrrdTypeChar: cerr<<"nrrdTypeChar"; break;
    case nrrdTypeShort: cerr<<"nrrdTypeShort"; break;
    case nrrdTypeUShort: cerr<<"nrrdTypeUShort"; break;
    case nrrdTypeInt: cerr<<"nrrdTypeInt"; break;
    case nrrdTypeUInt: cerr<<"nrrdTypeUInt"; break;
    case nrrdTypeFloat: cerr<<"nrrdTypeFloat"; break;
    case nrrdTypeLLong: cerr<<"nrrdTypeLLong"; break;
    case nrrdTypeULLong: cerr<<"nrrdTypeULLong"; break;
    case nrrdTypeDouble: cerr<<"nrrdTypeDouble"; break;
    default: cerr<<"unknown type"<<endl;
    }
    exit(1);
  }

  // Set up useful variables
  int width=texNrrd->axis[0].size;
  int height=texNrrd->axis[1].size;
  int ntextures=texNrrd->axis[2].size;
  int ndims=width*height;
  
  cout<<"Found "<<ntextures<<" "<<width<<"x"<<height<<" textures in \""
      <<in_fname<<"\""<<endl;
  cout<<endl;
  cout<<"-----------------------"<<endl;

  // Nix input nrrd junk
  texNrrd=nrrdNix(texNrrd);
  nios=nrrdIoStateNix(nios);

  // Allocate memory for input vector
  unsigned char* tex=new unsigned char[ndims];
  if (!tex) {
    cerr<<me<<":  error allocating memory for input vector"<<endl;
    exit(1);
  }

  // Allocate memory for index array
  int* idx=new int[ntextures];
  if (!idx) {
    cerr<<me<<":  error allocating memory for index array"<<endl;
    exit(1);
  }
  
  // Open raw input file
  size_t len=strlen(in_fname)-4;
  char* in_bname=new char[len];
  strncpy(in_bname, in_fname, len);
  sprintf(in_bname+len-1, "%c", '\0');
  char* in_rawfname=createFName(in_bname, ".raw");
  if (!in_rawfname) {
    cerr<<me<<":  error creating filename of raw input textures"<<endl;
    exit(1);
  }
  
  delete [] in_bname;

  FILE* in_rfile=fopen(in_rawfname, "r");
  if (!in_rfile) {
    cerr<<me<<":  error opening input raw file \""<<in_rawfname<<"\""<<endl;
    exit(1);
  }

  delete [] in_rawfname;

  // Open raw output file
  char* out_rawfname=createFName(out_bname, ".raw");
  if (!out_rawfname) {
    cerr<<me<<":  error creating filename of raw output textures"<<endl;
    exit(1);
  }

  FILE* out_rfile=fopen(out_rawfname, "w");
  if (!out_rfile) {
    cerr<<me<<":  error opening output raw file \""<<out_rawfname<<"\""<<endl;
    exit(1);
  }

  delete [] out_rawfname;

  cout<<"Removing textures that are within "<<threshold
      <<"% of maximum distance (";
  threshold/=100.0;
  double max_dist=threshold*sqrt((float)(MAX_VALUE2*ndims));
  double max_dist2=max_dist*max_dist;
  cout<<max_dist2<<" units)"<<endl;

  // Examine textures and write to output file, if necessary
  int write_cnt=0;
  for (int t=0; t<ntextures; t++) {
    // Read texture from input file
    size_t nread=fread(tex, sizeof(unsigned char), ndims, in_rfile);
    if (nread!=ndims) {
      cerr<<me<<":  error reading texture["<<t<<"]"<<endl;
      exit(1);
    }

    // Calculate squared distance
    float distance2=0.0;
    for (int d=0; d<ndims; d++) {
      distance2+=(float)(tex[d]*tex[d]);
      
      // Early termination check
      if (distance2>max_dist2)
        break;
    }

    if (distance2<max_dist2) {
      idx[t]=-1;
    } else {
      // Write texture to output file
      size_t nwritten=fwrite(tex, sizeof(unsigned char), ndims, out_rfile);
      if (nwritten!=ndims) {
	cerr<<me<<":  error writing texture["<<t<<"]"<<endl;
	exit(1);
      }

      idx[t]=write_cnt;
      write_cnt++;
    }
  }

  delete [] tex;
  
  // Close raw input and output files
  fclose(in_rfile);
  fclose(out_rfile);

  int delta=ntextures-write_cnt;
  cout<<"Removed "<<delta<<" of "<<ntextures<<" textures"<<endl;
  cout<<"-----------------------"<<endl;
  cout<<endl;

  if (delta>0 && write_cnt>0) {
    // Write nrrd header for output textures
    char* nhdr_fname=createFName(out_bname, ".nhdr");
    if (!nhdr_fname) {
      cerr<<me<<":  error creating header filename for output textures"<<endl;
      exit(1);
    }
  
    Nrrd* outNrrd=nrrdNew();
    nrrdAxisInfoSet(outNrrd, nrrdAxisInfoLabel, "width", "height", "texture");
    if (nrrdWrap(outNrrd, (void*)1, nrrdTypeUChar, 3, width, height, write_cnt)) {
      err=biffGet(NRRD);
      cerr<<me<<":  error creating header:  "<<err<<endl;
      free(err);
      biffDone(NRRD);
      exit(1);
    }

    NrrdIoState* nio=nrrdIoStateNew();
    nio->skipData=1;
    if (nrrdSave(nhdr_fname, outNrrd, nio)) {
      err=biffGet(NRRD);
      cerr<<me<<":  error writing header:  "<<err<<endl;
      free(err);
      biffDone(NRRD);
      exit(1);
    }

    cout<<"Wrote "<<write_cnt<<" "<<width<<"x"<<height<<" textures to \""
	<<nhdr_fname<<"\""<<endl;

    delete [] nhdr_fname;
    outNrrd=nrrdNix(outNrrd);
    nio=nrrdIoStateNix(nio);
  
    // Create index nrrd
    Nrrd* idxNrrd=nrrdNew();
    if (nrrdWrap(idxNrrd, idx, nrrdTypeInt, 1, ntextures)) {
      err=biffGet(NRRD);
      cerr<<me<<":  error creating index nrrd:  "<<err<<endl;
      free(err);
      biffDone(NRRD);
      exit(1);
    }

    if (saveNrrd(idxNrrd, "-idx", ".nrrd")) {
      cerr<<me<<":  error saving index array"<<endl;
      exit(1);
    }

    delete [] idx;
    idxNrrd=nrrdNix(idxNrrd);

    cout<<"Wrote texture indices to \""<<out_bname<<"-idx.nrrd\""<<endl;
  }

  return 0;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<": unknown argument \""<<unknown<<"\""<<endl;
  
  // Print the usage
  cerr<<"usage:  "<<me<<" [options] -i <filename>"<<endl;;
  cerr<<"options:"<<endl;
  cerr<<"  -o <basename>   basename of output files (\"rm\")"<<endl;
  cerr<<"  -t <float>      mean value threshold (0.0)"<<endl;
  cerr<<"  --help          print this message and exit"<<endl;

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

int saveNrrd(Nrrd* nin, char* type, char* ext) {
  char* me="saveNrrd";
  char* err;
  
  size_t outbname_len=strlen(out_bname);
  size_t type_len=strlen(type);
  size_t ext_len=strlen(ext);
  size_t len=outbname_len+type_len+ext_len;
  
  char* fname=new char[len];
  sprintf(fname, "%s%s%s", out_bname, type, ext);
  if (nrrdSave(fname, nin, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error saving to "<<fname<<":  "<<err<<endl;
    return 1;
  }
  
  return 0;
}
