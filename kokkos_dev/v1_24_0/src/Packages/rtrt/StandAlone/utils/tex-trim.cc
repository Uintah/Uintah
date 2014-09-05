#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <stdio.h>
#include <teem/nrrd.h>

using namespace std;

// Declare necessary functions
void usage(char* me, const char* unknown=0);
int saveNrrd(Nrrd* nin, char* type, char* ext);

// Declare global variables
char* outbname=0;

int main(int argc, char* argv[]) {
  char* me=argv[0];
  char* err;
  const float max_value=255.0;
  const float inv_max_value=1.0/max_value;
  char* infname=0;
  float threshold=0.0;

  for(int i=1; i<argc; i++) {
    string arg(argv[i]);
    if (arg=="-i") {
      infname=argv[++i];
    } else if (arg=="-o") {
      outbname=argv[++i];
    } else if (arg=="-thresh") {
      threshold=atof(argv[++i]);
    } else if (arg=="--help") {
      usage(me);
      exit(0);
    } else {
      usage(me, arg.c_str());
    }
  }

  // Verify the arguments
  if (!infname) {
    cerr<<me<<":  input filename not specified"<<endl;
    usage(me);
    exit(1);
  }
  
  if (!outbname) {
    cerr<<me<<":  output basename not specified"<<endl;
    usage(me);
    exit(1);
  }

  if (threshold<0.0 || threshold>1.0) {
    cerr<<me<<":  invalid threshold ("<<threshold
	<<"), resetting to zero"<<endl;
    threshold=0.0;
  }
  
  // Read the input header
  Nrrd* nin=nrrdNew();
  NrrdIO* nios=nrrdIONew();
  nios->skipData=1;
  if (nrrdLoad(nin, infname, nios)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error reading header:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Set up useful variables
  int width=nin->axis[0].size;
  int height=nin->axis[1].size;
  int ntextures=nin->axis[2].size;
  int ndims=width*height;
  const float max_distance=sqrt((float)ndims);
  threshold*=max_distance;
  
  cout<<"Found "<<ntextures<<" "<<width<<"x"<<height<<" textures in \""
      <<infname<<"\""<<endl;

  // Sanity check
  if (nin->type!=nrrdTypeUChar) {
    cerr<<me<<":  expecting nrrdTypeUChar, got ";
    switch(nin->type) {
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
    cerr<<" instead"<<endl;
    exit(1);
  }

  // Nix input nrrd junk
  nin=nrrdNix(nin);
  nios=nrrdIONix(nios);

  // Allocate necessary memory
  unsigned char* tex=new unsigned char[ndims];
  if (!tex) {
    cerr<<me<<":  error allocating memory for input vector"<<endl;
    exit(1);
  }

  int* idx=new int[ntextures];
  if (!idx) {
    cerr<<me<<":  error allocating memory for index array"<<endl;
    exit(1);
  }
  
  // Open input raw file
  size_t len=strlen(infname);
  char* in_rawfname=new char[len-1];
  strncpy(in_rawfname, infname, len-4);
  sprintf(in_rawfname+(len-4), "raw");
  FILE* in_rfile=fopen(in_rawfname, "r");
  if (!in_rfile) {
    cerr<<me<<":  error opening input raw file \""<<in_rawfname<<"\""<<endl;
    exit(1);
  }

  // Open output raw file
  len=strlen(outbname);
  char* out_rawfname=new char[len+4];
  strncpy(out_rawfname, outbname, len);
  sprintf(out_rawfname+len, ".raw");
  FILE* out_rfile=fopen(out_rawfname, "w");
  if (!out_rfile) {
    cerr<<me<<":  error opening output raw file \""<<out_rawfname<<"\""<<endl;
    exit(1);
  }

  // Write textures that are within threshold units of solid black to output file
  int cnt=0;
  float et_thresh=threshold*threshold;
  for (int t=0; t<ntextures; t++) {
    // Read texture from input file
    fseek(in_rfile, (long)t*ndims, SEEK_SET);
    size_t nread=fread(tex, sizeof(unsigned char), ndims, in_rfile);
    if (nread!=ndims) {
      cerr<<me<<":  error reading texture["<<t<<"]"<<endl;
      exit(1);
    }

    // Calculate distance to solid black texture
    float distance=0;
    for (int d=0; d<ndims; d++) {
      float tmp=(float)tex[d]*inv_max_value;
      distance+=tmp*tmp;

      if (distance>et_thresh)
	break;
    }

    if (sqrt(distance)<=threshold) {
      idx[t]=-1;
    } else {
      // Write texture to output file
      fseek(out_rfile, (long)cnt*ndims, SEEK_SET);
      size_t nwritten=fwrite(tex, sizeof(unsigned char), ndims, out_rfile);
      if (nwritten!=ndims) {
	cerr<<me<<":  error writing texture["<<t<<"]"<<endl;
	exit(1);
      }

      idx[t]=cnt;
      cnt++;
    }
  }
      
  cout<<"Removed "<<(1.0-cnt/(float)ntextures)*100.0<<"% of the textures"<<endl;
  
  // Close raw input and output files
  fclose(in_rfile);
  fclose(out_rfile);

  // Open nrrd header
  char* out_fname=new char[len+5];
  strncpy(out_fname, outbname, len);
  sprintf(out_fname+len, ".nhdr");
  FILE* out_file=fopen(out_fname, "w");
  if (!out_file) {
    cerr<<me<<":  error opening nrrd header file \""<<out_fname<<"\""<<endl;
    exit(1);
  }

  // Write nrrd header file
  fprintf(out_file, "NRRD0001\n");
  fprintf(out_file, "type: unsigned char\n");
  fprintf(out_file, "dimension: 3\n");
  fprintf(out_file, "sizes: %d %d %d\n", width, height, cnt);
  fprintf(out_file, "labels: \"x\" \"y\" \"texture\"\n");
  fprintf(out_file, "data file: ./%s\n", out_rawfname);
#ifdef __sgi
  fprintf(out_file, "endian: big\n");
#else
  fprintf(out_file, "endian: little\n");
#endif
  fprintf(out_file, "encoding: raw\n");

  // Close nrrd header
  fclose(out_file);

  cout<<"Wrote "<<cnt<<" "<<width<<"x"<<height<<" textures to \""
      <<outbname<<".nhdr\""<<endl;

  // Create index nrrd
  Nrrd* idxNrrd=nrrdNew();
  if (nrrdWrap(idxNrrd, idx, nrrdTypeInt, 1, ntextures)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error creating index nrrd:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  if (saveNrrd(idxNrrd, "-idx", ".nhdr")) {
    cerr<<me<<":  error saving index array"<<endl;
    exit(1);
  }

  cout<<"Wrote texture indices to \""<<outbname<<"-idx.nhdr\""<<endl;
  
  // Clean up memory
  delete [] in_rawfname;
  delete [] out_rawfname;
  delete [] tex;
  delete [] idx;
  delete [] out_fname;
  idxNrrd=nrrdNix(idxNrrd);

  return 0;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<": unknown argument \""<<unknown<<"\""<<endl;
  
  // Print the usage
  cerr<<"usage:  "<<me<<" [options] -i <filename> -o <basename>"<<endl;;
  cerr<<"options:"<<endl;
  cerr<<"  -thresh <float>    percent distance to solid black (0)"<<endl;
  cerr<<"  --help             print this message and exit"<<endl;

  if (unknown)
    exit(1);
}

int saveNrrd(Nrrd* nin, char* type, char* ext) {
  char* me="saveNrrd";
  char* err;
  
  size_t outbname_len=strlen(outbname);
  size_t type_len=strlen(type);
  size_t ext_len=strlen(ext);
  size_t len=outbname_len+type_len+ext_len;
  
  char* fname=new char[len];
  sprintf(fname, "%s%s%s", outbname, type, ext);
  if (nrrdSave(fname, nin, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error saving to "<<fname<<":  "<<err<<endl;
    return 1;
  }
  
  return 0;
}
