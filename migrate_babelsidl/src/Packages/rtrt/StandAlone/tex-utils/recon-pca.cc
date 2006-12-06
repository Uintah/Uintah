#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <stdio.h>
#include <teem/nrrd.h>

using namespace std;

// Declare necessary functions
void usage(char* me, const char* unknown=0);
int writeNrrdHeader(char* raw_fname, int type, int dim, int* size,
		    char* label[]);

// Declare global variables
int nbases=0;
int nvectors=0;
int ndims=0;
char* out_bname="recon";
char* nrrd_ext=".nrrd";

int main(int argc, char* argv[]) {
  char* me=argv[0];
  char* err;
  char* pca_bname="pca";
  char* b_fname=0;
  char* c_fname=0;
  char* m_fname=0;

  for (int i=1; i<argc; i++) {
    string arg(argv[i]);
    if (arg=="-i") {
      pca_bname=argv[++i];
    } else if (arg=="-o") {
      out_bname=argv[++i];
    } else if (arg=="-b") {
      b_fname=argv[++i];
    } else if (arg=="-c") {
      c_fname=argv[++i];
    } else if (arg=="-m") {
      m_fname=argv[++i];
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
  size_t name_len=strlen(pca_bname);
  if (!b_fname) {
    b_fname=new char[name_len+13];
    if (!b_fname) {
      cerr<<me<<":  error allocating "<<(name_len+11)*sizeof(char)
          <<" bytes for basis vector filename"<<endl;
      exit(1);
    }
      
    sprintf(b_fname, "%s-basis-q%s", pca_bname, nrrd_ext);
  }
    
  if (!c_fname) {
    c_fname=new char[name_len+13];
    if (!c_fname) {
      cerr<<me<<":  error allocating "<<(name_len+11)*sizeof(char)
          <<" bytes for coefficient matrix filename"<<endl;
      exit(1);
    }

    sprintf(c_fname, "%s-coeff-q%s", pca_bname, nrrd_ext);
  }
    
  if (!m_fname) {
    m_fname=new char[name_len+12];
    if (!m_fname) {
      cerr<<me<<":  error allocating "<<(name_len+10)*sizeof(char)
          <<" bytes for coefficient matrix filename"<<endl;
      exit(1);
    }
      
    sprintf(m_fname, "%s-mean-q%s", pca_bname, nrrd_ext);
  }
  
  // Load basis vectors
  Nrrd* basisNrrd=nrrdNew();
  if (nrrdLoad(basisNrrd, b_fname, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<": error loading basis vectors: "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Read header of coefficient matrix
  Nrrd* coeffNrrd=nrrdNew();
  NrrdIoState* nios=nrrdIoStateNew();
  nios->skipData=1;
  if (nrrdLoad(coeffNrrd, c_fname, nios)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error reading header:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Load mean vector
  Nrrd* meanNrrd=nrrdNew();
  if (nrrdLoad(meanNrrd, m_fname, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<": error loading mean vector: "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Initialize useful variables
  float basis_min=FLT_MAX;
  float basis_max=-FLT_MAX;
  float basis_diff=0;
  float coeff_min=FLT_MAX;
  float coeff_max=-FLT_MAX;
  float coeff_diff=0;
  if (basisNrrd->dim==2) {
    ndims=basisNrrd->axis[0].size;
    nbases=basisNrrd->axis[1].size;

    basis_min=basisNrrd->oldMin;
    basis_max=basisNrrd->oldMax;
    if (basis_min>=FLT_MAX) {
      cerr<<me<<":  error reading basis minimum from \""<<b_fname<<"\""<<endl;
      exit(1);
    } else if (basis_max<=-FLT_MAX) {
      cerr<<me<<":  error reading basis maximum from \""<<b_fname<<"\""<<endl;
      exit(1);
    }
    
    basis_diff=(basis_max-basis_min)/255.0;

    cout<<"Found "<<nbases<<" "<<ndims<<"-dimensional basis vectors in \""
        <<b_fname<<"\""<<endl;
  } else {
    cerr<<me<<":  invalid basis vectors:  expecting a 2-dimensional nrrd,"
        <<" but \""<<b_fname<<"\" has "<<basisNrrd->dim<<" dimensions"<<endl;
    exit(1);
  }
  
  if (coeffNrrd->dim==2) {
    if (coeffNrrd->axis[0].size!=nbases) {
      cerr<<me<<":  invalid coefficient matrix:  number of basis vectors ("
          <<coeffNrrd->axis[0].size<<") is not correct; expecting "
          <<nbases<<" basis vectors"<<endl;
      exit(1);
    }

    nvectors=coeffNrrd->axis[1].size;

    coeff_min=coeffNrrd->oldMin;
    coeff_max=coeffNrrd->oldMax;
    if (coeff_min>=FLT_MAX) {
      cerr<<me<<":  error reading coefficient minimum from \""
          <<c_fname<<"\""<<endl;
      return 1;
    } else if (coeff_max<=-FLT_MAX) {
      cerr<<me<<":  error reading coefficient maximum from \""
          <<c_fname<<"\""<<endl;
      return 1;
    }

    coeff_diff=(coeff_max-coeff_min)/255.0;
    
    cout<<"Found "<<nvectors<<"x"<<nbases<<" coefficient matrix in \""
        <<c_fname<<"\""<<endl;
  } else {
    cerr<<me<<":  invalid coefficient matrix:  expecting a 2-dimensional nrrd,"
        <<" but \""<<c_fname<<"\" has "<<coeffNrrd->dim<<" dimensions"<<endl;
    exit(1);
  }
      
  if (meanNrrd->dim==1) {
    if (meanNrrd->axis[0].size!=ndims) {
      cerr<<me<<":  invalid mean vector:  number of dimensions ("
          <<ndims<<") is not correct; expecting a "<<ndims<<"-dimensional "
          <<"mean vector"<<endl;
      exit(1);
    }

    cout<<"Found "<<ndims<<"-dimensional mean vector in \""
        <<m_fname<<"\""<<endl;
  } else {
    cerr<<me<<":  invalid mean vector:  expecting a 1-dimensional nrrd,"
        <<" but \""<<m_fname<<"\" has "<<meanNrrd->dim<<" dimensions"<<endl;
    exit(1);
  }

  // Clean up unnecessary memory
  coeffNrrd=nrrdNix(coeffNrrd);
  nios=nrrdIoStateNix(nios);
  if (b_fname)
    delete [] b_fname;
  if (c_fname)
    delete [] c_fname;
  if (m_fname)
    delete [] m_fname;

  // Open raw coefficients file
  size_t len=strlen(c_fname);
  char* c_rawfname=new char[len-1];
  if (!c_rawfname) {
    cerr<<me<<":  error allocating "<<(len-1)*sizeof(char)
	<<" bytes for raw coefficients filename"<<endl;
    exit(1);
  }
  
  strncpy(c_rawfname, c_fname, len-4);
  sprintf(c_rawfname+(len-4), "raw");
  FILE* c_rawfile=fopen(c_rawfname, "r");
  if (!c_rawfile) {
    cerr<<me<<":  error opening raw coefficients file \""<<c_rawfname<<"\""<<endl;
    exit(1);
  }

  delete [] c_rawfname;

  // Open raw vector file
  len=strlen(out_bname);
  char* v_rawfname=new char[len+4];
  if (!v_rawfname) {
    cerr<<me<<":  error allocating "<<(len+4)*sizeof(char)
	<<" bytes for raw vector filename"<<endl;
    exit(1);
  }
  
  sprintf(v_rawfname, "%s.raw", out_bname);
  FILE* v_rawfile=fopen(v_rawfname, "w");
  if (!v_rawfile) {
    cerr<<me<<":  error opening raw vector file \""<<v_rawfname<<"\""<<endl;
    exit(1);
  }

  // Allocate memory for coefficient and input vectors
  unsigned char* coeff=new unsigned char[nbases];
  if (!coeff) {
    cerr<<me<<":  error allocating "<<nbases*sizeof(float)
	<<" bytes for coefficient vector"<<endl;
    exit(1);
  }

  float* r_vec=new float[ndims];
  if (!r_vec) {
    cerr<<me<<":  error allocating "<<ndims*sizeof(float)
	<<" bytes for reconstructed vector"<<endl;
    exit(1);
  }

  cout<<endl;
  cout<<"-----------------------"<<endl;
  cout<<"Reconstructing "<<nvectors<<" vectors from "
      <<nbases<<" basis vectors"<<endl;
  
  unsigned char* basis=(unsigned char*)(basisNrrd->data);
  unsigned char* mean=(unsigned char*)(meanNrrd->data);
  for (int v=0; v<nvectors; v++) {
    // Reset reconstructed vector
    for (int d=0; d<ndims; d++)
      r_vec[d]=0.0;

    // Read coefficients from raw file
    size_t nread=fread(coeff, sizeof(unsigned char), nbases, c_rawfile);
    if (nread<nbases) {
      cerr<<me<<":  error reading coefficients for vector["<<v<<"]:  expected "
	  <<nbases*sizeof(unsigned char)<<" bytes but only read "<<nread<<" bytes"
	  <<endl;
      exit(1);
    }

    // Multiply basis vectors by coefficients
    for (int d=0; d<ndims; d++)
      for (int b=0; b<nbases; b++)
        r_vec[d]+=(coeff[b]*coeff_diff+coeff_min)*
          (basis[b*ndims+d]*basis_diff+basis_min);
    
    // Add mean vector
    for (int d=0; d<ndims; d++)
      r_vec[d]+=mean[d];

    // Write reconstructed vector to raw file
    size_t nwritten=fwrite(r_vec, sizeof(float), ndims, v_rawfile);
    if (nwritten!=ndims) {
      cerr<<me<<":  error writing reconstructed vector["<<v<<"]"<<endl;
      exit(1);
    }
  }

  // Close raw files
  fclose(c_rawfile);
  fclose(v_rawfile);

  // Write nrrd header for reconstructed vectors
  if (strcmp(nrrd_ext, ".nrrd")!=0) {
    cerr<<me<<":  warning:  must save reconstructed vectors as "
	<<"a detatched nrrd"<<endl;
  }
    
  int size[2]={ndims, nvectors};
  char* label[2]={"\"width*height\"", "\"vector\""};
  if (writeNrrdHeader(v_rawfname, nrrdTypeFloat, 2, size, label)) {
    cerr<<me<<":  error writing nrrd header for reconstructed vectors"<<endl;
    exit(1);
  }

  cout<<"Done reconstructing vectors"<<endl;
  cout<<"-----------------------"<<endl;
  cout<<endl;

  // Clean up remaining memory
  basisNrrd=nrrdNuke(basisNrrd);
  meanNrrd=nrrdNuke(meanNrrd);
  delete [] coeff;
  delete [] r_vec;
  delete [] v_rawfname;

  return 0;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<":  unknown argument \""<<unknown<<"\""<<endl;

  cerr<<"usage:  "<<me<<" [options]"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -i <basename>   basename of PCA input files (\"pca\")"<<endl;
  cerr<<"  -b <filename>   filename of basis vectors (null)"<<endl;
  cerr<<"  -c <filename>   filename of coefficient matrix (null)"<<endl;
  cerr<<"  -m <filename>   filename of mean vector (null)"<<endl;
  cerr<<"  -o <basename>   basename of output file (\"recon\")"<<endl;
  cerr<<"  -nhdr           use \".nhdr\" extension for nrrd files (false)"<<endl;
  cerr<<"  --help          print this message and exit"<<endl;
  
  if (unknown)
    exit(1);
}

int writeNrrdHeader(char* raw_fname, int type, int dim, int* size,
		    char* label[]) {
  char* me="writeNrrdHeader";
  
  // Open header file
  size_t len=strlen(raw_fname);
  char* nhdr_fname=new char[len+1];
  if (!nhdr_fname) {
    cerr<<me<<":  error allocating "<<(len+1)*sizeof(char)
	<<" bytes for nrrd header filename"<<endl;
    return 1;
  }
  
  strncpy(nhdr_fname, raw_fname, len-3);
  sprintf(nhdr_fname+(len-3), "nhdr");
  FILE* nhdr_file=fopen(nhdr_fname, "w");
  if (!nhdr_file) {
    cerr<<me<<":  error opening nrrd header file \""<<nhdr_fname<<"\""<<endl;
    return 1;
  }
  
  // Write nrrd characteristics
  fprintf(nhdr_file, "NRRD0001\n");
  fprintf(nhdr_file, "type: ");
  switch(type) {
  case nrrdTypeChar: fprintf(nhdr_file, "char\n"); break;
  case nrrdTypeUChar: fprintf(nhdr_file, "unsigned char\n"); break;
  case nrrdTypeShort: fprintf(nhdr_file, "short\n"); break;
  case nrrdTypeUShort: fprintf(nhdr_file, "unsigned short\n"); break;
  case nrrdTypeInt: fprintf(nhdr_file, "int\n"); break;
  case nrrdTypeUInt: fprintf(nhdr_file, "unsigned int\n"); break;
  case nrrdTypeFloat: fprintf(nhdr_file, "float\n"); break;
  case nrrdTypeLLong: fprintf(nhdr_file, "long long\n"); break;
  case nrrdTypeULLong: fprintf(nhdr_file, "unsigned long long\n"); break;
  case nrrdTypeDouble: fprintf(nhdr_file, "double\n"); break;
  case nrrdTypeUnknown:
  default:
    cerr<<me<<":  error writing nrrd header:  unknown nrrd type"<<endl;
    return 1;
  }
  fprintf(nhdr_file, "dimension: %d\n", dim);
  fprintf(nhdr_file, "sizes: ");
  for (int d=0; d<dim; d++)
    fprintf(nhdr_file, "%d ", size[d]);
  fprintf(nhdr_file, "\n");
  fprintf(nhdr_file, "labels: ");
  for (int d=0; d<dim; d++)
    fprintf(nhdr_file, "%s ", label[d]);
  fprintf(nhdr_file, "\n");
  fprintf(nhdr_file, "data file: ./%s\n", raw_fname);
#ifdef __sgi
  fprintf(nhdr_file, "endian: big\n");
#else
  fprintf(nhdr_file, "endian: little\n");
#endif
  fprintf(nhdr_file, "encoding: raw\n");

  // Close nrrd header
  fclose(nhdr_file);

  cout<<"Wrote data to \""<<nhdr_fname<<"\""<<endl;
  
  delete [] nhdr_fname;
  
  return 0;
}
