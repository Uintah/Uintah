#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <stdio.h>
#include <teem/nrrd.h>

using namespace std;

// Declare necessary functions
void usage(char* me, const char* unknown=0);

int main(int argc, char* argv[]) {
  char* me=argv[0];
  char* err;
  char* v_fname=0;
  char* pca_bname="pca";
  char* b_fname=0;
  char* c_fname=0;
  char* m_fname=0;
  char* out_fname="pca-error.nrrd";
  char* nrrd_ext=".nrrd";
  unsigned int nbases=0;
  unsigned int nvectors=0;
  unsigned int ndims=0;

  if (argc<3) {
    usage(me);
    exit(1);
  }

  for (int i=1; i<argc; i++) {
    string arg(argv[i]);
    if (arg=="-v") {
      v_fname=argv[++i];
    } else if (arg=="-pca") {
      pca_bname=argv[++i];
    } else if (arg=="-o") {
      out_fname=argv[++i];
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
  if (!v_fname) {
    cerr<<me<<":  filename of original vectors not specified"<<endl;
    exit(1);
  }

  size_t name_len=strlen(pca_bname);
  if (!b_fname) {
    b_fname=new char[name_len+11];
    if (!b_fname) {
      cerr<<me<<":  error allocating "<<(name_len+13)*sizeof(char)
          <<" bytes for basis vector filename"<<endl;
      exit(1);
    }
    
    sprintf(b_fname, "%s-basis-q%s", pca_bname, nrrd_ext);
  }
  
  if (!c_fname) {
    c_fname=new char[name_len+11];
    if (!c_fname) {
      cerr<<me<<":  error allocating "<<(name_len+13)*sizeof(char)
          <<" bytes for coefficient matrix filename"<<endl;
      exit(1);
    }
    
    sprintf(c_fname, "%s-coeff-q%s", pca_bname, nrrd_ext);
  }
  
  if (!m_fname) {
    m_fname=new char[name_len+10];
    if (!m_fname) {
      cerr<<me<<":  error allocating "<<(name_len+12)*sizeof(char)
          <<" bytes for coefficient matrix filename"<<endl;
      exit(1);
    }
    
    sprintf(m_fname, "%s-mean-q%s", pca_bname, nrrd_ext);
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
  NrrdIoState* coeffNios=nrrdIoStateNew();
  coeffNios->skipData=1;
  if (nrrdLoad(coeffNrrd, c_fname, coeffNios)) {
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
  if (vecNrrd->dim==3) {
    unsigned int width=vecNrrd->axis[0].size;
    unsigned int height=vecNrrd->axis[1].size;
    ndims=width*height;
    nvectors=vecNrrd->axis[2].size;

    cout<<"Found "<<nvectors<<" "<<ndims<<"-dimensional vectors in \""
        <<v_fname<<"\""<<endl;
  } else {
    cerr<<me<<":  invalid vectors:  expecting a 3-dimensional nrrd,"
        <<" but \""<<v_fname<<"\" has "<<vecNrrd->dim<<" dimensions"<<endl;
    exit(1);
  }

  float basis_min=FLT_MAX;
  float basis_max=-FLT_MAX;
  float basis_diff=0;
  float coeff_min=FLT_MAX;
  float coeff_max=-FLT_MAX;
  float coeff_diff=0;
  if (basisNrrd->dim==2) {  
    if (basisNrrd->axis[0].size!=ndims) {
      cerr<<me<<":  invalid basis vectors:  number of dimensions ("
          <<basisNrrd->axis[0].size<<") is not correct; expecting "
          <<ndims<<"-dimensional basis vectors"<<endl;
      exit(1);
    }

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

    if (coeffNrrd->axis[1].size!=nvectors) {
      cerr<<me<<":  invalid coefficient matrix:  number of vectors ("
          <<coeffNrrd->axis[1].size<<") is not correct; expecting "
          <<nvectors<<" vectors"<<endl;
      exit(1);
    }

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
  vecNrrd=nrrdNix(vecNrrd);
  vecNios=nrrdIoStateNix(vecNios);
  coeffNrrd=nrrdNix(coeffNrrd);
  coeffNios=nrrdIoStateNix(coeffNios);
  if (b_fname)
    delete [] b_fname;
  if (c_fname)
    delete [] c_fname;
  if (m_fname)
    delete [] m_fname;

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

  // Open raw coefficients file
  len=strlen(c_fname);
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

  // Allocate memory for input vector, coefficients
  // and reconstructed vector
  unsigned char* in_vec=new unsigned char[ndims];
  if (!in_vec) {
    cerr<<me<<":  error allocating "<<ndims*sizeof(unsigned char)
        <<" bytes for input vector"<<endl;
    exit(1);
  }

  unsigned char* coeff=new unsigned char[nbases];
  if (!coeff) {
    cerr<<me<<":  error allocating "<<nbases*sizeof(unsigned char)
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
  
  unsigned char* basis=(unsigned char*)(basisNrrd->data);
  unsigned char* mean=(unsigned char*)(meanNrrd->data);
  for (int v=0; v<nvectors; v++) {
    // Read original vector from raw file
    size_t nread=fread(in_vec, sizeof(unsigned char), ndims, v_rawfile);
    if (nread<ndims) {
      cerr<<me<<":  error reading elements of vector["<<v<<"]:  expected "
          <<ndims*sizeof(unsigned char)<<" bytes but only read "<<nread<<" bytes"
          <<endl;
      exit(1);
    }
    
    // Reset reconstructed vector
    for (int d=0; d<ndims; d++)
      r_vec[d]=0.0;
    
    // Read coefficients from raw file
    nread=fread(coeff, sizeof(unsigned char), nbases, c_rawfile);
    if (nread<nbases) {
      cerr<<me<<":  error reading coefficients for vector["<<v<<"]:  expected "
          <<nbases*sizeof(unsigned char)<<" bytes but only read "<<nread<<" bytes"
          <<endl;
      exit(1);
    }

#if 0
    if (v<10) {
      cout<<"coeff["<<v<<"]=( ";
      for (int i=0; i<16; i++)
        cout<<(float)coeff[i]<<" ";
      cout<<"... )"<<endl;
      
      cout<<"mean["<<v<<"]=( ";
      for (int i=0; i<10; i++)
        cout<<(float)mean[i]<<" ";
      cout<<"... )"<<endl;
    }
#endif

    // Multiply basis vectors by coefficients
    for (int d=0; d<ndims; d++)
      for (int b=0; b<nbases; b++)
        r_vec[d]+=(coeff[b]*coeff_diff+coeff_min)*
          (basis[b*ndims+d]*basis_diff+basis_min);
    
    // Add mean vector
    for (int d=0; d<ndims; d++)
      r_vec[d]+=mean[d];

#if 0
    if (v<10) {
      cout<<"in_vec["<<v<<"]=( ";
      for (int i=0; i<10; i++)
        cout<<(float)in_vec[i]<<" ";
      cout<<"... )"<<endl;
      cout<<endl;
      
      cout<<"r_vec["<<v<<"]=( ";
      for (int i=0; i<10; i++)
        cout<<r_vec[i]<<" ";
      cout<<"... )"<<endl;
    }
#endif

    // Compute distance between the two vectors
    for (int d=0; d<ndims; d++) {
      float residual=r_vec[d] - (float)in_vec[d];
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
  fclose(c_rawfile);
      
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
  
  // Clean up remaining memory
  basisNrrd=nrrdNuke(basisNrrd);
  meanNrrd=nrrdNuke(meanNrrd);
  delete [] in_vec;
  delete [] coeff;
  delete [] r_vec;
  nout=nrrdNuke(nout);

  return 0;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<":  unknown argument \""<<unknown<<"\""<<endl;

  cerr<<"usage:  "<<me<<" [options] -v <filename>"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -pca <basename>   basename of PCA input files (\"pca\")"<<endl;
  cerr<<"  -b <filename>     filename of basis vectors (null)"<<endl;
  cerr<<"  -c <filename>     filename of coefficient matrix (null)"<<endl;
  cerr<<"  -m <filename>     filename of mean vector (null)"<<endl;
  cerr<<"  -o <filename>     filename of output file (null)"<<endl;
  cerr<<"  -nhdr             use \".nhdr\" extension for nrrd files (false)"<<endl;
  cerr<<"  --help            print this message and exit"<<endl;
  
  if (unknown)
    exit(1);
}
