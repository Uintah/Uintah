#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <teem/nrrd.h>

using namespace std;

// Forward declare necessary functions
void usage(char* me, const char* unknown=0);
Nrrd* convertNrrdToFloat(Nrrd* nin);
float* computeError(Nrrd* cb, Nrrd* idx, Nrrd* tex);

// Declare global variables
bool use_mse=true;
int verbose=0;

int main(int argc, char *argv[]) {
  char* me=argv[0];
  char* err;
  char* inbasename=0;
  char* outfilename=0;
  char* cbfilename=0;
  char* idxfilename=0;
  char* texfilename=0;
  char* nrrd_ext=".nhdr";

  // Parse arguments
  for(int i=1; i<argc; i++) {
    string arg(argv[i]);
    if (arg=="-i") {
      inbasename=argv[++i];
    } else if (arg=="-o") {
      outfilename=argv[++i];
    } else if (arg=="-cb") {
      cbfilename=argv[++i];
    } else if (arg=="-idx") {
      idxfilename=argv[++i];
    } else if (arg=="-tex") {
      texfilename=argv[++i];
    } else if (arg=="-inf_norm") {
      use_mse=false;
    } else if (arg=="-nrrd") {
      nrrd_ext=".nrrd";
    } else if (arg=="-v") {
      verbose=atoi(argv[++i]);
    } else if (arg=="--help") {
      usage(me);
      exit(0);
    } else {
      usage(me, arg.c_str());
    }
  }

  // Verify the arguments
  if (!inbasename) {
    bool error=false;
    if (!cbfilename) {
      cerr<<me<<":  filename of codebook textures not specified"<<endl;
      error=true;
    }

    if (!idxfilename) {
      cerr<<me<<":  filename of texture indices not specified"<<endl;
      error=true;
    }

    if (!texfilename) {
      cerr<<me<<":  filename of original textures not specified"<<endl;
      error=true;
    }
    
    if (error) {
      usage(me);
      exit(1);
    }
  } else {
    size_t in_len=strlen(inbasename);
    size_t ext_len=strlen(nrrd_ext);
    
    if (!cbfilename) {
      cbfilename=new char[in_len+3+ext_len];
      sprintf(cbfilename, "%s-cb%s", inbasename, nrrd_ext);
    }
    
    if (!idxfilename) {
      idxfilename=new char[in_len+4+ext_len];
      sprintf(idxfilename, "%s-idx%s", inbasename, nrrd_ext);
    }
    
    if (!texfilename) {
      texfilename=new char[in_len+4+ext_len];
      sprintf(texfilename, "%s-tex%s", inbasename, nrrd_ext);
    }
  }
  
  if (!outfilename) {
    cerr<<me<<":  output filename not specified"<<endl;
    usage(me);
    exit(1);
  }

  // Load the input files
  Nrrd* cb=nrrdNew();
  Nrrd* idx=nrrdNew();
  Nrrd* tex=nrrdNew();
  int E=0;
  if (!E) E|=nrrdLoad(cb, cbfilename, 0);
  if (!E) E|=nrrdLoad(idx, idxfilename, 0);
  if (!E) E|=nrrdLoad(tex, texfilename, 0);
  if (E) {
    err=biffGet(NRRD);
    cerr<<me<<":  error loading input files:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  // Sanity check
  if (cb->dim!=3 || tex->dim!=3) {
    cerr<<me<<":  codebook and original textures must have"
	<<" three dimensions (width, height, texture)"<<endl;
    exit(1);
  }
  
  if (cb->axis[0].size!=tex->axis[0].size ||
      cb->axis[1].size!=tex->axis[1].size) {
    cerr<<me<<":  size of codebook textures ("<<cb->axis[0].size<<", "
	<<cb->axis[1].size<<") is not equal to the size of original textures ("
	<<tex->axis[0].size<<", "<<tex->axis[1].size<<")"<<endl;
    exit(1);
  }

  if (idx->axis[0].size!=tex->axis[2].size) {
    cerr<<me<<":  number of texture indices ("<<idx->axis[0].size
	<<") is not equal to the number of original textures ("
	<<tex->axis[2].size<<")"<<endl;
    exit(1);
  }

  if (verbose) {
    cout<<"Loaded "<<cb->axis[2].size<<" codebook textures from "
	<<cbfilename<<endl;
    cout<<"Loaded "<<idx->axis[0].size<<" texture indices from "
	<<idxfilename<<endl;
    cout<<"Loaded "<<tex->axis[2].size<<" original textures from "
	<<texfilename<<endl;
  }
  
  // Convert textures to nrrdTypeFloat, if necessary
  cb=convertNrrdToFloat(cb);
  if (!cb) {
    cerr<<me<<":  error converting codebook textures to nrrdTypeFloat"<<endl;
    exit(1);
  }

  tex=convertNrrdToFloat(tex);
  if (!tex) {
    cerr<<me<<":  error converting original textures to nrrdTypeFloat"<<endl;
    exit(1);
  }

  // Compute error between quantized and original textures
  float* error=computeError(cb, idx, tex);
  if (!error) {
    cerr<<me<<":  error computing quantization error"<<endl;
    exit(1);
  }

  // Write the output file
  Nrrd* nout=nrrdNew();
  if (nrrdWrap(nout, error, nrrdTypeFloat, 1, tex->axis[2].size)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error creating error nrrd:  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }
  
  if (nrrdSave(outfilename, nout, 0)) {
    err=biffGet(NRRD);
    cerr<<me<<":  error saving to "<<outfilename<<":  "<<err<<endl;
    free(err);
    biffDone(NRRD);
    exit(1);
  }

  if (verbose)
    cout<<"Wrote data to "<<outfilename<<endl;
  
  // Clean up memory
  cb=nrrdNuke(cb);
  idx=nrrdNuke(idx);
  tex=nrrdNuke(tex);
  delete [] error;
  // XXX - don't know why, but this causes a seg fault
  // nout=nrrdNuke(nout);
  
  return 0;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<":  unrecognized option \""<<unknown<<"\""<<endl;
  
  // Print out the usage
  cerr<<"usage:  "<<me<<" [options] -o <filename>"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -i <basename>     basename of input files (null)"<<endl;
  cerr<<"  -cb <filename>    filename of codebook textures (null)"<<endl;
  cerr<<"  -idx <filename>   filename of texture indices (null)"<<endl;
  cerr<<"  -tex <filename>   filename of original textures (null)"<<endl;
  cerr<<"  -inf_norm         use infinity norm to calculate error (false)"<<endl;
  cerr<<"  -nrrd             use .nrrd extension (false)"<<endl;
  cerr<<"  -v <int>          set verbosity level (0)"<<endl;
  cerr<<"  --help            print this message and exit"<<endl;

  if (unknown)
    exit(1);
}

Nrrd* convertNrrdToFloat(Nrrd* nin) {
  char* me="convertNrrdToFloat";
  char* err;
  
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

float* computeError(Nrrd* cb, Nrrd* idx, Nrrd* tex) {
  char* me="computeError";

  // Compute quantization error
  if (verbose)
    cout<<"Computing quantization error"<<endl;
  
  // Set up useful variables
  int width=cb->axis[0].size;
  int height=cb->axis[1].size;
  int ntextures=tex->axis[2].size;
  int npixels=width*height;

  // Allocate error array
  float* error=new float[ntextures];
  if (!error) {
    cerr<<me<<":  error allocating memory for error array"<<endl;
    exit(1);
  }

  // Compute error between codeword and original
  float* cb_data=(float*)(cb->data);
  int* idx_data=(int*)(idx->data);
  float* tex_data=(float*)(tex->data);
  float max_error=-FLT_MAX;
  float total_error=0;
  for (int tex=0; tex<ntextures; tex++) {
    if (use_mse) {
      // Using mean squared error
      for (int y=0; y<height; y++) {
	for (int x=0; x<width; x++) {
	  float residual=tex_data[x] - cb_data[idx_data[tex]];
	  error[tex]+=residual*residual;
	}

	tex_data+=width;
      }

      error[tex]/=npixels;
    } else{
      // Using infinity norm
      error[tex]=-FLT_MAX;
      for (int y=0; y<height; y++) {
	for (int x=0; x<width; x++) {
	  float residual=fabsf(tex_data[x] - cb_data[idx_data[tex]]);
	  if (residual>error[tex])
	    error[tex]=residual;
	}

	tex_data+=width;
      }
    }

    if (error[tex]>max_error)
      max_error=error[tex];
    
    total_error+=error[tex];
  }

  if (verbose) {
    cout<<"Maximum error:  "<<max_error<<endl;
    cout<<"Mean error:  "<<total_error/ntextures<<endl;
  }

  return error;
}
