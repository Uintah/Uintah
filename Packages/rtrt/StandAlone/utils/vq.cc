// #ifdef PEDANTIC
// #  undef PEDANTIC
// #endif
// #define PEDANTIC 1

#ifdef CHECK_VERBOSITY
#  undef CHECK_VERBOSITY
#endif
#define CHECK_VERBOSITY 1

#ifdef PARTIAL_DISTORTION
#  undef PARTIAL_DISTORTION
#endif
#define PARTIAL_DISTORTION 1

// #ifdef DOUBLE_TEST
// #  undef DOUBLE_TEST
// #endif
// #define DOUBLE_TEST 1

#ifdef COLLECT_STATS
#  undef COLLECT_STATS
#endif
#define COLLECT_STATS 1

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

#ifdef PARTIAL_DISTORTION
#ifdef DOUBLE_TEST
char *search_type="double test (with partial distortion)";
#else
char *search_type="standard test (with partial distortion)";
#endif
#else
#ifdef DOUBLE_TEST
char *search_type="double test";
#else
char *search_type="standard test";
#endif
#endif

int main(int argc, char *argv[]) {
  char *infilename=0;
  char *cw_outfilename=0;
  char *idx_outfilename=0;
  int num_dims=2;
  int num_vecs=10;
  int num_cwords=2;
  float thresh=1e-5;
  int max_iteration = 100;
  int seed=12;
#ifdef COLLECT_STATS
  bool stats=false;
#endif
  int verbosity=0;

  for (int i=1;i<argc;i++) {
    if (strcmp(argv[i], "-i")==0) {
      infilename=argv[++i];
    } else if (strcmp(argv[i], "-cw_out")==0) {
      cw_outfilename=argv[++i];
    } else if (strcmp(argv[i], "-idx_out")==0) {
      idx_outfilename=argv[++i];
    } else if (strcmp(argv[i], "-ndims")==0) {
      num_dims=atoi(argv[++i]);
    } else if (strcmp(argv[i], "-nvecs")==0) {
      num_vecs=atoi(argv[++i]);
    } else if (strcmp(argv[i], "-ncwords")==0) {
      num_cwords=atoi(argv[++i]);
    } else if (strcmp(argv[i], "-thresh")==0) {
      thresh=atof(argv[++i]);
    } else if (strcmp(argv[i], "-seed")==0) {
      seed=atoi(argv[++i]);
    } else if (strcmp(argv[i], "-niters")==0) {
      max_iteration=atoi(argv[++i]);
#ifdef COLLECT_STATS
    } else if (strcmp(argv[i], "-stats")==0) {
      stats=true;
#endif
    } else if (strcmp(argv[i], "-v")==0) {
      verbosity=atoi(argv[++i]);
    } else if (strcmp(argv[i], "--search")==0) {
      cout<<"using "<<search_type<<" for codebook search"<<endl;
    } else {
      if (strcmp(argv[i], "--help")!=0)
	cerr<<"unrecognized option \""<<argv[i]<<"\""<<endl;
      cerr<<"usage:  vq [options]"<<endl;
      cerr<<"  options:"<<endl;
      cerr<<"    -i <filename>         input filename (null)"<<endl;
      cerr<<"    -cw_out <filename>    write codewords to file (null)"<<endl;
      cerr<<"    -idx_out <filename>   write vector indices to file (null)"<<endl;
      cerr<<"    -ndims <int>          dimensionality of the vectors (2)"<<endl;
      cerr<<"    -nvecs <int>          number of vectors to quantize (10)"<<endl;
      cerr<<"    -ncwords <int>        number of codewords (2)"<<endl;
      cerr<<"    -thresh <float>       termination threshold (1e-5)"<<endl;
      cerr<<"    -niters <int>         maximum number of iterations (100)"<<endl;
      cerr<<"    -seed <int>           random number seed (12)"<<endl;
#ifdef COLLECT_STATS
      cerr<<"    -stats                report statistics (false)"<<endl;
#endif      
      cerr<<"    -v <int>              verbosity level (0)"<<endl;
      cerr<<"    --search              print codebook search method"<<endl;
      cerr<<"    --help                print this message and exit"<<endl;
      exit(1);
    }
  }

  // forward declare data structures for vectors
  float* vec;
  int* cluster_idx;
  int* ir_idx;

  // allocate memory for codewords
  float* prev_cw=new float[num_cwords*num_dims];
  float* current_cw=new float[num_cwords*num_dims];

  // seed the RNGs
  srand48(seed);
  srand(seed);

  int in_fd=-1;
  if (infilename) {
    // read vectors from a file
    in_fd=open(infilename, O_RDONLY);
    if (in_fd==-1) {
      cerr<<"failed to open file \""<<infilename<<"\" for reading"<<endl;
      exit(1);
    }
    
    struct stat statbuf;
    if (fstat(in_fd, &statbuf)==-1) {
      cerr<<"cannot stat file \""<<infilename<<"\""<<endl;
      exit(1);
    }
    
    num_vecs=(int)(statbuf.st_size/(num_dims*sizeof(float)));
  }

  if (num_cwords>num_vecs) {
    cerr<<"number of codewords ("<<num_cwords<<") is greater than the "
	<<"number of vectors ("<<num_vecs<<")"<<endl;
    exit(1);
  }
  
  // allocate the memory for vector structures
  vec=new float[num_vecs*num_dims];
  cluster_idx=new int[num_vecs];
  ir_idx=new int[num_vecs];

  if (in_fd>-1) {
    // slurp up the vector data
    float* data=&(vec[0]);
    size_t data_size=num_vecs*num_dims*sizeof(float);

    if (verbosity>4) {
      cout<<"slurping vector data ("<<num_vecs<<" vectors = " <<data_size
	  <<" bytes) from "<<infilename<<endl;
    }
    size_t num_read=0;
    num_read=read(in_fd, data, data_size);
    if(num_read==-1 || num_read!=data_size) {
      cerr<<"did not read "<<data_size<<" bytes from "
	  <<infilename<<endl;
      exit(1);
    }

    // close the file
    close(in_fd);
  }
  else {
    // create random vectors
    for (int i=0;i<num_vecs;i++) {
      for (int j=0;j<num_dims;j++) {
	vec[i*num_dims+j]=drand48();
      }
    }
  }

  // output the vectors
  if (verbosity>9) {
    cout<<"Input vectors"<<endl;
    cout<<"-----------------------"<<endl;
    for (int i=0;i<num_vecs;i++) {
      cout<<"vec["<<i<<"] = (";
      for (int j=0;j<num_dims-1;j++)
	cout<<vec[i*num_dims+j]<<", ";
      cout<<vec[i*num_dims+num_dims-1]<<")"<<endl;
    }
    cout<<endl;
  }
  
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

  // output the initial codewords
  if (verbosity>9) {
    cout<<"Initial codewords"<<endl;
    cout<<"-----------------------"<<endl;    
    for (int i=0;i<num_cwords;i++) {
      cout<<"codeword["<<i<<"] = (";
      for (int j=0;j<num_dims-1;j++)
	cout<<current_cw[i*num_dims+j]<<", ";
      cout<<current_cw[i*num_dims+num_dims-1]<<")"<<endl;
    }
    cout<<endl;
  }

#ifdef DOUBLE_TEST  
  // precompute the maximum component of each vector and
  // twice the sum of each vector's components
  float* vec_max=new float[num_vecs];
  float* vec_2sum=new float[num_vecs];
  for (int i=0;i<num_vecs;i++)
    vec_max[i]=0;
  
  for (int i=0;i<num_vecs;i++) {
    float tmp=0;
    for (int j=0;j<num_dims;j++) {
      if (vec[i*num_dims+j]>vec_max[i])
	vec_max[i]=vec[i*num_dims+j];
      
      tmp+=vec[i*num_dims+j];
    }
    
    vec_2sum[i]=2*tmp;
  }
#endif  
  
  // cluster the vectors around the codewords
  bool done=false;
  bool first=true;
  unsigned int iteration=0;
  float* tmp_mean=new float[num_dims];
#ifdef DOUBLE_TEST  
  float* codeword_mag=new float[num_cwords];
  float* codeword_max=new float[num_cwords];
  float* codeword_2sum=new float[num_cwords];
#endif
#ifdef COLLECT_STATS
  unsigned long real_distortion_cnt=0;
  unsigned long partial_distortion_cnt=0;
#ifdef DOUBLE_TEST  
  unsigned long early_exit_d1=0;
  unsigned long early_exit_d2=0;
#endif
#endif
  do {
#ifdef CHECK_VERBOSITY
    // output a status message
    if (verbosity>4)
      cout<<"Beginning iteration "<<iteration<<endl;
#endif
    
#ifdef DOUBLE_TEST    
    // compute the magnitude and maximum component of each codeword, and
    // twice the sum of each codeword's components
    for (int i=0;i<num_cwords;i++)
      codeword_max[i]=0;

    for (int i=0;i<num_cwords;i++) {
      float tmp=0;
      for (int j=0;j<num_dims;j++) {
        if (current_cw[i*num_dims+j]>codeword_max[i])
          codeword_max[i]=current_cw[i*num_dims+j];
	
	tmp+=current_cw[i*num_dims+j];
      }

      codeword_mag[i]=sqrt(tmp);
      codeword_2sum[i]=2*tmp;
    }
#endif    
    
    for (int i=0;i<num_vecs;i++) {
      float min=FLT_MAX;
      int my_cluster=-1;
      for (int j=0;j<num_cwords;j++) {
	float distance=0;
#ifdef DOUBLE_TEST	
	// compute the first distortion
	distance=codeword_mag[j]-vec_max[i]*codeword_2sum[j];
	if (distance>min) {
#ifdef COLLECT_STATS	  
	  early_exit_d1++;
#endif
	  continue;
	}

	// compute the second distortion
	distance=codeword_mag[j]-codeword_max[j]*vec_2sum[i];
	if (distance>min) {
#ifdef COLLECT_STATS	  
	  early_exit_d2++;
#endif
	  continue;
	}
#endif	

	// compute the real distortion
#ifdef COLLECT_STATS
	real_distortion_cnt++;
#endif
	distance=0;
	for (int k=0;k<num_dims;k++) {
	  float tmp=vec[i*num_dims+k]-current_cw[j*num_dims+k];
	  distance+=tmp*tmp;
#ifdef PARTIAL_DISTORTION
	  if (distance>min) {
#ifdef COLLECT_STATS
	    partial_distortion_cnt+=num_dims-k;
#endif
	    break;
	  }
#endif
        }
	if (distance>min)
	  continue;
	
	// set the cluster for the vector
	min=distance;
	my_cluster=j;
	
#ifdef PEDANTIC	
        if (verbosity > 4 && min==0 && iteration==0)
          cout<<"found initial codeword for cluster["<<j<<"]"<<endl;
#endif	
      }

#ifdef PEDANTIC      
      if (my_cluster>=0) {
        cluster_idx[i]=my_cluster;
        if (verbosity > 4)
          cout << "placing vector["<<i<<"] in cluster["<<cluster_idx[i]
	       <<"]"<<endl;
      } else {
	cerr << "couldn't determine a cluster for vector["<<i<<"]"<<endl;
	exit(2);
      }
#else
      cluster_idx[i]=my_cluster;
#endif
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

#ifdef PEDANTIC      
      // sanity check
      if (iteration == 0 && !count) {
	cerr<<"cluster["<<i<<"] is empty in iteration 0"<<endl;
	exit(2);
      }
#endif      
      
      // update codewords
      for (int k=0;k<num_dims;k++) {
	prev_cw[i*num_dims+k]=current_cw[i*num_dims+k];
	current_cw[i*num_dims+k]=tmp_mean[k]/(float)count;
	
	// compute the difference between iterations
	if (!first) {
	  done=true;
	  done&=((current_cw[i*num_dims+k]-prev_cw[i*num_dims+k])<thresh);
	} else {
	  first=false;
	  done=false;
	}
      }
    }

#ifdef CHECK_VERBOSITY    
    // output the codewords for the next iteration
    if (verbosity>4) {
      cout<<"Codewords (iteration="<<iteration<<")"<<endl;
      cout<<"-----------------------"<<endl;
      for (int i=0;i<num_cwords;i++) {
	cout<<"codeword["<<i<<"] = (";
	for (int j=0;j<num_dims-1;j++)
	  cout<<current_cw[i*num_dims+j]<<", ";
	cout<<current_cw[i*num_dims+num_dims-1]<<")"<<endl;
      }
      cout<<endl;
    }
#endif
    
    iteration++;
  } while (!done && iteration < max_iteration);

#ifdef COLLECT_STATS
  // output early exit statistics
  if (stats) {
    cout<<"----------------------------------------------------"<<endl;
#ifdef DOUBLE_TEST
    cout<<"           First distortion early exits = "<<early_exit_d1<<endl;
    cout<<"          Second distortion early exits = "<<early_exit_d2<<endl;
    cout<<"                      Total early exits = "
	<<(early_exit_d1+early_exit_d2)<<endl;
    cout<<endl;
#endif
    cout<<"           Real distortion computations = "
	<<real_distortion_cnt<<endl;
#ifdef PARTIAL_DISTORTION
    cout<<" Iterations saved by partial distortion = "
	<<partial_distortion_cnt<<endl;
#endif
    cout<<"----------------------------------------------------"<<endl;
    cout<<endl;
  }
#endif
  
  // output the final codewords
  if (verbosity>9) {
    cout<<"Codewords (after "<<iteration<<" iterations)";
    cout<<endl;
    cout<<"-----------------------"<<endl;
    for (int i=0;i<num_cwords;i++) {
      cout<<"codeword["<<i<<"] = (";
      for (int j=0;j<num_dims-1;j++)
	cout<<current_cw[i*num_dims+j]<<", ";
      cout<<current_cw[i*num_dims+num_dims-1]<<")"<<endl;
    }
  }

  // write out codebook
  if (cw_outfilename) {
    FILE *cw_outfile=fopen(cw_outfilename, "wb");
    if (cw_outfile) {
      unsigned long num_bytes=fwrite(current_cw, sizeof(float),
				     num_cwords*num_dims, cw_outfile);
      if (num_bytes!=num_cwords*num_dims) {
	cerr<<"couldn't write all "<<num_cwords<<" codewords"<<endl;
      } else {
	cout<<"wrote "<<num_cwords<<" codewords to \""<<cw_outfilename
	    <<"\""<<endl;
      }
      
      // close the file
      fclose(cw_outfile);
    }
  }

  // write out cluster indices
  if (idx_outfilename) {
    FILE *idx_outfile=fopen(idx_outfilename, "wb");
    if (idx_outfile) {
      unsigned long num_bytes=fwrite(cluster_idx, sizeof(int),
				     num_vecs, idx_outfile);
      if (num_bytes!=num_vecs) {
	cerr<<"couldn't write all "<<num_vecs<<" indices"<<endl;
      } else {
	cout<<"wrote "<<num_vecs<<" indices to \""<<idx_outfilename
	    <<"\""<<endl;
      }
      
      // close the file
      fclose(idx_outfile);
    }
  }
  
  // delete allocated memory
  delete vec;
  delete cluster_idx;
  delete ir_idx;
  delete prev_cw;
  delete current_cw;
  delete tmp_mean;
#ifdef DOUBLE_TEST  
  delete vec_max;
  delete vec_2sum;
  delete codeword_mag;
  delete codeword_max;
  delete codeword_2sum;
#endif
  
  return 0;
}
