#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CatmullRomSpline.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/VolumeVis.h>
#include <Packages/rtrt/Core/HVolumeVis.h>
#include <Packages/rtrt/Core/VolumeVisDpy.h>
//#include <Packages/rtrt/Core/CutPlane.h>
//#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/RegularColorMap.h>
#include <Packages/rtrt/Core/HVolumeVis.h>
#include <Packages/rtrt/Core/GridSpheres.h>
#include <Packages/rtrt/Core/GridSpheresDpy.h>

#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>

#include <teem/nrrd.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <math.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sci_values.h>

using namespace std;
using namespace rtrt;
using SCIRun::Thread;
using SCIRun::Semaphore;

// Whether or not to use the HVolumeVis code
static int np = 1;
static int depth = 2;

static Point minPin;
static Point maxPin;
static bool use_global_minmax = false;

void get_material(Array1<Color> &matls, Array1<AlphaPos> &alphas) {

  matls.add(Color(0,0,1));
  matls.add(Color(0,0.4,1));
  matls.add(Color(0,0.8,1));
  matls.add(Color(0,1,0.8));
  matls.add(Color(0,1,0.4));
  matls.add(Color(0,1,0));
  matls.add(Color(0.4,1,0));
  matls.add(Color(0.8,1,0));
  matls.add(Color(1,0.9176,0));
  matls.add(Color(1,0.8,0));
  matls.add(Color(1,0.4,0));
  matls.add(Color(1,0,0));

  alphas.add(AlphaPos(0       , 0));  // pos 0
  alphas.add(AlphaPos(28.0/255, 0));  // pos 28
  alphas.add(AlphaPos(64.0/255, 0.1));// pos 64
  alphas.add(AlphaPos(100.0/255,0));  // pos 100
  alphas.add(AlphaPos(156.0/255,0));  // pos 156
  alphas.add(AlphaPos(192.0/255,1));  // pos 192
  alphas.add(AlphaPos(228.0/255,0));  // pos 192 
  alphas.add(AlphaPos(1,        0));  // pos 192
}

void get_material2(Array1<Color> &matls, Array1<AlphaPos> &alphas) {

  float div = 1.0/255;
  matls.add(Color(0, 0, 0) * div);
  matls.add(Color(52, 0, 0) * div);
  matls.add(Color(102, 2, 0) * div);   
  matls.add(Color(153, 18, 0) * div);
  matls.add(Color(200, 41, 0) * div);   
  matls.add(Color(230, 71, 0) * div);
  matls.add(Color(255, 120, 0) * div);   
  matls.add(Color(255, 163, 20) * div);
  matls.add(Color(255, 204, 55) * div);   
  matls.add(Color(255, 228, 80) * div);
  matls.add(Color(255, 247, 120) * div);   
  matls.add(Color(255, 255, 180) * div);
  matls.add(Color(255, 255, 255) * div);


  alphas.add(AlphaPos(0       , 0));  // pos 0
  alphas.add(AlphaPos(0.109804, 0));  
  alphas.add(AlphaPos(0.328571, 0.216667));  
  alphas.add(AlphaPos(0.618367, 0.3375));  
  alphas.add(AlphaPos(1,        0.5));  
}

#if 0
template<DataType>
VolumeVisBase *get_data() {
  BrickArray3<DataType> data;
  DataType data_min = TypeInfo<DataType>::get_min();
  DataType data_max = TypeInfo<DataType>::get_max();
}
#endif

int get_material_nrrd(char * filename,
		       Array1<Color> &matls, Array1<AlphaPos> &alphas) {
  // Open the nrrd
  Nrrd *nrrd = nrrdNew();
  // load the nrrd in
  if (nrrdLoad(nrrd,filename,0)) {
    char *err = biffGet(NRRD);
    cerr << "Error reading nrrd "<< filename <<": "<<err<<"\n";
    free(err);
    biffDone(NRRD);
    nrrdNuke(nrrd);
    return 1;
  }

  // The nrrd needs to be of type float or double
  if (nrrd->type != nrrdTypeFloat) {
    if (nrrd->type != nrrdTypeDouble) {
      cerr << "Wrong type: \n";
      nrrdNuke(nrrd);
      return 1;
    } else {
      // convert it to float, since it will be later anyway.
      Nrrd *new_nrrd = nrrdNew();
      nrrdConvert(new_nrrd, nrrd, nrrdTypeFloat);
      // since the data was copied blow away the memory for the old nrrd
      nrrdNuke(nrrd);
      nrrd = new_nrrd;
    }
  }  

  // We need 2 dimentions
  // 1. For the r,g,b,a,pos
  // 2. For the transfer function element
  if (nrrd->dim != 2) {
    cerr << "Wrong number of dimenstions: "<<nrrd->dim<<"\n";
    nrrdNuke(nrrd);
    return 1;
  }

  // We need axis[0] to have size of 5
  if (nrrd->axis[0].size != 5) {
    cerr << "Axis[0] must have size of 5\n";
    nrrdNuke(nrrd);
    return 1;
  }

  // Ok, now we loop over the data and get ourselves a colormap.
  float *data = (float*)(nrrd->data);
  for(int i = 0; i < nrrd->axis[1].size; i++) {
    float r,g,b,a,pos;
    r = AIR_CLAMP(0,*data,1); data++;
    g = AIR_CLAMP(0,*data,1); data++;
    b = AIR_CLAMP(0,*data,1); data++;
    a = AIR_CLAMP(0,*data,1); data++;
    pos = AIR_CLAMP(0,*data,1); data++;
    cout << "color = ("<<r<<", "<<g<<", "<<b<<")\n";
    matls.add(Color(r,g,b));
    cout << "a = "<<a<<", pos = "<<pos<<endl;
    alphas.add(AlphaPos(pos,a));
  }

  nrrdNuke(nrrd);
  return 0;
}

int get_alpha_nrrd(char * filename, Array1<AlphaPos> &alphas) {
  // Open the nrrd
  Nrrd *nrrd = nrrdNew();
  // load the nrrd in
  if (nrrdLoad(nrrd,filename,0)) {
    char *err = biffGet(NRRD);
    cerr << "Error reading nrrd "<< filename <<": "<<err<<"\n";
    free(err);
    biffDone(NRRD);
    nrrdNuke(nrrd);
    return 1;
  }

  // The nrrd needs to be of type float or double
  if (nrrd->type != nrrdTypeFloat) {
    if (nrrd->type != nrrdTypeDouble) {
      cerr << "Wrong type: \n";
      nrrdNuke(nrrd);
      return 1;
    } else {
      // convert it to float, since it will be later anyway.
      Nrrd *new_nrrd = nrrdNew();
      nrrdConvert(new_nrrd, nrrd, nrrdTypeFloat);
      // since the data was copied blow away the memory for the old nrrd
      nrrdNuke(nrrd);
      nrrd = new_nrrd;
    }
  }  

  // We need 2 dimentions
  // 1. For the r,g,b,a,pos
  // 2. For the transfer function element
  if (nrrd->dim != 2) {
    cerr << "Wrong number of dimenstions: "<<nrrd->dim<<"\n";
    nrrdNuke(nrrd);
    return 1;
  }

  // We need axis[0] to have size of 2
  if (nrrd->axis[0].size != 2) {
    cerr << "Axis[0] must have size of 2\n";
    nrrdNuke(nrrd);
    return 1;
  }

  // Reset the alpha values
  alphas.remove_all();
  
  // Ok, now we loop over the data and get ourselves a colormap.
  float *data = (float*)(nrrd->data);
  for(int i = 0; i < nrrd->axis[1].size; i++) {
    float a, pos;
    a = AIR_CLAMP(0,*data,1); data++;
    pos = AIR_CLAMP(0,*data,1); data++;
    cout << "a = "<<a<<", pos = "<<pos<<endl;
    alphas.add(AlphaPos(pos,a));
  }

  nrrdNuke(nrrd);
  return 0;
}

HVolumeVis<float,VMCell<float> > *create_volume_from_nrrd(char *filename,
				   bool override_data_min, double data_min_in,
				   bool override_data_max, double data_max_in,
				   double spec_coeff, double ambient,
				   double diffuse, double specular,
				   VolumeVisDpy *dpy)
{
  BrickArray3<float> data;
  float data_min = FLT_MAX;
  float data_max = -FLT_MAX;
  Point minP, maxP;
  // Do the nrrd stuff
  Nrrd *n = nrrdNew();
  // load the nrrd in
  if (nrrdLoad(n,filename,0)) {
    char *err = biffGet(NRRD);
    cerr << "Error reading nrrd "<< filename <<": "<<err<<"\n";
    free(err);
    biffDone(NRRD);
    return 0;
  }
  // check to make sure the dimensions are good
  if (n->dim != 3) {
    cerr << "VolumeVisMod error: nrrd->dim="<<n->dim<<"\n";
    cerr << "  Can only deal with 3-dimensional scalar fields... sorry.\n";
    return 0;
  }
  // convert the type to floats if you need to
  size_t num_elements = nrrdElementNumber(n);
  cerr << "Number of data members = " << num_elements << endl;
  if (n->type != nrrdTypeFloat) {
    cerr << "Converting type from ";
    switch(n->type) {
    case nrrdTypeUnknown: cerr << "nrrdTypeUnknown"; break;
    case nrrdTypeChar: cerr << "nrrdTypeChar"; break;
    case nrrdTypeUChar: cerr << "nrrdTypeUChar"; break;
    case nrrdTypeShort: cerr << "nrrdTypeShort"; break;
    case nrrdTypeUShort: cerr << "nrrdTypeUShort"; break;
    case nrrdTypeInt: cerr << "nrrdTypeInt"; break;
    case nrrdTypeUInt: cerr << "nrrdTypeUInt"; break;
    case nrrdTypeLLong: cerr << "nrrdTypeLLong"; break;
    case nrrdTypeULLong: cerr << "nrrdTypeULLong"; break;
    case nrrdTypeDouble: cerr << "nrrdTypeDouble"; break;
    default: cerr << "Unknown!!";
    }
    cerr << " to nrrdTypeFloat\n";
    Nrrd *new_n = nrrdNew();
    nrrdConvert(new_n, n, nrrdTypeFloat);
    // since the data was copied blow away the memory for the old nrrd
    nrrdNuke(n);
    n = new_n;
    cerr << "Number of data members = " << num_elements << endl;
  }
  // get the dimensions
  int nx, ny, nz;
  nx = n->axis[0].size;
  ny = n->axis[1].size;
  nz = n->axis[2].size;
  cout << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
  cout << "total = " << nz * ny * nz << endl;
  cout << "spacing = " << n->axis[0].spacing << " x "<<n->axis[1].spacing<< " x "<<n->axis[2].spacing<< endl;
  for (int i = 0; i<n->dim; i++)
    if (!(AIR_EXISTS_D(n->axis[i].spacing))) {
      cout <<"spacing for axis "<<i<<" does not exist.  Setting to 1.\n";
      n->axis[i].spacing = 1;
    }
  data.resize(nx,ny,nz); // resize the bricked data
  if (!use_global_minmax) {
    // get the physical bounds
    minP = Point(0,0,0);
    maxP = Point((nx - 1) * n->axis[0].spacing,
		 (ny - 1) * n->axis[1].spacing,
		 (nz - 1) * n->axis[2].spacing);
    // lets normalize the dimensions to 1
    Vector size = maxP - minP;
    // find the biggest dimension
    double max_dim = Max(Max(size.x(),size.y()),size.z());
    maxP = ((maxP-minP)/max_dim).asPoint();
    minP = Point(0,0,0);
  } else {
    minP = minPin;
    maxP = maxPin;
  }
  // copy the data into the brickArray
  cerr << "Number of data members = " << num_elements << endl;
  float *p = (float*)n->data; // get the pointer to the raw data
  for (int z = 0; z < nz; z++)
    for (int y = 0; y < ny; y++)
      for (int x = 0; x < nx; x++) {
	float val = *p++;
	data(x,y,z) = val;
	// also find the min and max
	if (val < data_min)
	  data_min = val;
	else if (val > data_max)
	  data_max = val;
      }
#if 0
  // compute the min and max of the data
  double dmin,dmax;
  nrrdMinMaxFind(&dmin,&dmax,n);
  data_min = (float)dmin;
  data_max = (float)dmax;
#endif
  // delete the memory that is no longer in use
  nrrdNuke(n);


  // override the min and max if it was passed in
  if (override_data_min)
    data_min = data_min_in;
  if (override_data_max)
    data_max = data_max_in;

  cout << "minP = "<<minP<<", maxP = "<<maxP<<endl;

  return new HVolumeVis<float,VMCell<float> >(data, data_min, data_max,
                                              depth, minP, maxP, dpy,
                                              spec_coeff, ambient, diffuse,
                                              specular, np);
}  

GridSpheres* read_spheres(char* spherefile, int datanum,
			  int gridcellsize, int griddepth,
			  float radius_in, float radius_factor,
			  int numvars, RegularColorMap* cmap)
{
  double time=SCIRun::Time::currentSeconds();

  //------------------------------------
  // open the header file
  bool found_header = false;
  
  char buf[300];
  sprintf(buf, "%s.meta", spherefile);
  ifstream in(buf);
  int nspheres;
  double radius=radius_in;
  float* mins = 0;
  float* maxs = 0;
  vector<float> mins_vec,maxs_vec;
  
  if (in) {
    // the file existed
    found_header = true;
    in >> nspheres;
    in >> radius;
    if (radius_in != 0)
      radius = radius_in;
    radius*=radius_factor;
    int varcount = 0;
    while(in) {
      float min,max;
      in >> min >> max;
      cerr << "min = " << min << ", max = " << max << endl;
      if (in) {
#if 0
	// this takes into account of
	// min/max equaling each other
	if (min == max) {
	  if (max > 0) {
	    max*=1.1;
	  } else {
	    if (max < 0)
	      max*=0.9;
	    else
	      max=1;
	  }
	}
#endif
	mins_vec.push_back(min);
	maxs_vec.push_back(max);
	varcount++;
      }
    }
    numvars = varcount;
    cerr << "Num Variables found in " << buf << " is " << numvars << endl;
    mins = (float*)malloc(numvars*sizeof(float));
    maxs = (float*)malloc(numvars*sizeof(float));
    if (numvars != mins_vec.size() || numvars != maxs_vec.size()) {
      cerr << "There was a problem in read_spheres\n";
      return 0;
    }
    for (int i = 0; i < numvars; i++) {
      mins[i] = mins_vec[i];
      maxs[i] = maxs_vec[i];
    }
    in.close();
  }
  else {
    // the file did not exist
    cerr << "Metafile does not exist, recreating after data file is read.\n";
  }

  //------------------------------------
  // setup the colors
  
  if(datanum<1 || datanum>numvars){
    cerr << "colordata must be between 1 and " << numvars << ".\n";
    abort();
  }
  datanum--;
  
  //------------------------------------
  // open the data file
  static const int nsph=512*400;
  
#ifdef __sgixx
  //int d_maxiosz;
  int d_mem;
  struct dioattr s;
  int in_fd=open(spherefile, O_RDONLY|O_DIRECT);
  if(in_fd==-1){
    in_fd=open(spherefile, O_RDONLY);
    if(in_fd==-1){
      perror("open");
      exit(1);
    }
    s.d_maxiosz=1024*1024;
    s.d_mem=512;
    cerr << "Couldn't open with O_DIRECT, reading slow\n";
  } else {
    if(fcntl(in_fd, F_DIOINFO, &s) == 0){
      fprintf(stderr, "direct io: d_mem=%d, d_miniosz=%d, d_maxiosz=%d\n", s.d_mem, s.d_miniosz, s.d_maxiosz);
      fprintf(stderr, "using: %ld\n", nsph*sizeof(float)*numvars);
    } else {
      s.d_mem=512;
    }
  }
  //d_maxiosz=s.d_maxiosz;
  d_mem=s.d_mem;
#else
  int in_fd=open(spherefile, O_RDONLY);
  if(in_fd==-1){
    perror("open");
    exit(1);
  }
  //d_maxiosz=1024*1024;
  int d_mem=512;
  cerr << "Couldn't open with O_DIRECT, reading slow\n";
#endif
  struct stat64 statbuf;
  if(fstat64(in_fd, &statbuf) == -1){
    perror("fstat");
    cerr << "cannot stat file\n";
    exit(1);
  } else {
    cerr << "size of " << spherefile << " is " << statbuf.st_size << " bytes\n";
  }

  // make sure the data file is the correct size
  if (found_header) {
    if (nspheres != (int)(statbuf.st_size/(numvars*sizeof(float)))) {
      cerr << "Size of file does not match that for " << nspheres << " spheres.\nIf the number of variables is not 3 please specify -numvars [number] on the command line\n";
      exit(1);
    }
  }
  else {
    nspheres = (int)(statbuf.st_size/(numvars*sizeof(float)));
  }

  //-----------------------------------------
  // read in the data
  cerr << "Reading " << nspheres << " spheres\n";
  char* odata=(char*)malloc(numvars*(nspheres+nsph)*sizeof(float)+d_mem);
  unsigned long addr=(unsigned long)odata;
  unsigned long off=addr%d_mem;
  if(off){
    addr+=d_mem-off;
  }
  float* data=(float*)addr;
  
  int total=0;
  float* p=data;
  for(;;){
    long s=read(in_fd, p, numvars*nsph*sizeof(float));
    if(s==0)
      break;
    if(s==-1){
      perror("read");
      exit(1);
    }
    int n=(int)(s/(numvars*sizeof(float)));
    total+=n;
    p+=n*numvars;
  }
  if(total != nspheres){
    cerr << "Wrong number of spheres!\n";
    cerr << "Wanted: " << nspheres << '\n';
    cerr << "Got: " << total << '\n';
  }
  
  //--------------------------------------------
  // create header file if it was not found
  if(!found_header){
    mins = (float*)malloc(numvars*sizeof(float));
    maxs = (float*)malloc(numvars*sizeof(float));
    // setup min/max values
    for(int j=0;j<numvars;j++){
      mins[j]=MAXFLOAT;
      maxs[j]=-MAXFLOAT;
    }

    // loop through the data and find min/max
    float* p=data;
    for(int i=0;i<nspheres;i++){
      for(int j=0;j<numvars;j++){
	mins[j]=Min(mins[j], p[j]);
	maxs[j]=Max(maxs[j], p[j]);
      }
      p+=numvars;
    }

    // write the output file
    ofstream out(buf);
    if(out){
      out << nspheres << '\n';
      out << radius << '\n';
      for(int i=0;i<numvars;i++){
	out << mins[i] << " " << maxs[i] << '\n';
      }
      out.close();
    }
    else {
      cerr << "Warning: could not create metafile, it will be recomputed next time!\n";
    }
  }
  
  
  
  double dt=SCIRun::Time::currentSeconds()-time;
  cerr << "Read " << nspheres << " spheres in " << dt << " seconds (" << nspheres/dt << " spheres/sec)\n";
  close(in_fd);
  return new GridSpheres(data, mins, maxs, nspheres, numvars, gridcellsize, griddepth, radius, cmap);
}

/////////////////////////////////////////////////////////////
//////////////////
//////             Parallel code
/////////////////////////////////////////////////////////////
class Preprocessor: public Runnable {
  GridSpheres *grid;
  Semaphore *sema;

public:
  Preprocessor(GridSpheres* grid, Semaphore *sema):
    grid(grid), sema(sema)
  {}
  ~Preprocessor() {}
  void run() {
    int a,b;
    grid->preprocess(0,a,b);
    sema->up();
  }
};

/////////////////////////////////////////////////

#define RAINBOW_COLOR_MAP 0
#define INVERSE_BLACK_BODY_COLOR_MAP 1
#define NRRD_COLOR_MAP 2

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
  int scene_type = 0;
  vector<string> data_files;
  int ncolors=256;
  float t_inc = 0.00125;
  double spec_coeff = 64;
  double ambient = 0.5;
  double diffuse = 1.0;
  double specular = 1.0;
  bool override_data_min = false;
  bool override_data_max = false;
  float data_min_in = -1;
  float data_max_in = -1;
  float frame_rate = 3;
  int color_map_type = INVERSE_BLACK_BODY_COLOR_MAP;
  char *nrrd_color_map_file = 0;
  char *nrrd_alpha_map = 0;
  Color bgcolor(0,0,0);

  // Options for the sphere stuff
  int gridcellsize=4;
  int griddepth=2;
  int colordata=5;
  float radius_factor=1;
  // This is the index to use for the radius.  -1 means don't use it.
  int radius_index = -1;
  int numvars=6;
  float radius=0.0004;
  char *cmap_file = 0; // Non zero when a file has been specified
  char *cmap_type = "InvRainbow";
  char *gridconfig = 0;
  string *var_names = 0;

  // This is number of times to repeat the last timestep
  int repeat_last_timestep = 0;
  
  for(int i=1;i<argc;i++){
    if (strcmp(argv[i], "-type")==0) {
      i++;
      if (strcmp(argv[i], "nrrd") == 0) {
	scene_type = 6;
	i++;
	data_files.push_back(argv[i]);
      } else if (strcmp(argv[i], "nrrdlist") == 0) {
	scene_type = 6;
	// open up the file and then suck in all the files
	i++;
	cout << "Reading nrrd file list from " << argv[i] << endl;
	ifstream in(argv[i]);
	while (in) {
	  string file;
	  in >> file;
	  data_files.push_back(file);
	  cout << "Nrrd file: "<<file<<"\n";
	}
	cout << "Read "<<data_files.size()<<" nrrd files.\n";
      }
    } else if(strcmp(argv[i], "-ncolors")==0){
      i++;
      ncolors = atoi(argv[i]);
    } else if(strcmp(argv[i], "-tinc")==0){
      i++;
      t_inc = atof(argv[i]);
    } else if(strcmp(argv[i], "-min")==0){
      i++;
      data_min_in = atof(argv[i]);
      override_data_min = true;
    } else if(strcmp(argv[i], "-max")==0){
      i++;
      data_max_in = atof(argv[i]);
      override_data_max = true;
    } else if(strcmp(argv[i], "-spow")==0){
      i++;
      spec_coeff = atof(argv[i]);
    } else if(strcmp(argv[i], "-ambient")==0){
      i++;
      ambient = atof(argv[i]);
    } else if(strcmp(argv[i], "-diffuse")==0){
      i++;
      diffuse = atof(argv[i]);
    } else if(strcmp(argv[i], "-specular")==0){
      i++;
      specular = atof(argv[i]);
    } else if(strcmp(argv[i], "-rate")==0){
      i++;
      frame_rate = atof(argv[i]);
    } else if(strcmp(argv[i], "-colormap")==0){
      i++;
      if (strcmp(argv[i], "rainbow")==0){
	color_map_type = RAINBOW_COLOR_MAP;
      } else if (strcmp(argv[i], "inversebb")==0){
	color_map_type = INVERSE_BLACK_BODY_COLOR_MAP;
      } else if (strcmp(argv[i], "nrrd")==0){
	color_map_type = NRRD_COLOR_MAP;
	nrrd_color_map_file = argv[++i];
      } else {
	cerr << "Unknown color map type.  Using rainbow.\n";
	color_map_type = RAINBOW_COLOR_MAP;
      }
    } else if(strcmp(argv[i], "-alpha")==0){
      nrrd_alpha_map = argv[++i];
    } else if(strcmp(argv[i], "-bgcolor")==0){
      float r,g,b;
      r = atof(argv[++i]);
      g = atof(argv[++i]);
      b = atof(argv[++i]);
      bgcolor = Color(r,g,b);
    } else if(strcmp(argv[i], "-depth")==0) {
      depth = atoi(argv[++i]);
      if (depth < 2) {
	cerr << "depth should be greater than 1\n";
	return 0;
      }
      if (depth > 5) {
	cerr << "depth is larger than 5 which it really too much for most applications.  If you want more than 5 recompile this scene file. :)\n";
	return 0;
      }
    } else if(strcmp(argv[i], "-minmax")==0){
      float x,y,z;
      x = atof(argv[++i]);
      y = atof(argv[++i]);
      z = atof(argv[++i]);
      minPin = Point(x,y,z);
      x = atof(argv[++i]);
      y = atof(argv[++i]);
      z = atof(argv[++i]);
      maxPin = Point(x,y,z);
      use_global_minmax = true;
    } else if(strcmp(argv[i], "-gridcellsize")==0){
      i++;
      gridcellsize=atoi(argv[i]);
    } else if(strcmp(argv[i], "-griddepth")==0){
      i++;
      griddepth=atoi(argv[i]);
    } else if(strcmp(argv[i], "-colordata")==0){
      i++;
      colordata=atoi(argv[i]);
    } else if(strcmp(argv[i], "-radiusfactor")==0) {
      i++;
      radius_factor=atof(argv[i]);
    } else if(strcmp(argv[i], "-radius")==0) {
      i++;
      radius=atof(argv[i]);
    } else if(strcmp(argv[i], "-radius_index")==0) {
      radius_index = atoi(argv[++i]);
    } else if(strcmp(argv[i], "-numvars")==0) {
      i++;
      numvars=atoi(argv[i]);
    } else if (strcmp(argv[i], "-cmap") == 0) {
      cmap_file = argv[++i];
    } else if (strcmp(argv[i], "-cmaptype") == 0) {
      cmap_type = argv[++i];
    } else if (strcmp(argv[i], "-gridconfig") == 0) {
      gridconfig = argv[++i];
    } else if (strcmp(argv[i], "-varnames") == 0) {
      int num_varnames = atoi(argv[++i]);
      cerr << "Reading "<<num_varnames << " variable names\n";
      var_names = new string[num_varnames];
      for(int v = 0; v < num_varnames; v++)
        var_names[v] = string(argv[++i]);
    } else if (strcmp(argv[i], "-repeatlast") == 0) {
      repeat_last_timestep = atoi(argv[++i]);
    } else {
      cerr << "Unknown option: " << argv[i] << '\n';
      cerr << "Valid options for scene: " << argv[0] << '\n';
      cerr << " -dim [int]x[int]x[int]";
      cerr << " -type [\"nrrd\" or \"nrrdlist\"]\n";
      cerr << "\t\tnrrd [path to nrrd file] (-dim parameter ignored)\n";
      cerr << " -lam - use a lambertian surface\n";
      cerr << " -ncolors [int] - the size of the transfer function\n";
      cerr << " -tinc [float] - the number of samples per unit\n";
      cerr << " -spow [float] - the spectral exponent\n";
      cerr << " -ambient [float] - the ambient factor\n";
      cerr << " -diffuse [float] - the diffuse factor\n";
      cerr << " -specular [float] - the specular factor\n";
      cerr << " -rate [float] - frame rate of the time steps\n";
      cerr << " -colormap [string] - \"rainbow\" or \"inversebb\"\n";
      cerr << " -colormap nrrd [filename.nrrd] - read in a nrrd for the colormap with alphas\n";
      cerr << " -alpha [filename.nrrd] - read in a nrrd with just the alpha transfer function\n";
      cerr << " -bgcolor [float] [float] [float] - the three floats are r, g, b\n";
      cerr << " -depth - the number of depths to use for the HVolumeVis.\n";
      cerr << "          [defaults to "<<depth<<"]\n";
      cerr << " GridSphere's options\n";
      cerr << " -gridcellsize [int]\n";
      cerr << " -griddepth [int]\n";
      cerr << " -radiusfactor [float]\n";
      cerr << " -radius [float]\n";
      cerr << " -radius_index [int]\n";
      cerr << " -numvars [int]\n";
      cerr << " -colordata [int]\n";
      cerr<<"  -cmap <filename>     defaults to inverse rainbow"<<endl;
      cerr<<"  -cmaptype <type>     type of colormap\n";
      cerr << " -gridconfig <filename> use this file as the config file.\n";
      cerr << " -varnames [number] vname1 \"v name 2\"\n";
      cerr << " -repeatlast [number]  number of times to repeat the last timestep\n";
      return 0;
    }
  }

  // check the parameters
  if (scene_type == 6)
    if (data_files.size() == 0) {
      cerr << "The file was not set.\n";
      return 0;
    }
  
  Camera cam(Point(0.514848, 0.133841, 0.411278),
             Point(0.0540672, 0.0518796, 0.0558233),
	     Vector(-0.135368, 0.989395, -0.0526561),
             29.4);

  // Make this global so that we don't have to pass a million
  // parameters around.
  np = nworkers;
  
  double ambient_scale=0.5;
  
  Array1<Color> matls;
  Array1<AlphaPos> alphas;
  switch (color_map_type) {
  case INVERSE_BLACK_BODY_COLOR_MAP:
    get_material2(matls,alphas);
    break;
  case NRRD_COLOR_MAP:
    if (get_material_nrrd(nrrd_color_map_file, matls, alphas) == 0)
      break;
    // else do the rainbow color map
  case RAINBOW_COLOR_MAP:
  default:
    get_material(matls,alphas);
    break;
  }
  if (nrrd_alpha_map) {
    get_alpha_nrrd(nrrd_alpha_map, alphas);
  }
  
  VolumeVisDpy *dpy = new VolumeVisDpy(matls, alphas, ncolors, t_inc);

  // Generate the data
  Object *obj;

  Array1<GridSpheres*> spheres;

  // For the GridSpheres
  RegularColorMap *cmap = 0;
  if (cmap_file)
    cmap = new RegularColorMap(cmap_file);
  else {
    int cmap_type_index = RegularColorMap::parseType(cmap_type);
    cmap = new RegularColorMap(cmap_type_index);
  }

  if (scene_type == 6) {
    if (data_files.size() > 1) {
      // create a timeobj
      //      TimeObj *timeobj = new TimeObj(frame_rate);
      SelectableGroup *timeobj = new SelectableGroup(1.0/frame_rate);
      timeobj->set_name("Volume Data");
      for(unsigned int i = 0; i < data_files.size(); i++) {
	char *myfile = strdup(data_files[i].c_str());
        if (strcmp(myfile, "") == 0) continue;
        HVolumeVis<float,VMCell<float> > *volume = create_volume_from_nrrd
	  (myfile, override_data_min, data_min_in,
	   override_data_max, data_max_in, spec_coeff,
	   ambient, diffuse, specular, dpy);
	// add the volume to the timeobj
	if (myfile)
	  free(myfile);
	if (volume == 0) {
          cerr << "There was a problem reading volume from "<<myfile<<"\n";
	  // there was a problem
	  return 0;
        }
	timeobj->add(volume);

        // Add the GridSpheres
	myfile = strdup(data_files[++i].c_str());
        if (strcmp(myfile, "") == 0) continue;
        GridSpheres* gsphere = read_spheres(myfile, colordata, gridcellsize, griddepth, radius, radius_factor, numvars, cmap);
        spheres.add(gsphere);
        volume->set_child(gsphere);
        if (myfile) free(myfile);
      }
      // Repeat the last timestep if need be.
      for(int repeat = 0; repeat < repeat_last_timestep; repeat++)
        timeobj->add(timeobj->objs[timeobj->numObjects()-1]);
      // Set the object
      obj = (Object*)timeobj;
    } else {
      char *myfile = strdup(data_files[0].c_str());
      obj = (Object*)create_volume_from_nrrd
	(myfile, override_data_min, data_min_in,
	 override_data_max, data_max_in, spec_coeff,
	 ambient, diffuse, specular, dpy);
      if (myfile)
	free(myfile);
    }
  } else {
    cerr << "Unknown scene type\n";
    return 0;
  }

  (new Thread(dpy, "VolumeVis display thread"))->detach();

  GridSpheresDpy* display;
  if (gridconfig)
    display = new GridSpheresDpy(colordata-1, gridconfig);
  else
    display = new GridSpheresDpy(colordata-1);

  if (radius_index >= 0)
    display->set_radius_index(radius_index);

  for(int i = 0; i < spheres.size(); i++) {
    display->attach(spheres[i]);
  }

  // Give it time to compute the histogram while the rest of the data
  // is preprocessed.
  if (var_names) display->set_var_names(var_names);
  (new Thread(display, "GridSpheres display thread\n"))->detach();

  // Do the preprocessing here
  int num_threads =  Min(nworkers, 16);
  Semaphore* prepro_sema = new Semaphore("rtrt::tstdemo preprocess semaphore", num_threads);
  for(int i = 0; i < spheres.size(); i++) {
    prepro_sema->down();
    cerr << "=====================================\n";
    cerr << "Proprocessing GridSpheres #"<<i<<"\n";
    Thread *thrd = new Thread(new Preprocessor(spheres[i], prepro_sema),
                                 "rtrt::tstdemo:Preprocessor Thread");
    thrd->detach();
  }
  // Wait for everyone to finish
  prepro_sema->down(num_threads);
  if (prepro_sema) delete prepro_sema;
  
  Group* all = new Group();
  all->add(obj);

  Plane groundplane ( Point(-500, 300, 0), Vector(7, -3, 2) );
  Color cup(0.9, 0.7, 0.3);
  Color cdown(0.0, 0.0, 0.2);

  Scene* scene=new Scene(all, cam,
			 bgcolor, cdown, cup, groundplane, 
			 ambient_scale);

  Light *light0 = new Light(Point(-500,300,-300), Color(.8,.8,.8), 0);
  light0->name_ = "light 0";
  //  light0->turnOff();
  light0->updateIntensity(0.7);
  scene->add_light(light0);
  scene->attach_display(dpy);
  scene->attach_display(display);
  scene->addGuiObject("Fire", obj);
  scene->select_shadow_mode( No_Shadows );
  
  return scene;
}



