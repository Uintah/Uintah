

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CatmullRomSpline.h>
#include <Packages/rtrt/Core/GridSpheres.h>
#include <Packages/rtrt/Core/GridSpheresDpy.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/TimeObj.h>
#include <Core/Thread/Time.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <values.h>
#include <vector.h>

using namespace rtrt;
using namespace std;
//using SCIRun::Time;

using SCIRun::Thread;

//#define NDATA 3

#ifndef O_DIRECT
#define O_DIRECT 0
#endif

class SphereData {
public:
  float* data;
  int nspheres;
  int numvars;
  float radius;
  Array1<float> mins;
  Array1<float> maxs;
};


//////////////////////////////////////////////////////////////////
// npsheres must be the number of spheres or -1 if it is not known

//void read_spheres(char* spherefile, int numvars, int &nspheres, float *&data) {
void read_spheres(char* spherefile, SphereData &sphere_data) {
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
      fprintf(stderr, "using: %ld\n", nsph*sizeof(float)*sphere_data.numvars);
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
  struct stat statbuf;
  if(fstat(in_fd, &statbuf) == -1){
    perror("fstat");
    cerr << "cannot stat file\n";
    exit(1);
  }

  // make sure the data file is the correct size
  if (sphere_data.nspheres != -1) {
    if (sphere_data.nspheres != (int)(statbuf.st_size/(sphere_data.numvars*sizeof(float)))) {
      cerr << "Size of file does not match that for " << sphere_data.nspheres << " spheres.\nIf the number of variables is not 3 please specify -numvars [number] on the command line\n";
      exit(1);
    }
  }
  else {
    sphere_data.nspheres = (int)(statbuf.st_size/(sphere_data.numvars*sizeof(float)));
  }

  //-----------------------------------------
  // read in the data
  cerr << "Reading " << sphere_data.nspheres << " spheres\n";
  char* odata=(char*)malloc(sphere_data.numvars*(sphere_data.nspheres+nsph)*sizeof(float)+d_mem);
  unsigned long addr=(unsigned long)odata;
  unsigned long off=addr%d_mem;
  if(off){
    addr+=d_mem-off;
  }
  sphere_data.data=(float*)addr;
  
  int total=0;
  float* p=sphere_data.data;
  for(;;){
    long s=read(in_fd, p, sphere_data.numvars*nsph*sizeof(float));
    if(s==0)
      break;
    if(s==-1){
      perror("read");
      exit(1);
    }
    int n=(int)(s/(sphere_data.numvars*sizeof(float)));
    total += n;
    p += n * sphere_data.numvars;
  }
  if(total != sphere_data.nspheres){
    cerr << "Wrong number of spheres!\n";
    cerr << "Wanted: " << sphere_data.nspheres << '\n';
    cerr << "Got: " << total << '\n';
  }
  
}

bool read_header(char* header, SphereData &sphere_data) {
  ifstream in(header);
  
  if (in) {
    in >> sphere_data.nspheres;
    in >> sphere_data.radius;
    int varcount = 0;
    while(in) {
      float min,max;
      in >> min >> max;
      if (in) {
	sphere_data.mins.add(min);
	sphere_data.maxs.add(max);
	varcount++;
      }
    }
    sphere_data.numvars = varcount;
    cerr << "Num Variables found in " << header << " is " << sphere_data.numvars << endl;
    in.close();
    // the file existed
    return true;
  }
  else {
    // the file did not exist
    cerr << "Metafile does not exist, recreating after data file is read.\n";
    return false;
  }
}

//--------------------------------------------
// create header file if it was not found
void write_header(char* header, SphereData &sphere_data) {
  sphere_data.mins.resize(sphere_data.numvars);
  sphere_data.maxs.resize(sphere_data.numvars);
  // setup min/max values
  for(int j = 0; j < sphere_data.numvars; j++){
    sphere_data.mins[j] =  MAXFLOAT;
    sphere_data.maxs[j] = -MAXFLOAT;
  }
  
  // loop through the data and find min/max
  float* p = sphere_data.data;
  for(int i = 0; i < sphere_data.nspheres; i++){
    for(int j = 0; j < sphere_data.numvars; j++){
      sphere_data.mins[j] = Min(sphere_data.mins[j], p[j]);
      sphere_data.maxs[j] = Max(sphere_data.maxs[j], p[j]);
    }
    p += sphere_data.numvars;
  }
  
  // write the output file
  ofstream out(header);
  if(out){
    out << sphere_data.nspheres << '\n';
    out << sphere_data.radius << '\n';
    for(int i = 0; i < sphere_data.numvars; i++){
      out << sphere_data.mins[i] << " " << sphere_data.maxs[i] << '\n';
    }
    out.close();
  }
  else {
    cerr << "Warning: could not create metafile, it will be recomputed next time!\n";
  }
}

void append_spheres(char* spherefile, Array1<SphereData> &data_group,
		    float radius_in, float radius_factor) {

  // try to read the header
  char header[300];
  sprintf(header, "%s.meta", spherefile);
  SphereData sphere_data;
  sphere_data.radius = radius_in;
  bool found_header = read_header(header,sphere_data);
  if (radius_in != 0)
    sphere_data.radius = radius_in;
  sphere_data.radius *= radius_factor;
  
  // read the spheres
  read_spheres(spherefile, sphere_data);
  // write the header if you need to
  if (!found_header)
    write_header(header, sphere_data);

  // append the data to the group of spheres and update the total sphere count
  data_group.add(sphere_data);
}

void get_material(Array1<Material*> &matls) {
  CatmullRomSpline<Color> spline(0);
  spline.add(Color(.4,.4,.4));
  spline.add(Color(.4,.4,1));
  //    for(int i=0;i<2;i++)
  spline.add(Color(.4,1,.4));
  //    for(int i=0;i<3;i++)
  spline.add(Color(1,1,.4));
  //    for(int i=0;i<300;i++)
  spline.add(Color(1,.4,.4));
  int ncolors=5000;
  matls.resize(ncolors);
  float Ka=.8;
  float Kd=.8;
  float Ks=.8;
  float refl=0;
  float specpow=40;
  for(int i=0;i<ncolors;i++){
    float frac=float(i)/(ncolors-1);
    Color c(spline(frac));
    matls[i]=new Phong(c*Kd, c*Ks, specpow, refl);
    //matls[i]=new LambertianMaterial(c*Kd);
  }
}

GridSpheres* create_GridSpheres(Array1<SphereData> data_group, int colordata,
				int gridcellsize, int griddepth) {
  // need from the group
  // 1. total number of spheres
  // 2. make sure the numvars is the same
  // 3. average radius

  int total_spheres = 0;
  cerr << "Size of data_group = " << data_group.size() << endl;
  int numvars = data_group[0].numvars;
  float radius = 0;
  for (int i = 0; i < data_group.size(); i++) {
    total_spheres += data_group[i].nspheres;
    if (numvars != data_group[i].numvars) {
      cerr << "numvars does not match: Goodbye!\n";
      abort();
    }
    radius += data_group[i].radius;
  }
  radius /= data_group.size();
  
  if(colordata < 1 || colordata > numvars){
    cerr << "colordata must be between 1 and " << numvars << ".\n";
    abort();
  }

  float *mins, *maxs;
  mins = (float*)malloc(numvars * sizeof(float));
  maxs = (float*)malloc(numvars * sizeof(float));
  // initialize the mins and maxs
  for (int i = 0; i < numvars; i++) {
    mins[i] =  MAXFLOAT;
    maxs[i] = -MAXFLOAT;
  }
  // now concatenate the spheres and compute the mins and maxs
  for (int i = 0; i < numvars; i++) {
    mins[i] =  MAXFLOAT;
    maxs[i] = -MAXFLOAT;
  }

  // allocate memory for the data
  static const int d_mem = 512;
  static const int nsph = (512*400);
  char* odata=(char*)malloc(numvars*(total_spheres+nsph)*sizeof(float)+d_mem);
  unsigned long addr=(unsigned long)odata;
  unsigned long off=addr%d_mem;
  if(off){
    addr+=d_mem-off;
  }

  // the data
  float *data = (float*)addr;
  // the index to the data array
  int index = 0;
  
  for (int g = 0; g < data_group.size(); g++) {
    // compute the mins and maxs
    for (int i = 0; i < numvars; i++) {
      mins[i] = Min(mins[i], data_group[g].mins[i]);
      maxs[i] = Max(maxs[i], data_group[g].maxs[i]);
    }
    // copy the data
    // this may be done more efficient using mcopy or something like it.
    int ndata = data_group[g].nspheres * numvars;
    for (int j = 0; j < ndata; j++) {
      data[index++] = data_group[g].data[j];
    }
  }
  if (index != total_spheres * numvars) {
    cerr << "Wrong number of vars copied: index = " << index << ", total_spheres * numvars = " << total_spheres * numvars << endl;
  }
  Array1<Material*> matls;
  get_material(matls);
  cout << "Using radius "<<radius<<"\n";
  return new GridSpheres(data, mins, maxs, total_spheres, numvars-3, gridcellsize, griddepth, radius, matls.size(), &matls[0]);  
}

GridSpheres* read_spheres(char* spherefile, int datanum,
			  int gridcellsize, int griddepth,
			  float radius_in, float radius_factor,
			  int numvars)
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
  float* mins;
  float* maxs;
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
  
  CatmullRomSpline<Color> spline(0);
#if 0
  spline.add(Color(.4,.4,.4));
  spline.add(Color(.4,.4,1));
  //    for(int i=0;i<2;i++)
  spline.add(Color(.4,1,.4));
  //    for(int i=0;i<3;i++)
  spline.add(Color(1,1,.4));
  //    for(int i=0;i<300;i++)
  spline.add(Color(1,.4,.4));
#else
  spline.add(Color(0,0,1));
  spline.add(Color(0,0.4,1));
  spline.add(Color(0,0.8,1));
  spline.add(Color(0,1,0.8));
  spline.add(Color(0,1,0.4));
  spline.add(Color(0,1,0));
  spline.add(Color(0.4,1,0));
  spline.add(Color(0.8,1,0));
  spline.add(Color(1,0.9176,0));
  spline.add(Color(1,0.8,0));
  spline.add(Color(1,0.4,0));
  spline.add(Color(1,0,0));
  //{ 0 0 255}   { 0 102 255}
  //{ 0 204 255}  { 0 255 204}
  //{ 0 255 102}  { 0 255 0}
  //{ 102 255 0}  { 204 255 0}
  //{ 255 234 0}  { 255 204 0}
  //{ 255 102 0}  { 255 0 0} }}
#endif  
  int ncolors=5000;
  Array1<Material*> matls(ncolors);
  //float Ka=.8;
  float Kd=.8;
  //float Ks=.8;
  //float refl=0;
  //float specpow=40;
  for(int i=0;i<ncolors;i++){
    float frac=float(i)/(ncolors-1);
    Color c(spline(frac));
    //matls[i]=new Phong(c*Kd, c*Ks, specpow, refl);
    matls[i]=new LambertianMaterial(c*Kd);
  }
  
  
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
  struct stat statbuf;
  if(fstat(in_fd, &statbuf) == -1){
    perror("fstat");
    cerr << "cannot stat file\n";
    exit(1);
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
  return new GridSpheres(data, mins, maxs, nspheres, numvars-3, gridcellsize, griddepth, radius, matls.size(), &matls[0]);
}

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  char* file=0;
  int gridcellsize=4;
  int griddepth=3;
  int colordata=2;
  bool timevary=false;
  int timeblockmax=0;
  bool counttimeblock=false;
  float radius_factor=1;
  float rate=3;
  int numvars=3;
  float radius=0;
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-gridcellsize")==0){
      i++;
      gridcellsize=atoi(argv[i]);
    }
    else if(strcmp(argv[i], "-griddepth")==0){
      i++;
      griddepth=atoi(argv[i]);
    }
    else if(strcmp(argv[i], "-colordata")==0){
      i++;
      colordata=atoi(argv[i]);
    }
    else if(strcmp(argv[i], "-timevary")==0) {
      timevary = true;
      i++;
      if (strcmp(argv[i], "all")!=0) {
	counttimeblock = true;
	timeblockmax = atoi(argv[i]);
      }
    }
    else if(strcmp(argv[i], "-radiusfactor")==0) {
      i++;
      radius_factor=atof(argv[i]);
    }
    else if(strcmp(argv[i], "-radius")==0) {
      i++;
      radius=atof(argv[i]);
    }
    else if(strcmp(argv[i], "-rate")==0){
      i++;
      rate=atof(argv[i]);
    }
    else if(strcmp(argv[i], "-numvars")==0) {
      i++;
      numvars=atoi(argv[i]);
    }
    else {
      if(file){
	cerr << "Unknown option: " << argv[i] << '\n';
	cerr << "Valid options for scene: " << argv[0] << '\n';
	cerr << " -gridcellsize [int]\n";
	cerr << " -griddepth [int]\n";
	cerr << " -colordata [int]\n";
	cerr << " -timevary [all or int]\n";
	cerr << " -radiusfactor [float]";
	cerr << " -radius [float]";
	cerr << " -rate [float]";
	cerr << " -numvars [int]";
	return 0;
      }
      file=argv[i];
    }
  }
  
  Camera cam(Point(0,0,400), Point(0,0,0),
	     Vector(0,1,0), 60.0);
  
  //double bgscale=0.5;
  //Color groundcolor(0,0,0);
  //Color averagelight(0,0,0);
  double ambient_scale=1.0;
  
  //  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  Color bgcolor(0,0,0);
  
  Group* all = new Group();
  // the value will be checked later and the program will abort
  // if the value is not correct.
  GridSpheresDpy* display = new GridSpheresDpy(colordata-1);
  
  if (timevary) {
    ifstream in(file);
    TimeObj* alltime = new TimeObj(rate);
    //Group* timeblock;
    int numtimeblock = 0;
    Array1<SphereData> sphere_data;
    while(in){
      char file[1000];
      // stick the next line in file
      in >> file;
      if (in) {
	if (strcmp(file,"<TIMESTEP>") == 0) {
	  cerr << "-------------Starting timestep----------\n";
	  //timeblock = new Group();
	  sphere_data.remove_all();
	}
	else if (strcmp(file,"</TIMESTEP>") == 0) {
	  cerr << "=============Ending timestep============\n";
	  GridSpheres* obj = create_GridSpheres(sphere_data, colordata,
						gridcellsize, griddepth);
	  display->attach(obj);
	  alltime->add((Object*)obj);
	  //alltime->add(timeblock);
	  if (counttimeblock && ++numtimeblock >= timeblockmax)
	    break;
	}
	else if (strcmp(file,"<PATCH>") == 0) {
	  //
	}
	else if (strcmp(file,"</PATCH>") == 0) {
	  //
	}
	else {
	  cerr << "Reading " << file << "\n";
	  append_spheres(file,sphere_data,radius, radius_factor);
	  //GridSpheres* obj=read_spheres(file, colordata, gridcellsize, griddepth, radius, radius_factor, numvars);
	  //display->attach(obj);
	  //timeblock->add((Object*)obj);
	}
      }
    }
    all->add(alltime);
  }
  else {
    GridSpheres* obj=read_spheres(file, colordata, gridcellsize, griddepth, radius, radius_factor, numvars);
    display->attach(obj);
    all->add((Object*)obj);
  }

  Plane groundplane ( Point(-500, 300, 0), Vector(7, -3, 2) );
  //Color cup(0,0,0.3);
  //Color cdown(0.4, 0.2, 0);
  Color cup(0.9, 0.7, 0.3);
  Color cdown(0.0, 0.0, 0.2);


#if 0
  Group *all2 = new Group();
  TimeObj* alltime2 = new TimeObj(rate);
  Group* timeblock2;
  for (int t = 0; t < 5; t++) {
    timeblock2 = new Group();
    Material* matl0=new Phong(Color(.5,.2,.2), Color(.8,.3,.3), 10, .5);
    timeblock2->add((Object*)new Sphere(matl0,::Point(t,t,t),1));
    alltime2->add((Object*)timeblock2);
  }
  all2->add((Object*)alltime2);
#endif
  Scene* scene=new Scene(all, cam,
			 bgcolor, cdown, cup, groundplane, 
			 ambient_scale);
  
  scene->add_light(new Light(Point(500,-300,300), Color(.8,.8,.8), 0));
  scene->select_shadow_mode( No_Shadows );
  scene->addObjectOfInterest(all, true);

#if 0 // GridSpheresDpy needs to be made to inherit DpyBase.
  scene->attach_display(display);
  display->setName("Particle Vis");
  scene->attach_auxiliary_display(display);
#endif
  (new Thread(display, "GridSpheres display thread\n"))->detach();

  return scene;
}



