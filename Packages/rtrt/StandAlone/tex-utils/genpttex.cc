
/*

 A useful genpttex command:

 ./genpttex -nsamples 100 -depth 3 -tex_size 16 -light 5 10 7.5 -intensity 5000

 Some useful unu commands

 # Use these to quantize the textures and copy them to ppms
 
 unu gamma -g 2 -i sphere00000.nrrd | unu quantize -b 8 | unu dice -a 3 -o sphere
 echo 'for T in sphere*.png; do unu save -f pnm -i $T -o `basename $T .png`.ppm; done' | bash

 # This copies the first column to the end to help with texture blending.
 
 echo 'for T in sphere?.ppm; do unu slice -a 1 -p M -i $T | unu reshape -s 3 1 64 | unu join -a 1 -i - $T -o $T;done' | bash
*/

#include <Packages/rtrt/Core/CatmullRomSpline.h>
#include <Packages/rtrt/Core/GridSpheres.h>
#include <Packages/rtrt/Core/GridSpheresDpy.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/PathTracer/PathTraceEngine.h>

#include <Core/Thread/Thread.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <unistd.h> // For read and close of files.
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef WRITE_SHPEREDATA
#  undef WRITE_SPHEREDATA
#endif
#define WRITE_SPHEREDATA 1

using namespace rtrt;
using namespace SCIRun;
using namespace std;

size_t work_load_max=1000;
int tex_size=16;
double radius=0;
double radius_factor=1.0;
int total_num_spheres=0;
Array1<float *> spheredata;
Array1<char *> deleteme;
Array1<int> spheredatasize;
float* alldata=0;
size_t next_sphere=0;
#define NUM_DATA 3
int gridcellsize=6;
int griddepth=2;
int numvars=3;

void readSpheres(char *spherefile);

Object *make_geometry( )
{
  Group* group=new Group();

  if (radius<=0) {
    cout<<"Setting radius=1.0"<<endl;
    radius=1;
  }

#if 0
  total_num_spheres=8;
  alldata=new float[total_num_spheres*NUM_DATA];
  float *data=alldata;
  spheredata.add(data);
  spheredatasize.add(total_num_spheres);
  for(int z=-1; z<=1; z+=2)
    for(int y=-1; y<=1; y+=2)
      for(int x=-1; x<=1; x+=2)
	{
	  *data++=x;
	  *data++=y;
	  *data++=z;
	  data += NUM_DATA-3;
	}
#else
  total_num_spheres=125000;
  alldata=new float[total_num_spheres*NUM_DATA];
  float *data=alldata;
  spheredata.add(data);
  spheredatasize.add(total_num_spheres);
  for(int z=-25; z<=25; z+=2)
    for(int y=-25; y<=25; y+=2)
      for(int x=-25; x<=25; x+=2)
	{
	  *data++=x;
	  *data++=y;
	  *data++=z;
	  data += NUM_DATA-3;
	}
#endif
  
  // Now generate some geometry
  data=spheredata[0];
  
#if WRITE_SPHEREDATA
  // Write out sphere data into a file
  FILE *out=fopen("spheredata", "wb");
  if (out) {
    size_t wrote=fwrite(data, sizeof(float), total_num_spheres*NUM_DATA, out);
    if (wrote != total_num_spheres*NUM_DATA) {
      cerr << "Couldn't write out all the values.  Wrote "<<wrote<<" items.\n";
    }
    fclose(out);
  }
  cout<<"Wrote sphere data to file \"spheredata\""<<endl;
#endif

  for(int i=0; i < total_num_spheres; i++) {
    double x=*data; data++;
    double y=*data; data++;
    double z=*data; data++;
    Point center(x, y, z);
    data += NUM_DATA-3;
    //    group->add( new Sphere(0, center, radius) );
    group->add( new TextureSphere(center, radius, tex_size) );
  }
  
  return group;
}

Object *make_geometry_fromfile(char *filename, bool timevary)
{
  if (timevary) {
    ifstream flist(filename);
    char pfile[128];
    flist.getline(pfile,128);
    while(!flist.eof())
      {
	if(strcmp(pfile,"<TIMESTEP>")==0)
	  {
	    // skip line
	  }
	else if(strcmp(pfile,"</TIMESTEP>")==0)
	  {
	    // skip line
	  }
	else if(strcmp(pfile,"<PATCH>")==0)
	  {
	    // skip line
	  }
	else if(strcmp(pfile,"</PATCH>")==0)
	  {
	    // skip line
	  }
	else
	  readSpheres(pfile);
	flist.getline(pfile,128);
      }
  } else {
    readSpheres(filename);
  }

  // Now we need an accelaration structure.
  //
  // Discussing this, it would be best to use a GridSpheres.  In order
  // to use one we need to create a large array with our floating
  // point data.

  // Allocate memory
  alldata=new float[total_num_spheres*numvars];
  float *data=alldata;
  for(int i=0; i < spheredata.size(); i++) {
    size_t size=spheredatasize[i] * sizeof(float) * numvars;
    memcpy(data, spheredata[i], size);
    // Free the old memory
    free(deleteme[i]);
    // Set the pointer back into our large array
    spheredata[i]=data;
    // Move our pointer to the next set of spheres
    data += size/sizeof(float);
  }

  // Here is our geometry
  cout << "total_num_spheres="<<total_num_spheres<<endl;
  GridSpheres *gridspheres=new GridSpheres(alldata, 0, 0, total_num_spheres,
					     numvars,
					     gridcellsize, griddepth,
					     radius*radius_factor,
					     0, 0);
  
  // We need a GridSphereDpy as GridSphere needs one to compute intersections
  GridSpheresDpy *griddpy=new GridSpheresDpy(0);
  griddpy->attach(gridspheres);
  // This will set up the rendering parameters
  griddpy->setup_vars();
  
  return gridspheres;
}

Group* get_next_sphere_set(size_t num_spheres) {
  Group *group=new Group();

  // Fix num_spheres to prevent running off the array
  if (num_spheres + next_sphere > total_num_spheres)
    num_spheres=total_num_spheres - next_sphere;
  
  // create spheres
  float *data=alldata + (next_sphere*numvars);
  for (int i=0;i<num_spheres;i++) {
    double x=*data; data++;
    double y=*data; data++;
    double z=*data; data++;
    Point center(x, y, z);
    data += numvars-3;
    // group->add( new Sphere(0, center, radius) );
    group->add( new TextureSphere(center, radius, tex_size) );
  }

  next_sphere += num_spheres;
  
  return group;
}

void
readSpheres(char* spherefile)
{
  // open the header file
  bool found_header=false;
  
  char buf[300];
  sprintf(buf, "%s.meta", spherefile);
  ifstream in(buf);
  int nspheres=0;
  double file_radius=radius;
  float* mins;
  float* maxs;
  vector<float> mins_vec,maxs_vec;
  
  if (in)
  {
    // the file existed
    found_header=true;
    in >> nspheres;
    in >> file_radius;
    if (radius != 0) {
      // Use the radius specified globally
      file_radius=radius;
    } else {
      // Use the file radius
      radius=file_radius;
    }
    int varcount=0;
    while(in)
    {
      float min,max;
      in >> min >> max;
      if (in) {
        mins_vec.push_back(min);
        maxs_vec.push_back(max);
        varcount++;
      }
    }
    numvars=varcount;
#if 0
    if (numvars != NUM_DATA) {
      cerr << "readSpheres::Can only deal with point data (3 vars)\n";
      return;
    }
#endif
    mins=(float*)malloc(numvars*sizeof(float));
    maxs=(float*)malloc(numvars*sizeof(float));
    for (int i=0; i < numvars; i++) {
      mins[i]=mins_vec[i];
      maxs[i]=maxs_vec[i];
    }
    in.close();
  }
  else
  {
    // the file did not exist
    cerr << "Metafile " << buf << " does not exist." << endl;
  }

  // open the data file
  static const int nsph=512*400;
  int in_fd=open(spherefile, O_RDONLY);
  if (in_fd==-1){
    perror("readSpheres::open");
    exit(1);
  }
  int d_mem=512;

  struct stat statbuf;
  if (fstat(in_fd, &statbuf)==-1){
    perror("readSpheres::fstat");
    cerr << "cannot stat file\n";
    exit(1);
  }

  // make sure the data file is the correct size
  if (found_header) {
    if (nspheres != (int)(statbuf.st_size/(numvars*sizeof(float)))) {
      cerr << "Size of file does not match that for " << nspheres
           << " spheres.\nIf the number of variables is not 3 please specify -numvars [number] on the command line"
           << endl;
      exit(1);
    }
  }
  else {
    nspheres=(int)(statbuf.st_size/(numvars*sizeof(float)));
  }

  cerr << "Reading "<<nspheres<<" spheres\n";
  
  // read in the data
  char* odata=(char*)malloc(numvars*(nspheres+nsph)*sizeof(float)+d_mem);
  unsigned long addr=(unsigned long)odata;
  unsigned long off=addr%d_mem;
  if (off){
    addr+=d_mem-off;
  }
  
  float *data=(float*)addr;
  int total=0;
  float* p=data;
  for (;;){
    long s=read(in_fd, p, numvars*nsph*sizeof(float));
    if (s==0)
      break;
    if (s==-1){
      perror("readSpheres::read");
      exit(1);
    }
    
    int n=(int)(s/(numvars*sizeof(float)));
    total+=n;
    p+=n*numvars;
  }
  
  close(in_fd);

  if (total != nspheres){
    cerr << "Wrong number of spheres!\n";
    cerr << "Wanted: " << nspheres << '\n';
    cerr << "Got: " << total << '\n';
  }

  total_num_spheres += nspheres;

  spheredata.add(data);
  deleteme.add(odata);
  spheredatasize.add(nspheres);
}


/**************************************************************************/

int main(int argc, char** argv)
{
  double lx=-0.25, ly=0.5, lz=-0.1;
  double lr=0.01;
  double intensity=1000.0;
  int nsamples=49;
  int depth=3;
  char *filename=0;
  bool timevary=false;
  char *bg=0;
  char *outfile=0;
  int nworkers=1;
  bool dilate=true;
  int support=2;
  int use_weighted_ave=0;
  float threshold=0.3;
  float luminance=1.0;
  int num_sample_divs=0;
  bool noShadows=false;
  bool noDirectL=false;
  int nldivs=0;
  float ldistance=10.0;
  
  for(int i=1;i<argc;i++) {
    if(strcmp(argv[i], "-light")==0) {
      lx=atof(argv[++i]);
      ly=atof(argv[++i]);
      lz=atof(argv[++i]);
    } else if(strcmp(argv[i], "-lr")==0) {
      lr=atof(argv[++i]);
    } else if(strcmp(argv[i],"-intensity")==0) {
      intensity=atof(argv[++i]);
    } else if (strcmp(argv[i], "-nldivs")==0) {
      nldivs=atoi(argv[++i]);
    } else if(strcmp(argv[i],"-nsamples")==0) {
      nsamples=atoi(argv[++i]);
    } else if (strcmp(argv[i],"-nsdivs")==0) {
      num_sample_divs=atoi(argv[++i]);
    } else if(strcmp(argv[i],"-depth")==0) {
      depth=atoi(argv[++i]);
    }
    else if(strcmp(argv[i],"-tex_size")==0 || strcmp(argv[i],"-tex_res")==0) {
      tex_size=atoi(argv[++i]);
    }
    else if(strcmp(argv[i],"-radius")==0) {
      radius=atof(argv[++i]);
    }
    else if(strcmp(argv[i],"-file")==0 || strcmp(argv[i],"-i")==0) {
      filename=argv[++i];
      cerr << "Reading from file "<<filename<<endl;
    } else if(strcmp(argv[i],"-numvars")==0) {
      numvars=atoi(argv[++i]);
      cerr << "Operating with "<<numvars<<" number of variables per sphere.\n";
    } else if(strcmp(argv[i],"-timevary")==0) {
      filename=argv[++i];
      timevary=true;
      cerr << "Reading from timelist "<<filename<<endl;
    } else if (strcmp(argv[i],"-bg")==0) {
      bg=argv[++i];
    } else if (strcmp(argv[i],"-o")==0) {
      outfile=argv[++i];
    } else if (strcmp(argv[i],"-np")==0) {
      nworkers=atoi(argv[++i]);
    } else if (strcmp(argv[i],"-nsides")==0) {
      gridcellsize=atoi(argv[++i]);
    } else if (strcmp(argv[i],"-gdepth")==0) {
      griddepth=atoi(argv[++i]);
    } else if (strcmp(argv[i],"-no_dilate")==0) {
      dilate=false;
    } else if (strcmp(argv[i],"-s")==0) {
      support=atoi(argv[++i]);
    } else if (strcmp(argv[i],"-wa")==0) {
      use_weighted_ave=1;
    } else if (strcmp(argv[i],"-thresh")==0) {
      threshold=atof(argv[++i]);
    } else if (strcmp(argv[i],"-lum")==0) {
      luminance=atof(argv[++i]);
    } else if (strcmp(argv[i],"-workload")==0) {
      work_load_max=atoi(argv[++i]);
    } else if (strcmp(argv[i],"-start")==0) {
      next_sphere=(size_t)atoi(argv[++i]);
    } else if (strcmp(argv[i],"-ambientonly")==0) {
      noShadows=true;
      noDirectL=true;
    } else if (strcmp(argv[i],"-no_shadows")==0) {
      noShadows=true;
    } else {
      if (strcmp(argv[i], "--help")!=0) {
        cerr<<"unrecognized option \""<<argv[i]<<"\""<<endl;
        cerr<<"valid options are: \n";
      }

      cerr<<"  -light <lx> <ly> <lz>   position of light source (-0.25, 0.2, -0.1)\n";
      cerr<<"  -lr <float>             radius of light source (0.01)\n";
      cerr<<"  -intensity <float>      intensity of light source (1000.0)\n";
      cerr<<"  -nldivs <int>           number of samples for each of <phi, theta> (0)\n";
      cerr<<"  -ldistance <float>      distance of light source from geometry center (10.0)\n";
      cerr<<"  -nsamples <int>         maximum number of samples per texel (49)\n";
      cerr<<"  -nsdivs <int>           number of samples divisions (0)\n";
      cerr<<"  -depth <int>            maximum ray depth (3)\n";
      cerr<<"  -tex_res <int>          texture resolution (16)\n";
      cerr<<"  -radius <float>         sphere radius (0.0)\n";
      cerr<<"  -i <filename>           input filename (null)\n";
      cerr<<"  -numvars <int>          number of variables in file (3)"<<endl;
      cerr<<"  -timevary <filename>    input filename timelist  (null)\n";
      cerr<<"  -bg <filename>          background image name (null)\n";
      cerr<<"  -o <filename>           basename of texture files (null)\n";
      cerr<<"  -np <int>               number of processors to use (1)\n";
      cerr<<"  -nsides <int>           grid cell size (6)\n";
      cerr<<"  -gdepth <int>           grid depth (2)\n";
      cerr<<"  -no_dilate              do not dilate textures before writing (false)\n";
      cerr<<"  -s <int>                size of support kernel for dilation (2)\n";
      cerr<<"  -wa                     use weighted averaging during dilation (false)\n";
      cerr<<"  -t <float>              threshold for contribution determination (0.3)\n";
      cerr<<"  -lum <float>            luminance value (1.0)"<<endl;
      cerr<<"  -workload <int>         size of the maximum work load (1000)"<<endl;
      cerr<<"  -start <int>            start at given sphere (0)"<<endl;
      cerr<<"  -ambientonly            only compute the ambient term\n";
      cerr<<"  -no_shadows             don't compute shadows\n";
      exit(1);
    }
  }

  // Create the light
  PathTraceLight ptlight(Point(lx, ly, lz), lr, intensity);
  
  // Create the geometry
  Object *geometry;
  if (filename)
    geometry=make_geometry_fromfile(filename, timevary);
  else
    geometry=make_geometry();
  
  // Create the background
  EnvironmentMapBackground *emap=new EnvironmentMapBackground(bg, Vector(0,0,-1));
  if (emap->valid() != true) {
    // try a local copy
    delete emap;
    emap=new EnvironmentMapBackground("up-down.ppm", Vector(0,0,-1));
    if (emap->valid() != true) {
      return 0;
    }
  }

  // Create the context for rendering
  Semaphore sem("genpttex::Semaphore", nworkers);

  PathTraceContext* ptcontext;
  if (nldivs<=0) {
    // Sample the sphere of light positions
    cerr << "genpttex::There's no constructure for this yet\n";
    return 0;
    //    ptcontext = new PathTraceContext(luminance, nldivs, ldistance,
    ptcontext = new PathTraceContext(luminance, ptlight,
                                     geometry, emap,
                                     nsamples, num_sample_divs,
                                     depth, dilate,
                                     support, use_weighted_ave,
                                     threshold, &sem);
    
  } else {
    // Single, fixed light source
    ptcontext = new PathTraceContext(luminance, ptlight,
                                     geometry, emap,
                                     nsamples, num_sample_divs,
                                     depth, dilate,
                                     support, use_weighted_ave,
                                     threshold, &sem);
  }
  
  if (outfile==0)
    outfile="particle";
  
  if (noShadows) ptcontext->shadowsOff();
  if (noDirectL) ptcontext->directLightingOff();
  
  // Partition the spheres out and generate textures
  if (nworkers<=1) {
    Group *texture_spheres;
    
    if (filename)
      texture_spheres=get_next_sphere_set(total_num_spheres);
    else
      texture_spheres=dynamic_cast<Group*>(geometry);
    
    PathTraceWorker ptworker(texture_spheres, ptcontext, outfile);
    
    ptworker.run();
  } else {
    // We need to determine how many spheres to do per work unit
    size_t work_load=(size_t)ceil((double)total_num_spheres/nworkers);
    
    // This prevents the work load from getting too large
    if (work_load > work_load_max)
      work_load=work_load_max;
    
    while (next_sphere < total_num_spheres) {
      sem.down();
      
      // Get the next range of spheres
      size_t last_work=next_sphere;
      Group *work_unit=get_next_sphere_set(work_load);
      
      // Create a thread
      PathTraceWorker *ptworker=new PathTraceWorker(work_unit, ptcontext,
						      outfile, last_work);
      Thread *thread=new Thread(ptworker, "PathTraceWorker");
      thread->detach();
    }
    sem.down(nworkers);
  }
      
  return 0;
}
