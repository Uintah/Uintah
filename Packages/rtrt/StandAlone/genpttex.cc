
/*

 A useful genpttex command:

 ./genpttex -num_samples 100 -depth 3 -tex_size 16 -light_pos 5 10 7.5 -intensity 5000

 Some useful unu commands

 # Use these to quantize the textures and copy them to ppms
 
 unu gamma -g 2 -i sphere00000.nrrd | unu quantize -b 8 | unu dice -a 3 -o sphere
 echo 'for T in sphere*.png; do unu save -f pnm -i $T -o `basename $T .png`.ppm; done' | bash

 # This copies the first column to the end to help with texture blending.
 
 echo 'for T in sphere?.ppm; do unu slice -a 1 -p M -i $T | unu reshape -s 3 1 64 | unu join -a 1 -i - $T -o $T;done' | bash
*/

#include <Packages/rtrt/Core/PathTracer/PathTraceEngine.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/GridSpheres.h>
#include <Packages/rtrt/Core/GridSpheresDpy.h>

#include <Core/Thread/Thread.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

int tex_size = 16;
double radius = 0;
double radius_factor = 1.0;
size_t total_num_spheres = 0;
Array1<float *> spheredata;
Array1<char *> deleteme;
Array1<size_t> spheredatasize;
size_t last_sphere = 0;
#define NUM_DATA 3
int gridcellsize = 6;
int griddepth = 2;

void readSpheres(char *spherefile);

Object *make_geometry( )
{
  Group* group=new Group();

  if (radius <= 0)
    radius = 1;

  total_num_spheres = 8;
  float *data = new float[total_num_spheres*NUM_DATA];
  spheredata.add(data);
  spheredatasize.add(total_num_spheres);
  for(int z = -1; z <= 1; z+=2)
    for(int y = -1; y <= 1; y+=2)
      for(int x = -1; x <= 1; x+=2)
	{
	  *data++ = x;
	  *data++ = y;
	  *data++ = z;
	  data += NUM_DATA-3;
	}

  // Now generate some geometry
  data = spheredata[0];

#if 0
  // Write out sphere data into a file
  FILE *out = fopen("spheredata", "wb");
  if (out) {
    size_t wrote = fwrite(data, sizeof(float), total_num_spheres*NUM_DATA, out);
    if (wrote != total_num_spheres*NUM_DATA) {
      cerr << "Couldn't write out all the values.  Wrote "<<wrote<<" items.\n";
    }
    fclose(out);
  }
#endif

  for(int i = 0; i < total_num_spheres; i++) {
    double x = *data; data++;
    double y = *data; data++;
    double z = *data; data++;
    Point center(x, y, z);
    data += NUM_DATA-3;
    //    group->add( new Sphere(0, center, radius) );
    group->add( new TextureSphere(center, radius, tex_size) );
  }
  
  return group;
}

Object *make_geometry_fromfile(char *filename)
{
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

  // Now we need an accelaration structure.
  //
  // Discussing this, it would be best to use a GridSpheres.  In order
  // to use one we need to create a large array with our floating
  // point data.

  // Allocate memory
  float *alldata = new float[total_num_spheres*NUM_DATA];
  float *data = alldata;
  for(int i = 0; i < spheredata.size(); i++) {
    size_t size = spheredatasize[i] * sizeof(float) * NUM_DATA;
    memcpy(data, spheredata[i], size);
    // Free the old memory
    free(deleteme[i]);
    // Set the pointer back into our large array
    spheredata[i] = data;
    // Move our pointer to the next set of spheres
    data += size/sizeof(float);
  }

  // A fake material list
  Material *matls[1] = { 0 };
  // Here is our geometry
  cout << "total_num_spheres = "<<total_num_spheres<<endl;
  GridSpheres *gridspheres = new GridSpheres(alldata, 0, 0, total_num_spheres,
					     NUM_DATA-3,
					     gridcellsize, griddepth,
					     radius*radius_factor,
					     1, matls, 0);
  // We need a GridSphereDpy as GridSphere needs one to compute intersections
  GridSpheresDpy *griddpy = new GridSpheresDpy(0);
  griddpy->attach(gridspheres);
  // This will set up the rendering parameters
  griddpy->setup_vars();
  
  return gridspheres;
}

Group* get_next_sphere_set() {
  Group *group = new Group();

  // create spheres
  float *data = spheredata[0];
  for (int i=0;i<total_num_spheres;i++) {
    double x = *data; data++;
    double y = *data; data++;
    double z = *data; data++;
    Point center(x, y, z);
    data += NUM_DATA-3;
    //    group->add( new Sphere(0, center, radius) );
    group->add( new TextureSphere(center, radius, tex_size) );
  }

  return group;
}

void
readSpheres(char* spherefile)
{
  // open the header file
  bool found_header = false;
  
  char buf[300];
  sprintf(buf, "%s.meta", spherefile);
  ifstream in(buf);
  int nspheres = 0;
  int numvars = 3;
  double file_radius=radius;
  float* mins;
  float* maxs;
  vector<float> mins_vec,maxs_vec;
  
  if (in)
  {
    // the file existed
    found_header = true;
    in >> nspheres;
    in >> file_radius;
    if (radius != 0) {
      // Use the radius specified globally
      file_radius = radius;
    } else {
      // Use the file radius
      radius = file_radius;
    }
    int varcount = 0;
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
    numvars = varcount;
    if (numvars != NUM_DATA) {
      cerr << "readSpheres::Can only deal with point data (3 vars)\n";
      return;
    }
    mins = (float*)malloc(numvars*sizeof(float));
    maxs = (float*)malloc(numvars*sizeof(float));
    for (int i = 0; i < numvars; i++) {
      mins[i] = mins_vec[i];
      maxs[i] = maxs_vec[i];
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
  if(in_fd==-1){
    perror("readSpheres::open");
    exit(1);
  }
  int d_mem=512;

  struct stat statbuf;
  if(fstat(in_fd, &statbuf) == -1){
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
    nspheres = (int)(statbuf.st_size/(numvars*sizeof(float)));
  }

  cerr << "Reading "<<nspheres<<" spheres\n";
  
  // read in the data
  char* odata=(char*)malloc(numvars*(nspheres+nsph)*sizeof(float)+d_mem);
  unsigned long addr=(unsigned long)odata;
  unsigned long off=addr%d_mem;
  if(off){
    addr+=d_mem-off;
  }
  float *data=(float*)addr;
  
  int total=0;
  float* p=data;
  for(;;){
    long s=read(in_fd, p, numvars*nsph*sizeof(float));
    if(s==0)
      break;
    if(s==-1){
      perror("readSpheres::read");
      exit(1);
    }
    int n=(int)(s/(numvars*sizeof(float)));
    total+=n;
    p+=n*numvars;
  }
  close(in_fd);

  if(total != nspheres){
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
  double lx=-0.25, ly=0.2, lz=-0.1;
  double lr = 0.01;
  double intensity=1000.0;
  int num_samples=10000;
  int depth=3;
  char *filename = 0;
  char *bg="/home/sci/cgribble/research/datasets/mpm/misc/envmap.ppm";

  for(int i=1;i<argc;i++) {
    if(strcmp(argv[i], "-light_pos")==0) {
      lx=atof(argv[++i]);
      ly=atof(argv[++i]);
      lz=atof(argv[++i]);
    } else if(strcmp(argv[i], "-lr")==0) {
      lr = atof(argv[++i]);
    } else if(strcmp(argv[i],"-intensity")==0) {
      intensity=atof(argv[++i]);
    }
    else if(strcmp(argv[i],"-num_samples")==0) {
      num_samples=atoi(argv[++i]);
    }
    else if(strcmp(argv[i],"-depth")==0) {
      depth=atoi(argv[++i]);
    }
    else if(strcmp(argv[i],"-tex_size")==0) {
      tex_size=atoi(argv[++i]);
    }
    else if(strcmp(argv[i],"-radius")==0) {
      radius = atof(argv[++i]);
    }
    else if(strcmp(argv[i],"-file")==0) {
      filename = argv[++i];
      cerr << "Reading from file "<<filename<<endl;
    } else if (strcmp(argv[i],"-bg")==0) {
      bg = argv[++i];
    }
    else {
      cerr<<"unrecognized option \""<<argv[i]<<"\""<<endl;

      cerr << "valid options are: \n";
      cerr << "-light_pos <lx> <ly> <lz>\n";
      cerr << "-lr <float>\n";
      cerr << "-intensity <float>\n";
      cerr << "-num_samples <int>\n";
      cerr << "-depth <int>\n";
      cerr << "-tex_size <int>\n";
      cerr << "-file <filename>\n";
      cerr << "-bg <background image>\n";
      exit(1);
    }
  }

  // Create the light
  PathTraceLight ptlight(Point(lx, ly, lz), lr, intensity*Color(1,1,1));

  // Create the geometry
  Object *geometry;
  if (filename)
    geometry = make_geometry_fromfile(filename);
  else
    geometry = make_geometry();

  // Create the background
  EnvironmentMapBackground *emap=new EnvironmentMapBackground(bg, Vector(0,1,0));
  if (emap->valid() != true) {
    // try a local copy
    delete emap;
    emap = new EnvironmentMapBackground("./envmap.ppm", Vector(0,1,0));
    if (emap->valid() != true) {
      return 0;
    }
  }

  // Create the context for rendering
  PathTraceContext ptcontext(Color(0.1,0.7,0.2), ptlight, geometry, emap,
			     num_samples, depth);
  
  // Partition the spheres out and generate textures
  Group *texture_spheres;
  if (filename)
    texture_spheres = get_next_sphere_set();
  else
    texture_spheres = dynamic_cast<Group*>(geometry);
  
  PathTraceWorker ptworker(texture_spheres, &ptcontext, "sphere");

  ptworker.run();
      
  return 0;
}


