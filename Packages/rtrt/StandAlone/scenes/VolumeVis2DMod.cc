//#include <Packages/rtrt/Core/TimeObj.h>
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CatmullRomSpline.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Volvis2DDpy.h>
#include <Packages/rtrt/Core/VolumeVis2D.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Core/Thread/Thread.h>
#include <nrrd.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <string>

using namespace std;
using namespace rtrt;
using SCIRun::Thread;

// template<class T>
VolumeVis2D *create_volume_from_nrrd2( char *filename1, char *filename2,
				       bool override_data_min,
				       double data_min_in,
				       bool override_data_max,
				       double data_max_in,
				       double spec_coeff, double ambient,
				       double diffuse, double specular,
				       Volvis2DDpy *dpy )
{
  BrickArray3<Voxel2D<float> > data;
  float v_data_min = FLT_MAX;
  float v_data_max = -FLT_MAX;
  float g_data_min = FLT_MAX;
  float g_data_max = -FLT_MAX;
  Point minP, maxP;
  // Do the nrrd stuff
  Nrrd *n1 = nrrdNew();
  // load the nrrd in
  if( nrrdLoad( n1, filename1 )) {
    char *err = biffGet(NRRD);
    cerr << "Error reading nrrd " << filename1 << ": " << err << "\n";
    free( err );
    biffDone(NRRD);
    return 0;
  }
  Nrrd *n2 = nrrdNew();
  // load the second nrrd in
  if( nrrdLoad( n2, filename2 )) {
    char *err = biffGet(NRRD);
    cerr << "Error reading nrrd " << filename2 << ": " << err << "\n";
    free( err );
    biffDone(NRRD);
    return 0;
  }
  // check to make sure the dimensions are good
  if ( !(n1->dim == 3 && n2->dim == 3) ) {
    cerr << "VolumeVis2DMod error: nrrd->dim="<<n1->dim<<"\n";
    cerr << "VolumeVis2DMod error: nrrd->dim="<<n2->dim<<"\n";
    cerr << "  Can only deal with 3-dimensional scalar fields... sorry.\n";
    return 0;
  }

  // convert the type to floats if you need to
  size_t num_elements = nrrdElementNumber(n1);
  size_t num_elements2 = nrrdElementNumber(n2);
  if( num_elements != num_elements2 ) {
    cerr << "Both scalar fields must be of the same size.\n";
    return 0;
  }

  cerr << "Number of data members = " << num_elements << endl;

  if (n1->type != nrrdTypeFloat) {
    cerr << "Converting type from ";
    switch(n1->type) {
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
      cerr << " to nrrdTypeFloat\n";
  }
    Nrrd *new_n1 = nrrdNew();
    nrrdConvert( new_n1, n1, nrrdTypeFloat );
    // since the data was copied, blow away the memory for the old nrrd
    nrrdNuke(n1);
    n1 = new_n1;
    cerr << "Number of data members = " << num_elements << endl;
  }
  if (n2->type != nrrdTypeFloat) {
    cerr << "Converting type from ";
    switch(n2->type) {
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
      cerr << " to nrrdTypeFloat\n";
    }
    Nrrd *new_n2 = nrrdNew();
    nrrdConvert( new_n2, n2, nrrdTypeFloat );
    // since the data was copied, blow away the memory for the old nrrd
    nrrdNuke(n2);
    n2 = new_n2;
    cerr << "Number of data members = " << num_elements << endl;
  }

  // get the dimensions
  int nx, ny, nz;
  int x = 0;
  int y = 1;
  int z = 2;
  nx = n1->axis[x].size;
  ny = n1->axis[y].size;
  nz = n1->axis[z].size;
  cout << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
  cout << "total = " << nz * ny * nz << endl;
  cout << "spacing = " << n1->axis[x].spacing << " x "<<n1->axis[y].spacing
       << " x "<<n1->axis[z].spacing<< endl;
  // Don't need to check the spacing of dimension 0 because it is the voxel.
  for (int i = x; i <= z; i++) {
    if (!(AIR_EXISTS(n1->axis[i].spacing))) {
      cout <<"spacing for axis "<<i<<" does not exist.  Setting to 1.\n";
      n1->axis[i].spacing = 1;
    } // if()
    if (!(AIR_EXISTS(n2->axis[i].spacing))) {
      cout <<"spacing for axis "<<i<<" does not exist.  Setting to 1.\n";
      n2->axis[i].spacing = 1;
    } // if()
  } // for()
  data.resize(nx,ny,nz); // resize the bricked data
  // get the physical bounds
  minP = Point(0,0,0);
  maxP = Point((nx - 1) * n1->axis[x].spacing,
	       (ny - 1) * n1->axis[y].spacing,
	       (nz - 1) * n1->axis[z].spacing);
  // lets normalize the dimensions to 1
  Vector size = maxP - minP;
  // find the biggest dimension
  double max_dim = Max(Max(size.x(),size.y()),size.z());
  maxP = ((maxP-minP)/max_dim).asPoint();
  minP = Point(0,0,0);
  // copy the data into the brickArray
  cerr << "Number of data members = " << num_elements << endl;
  float *p1 = (float*)n1->data; // get pointer to raw data in the first nrrd
  float *p2 = (float*)n2->data; // get pointer to raw data in the second nrrd

  // merge the two scalar fields...
  for( int z = 0; z < nz; z++ ) {
    for( int y = 0; y < ny; y++ ) {
      for( int x = 0; x < nx; x++ ) {
	float v_val = *p1++;
	float g_val = *p2++;
	data(x,y,z) = Voxel2D<float>(v_val, g_val);

	// also find the min and max
	if (v_val < v_data_min)
	  v_data_min = v_val;
	else if (v_val > v_data_max)
	  v_data_max = v_val;
	if (g_val < g_data_min)
	  g_data_min = g_val;
	else if (g_val > g_data_max)
	  g_data_max = g_val;
      } // for( x )
    } // for( y )
  } // for( z )

#if 0
  // compute the min and max of the data
  double dmin1, dmax1, dmin2, dmax2;
  nrrdMinMaxFind( &dmin1, &dmax1, n1 );
  nrrdMinMaxFind( &dmin2, &dmax2, n2 );
  data_min = <float>min( dmin1, dmin2 );
  data_max = <float>Max( dmax1, dmax2 );
#endif
  // delete the memory that is no longer in use
  nrrdNuke(n1);
  nrrdNuke(n2);

  // override the min and max if it was passed in
  if (override_data_min)
    v_data_min = data_min_in;
  if (override_data_max)
    v_data_max = data_max_in;

  cout << "minP = "<<minP<<", maxP = "<<maxP<<endl;

  return new VolumeVis2D(data,
			 Voxel2D<float>(v_data_min, g_data_min),
			 Voxel2D<float>(v_data_max, g_data_max),
			 nx, ny, nz,
			 minP, maxP,
			 spec_coeff, ambient, diffuse,
			 specular, dpy);
}


// template<class T>
VolumeVis2D *create_volume_from_nrrd(char *filename,
				     bool override_data_min,double data_min_in,
				     bool override_data_max,double data_max_in,
				     double spec_coeff, double ambient,
				     double diffuse, double specular,
				     Volvis2DDpy *dpy)
{
  BrickArray3<Voxel2D<float> > data;
  float v_data_min = FLT_MAX;
  float v_data_max = -FLT_MAX;
  float g_data_min = FLT_MAX;
  float g_data_max = -FLT_MAX;
  Point minP, maxP;
  // Do the nrrd stuff
  Nrrd *n = nrrdNew();
  // load the nrrd in
  if (nrrdLoad(n,filename)) {
    char *err = biffGet(NRRD);
    cerr << "Error reading nrrd "<< filename <<": "<<err<<"\n";
    free(err);
    biffDone(NRRD);
    return 0;
  }
  // check to make sure the dimensions are good
  if ( !(n->dim == 4 || n->dim == 3) ) {
    cerr << "VolumeVis2DMod error: nrrd->dim="<<n->dim<<"\n";
    cerr << "  Can only deal with 3- and 4-dimensional scalar fields... sorry.\n";
    return 0;
  }
  if ( n->dim == 4 && n->axis[0].size != 2 ) {
    cerr << "Can only handle bivariate data.\n";
    //    return 0;
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
    Nrrd *new_n = nrrdNew();
    nrrdConvert( new_n, n, nrrdTypeFloat );
    // since the data was copied, blow away the memory for the old nrrd
    nrrdNuke(n);
    n = new_n;
    cerr << "Number of data members = " << num_elements << endl;
  }

  // get the dimensions
  int nx, ny, nz;
  int x, y, z;
  if( n->dim == 4 ) {
    x = 1;
    y = 2;
    z = 3;
  }
  else if( n->dim == 3 ) {
    x = 0;
    y = 1;
    z = 2;
  }
  nx = n->axis[x].size;
  ny = n->axis[y].size;
  nz = n->axis[z].size;
  cout << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
  cout << "total = " << nz * ny * nz << endl;
  cout << "spacing = " << n->axis[x].spacing << " x "<<n->axis[y].spacing<< " x "<<n->axis[z].spacing<< endl;
  // Don't need to check the spacing of dimension 0 because it is the voxel.
  for (int i = x; i <= z; i++)
    if (!(AIR_EXISTS(n->axis[i].spacing))) {
      cout <<"spacing for axis "<<i<<" does not exist.  Setting to 1.\n";
      n->axis[i].spacing = 1;
    }
  data.resize(nx,ny,nz); // resize the bricked data
  // get the physical bounds
  minP = Point(0,0,0);
  maxP = Point((nx - 1) * n->axis[x].spacing,
	       (ny - 1) * n->axis[y].spacing,
	       (nz - 1) * n->axis[z].spacing);
  // lets normalize the dimensions to 1
  Vector size = maxP - minP;
  // find the biggest dimension
  double max_dim = Max(Max(size.x(),size.y()),size.z());
  maxP = ((maxP-minP)/max_dim).asPoint();
  minP = Point(0,0,0);
  // copy the data into the brickArray
  cerr << "Number of data members = " << num_elements << endl;
  float *p = (float*)n->data; // get the pointer to the raw data
  float *raw_data = (float*)n->data;
  // if no gradient field, we must create one...
  if( n->dim == 3 ) {
      float dx;
      float dy;
      float dz;
      for( int z = 0; z < nz; z++ ) {
	int xyz = z*ny*nx;
	for( int y = 0; y < ny; y++ )
	  {
	    int xy = y*nx;
	    for( int x = 0; x < nx; x++ ) {
	      float v_val = *p++;
	      if( !(x == 0 || x == nx-1) )
		dx = raw_data[xyz+xy+x+1] - raw_data[xyz+xy+x-1];
	      else if( x == 0 )
		dx = raw_data[xyz+xy+x+2] - raw_data[xyz+xy+x];
	      else if( x == nx-1 )
		dx = raw_data[xyz+xy+x] - raw_data[xyz+xy+x-2];
	      
	      if( !(y == 0 || y == ny-1) )
		dy = raw_data[xyz+xy+x+nx] - raw_data[xyz+xy+x-nx];
	      else if( y == 0 )
		dy = raw_data[xyz+xy+x+2*nx] - raw_data[xyz+xy+x];
	      else if( y == ny-1 )
		dy = raw_data[xyz+xy+x] - raw_data[xyz+xy+x-2*nx];
	      
	      if( !(z == 0 || z == nz-1) )
		dz = raw_data[xyz+xy+x+nx*ny] - raw_data[xyz+xy+x-nx*ny];
	      else if( z == 0 )
		dz = raw_data[xyz+xy+x+2*nx*ny] - raw_data[xyz+xy+x];
	      else if( z == nz-1 )
		dz = raw_data[xyz+xy+x] - raw_data[xyz+xy+x-2*nx*ny];
	      
	      float g_val = sqrt( dx*dx + dy*dy + dz*dz );
	      data(x,y,z) = Voxel2D<float>(v_val, g_val);
	      // also find the min and max
	      if (v_val < v_data_min)
		v_data_min = v_val;
	      else if (v_val > v_data_max)
		v_data_max = v_val;
	      if (g_val < g_data_min)
		g_data_min = g_val;
	      else if (g_val > g_data_max)
		g_data_max = g_val;
	    } // for( z )
	  } // for( y )
      } // for( x )
  } // if( p_size == 1 )

  // ...direct access to gradient field otherwise
  else
    for (int z = 0; z < nz; z++)
      for (int y = 0; y < ny; y++)
	for (int x = 0; x < nx; x++) {
	  float v_val = *p++;
	  float g_val = *p++;
	data(x,y,z) = Voxel2D<float>(v_val,g_val);
	// also find the min and max
	if (v_val < v_data_min)
	  v_data_min = v_val;
	else if (v_val > v_data_max)
	  v_data_max = v_val;
	if (g_val < g_data_min)
	  g_data_min = g_val;
	else if (g_val > g_data_max)
	  g_data_max = g_val;
	} // for( x )
#if 0
  // compute the min and max of the data
  double dmin,dmax;
  nrrdMinMaxFind(&dmin,&dmax,n);
  data_min = <T>dmin;
  data_max = <T>dmax;
#endif
  // delete the memory that is no longer in use
  nrrdNuke(n);


  // override the min and max if it was passed in
  if (override_data_min)
    v_data_min = data_min_in;
  if (override_data_max)
    v_data_max = data_max_in;

  cout << "minP = "<<minP<<", maxP = "<<maxP<<endl;

  return new VolumeVis2D(data,
			 Voxel2D<float>(v_data_min, g_data_min),
			 Voxel2D<float>(v_data_max, g_data_max),
			 nx, ny, nz,
			 minP, maxP,
			 spec_coeff, ambient, diffuse,
			 specular, dpy);
}  




extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  int nx = 20;
  int ny = 30;
  int nz = 40;
  int scene_type = 0;
  vector<string> data_files;
  bool cut=false;
  int ncolors=256;
  float t_inc = 0.01;
  double spec_coeff = 64;
  double ambient = 0.5;
  double diffuse = 1.0;
  double specular = 1.0;
  float val=1;
  bool override_data_min = false;
  bool override_data_max = false;
  float data_min_in = -1;
  float data_max_in = -1;
  float frame_rate = 3;
//    int color_map_type = 0;
//    char *nrrd_color_map_file = 0;
//    char *nrrd_alpha_map = 0;
  bool loadWidgetFile = false;
  char* widgetFile;
  Color bgcolor(1.,1.,1.);
  
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-dim")==0){
      i++;
      if(sscanf(argv[i], "%dx%dx%d", &nx, &ny, &nz) != 3){
	cerr << "Error parsing dimensions: " << argv[i] << '\n';
	cerr << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
	exit(1);
      } 
    }
    else if (strcmp(argv[i], "-type")==0) {
      i++;
      if (strcmp(argv[i], "nrrd") == 0) {
	scene_type = 6;
	i++;
	data_files.push_back(argv[i]);
      }
      else if (strcmp(argv[i], "nrrd2") == 0) {
	scene_type = 7;
	data_files.push_back(argv[++i]);
	data_files.push_back(argv[++i]);
      }
      else if (strcmp(argv[i], "nrrdlist") == 0) {
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
      } else {
	scene_type = atoi(argv[i]);
	if (scene_type == 5) {
	  i++;
	  val = atof(argv[i]);
	}
      }
    } else if(strcmp(argv[i], "-cut")==0){
      cut=true;
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
    } else if(strcmp(argv[i], "-bgcolor")==0){
      float r,g,b;
      r = atof(argv[++i]);
      g = atof(argv[++i]);
      b = atof(argv[++i]);
      bgcolor = Color(r,g,b);
    } else if(strcmp(argv[i], "-loadfile")==0){
      loadWidgetFile = true;
      widgetFile = argv[++i];
//        ifstream in(argv[i]);
//        in >> widgetFile;
    } else {
      cerr << "Unknown option: " << argv[i] << '\n';
      cerr << "Valid options for scene: " << argv[0] << '\n';
      cerr << " -dim [int]x[int]x[int]";
      cerr << " -type [int or \"nrrd\"]\n";
      cerr << "\t\t0 - data = z\n";
      cerr << "\t\t1 - data = y*z\n";
      cerr << "\t\t2 - data = x*y*z\n";
      cerr << "\t\t3 - data = y+z\n";
      cerr << "\t\t4 - data = x+y+z\n";
      cerr << "\t\t5 [val=float] - data = val\n";
      cerr << "\t\tnrrd [path to nrrd file] (-dim parameter ignored)\n";
      cerr << " -cut - turn on the cutting plane\n";
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
      cerr << " -loadfile [filename.txt] - read in a widget configuration to load upon initialization\n";
      return 0;
    }
  }

  if (scene_type == 6 && data_files.size() < 1) {
    cerr << "Need at least one nrrd data file.\n";
    return 0;
  }
  else if( scene_type == 7 && data_files.size() < 2 ) {
    cerr << "Need at least two nrrd data files.\n";
    return 0;
  }
  
  Camera cam(Point(0.5,0.5,3), Point(0.5,0.5,0.5),
	     Vector(0,1,0), 40.0);
  
  double ambient_scale=1.0;
  
  //  double bgscale=0.5;
  //  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  
  Volvis2DDpy *dpy = new Volvis2DDpy(t_inc, cut);
  if(loadWidgetFile)
    dpy->loadWidgets( widgetFile );

  PlaneDpy *cdpy;
  if(cut) {
    cdpy = new PlaneDpy(Vector(0,0,1), Point(0,0,0.5));
    //    obj = new VolumeCutPlane( obj, cdpy );
    //    obj->set_matl(matl0);
    (new Thread( cdpy, "Cutting plane display thread" ))->detach();
  }
  // Generate the data
  Object *obj;
  if( scene_type == 7 ) {
    if( data_files.size() > 1 ) {
      SelectableGroup *timeobj = new SelectableGroup(1.0/frame_rate);
      timeobj->set_name( "Volume Data" );
      char *myfile1 = strdup( data_files[0].c_str());
      char *myfile2 = strdup( data_files[1].c_str());
      VolumeVis2D *vv2d = create_volume_from_nrrd2
	(myfile1, myfile2, override_data_min, data_min_in,
	 override_data_max, data_max_in, spec_coeff,
	 ambient, diffuse, specular, dpy);
      if(cut) {
	vv2d->initialize_cuttingPlane( cdpy );
	vv2d->initialize_callbacks();
      }

      Object *volume = (Object*)vv2d;
      volume->set_name("VolumeVis2D from "+ string(myfile1) + " and " + string(myfile2));
      // add the volume to the timeobj
      if( myfile1 )
	free( myfile1 );
      if( myfile2 )
	free( myfile2 );
      if( volume != 0 ) {
	// there was a problem
      timeobj->add(volume);
      obj = (Object*)timeobj;
      } // if()
    } // if()
  }

  else if (scene_type == 6) {
    if (data_files.size() > 1) {
      // create a timeobj
      //      TimeObj *timeobj = new TimeObj(frame_rate);
      SelectableGroup *timeobj = new SelectableGroup(1.0/frame_rate);
      timeobj->set_name("Volume Data");
      for(unsigned int i = 0; i < data_files.size(); i++) {
	char *myfile = strdup(data_files[i].c_str());
	VolumeVis2D *vv2d = create_volume_from_nrrd
	  (myfile, override_data_min, data_min_in,
	   override_data_max, data_max_in, spec_coeff,
	   ambient, diffuse, specular, dpy);
	if(cut) {
	  vv2d->initialize_cuttingPlane( cdpy );
	  vv2d->initialize_callbacks();
	}

	Object *volume = (Object*)vv2d;
	volume->set_name("VolumeVis2D from "+ string(myfile));
	// add the volume to the timeobj
	if (myfile)
	  free(myfile);
	if (volume == 0)
	  // there was a problem
	  continue;
	timeobj->add(volume);
      }
      obj = (Object*)timeobj;
    } else {
      char *myfile = strdup(data_files[0].c_str());
      VolumeVis2D *vv2d = create_volume_from_nrrd
	(myfile, override_data_min, data_min_in,
	 override_data_max, data_max_in, spec_coeff,
	 ambient, diffuse, specular, dpy);
      if(cut) {
	vv2d->initialize_cuttingPlane( cdpy );
	vv2d->initialize_callbacks();
      }

      obj = (Object*)vv2d;
      obj->set_name("VolumeVis2D from "+ string(myfile));
      if (myfile)
	free(myfile);
    }
  } else {
#if 1
    cerr << "Don't know how to make a scene of type "<<scene_type<<".\n";
    return 0;
#else


    VolumeVis2D *vv2d =  (Object*) create_volume_default
	(scene_type, val, nx, ny, nz, override_data_min, data_min_in,
	 override_data_max, data_max_in, spec_coeff, ambient,
	 diffuse, specular, dpy);
    if(cut) {
      vv2d->initialize_cuttingPlane( cdpy );
      vv2d->initialize_callbacks();
    }
    obj = (Object*)vv2d;

#endif
  }

  Group* all = new Group();
  all->add(obj);

  (new Thread(dpy, "VolumeVis2D display thread"))->detach();
  //dpy->run();

  Plane groundplane ( Point(-500, 300, 0), Vector(7, -3, 2) );
  Color cup(0.9, 0.7, 0.3);
  Color cdown(0.0, 0.0, 0.2);

  Scene* scene=new Scene(obj, cam,
			 bgcolor, cdown, cup, groundplane, 
			 ambient_scale);

  Light *light0 = new Light(Point(500,-300,300), Color(.8,.8,.8), 0);
  light0->name_ = "light 0";
  scene->add_light(light0);
  //scene->shadow_mode=1;
  scene->attach_display(dpy);
  if (cut)
    scene->attach_display(cdpy);
  scene->addObjectOfInterest(obj,true);
  
  return scene;
}





