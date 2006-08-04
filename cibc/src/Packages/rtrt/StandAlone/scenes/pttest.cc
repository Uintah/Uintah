#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/GridSpheresDpy.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/UVSphere2.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Core/TextureGridSpheres.h>
#include <Packages/rtrt/Core/PCAGridSpheres.h>
#include <Packages/rtrt/Core/RegularColorMap.h>

#include <Core/Thread/Thread.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace SCIRun;
using namespace rtrt;
using namespace std;

using std::cerr;
using std::endl;

#define NUM_TEXTURES 8
#define MAX_LINE_LEN 256

TextureGridSpheres* texGridFromFile(char *fname, int tex_res, float radius,
				    int numvars, int nsides, int gdepth,
				    RegularColorMap *cmap, const Color& color);

float radius = 1.0;

/*
  -eye -4.05479 0.0260826 7.411 -lookat 0 0 0 -up 0.125951 0.989877 0.065428 -fov 42.5676

*/

// Returns 1 if there was an error.  This is based on if the texture
// was found.
int add_sphere(char *tex_name, const Point& center, Group *group) {
  ImageMaterial* matl = new ImageMaterial(tex_name, ImageMaterial::Clamp,
					  ImageMaterial::Clamp, 1.0,
					  Color(1.0, 1.0, 1.0), 0);
  if (!matl->valid())
  {
    cerr << "AddSphere::texture is bad :" << tex_name << endl;
    return 1;
  }

  group->add( new UVSphere2(matl, center, radius) );
  return 0;
}

Group *make_geometry(char* tex_names[NUM_TEXTURES])
{
  Group* group=new Group();

  int E = 0;
  int tex_index = 0;
  for(int z = -1; z <= 1; z+=2)
    for(int y = -1; y <= 1; y+=2)
      for(int x = -1; x <= 1; x+=2)
	{
	  if (!E) E |= add_sphere(tex_names[tex_index++],
				  Point(x, y, z),
				  group);
	}

  if (!E)
    return group;
  else
    return 0;
}

Group *make_geometry_tg(char* tex_names[NUM_TEXTURES], int tex_res,
			int nsides, int gdepth,
                        RegularColorMap* cmap, const Color& color)
{
  Group* group = new Group();
  
  float* spheres = new float[NUM_TEXTURES*3];
  unsigned char* tex_data = new unsigned char[NUM_TEXTURES*tex_res*tex_res];
  int nspheres = 0;
  for(int z = -1; z <= 1; z+=2)
    for(int y = -1; y <= 1; y+=2)
      for(int x = -1; x <= 1; x+=2)
	{
	  float *sphere = spheres + nspheres * 3;
	  sphere[0] = x;
	  sphere[1] = y;
	  sphere[2] = z;
	  PPMImage image(tex_names[nspheres]);
	  if (!image.valid()) {
	    cerr << "Error loading texture "<<tex_names[nspheres]<<endl;
	    return 0;
	  }
	  Array2<Color> image_data;
	  int width = 0, height = 0;
	  image.get_dimensions_and_data(image_data, width, height);
	  if (width != tex_res) {
	    cerr << "Texture width ("<<width<<") does not match tex_res ("<<tex_res<<").\n";
	    return 0;
	  }
	  if (height != tex_res) {
	    cerr << "Texture height ("<<height<<") does not match tex_res ("<<tex_res<<").\n";
	    return 0;
	  }

	  // Copy the data over
	  unsigned char* pixel = tex_data + (nspheres * tex_res * tex_res);
	  for (int j = 0; j < height; j++)
	    for (int i = 0; i < width; i++) {
	      Color c = image(i,j) * 255;
	      // Expecting a gray-scale texture, so just take the first channel
	      *pixel = (unsigned char) c.red();
	      pixel++;
	    }
	  
          nspheres++;
	}

  int *tex_indices = 0;
  group->add(new 
	     TextureGridSpheres(spheres, nspheres, 3, radius, tex_indices,
				tex_data, nspheres, tex_res,
				nsides, gdepth, cmap, color));
  return group;
}

extern "C" 
Scene* make_scene(int argc, char** argv, int /*nworkers*/)
{
  double lx=-0.25, ly=0.5, lz=-0.1;
  char *bg="/home/sci/cgribble/research/datasets/mpm/misc/envmap.ppm";
  char *tex_basename="./sphere";
  int tex_res = 16;
  char *infilename=0;
  int nsides = 6;
  int gdepth = 2;
  int numvars = 3;
  Color color(1.0, 1.0, 1.0);
  bool display=false;
  char *cmap_file = 0; // Non zero when a file has been specified
  char *cmap_type = "InvRIsoLum";

  for (int i=1;i<argc;i++)
  {
    if (strcmp(argv[i],"-bg")==0)
      bg = argv[++i];
    else if (strcmp(argv[i],"-i")==0)
      infilename=argv[++i];
    else if (strcmp(argv[i],"-tex")==0)
      tex_basename = argv[++i];
    else if(strcmp(argv[i],"-radius")==0)
      radius=atof(argv[++i]);
    else if(strcmp(argv[i],"-tex_res")==0)
      tex_res=atoi(argv[++i]);
    else if(strcmp(argv[i],"-numvars")==0)
      numvars=atoi(argv[++i]);
    else if(strcmp(argv[i],"-nsides")==0)
      nsides=atoi(argv[++i]);
    else if(strcmp(argv[i],"-gdepth")==0)
      gdepth=atoi(argv[++i]);
    else if(strcmp(argv[i],"-color")==0) {
      color=Color(atof(argv[++i]),
		  atof(argv[++i]),
		  atof(argv[++i]));
    } else if (strcmp(argv[i], "-cmap") == 0) {
      cmap_file = argv[++i];
    } else if (strcmp(argv[i], "-cmaptype") == 0) {
      cmap_type = argv[++i];
    } else if(strcmp(argv[i], "-light_pos")==0) {
      lx=atof(argv[++i]);
      ly=atof(argv[++i]);
      lz=atof(argv[++i]);
    } else if(strcmp(argv[i],"-display")==0)
      display=true;
    else {
      cerr<<"unrecognized option \""<<argv[i]<<"\""<< endl;
      cerr<<"valid options are:"<<endl;
      cerr<<"  -bg <filename>       environment map image file (envmap.ppm)"<<endl;
      cerr<<"  -i <filename>        input sphere file name (null)"<<endl;
      cerr<<"  -tex <filename>      basename of gray-scale texture files (./sphere)"<<endl;
      cerr<<"  -tex_res <int>       resolution of the textures (16)"<<endl;
      cerr<<"  -radius <float>      sphere radius (1.0)"<<endl;
      cerr<<"  -numvars <int>       number of variables (3)"<<endl;
      cerr<<"  -nsides <int>        number of sides for grid cells (6)"<<endl;
      cerr<<"  -gdepth <int>        gdepth of grid cells (2)"<<endl;
      cerr<<"  -color <r> <g> <b>   surface color (1.0, 1.0, 1.0)"<<endl;
      cerr<<"  -cmap <filename>     defaults to inverse rainbow"<<endl;
      cerr<<"  -cmaptype <type>     type of colormap\n";
      cerr<<"  -light_pos <lx> <ly> <lz>   position of light source (-0.25, 0.5, -0.1)\n";
      cerr<<"  -display             use GridSpheresDpy display (false)"<<endl;
      exit(1);
    }
  }

  RegularColorMap *cmap = 0;
  if (cmap_file)
    cmap = new RegularColorMap(cmap_file);
  else {
    int cmap_type_index = RegularColorMap::parseType(cmap_type);
    cmap = new RegularColorMap(cmap_type_index);
  }

  Object *group=0;
  if (infilename) {
    GridSpheresDpy *dpy = new GridSpheresDpy(1);
    GridSpheres *grid = texGridFromFile(infilename, tex_res, radius, numvars,
					nsides, gdepth, cmap, color);
    
    dpy->attach(grid);
    if (display) {
      (new Thread(dpy, "GridSpheres display thread\n"))->detach();
    } else {
      // This will set up the rendering parameters
      dpy->setup_vars();
    }

    group = grid;
  } else {
    char *tex_names[NUM_TEXTURES];
    // Make the tex_names
    size_t name_length = strlen(tex_basename) + 15;
    for(int i = 0; i < NUM_TEXTURES; i++) {
      tex_names[i] = new char[name_length];
      sprintf(tex_names[i], "%s%d.ppm", tex_basename, i); 
    }
  
    if (tex_res > 0)
      group = make_geometry_tg(tex_names, tex_res, nsides, gdepth, cmap,color);
    else
      group = make_geometry(tex_names);
  }
  
  if (!group) {
    cerr << "Could not generate geometry successfully.\n";
    // Then something went wrong and you should kill the scene
    return 0;
  }

  Camera cam(Point(-0.25,-0.1,0.1), Point(0.1,0.075,0.2), Vector(0,0,-1), 15.0);

  double ambient_scale=0.7;
  Color bgcolor(0,0,0);
  Color cdown(1,1,1);
  Color cup(1,1,1);

  rtrt::Plane groundplane(Point(0,0,0), Vector(0,0,-1));
  Scene* scene=new Scene(group, cam, bgcolor, cdown, cup, groundplane,
    ambient_scale, Arc_Ambient);

  EnvironmentMapBackground *emap=new EnvironmentMapBackground(bg, Vector(0,0,-1));
  if (emap->valid() != true) {
    // try a local copy
    delete emap;
    emap = new EnvironmentMapBackground("./envmap.ppm", Vector(0,0,-1));
    if (emap->valid() != true) {
      return 0;
    }
  }
  scene->set_background_ptr(emap);
    
  Light* mainLight = new Light(Point(lx, ly, lz), Color(1,1,1), 0.01);
  mainLight->name_ = "main light";
  scene->add_light( mainLight );

  scene->select_shadow_mode(No_Shadows);

  if (display)
    scene->addAnimateObject(group);
  
  return scene;
}

int countData(char* fname, size_t dsize, int multiplier=1) {
  // Open the data file
  int in_fd=open(fname, O_RDONLY);
  if (in_fd==-1) {
    cerr<<"failed to open \""<<fname<<"\""<<endl;
    return 0;
  }
  
  struct stat statbuf;
  if (fstat(in_fd, &statbuf) == -1) {
    cerr<<"cannot stat \""<<fname<<"\""<<endl;
    return 0;
  }
  
  // Close the data file
  close(in_fd);

  // Return number of data items
  return (int)(statbuf.st_size/(multiplier*dsize));
}

int slurpFloatData(char* fname, float* data, int multiplier=1) {
  // Open the data file
  int in_fd=open(fname, O_RDONLY);
  if (in_fd==-1) {
    cerr<<"failed to open \""<<fname<<"\""<<endl;
    return 0;
  }
  
  struct stat statbuf;
  if (fstat(in_fd, &statbuf) == -1) {
    cerr<<"cannot stat \""<<fname<<"\""<<endl;
    return 0;
  }
  
  // Slurp the data
  int ndata=(int)(statbuf.st_size/(multiplier*sizeof(float)));
  unsigned long total_dsize=ndata*multiplier*sizeof(float);
  unsigned long num_read;
  num_read=read(in_fd, data, total_dsize);
  if(num_read!=total_dsize) {
    cerr<<"did not read "<<total_dsize<<" bytes from "
	<<fname<<endl;
    return 0;
  }
  
  // Close the data file
  close(in_fd);

  // Return number of data items read
  return (int)(num_read/sizeof(float));
}

int slurpIntegerData(char* fname, int* data, int multiplier=1) {
  // Open the data file
  int in_fd=open(fname, O_RDONLY);
  if (in_fd==-1) {
    cerr<<"failed to open \""<<fname<<"\""<<endl;
    return 0;
  }
  
  struct stat statbuf;
  if (fstat(in_fd, &statbuf) == -1) {
    cerr<<"cannot stat \""<<fname<<"\""<<endl;
    return 0;
  }
  
  // Slurp the data
  int ndata=(int)(statbuf.st_size/(multiplier*sizeof(int)));
  unsigned long total_dsize=ndata*multiplier*sizeof(int);
  unsigned long num_read;
  num_read=read(in_fd, data, total_dsize);
  if(num_read!=total_dsize) {
    cerr<<"did not read "<<total_dsize<<" bytes from "
	<<fname<<endl;
    return 0;
  }
  
  // Close the data file
  close(in_fd);

  // Return number of data items read
  return (int)(num_read/sizeof(int));
}

int slurpUCharData(char* fname, unsigned char* data, int multiplier=1) {
  // Open the data file
  int in_fd=open(fname, O_RDONLY);
  if (in_fd==-1) {
    cerr<<"failed to open \""<<fname<<"\""<<endl;
    return 0;
  }
  
  struct stat statbuf;
  if (fstat(in_fd, &statbuf) == -1) {
    cerr<<"cannot stat \""<<fname<<"\""<<endl;
    return 0;
  }
  
  // Slurp the data
  int ndata=(int)(statbuf.st_size/(multiplier*sizeof(unsigned char)));
  unsigned long total_dsize=ndata*multiplier*sizeof(unsigned char);
  unsigned long num_read;
  num_read=read(in_fd, data, total_dsize);
  if(num_read!=total_dsize) {
    cerr<<"did not read "<<total_dsize<<" bytes from "
	<<fname<<endl;
    return 0;
  }
  
  // Close the data file
  close(in_fd);

  // Return number of data items read
  return (int)(num_read/sizeof(unsigned char));
}

// Parse input file and populate data structues
// Returns a pointer to a newly allocated TextureGridSpheres
//   on success, NULL on any failure
TextureGridSpheres* texGridFromFile(char *fname, int tex_res, float radius,
				    int numvars, int nsides, int gdepth,
				    RegularColorMap* cmap, const Color& color)
{
  // Declare a few variables
  float* sphere_data=0;
  int* index_data=0;
  unsigned char* tex_data=0;
  unsigned char* mean_data=0;
  unsigned char* coeff_data=0;
  int total_nspheres=0;
  int total_nindices=0;
  int total_ntextures=0;
  int total_nmeans=0;
  int total_nxforms=0;
  float tex_min = 1;
  float tex_max = 0;
  float coeff_min = 1;
  float coeff_max = 0;
  
  // Open file for reading
  ifstream infile(fname);
  if (!infile.is_open()) {
    cerr<<"failed to open \""<<fname<<"\""<<endl;
    return 0;
  }

  // First pass:  Determine the number of spheres, indices, and textures
  bool group_flag=false;
  bool s_flag=false;
  bool idx_flag=false;
  bool tex_flag=false;
  bool m_flag=false;
  bool coeff_flag=false;
  bool basis_minmax_flag=false;
  bool coeff_minmax_flag=false;
  char line[MAX_LINE_LEN];
  infile.getline(line,MAX_LINE_LEN);
  while(!infile.eof()) {
    // Parse the line
    char *token=strtok(line, " ");
    if (!token || strcmp(token, "#")==0) {
      // Skip blank lines and comments
      infile.getline(line,MAX_LINE_LEN);
      continue;
    } else if (strcmp(token, "sphere_group")==0) {
      if (group_flag==true) {
	cerr<<"already within a valid sphere group's scope"<<endl;
	return 0;
      }
      
      token=strtok(0, " ");
      if (!token) {
	cerr<<"expecting \"{\", '\\n' found instead"<<endl;
	return 0;
      } else if (strcmp(token, "{")==0) {
	group_flag=true;
      } else {
	cerr<<"expecting \"{\", \""<<token<<"\" found instead"<<endl;
	return 0;
      }
    } else if (strcmp(token, "sphere_file:")==0) {
      if (!group_flag) {
	cerr<<"encountered \"sphere_file\" without a valid sphere group"<<endl;
	return 0;
      }
      
      // Check for a sphere data file
      char *s_fname=strtok(0, " ");
      if (!s_fname) {
	cerr<<"sphere_file requires a filename"<<endl;
	return 0;
      }

      total_nspheres+=countData(s_fname, sizeof(float), numvars);
      s_flag = true;
    } else if (strcmp(token, "index_file:")==0) {
      if (!group_flag) {
	cerr<<"encountered \"index_file\" without a valid sphere group"<<endl;
	return 0;
      }
      
      // Check for a index data file
      char *idx_fname=strtok(0, " ");
      if (!idx_fname) {
	cerr<<"index_file requires a filename"<<endl;
	return 0;
      }

      total_nindices+=countData(idx_fname, sizeof(int));      
      idx_flag = true;
    } else if (strcmp(token, "texture_file:")==0) {
      if (!group_flag) {
	cerr<<"encountered \"texture_file\" without a valid sphere group"<<endl;
	return 0;
      }
      
      // Check for a texture data file
      char *tex_fname=strtok(0, " ");
      if (!tex_fname) {
	cerr<<"texture_file requires a filename"<<endl;
	return 0;
      }

      total_ntextures+=countData(tex_fname, sizeof(unsigned char), tex_res*tex_res);
      tex_flag=true;
    } else if (strcmp(token, "basis_file:")==0) {
      if (!group_flag) {
	cerr<<"encountered \"basis_file\" without a valid sphere group"<<endl;
	return 0;
      }
      
      // Check for a texture data file
      char *tex_fname=strtok(0, " ");
      if (!tex_fname) {
	cerr<<"basis_file requires a filename"<<endl;
	return 0;
      }

      total_ntextures+=countData(tex_fname, sizeof(unsigned char), tex_res*tex_res);
      tex_flag=true;
    } else if (strcmp(token, "mean_file:")==0) {
      if (!group_flag) {
	cerr<<"encountered \"mean_file\" without a valid sphere group"<<endl;
	return 0;
      }
      
      // Check for a mean data file
      char *m_fname=strtok(0, " ");
      if (!m_fname) {
	cerr<<"mean_file requires a filename"<<endl;
	return 0;
      }

      // Open the mean data file
      int in_fd=open(m_fname, O_RDONLY);
      if (in_fd==-1) {
	cerr<<"failed to open \""<<m_fname<<"\""<<endl;
	return 0;
      }
      
      struct stat statbuf;
      if (fstat(in_fd, &statbuf) == -1) {
	cerr<<"cannot stat \""<<m_fname<<"\""<<endl;
	return 0;
      }

      total_nmeans+=countData(m_fname, sizeof(unsigned char));      
      m_flag = true;
    } else if (strcmp(token, "coeff_file:")==0) {
      if (!group_flag) {
	cerr<<"encountered \"coeff_file\" without a valid sphere group"<<endl;
	return 0;
      }
      
      // Check for a coefficient data file
      char *coeff_fname=strtok(0, " ");
      if (!coeff_fname) {
	cerr<<"coeff_file requires a filename"<<endl;
	return 0;
      }

      total_nxforms+=countData(coeff_fname, sizeof(unsigned char));
      coeff_flag = true;
    } else if (strcmp(token, "basis_minmax:")==0) {
      if (basis_minmax_flag) {
	cerr << "Can only have one basis_minmax per file.\n";
	return 0;
      }
      
      char *min=strtok(0, " ");
      if (min)
	tex_min = atof(min);
      else {
	cerr << "Expected a min for basis_minmax, but not found.\n";
	return 0;
      }
      
      char *max=strtok(0, " ");
      if (max)
	tex_max = atof(max);
      else {
	cerr << "Expected a max for basis_minmax, but not found.\n";
	return 0;
      }

      basis_minmax_flag = true;
    } else if (strcmp(token, "coeff_minmax:")==0) {
      if (coeff_minmax_flag) {
	cerr << "Can only have one coeff_minmax per file.\n";
	return 0;
      }
      
      char *min=strtok(0, " ");
      if (min)
	coeff_min = atof(min);
      else {
	cerr << "Expected a min for coeff_minmax, but not found.\n";
	return 0;
      }
      
      char *max=strtok(0, " ");
      if (max)
	coeff_max = atof(max);
      else {
	cerr << "Expected a max for coeff_minmax, but not found.\n";
	return 0;
      }

      coeff_minmax_flag = true;
    } else if (strcmp(token, "}")==0) {
      if (!group_flag) {
	cerr<<"encountered \"}\" without a valid sphere group"<<endl;
	return 0;
      }
      
      group_flag=false;
    }
    
    // Get the next line
    infile.getline(line,MAX_LINE_LEN);
  }

  // Close the input file
  infile.close();
  
  // Final error checking
  if (group_flag) {
    cerr<<"warning:  missing \"}\" at the end of the current sphere group"<<endl;
  }
  
  if (!s_flag) {
    cerr<<"no sphere file specified"<<endl;
    return 0;
  }

  if (!idx_flag) {
    cerr<<"warning:  no index file specified";
    if (tex_flag) {
      cerr<<"; using sphere index for texture index";
    }
    cerr<<endl;
  } else if (total_nspheres!=total_nindices) {
    cerr<<"total number of spheres ("<<total_nspheres<<") does not "
	<<"equal total number of texture indices ("<<total_nindices<<")"
	<<endl;
    return 0;
  }

  if (!tex_flag) {
    cerr<<"warning:  no texture file specified";
    if (idx_flag) {
      cerr<<"; ignoring index file and using default color"<<endl;
      idx_flag=false;
    } else {
      cerr<<"; using default color"<<endl;
    }
  }

  if (m_flag && !coeff_flag) {
    cerr<<"mean_file specified, but no coeff_file found"<<endl;
    return 0;
  } else if (!m_flag && coeff_flag) {
    cerr<<"coeff_file specified, but no mean_file found"<<endl;
    return 0;
  } else if (m_flag && coeff_flag) {
    if (!basis_minmax_flag) {
      cerr<<"basis_minmax was not found"<<endl;
      return 0;
    } else if (!coeff_minmax_flag) {
      cerr<<"coeff_minmax was not found"<<endl;
      return 0;
    }
  }

  // XXX - fix dimensionality checks
#if 0
  if (total_nxforms != total_nmeans*total_ntextures) {
    cerr<<"number of elements in transform ("<<total_nxforms
	<<") is not equal to the number of channels times the number of basis ("
	<<(total_nmeans*total_ntextures)<<")"<<endl;
    return 0;
  }
#endif

  // Allocate memory for the necessary data structures
  cout<<"Allocating space for "<<total_nspheres<<" spheres"<<endl;
  sphere_data=new float[numvars*total_nspheres];
  if (!sphere_data) {
    cerr<<"failed to allocate "<<numvars*sizeof(float)*total_nspheres<<" bytes "
	<<"for sphere data"<<endl;
    return 0;
  }
  
  if (idx_flag) {
    cout<<"Allocating space for "<<total_nindices<<" indices"<<endl;
    index_data=new int[total_nindices];
    if (!index_data) {
      cerr<<"failed to allocate "<<sizeof(int)*total_nindices<<" bytes "
	  <<"for index data"<<endl;
      return 0;
    }
  }
  
  cout<<"Allocating space for "<<total_ntextures<<" textures"<<endl;
  if (tex_flag) {
    tex_data=new unsigned char[tex_res*tex_res*total_ntextures];
    if (!tex_data) {
      cerr<<"failed to allocate "<<tex_res*tex_res*total_ntextures<<" bytes "
	  <<"for texture data"<<endl;
      return 0;
    }
  }
  
  if (m_flag) {
    cout<<"Allocating space for "<<total_nmeans<<" elements of the mean vector"
	<<endl;
    mean_data=new unsigned char[total_nmeans];
    if (!mean_data) {
      cerr<<"failed to allocate "<<total_nmeans*sizeof(unsigned char)<<" bytes "
	  <<"for mean data"<<endl;
      return 0;
    }
  }
  
  if (coeff_flag) {
    cout<<"Allocating space for "<<total_nxforms<<" PCA coefficients"<<endl;
    coeff_data=new unsigned char[total_nxforms];
    if (!coeff_data) {
      cerr<<"failed to allocate "<<total_nxforms*sizeof(unsigned char)<<" bytes "
	  <<"for coefficient data"<<endl;
      return 0;
    }
  }
  
  cout<<"Done allocating memory"<<endl;
  
  // Second pass:  Populate the data structures
  // XXX - Ugly hack!  For some reason, simply repositioning the stream's get
  //       pointer to the beginning of the file, as in:
  //
  //         infile.seekg(0, ios::beg);
  //
  //       and rereading doesn't work (EOF==true), so just create a new
  //       ifstream and use it instead
  ifstream infile2(fname);
  if (!infile2.is_open()) {
    cerr<<"failed to open \""<<fname<<"\""<<endl;
    return 0;
  }
  
  int s_index = 0;
  int idx_index = 0;
  int tex_index = 0;
  int m_index = 0;
  int coeff_index = 0;
  infile2.getline(line,MAX_LINE_LEN);
  while(!infile2.eof()) {
    // Parse the line
    char *token=strtok(line, " ");
    if (!token || strcmp(token, "#")==0) {
      // Skip blank lines and comments
      infile2.getline(line,MAX_LINE_LEN);
      continue;
    } else if (strcmp(token, "sphere_file:")==0) {
      // Get the sphere data file
      char *s_fname=strtok(0, " ");
      if (!s_fname) {
	cerr<<"sphere_file requires a filename"<<endl;
	return 0;
      }

      // Slurp the sphere data
      s_index+=slurpFloatData(s_fname, &(sphere_data[s_index]), numvars);
    } else if (idx_flag && strcmp(token, "index_file:")==0) {
      // Get the index data file
      char *idx_fname=strtok(0, " ");
      if (!idx_fname) {
	cerr<<"index_file requires a filename"<<endl;
	return 0;
      }
      
      // Slurp the index data
      idx_index+=slurpIntegerData(idx_fname, &(index_data[idx_index]));
    } else if (strcmp(token, "texture_file:")==0 ||
	       strcmp(token, "basis_file:") == 0) {
      // Get the texture data file
      char *tex_fname=strtok(0, " ");
      if (!tex_fname) {
	cerr<<"texture_file requires a filename"<<endl;
	return 0;
      }
      
      // Open the texture data file
      int in_fd=open(tex_fname, O_RDONLY);
      if (in_fd==-1) {
	cerr<<"failed to open \""<<tex_fname<<"\""<<endl;
	return 0;
      }

      struct stat statbuf;
      if (fstat(in_fd, &statbuf) == -1) {
	cerr<<"cannot stat \""<<tex_fname<<"\""<<endl;
	return 0;
      }

      // Slurp the texture data
      tex_index+=slurpUCharData(tex_fname, &(tex_data[tex_index]), tex_res*tex_res);
    } else if (strcmp(token, "mean_file:")==0) {
      // Get the mean data file
      char *m_fname=strtok(0, " ");
      if (!m_fname) {
	cerr<<"mean_file requires a filename"<<endl;
	return 0;
      }

      // Slurp the mean data
      m_index+=slurpUCharData(m_fname, &(mean_data[m_index]));
    } else if (strcmp(token, "coeff_file:")==0) {
      // Get the coefficient data file
      char *coeff_fname=strtok(0, " ");
      if (!coeff_fname) {
	cerr<<"coeff_file requires a filename"<<endl;
	return 0;
      }
      
      // Slurp the coefficient data
      coeff_index+=slurpUCharData(coeff_fname, &(coeff_data[coeff_index]));
    }
    
    // Get the next line
    infile2.getline(line,MAX_LINE_LEN);
  }

  // Close the input file
  infile2.close();

  // Create the appropriate structure
  TextureGridSpheres* tex_grid;
  if (m_flag && coeff_flag) {
    tex_grid = new PCAGridSpheres(sphere_data, total_nspheres, numvars,
				  radius, index_data,
				  tex_data, total_ntextures, tex_res,
				  coeff_data, mean_data,
				  total_nxforms/total_ntextures,
				  tex_min, tex_max, coeff_min, coeff_max,
				  nsides, gdepth, cmap, color);
  } else {
    tex_grid = new TextureGridSpheres(sphere_data, total_nspheres, numvars,
				      radius, index_data, tex_data,
				      total_ntextures, tex_res,
				      nsides, gdepth, cmap, color);
  }
  
  return tex_grid;
}
