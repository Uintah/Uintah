

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/UVSphere2.h>
#include <Packages/rtrt/Core/SharedTexture.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Core/TextureGridSpheres.h>

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

using namespace rtrt;

#define NUM_TEXTURES 8
#define MAX_LINE_LEN 256

TextureGridSpheres* texGridFromFile(char *fname, int tex_res, float radius,
				    int nsides, int gdepth);

float radius = 1.0;

/*
  -eye -4.05479 0.0260826 7.411 -lookat 0 0 0 -up 0.125951 0.989877 0.065428 -fov 42.5676

*/

// Returns 1 if there was an error.  This is based on if the texture
// was found.
int add_sphere(char *tex_name, const Point& center, Group *group) {
  SharedTexture* matl = new SharedTexture(tex_name);
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
			int nsides, int gdepth) {
  Group* group = new Group();
  
  float* spheres = new float[NUM_TEXTURES*3];
  unsigned char *tex_data = new unsigned char[NUM_TEXTURES*3*tex_res*tex_res];
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
	  unsigned char *pixel = tex_data + (nspheres * tex_res * tex_res * 3);
	  for (int j = 0; j < height; j++)
	    for (int i = 0; i < width; i++) {
	      Color c = image(i,j) * 255;
	      pixel[0] = c.red();
	      pixel[1] = c.green();
	      pixel[2] = c.blue();
	      pixel+=3;
	    }
	  
          nspheres++;
	}

  int *tex_indices = 0;
  group->add(new 
	     TextureGridSpheres(spheres, nspheres, radius, tex_indices,
				tex_data, nspheres, tex_res,
				nsides, gdepth));
  return group;
}

extern "C" 
Scene* make_scene(int argc, char** argv, int /*nworkers*/)
{
  char *bg="/home/sci/cgribble/research/datasets/mpm/misc/envmap.ppm";
  char *tex_basename="./sphere";
  int tex_res = -1;
  char *infilename=0;
  int nsides = 6;
  int gdepth = 2;

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
    else if(strcmp(argv[i],"-nsides")==0)
      nsides=atoi(argv[++i]);
    else if(strcmp(argv[i],"-gdepth")==0)
      gdepth=atoi(argv[++i]);
    else
    {
      cerr<<"unrecognized option \""<<argv[i]<<"\""<< endl;
      cerr<<"valid options are:"<<endl;
      cerr<<"  -bg <filename>      environment map image file (envmap.ppm)"<<endl;
      cerr<<"  -i <filename>       input sphere file name (null)"<<endl;
      cerr<<"  -tex <filename>     basename of texture files (./sphere)"<<endl;
      cerr<<"  -tex_res <int>      resolution of the textures (-1)"<<endl;
      cerr<<"  -radius <float>     sphere radius (1.0)"<<endl;
      cerr<<"  -nsides <int>       number of sides for grid cells (6)"<<endl;
      cerr<<"  -gdepth <int>       gdepth of grid cells (2)"<<endl;
      exit(1);
    }
  }

  Object *group=0;
  if (infilename) {
    group=texGridFromFile(infilename, tex_res, radius, nsides, gdepth);
  } else {
    char *tex_names[NUM_TEXTURES];
    // Make the tex_names
    size_t name_length = strlen(tex_basename) + 15;
    for(int i = 0; i < NUM_TEXTURES; i++) {
      tex_names[i] = new char[name_length];
      sprintf(tex_names[i], "%s%d.ppm", tex_basename, i); 
    }
  
    if (tex_res > 0)
      group = make_geometry_tg(tex_names, tex_res, nsides, gdepth);
    else
      group = make_geometry(tex_names);
  }
  
  if (!group) {
    cerr << "Could not generate geometry successfully.\n";
    // Then something went wrong and you should kill the scene
    return 0;
  }

  Camera cam(Point(-0.25,-0.1,0.1), Point(0.1,0.075,0.2), Vector(0,0,-1), 15.0);

  double ambient_scale=2;
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
    
  Light* mainLight = new Light(Point(-5,10,7.5), Color(1,1,1), 0.01);
  mainLight->name_ = "main light";
  scene->add_light( mainLight );
  scene->turnOffAllLights( 0.0 ); 

  scene->select_shadow_mode(No_Shadows);
  
  return scene;
}

// Parse input file and populate data structues
// Returns a pointer to a newly allocated TextureGridSpheres
//   on success, NULL on any failure
TextureGridSpheres* texGridFromFile(char *fname, int tex_res, float radius,
				    int nsides, int gdepth) {
  // Declare a few variables
  float* sphere_data=0;
  int* index_data=0;
  unsigned char* tex_data=0;
  size_t total_nspheres=0;
  size_t total_nindices=0;
  size_t total_ntextures=0;
  
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

      // Open the sphere data file
      int in_fd=open(s_fname, O_RDONLY);
      if (in_fd==-1) {
	cerr<<"failed to open \""<<s_fname<<"\""<<endl;
	return 0;
      }
      
      struct stat statbuf;
      if (fstat(in_fd, &statbuf) == -1) {
	cerr<<"cannot stat \""<<s_fname<<"\""<<endl;
	return 0;
      }
      
      total_nspheres+=(int)(statbuf.st_size/(3*sizeof(float)));
      
      // Close the sphere data file
      close(in_fd);
      
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

      // Open the index data file
      int in_fd=open(idx_fname, O_RDONLY);
      if (in_fd==-1) {
	cerr<<"failed to open \""<<idx_fname<<"\""<<endl;
	return 0;
      }
      
      struct stat statbuf;
      if (fstat(in_fd, &statbuf) == -1) {
	cerr<<"cannot stat \""<<idx_fname<<"\""<<endl;
	return 0;
      }
      
      total_nindices+=(int)(statbuf.st_size/sizeof(int));
      
      // Close the index data file
      close(in_fd);
      
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

      total_ntextures+=(int)(statbuf.st_size/(tex_res*tex_res*3*sizeof(unsigned char)));
      
      // Close the texture data file
      close(in_fd);
            
      tex_flag=true;
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

  cout << "attempting to read "<<total_nspheres<<" spheres\n";
  
  // Allocate memory for the necessary data structures
  sphere_data=new float[3*total_nspheres];
  if (!sphere_data) {
    cerr<<"failed to allocate "<<3*sizeof(float)*total_nspheres<<" bytes"
	<<"for sphere data"<<endl;
    return 0;
  }
  if (idx_flag) {
    index_data=new int[total_nindices];
    if (!index_data) {
      cerr<<"failed to allocate "<<sizeof(int)*total_nindices<<" bytes"
	  <<"for index data"<<endl;
      return 0;
    }
  }

  cout << "attempting to read "<<total_ntextures<<" textures.\n";
  
  if (tex_flag) {
    tex_data=new unsigned char[3*tex_res*tex_res*total_ntextures];
    if (!tex_data) {
      cerr<<"failed to allocate "<<3*tex_res*tex_res*total_ntextures<<" bytes"
	  <<"for texture data"<<endl;
      return 0;
    }
  }

  cout << "Done allocating sphere and texture data\n";
  
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
  infile2.getline(line,MAX_LINE_LEN);
  int s_index = 0;
  int idx_index = 0;
  int tex_index = 0;
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

      // Open the sphere data file
      int in_fd=open(s_fname, O_RDONLY);
      if (in_fd==-1) {
	cerr<<"failed to open \""<<s_fname<<"\""<<endl;
	return 0;
      }

      struct stat statbuf;
      if (fstat(in_fd, &statbuf) == -1) {
	cerr<<"cannot stat \""<<s_fname<<"\""<<endl;
	return 0;
      }

      // Slurp the sphere data
      float* data=&(sphere_data[s_index]);
      int nspheres=(int)(statbuf.st_size/(3*sizeof(float)));
      unsigned long data_size=nspheres*3*sizeof(float);

      cerr<<"slurping sphere data ("<<nspheres<<" spheres = " <<data_size
	  <<" bytes) from "<<s_fname<<endl;
      unsigned long num_read;
      num_read=read(in_fd, data, data_size);
      if(num_read==-1 || num_read!=data_size) {
	cerr<<"did not read "<<data_size<<" bytes from "
	    <<s_fname<<endl;
	return 0;
      }
      
      // Set start of next slurp
      s_index+=(int)(num_read/sizeof(float));

      // Close the sphere data file
      close(in_fd);
    } else if (idx_flag && strcmp(token, "index_file:")==0) {
      // Get the index data file
      char *idx_fname=strtok(0, " ");
      if (!idx_fname) {
	cerr<<"index_file requires a filename"<<endl;
	return 0;
      }

      // Open the index data file
      int in_fd=open(idx_fname, O_RDONLY);
      if (in_fd==-1) {
	cerr<<"failed to open \""<<idx_fname<<"\""<<endl;
	return 0;
      }

      struct stat statbuf;
      if (fstat(in_fd, &statbuf) == -1) {
	cerr<<"cannot stat \""<<idx_fname<<"\""<<endl;
	return 0;
      }

      // Slurp the index data
      int* data=&(index_data[idx_index]);
      int nindices=(int)(statbuf.st_size/sizeof(int));
      unsigned long data_size=nindices*sizeof(int);

      cerr<<"slurping index data ("<<nindices<<" indices = " <<data_size
	  <<" bytes) from "<<idx_fname<<endl;
      unsigned long num_read;
      num_read=read(in_fd, data, data_size);
      if(num_read==-1 || num_read!=data_size) {
	cerr<<"did not read "<<data_size<<" bytes from "
	    <<idx_fname<<endl;
	return 0;
      }

      // Set start of next slurp
      idx_index+=(int)(num_read/sizeof(int));

      // Close the index data file
      close(in_fd);
    } else if (strcmp(token, "texture_file:")==0) {
      // Get the texture data file
      char *tex_fname=strtok(0, " ");
      if (!tex_fname) {
	cerr<<"texture_file requires a filename"<<endl;
	return 0;
      }
      
      // Open the index data file
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
      unsigned char* data=&(tex_data[tex_index]);
      int ntextures=(int)(statbuf.st_size/(3*tex_res*tex_res*sizeof(unsigned char)));
      unsigned long data_size=ntextures*3*tex_res*tex_res*sizeof(unsigned char);

      cerr<<"slurping texture data ("<<ntextures<<" textures = " <<data_size
	  <<" bytes) from "<<tex_fname<<endl;
      unsigned long num_read;
      num_read=read(in_fd, data, data_size);
      if(num_read==-1 || num_read!=data_size) {
	cerr<<"did not read "<<data_size<<" bytes from "
	    <<tex_fname<<endl;
	return 0;
      }
      
      // Set start of next slurp
      tex_index+=num_read;

      // Close the texture data file
      close(in_fd);
    }
    
    // Get the next line
    infile2.getline(line,MAX_LINE_LEN);
  }

  // Close the input file
  infile2.close();

  // Create the TextureGridSpheres structure
  TextureGridSpheres* tex_grid;
  tex_grid = new TextureGridSpheres(sphere_data, total_nspheres,
				    radius, index_data, tex_data,
				    total_ntextures, tex_res,
				    nsides, gdepth);
  return tex_grid;
}
