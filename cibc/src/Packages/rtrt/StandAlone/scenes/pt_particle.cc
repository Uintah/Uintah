
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/GridSpheresDpy.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/PCAGridSpheres.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Core/RegularColorMap.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/TextureGridSpheres.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Thread/Thread.h>

#include <teem/nrrd.h>

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

#define MAX_LINE_LEN 1024
#define DEFAULT_RADIUS 0.001
#define DEFAULT_TEXRES 16
#define DEFAULT_NSIDES 6
#define DEFAULT_GDEPTH 2
#define DEFAULT_RATE 4

void usage(char* me, const char* unknown=0);
int parseFile(char* fname, GridSpheresDpy* dpy, SelectableGroup* timesteps);

float radius=DEFAULT_RADIUS;
int radius_index=-1;
int tex_res=DEFAULT_TEXRES;
Color color(1.0, 1.0, 1.0);
int nsides=DEFAULT_NSIDES;
int gdepth=DEFAULT_GDEPTH;
char* cmap_fname=0;
char *cmap_type = "InvRIsoLum";
bool display=true;
int ts_start=0;
int ts_inc=1;
int ntsteps=1;
char *gridconfig = 0;
string *var_names = 0;
int colordata = 0;

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/) {
  char* me="make_scene";
  char* envmap_fname=0;
  int rate=DEFAULT_RATE;
  char* in_fname=0;

  for (int i=1;i<argc;i++) {
    if (strcmp(argv[i],"-i")==0)
      in_fname=argv[++i];
    else if (strcmp(argv[i],"-radius")==0)
      radius=atof(argv[++i]);
    else if (strcmp(argv[i],"-radius_index")==0)
      radius_index=atoi(argv[++i]);
    else if (strcmp(argv[i],"-tex_res")==0)
      tex_res=atoi(argv[++i]);
    else if (strcmp(argv[i],"-color")==0) {
      color=Color(atof(argv[++i]),
		  atof(argv[++i]),
		  atof(argv[++i]));
    } else if (strcmp(argv[i],"-nsides")==0)
      nsides=atoi(argv[++i]);
    else if (strcmp(argv[i],"-gdepth")==0)
      gdepth=atoi(argv[++i]);
    else if (strcmp(argv[i], "-cmap") == 0)
      cmap_fname=argv[++i];
    else if (strcmp(argv[i], "-cmaptype") == 0)
      cmap_type = argv[++i];
    else if(strcmp(argv[i], "-colordata")==0) 
      colordata=atoi(argv[++i]);
    else if (strcmp(argv[i],"-envmap")==0)
      envmap_fname=argv[++i];
    else if (strcmp(argv[i],"-no_dpy")==0)
      display=false;
    else if (strcmp(argv[i], "-start")==0)
      ts_start=atoi(argv[++i]);
    else if (strcmp(argv[i], "-inc")==0)
      ts_inc=atoi(argv[++i]);
    else if (strcmp(argv[i], "-ntsteps")==0)
      ntsteps=atoi(argv[++i]);
    else if (strcmp(argv[i], "-rate")==0)
      rate=atoi(argv[++i]);
    else if (strcmp(argv[i], "-gridconfig") == 0)
      gridconfig = argv[++i];
    else if (strcmp(argv[i], "-varnames") == 0) {
      int num_varnames = atoi(argv[++i]);
      cerr << "Reading "<<num_varnames << " variable names\n";
      var_names = new string[num_varnames];
      for(int v = 0; v < num_varnames; v++)
        var_names[v] = string(argv[++i]);
    } else if(strcmp(argv[i],"--help")==0) {
      usage(me);
      exit(0);
    } else {
      usage(me, argv[i]);
      exit(1);
    }
  }

  // Validate arguments
  if (!in_fname) {
    cerr<<me<<":  no input file specified"<<endl;
    return 0;
  }

  if (ts_start<0) {
    cerr<<me<<":  invalid starting timestep ("
	<<ts_start<<"):  resetting to zero"<<endl;
    ts_start=0;
  }
  
  if (ts_inc<1) {
    cerr<<me<<":  invalid timestep increment ("
	<<ts_inc<<"):  resetting to one"<<endl;
    ts_inc=1;
  }
  
  if (ntsteps<=0) {
    cerr<<me<<":  invalid number of timesteps ("
	<<ntsteps<<"):  resetting to one"<<endl;
    ntsteps=1;
  }

  // Create geometry
  Group* world=new Group();
  GridSpheresDpy* dpy;
  if (gridconfig)
    dpy = new GridSpheresDpy(colordata, gridconfig);
  else
    dpy = new GridSpheresDpy(colordata);
  SelectableGroup* timesteps=new SelectableGroup(1.0/(float)rate);

  if (parseFile(in_fname, dpy, timesteps)) {
    cerr<<me<<":  error parsing \""<<in_fname<<"\""<<endl;
    return 0;
  }

  if (var_names) dpy->set_var_names(var_names);
  if (radius_index>=0)
    dpy->set_radius_index(radius_index);
  
  if (display)
    (new Thread(dpy, "GridSpheres display thread"))->detach();
  else
    dpy->setup_vars();

  world->add(timesteps);

  // Create scene
  Camera cam(Point(-0.25, -0.1, 0.1), Point(0.1, 0.075, 0.2),
	     Vector(0, 0, -1), 15.0);

  double ambient_scale=0.7;
  Color bgcolor(0, 0, 0);
  Color cdown(1, 1, 1);
  Color cup(1, 1, 1);

  rtrt::Plane groundplane(Point(0, 0, 0), Vector(0, 0, -1));
  Scene* scene=new Scene(world, cam, bgcolor, cdown, cup, groundplane,
			 ambient_scale, Arc_Ambient);

  if (envmap_fname) {
    EnvironmentMapBackground* emap=new EnvironmentMapBackground(envmap_fname,
								Vector(0, 0, -1));
    if (emap->valid())
      scene->set_background_ptr(emap);
    else {
      cerr<<me<<":  warning:  ignoring invalid environment map \""
	  <<envmap_fname<<"\""<<endl;
    }
  }
  
  Light* mainLight=new Light(Point(0, 0, 0), Color(1, 1, 1), 0.01, 0.6);
  mainLight->name_="main light";
  scene->add_light(mainLight);
  scene->turnOffAllLights(0.0); 
  scene->select_shadow_mode(No_Shadows);

  // Add geometry
  if (display)
    scene->addAnimateObject(world);
  scene->addGuiObject("Timesteps", timesteps);

  return scene;
}

void usage(char* me, const char* unknown) {
  if (unknown)
    cerr<<me<<":  unknown argument \""<<unknown<<"\""<<endl;

  cerr<<"usage:  "<<me<<" [options] -i <filename>"<<endl;
  cerr<<"options:"<<endl;
  cerr<<"  -radius <float>         particle radius ("<<DEFAULT_RADIUS<<")"<<endl;
  cerr<<"  -radius_index <int>     particle radius index (-1)"<<endl;
  cerr<<"  -tex_res <int>          texture resolution ("
      <<DEFAULT_TEXRES<<")"<<endl;
  cerr<<"  -color <r> <g> <b>      surface color (1.0, 1.0, 1.0)"<<endl;
  cerr<<"  -nsides <int>           number of sides for grid cells ("
      <<DEFAULT_NSIDES<<")"<<endl;
  cerr<<"  -gdepth <int>           gdepth of grid cells ("
      <<DEFAULT_GDEPTH<<")"<<endl;
  cerr<<"  -cmap <filename>        filename of color map (null)"<<endl;
  cerr<<"  -envmap <filename>      filename of environment map (null)"<<endl;
  cerr<<"  -no_dpy                 turn off GridSpheresDpy display (false)"<<endl;
  cerr<<"  -start <int>            index of starting timestep (0)"<<endl;
  cerr<<"  -inc <int>              timestep load increment  (1)"<<endl;
  cerr<<"  -ntsteps <int>          number of timesteps to load (1)"<<endl;
  cerr<<"  -rate <int>             target number of timesteps to render per second ("
      <<DEFAULT_RATE<<")"<<endl;

  cerr<<"  -colordata [int]        defaults to "<<colordata<<"\n";
  cerr<<"  -cmap <filename>        \n";
  cerr<<"  -cmaptype <type>        defaults to "<<cmap_type<<"\n";
  cerr<<"  -gridconfig <filename>  use this file as the config file for GridSpheresDpy.\n";
  cerr<<"  -varnames [number] vname1 \"v name 2\"\n";

  cerr<<"  --help                  print this message and exit"<<endl;
}

int parseFile(char* fname, GridSpheresDpy* dpy, SelectableGroup* timesteps) {
  char* me="parseFile";
  
  // Open file for reading
  ifstream infile(fname);
  if (!infile.is_open()) {
    cerr<<me<<":  failed to open \""<<fname<<"\""<<endl;
    return 1;
  }

  // Parse file
  bool find_first=true;
  int nloaded=0;
  for (int ts=0; ts<ntsteps; ts++) {
    bool ts_flag=false;
    char* s_fname=0;
    char* i_fname=0;
    char* t_fname=0;
    char* b_fname=0;
    char* m_fname=0;
    char* c_fname=0;
    char line[MAX_LINE_LEN];
    infile.getline(line, MAX_LINE_LEN);
    while (!infile.eof()) {
      char* token=strtok(line, " ");
      if (!token || strcmp(token, "#")==0) {
	// Skip blank lines and comments
	infile.getline(line, MAX_LINE_LEN);
	continue;
      } else if (strcmp(token, "timestep")==0) {
	if (ts_flag) {
	  cerr<<me<<":  encountered \"timestep\" while in scope of Timestep "
	      <<ts<<endl;
	  return 1;
	}

	token=strtok(0, " ");
	if (!token) {
	  cerr<<me<<":  expecting \"{\", '\\n' found instead"<<endl;
	  return 1;
	} else if (strcmp(token, "{")==0)
	  ts_flag=true;
	else {
	  cerr<<me<<":  expecting \"{\", \""<<token<<"\" found instead"<<endl;
	  return 1;
	}

	if (ts_start>0 && find_first) {
	  // Position stream for reading first requested timestep
	  int ts_current=0;
	  infile.getline(line, MAX_LINE_LEN);
	  while (!infile.eof()) {
	    char* token=strtok(line, " ");
	    if (strcmp(token, "timestep")==0) {
	      token=strtok(0, " ");
	      if (!token) {
		cerr<<me<<":  expecting \"{\", '\\n' found instead"<<endl;
		return 1;
	      } else if (strcmp(token, "{")==0)
		ts_flag=true;
	      else {
		cerr<<me<<":  expecting \"{\", \""<<token
		    <<"\" found instead"<<endl;
		return 1;
	      }
	    } else if (strcmp(token, "}")==0) {
	      if (ts_flag) {
		ts_flag=false;
		ts_current++;
		if (ts_current==ts_start)
		  break;
	      } else {
		cerr<<me<<":  encountered \"}\" outside of a valid timestep"<<endl;
		return 1;
	      }
	    }

	    // Read next line
	    infile.getline(line, MAX_LINE_LEN);
	  }

	  if (ts_current>=ts_start)
	    find_first=false;
	  else {
	    cerr<<me<<":  encountered EOF before first requested timestep (start="
		<<ts_start<<")"<<endl;
	    return 1;
	  }
	}
      } else if (strcmp(token, "radius:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"radius\" outside of a valid timestep"<<endl;
	  return 1;
	}

	// Check for a particle radius
	char* r=strtok(0, " ");
	if (!r) {
	  cerr<<me<<":  \"radius\" requires a particle radius"<<endl;
	  return 1;
	}

	radius=atof(r);
      } else if (strcmp(token, "radius_index:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"radius_index\" outside of a valid timestep"<<endl;
	  return 1;
	}

	// Check for a particle radius index
	char* r=strtok(0, " ");
	if (!r) {
	  cerr<<me<<":  \"radius_index\" requires a particle radius"<<endl;
	  return 1;
	}

	radius_index=atoi(r);
      } else if (strcmp(token, "tex_res:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"tex_res\" outside of a valid timestep"<<endl;
	  return 1;
	}

	// Check for a texture resolution
	char* res=strtok(0, " ");
	if (!res) {
	  cerr<<me<<":  \"tex_res\" requires a texture resolution"<<endl;
	  return 1;
	}

	tex_res=atoi(res);
      } else if (strcmp(token, "color:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"color\" outside of a valid timestep"<<endl;
	  return 1;
	}

	// Check for a surface color
	char* red=strtok(0, " ");
	char* green=strtok(0, " ");
	char* blue=strtok(0, " ");
	if (!red || !green || !blue) {
	  cerr<<me<<":  \"color\" requires a surface color"<<endl;
	  return 1;
	}

	color=Color(atof(red), atof(green), atof(blue));
      } else if (strcmp(token, "nsides:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"nsides\" outside of a valid timestep"<<endl;
	  return 1;
	}

	// Check for a number of sides
	char* num=strtok(0, " ");
	if (!num) {
	  cerr<<me<<":  \"nsides\" requires a number of variables"<<endl;
	  return 1;
	}

	nsides=atoi(num);
      } else if (strcmp(token, "gdepth:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"gdepth\" outside of a valid timestep"<<endl;
	  return 1;
	}

	// Check for a grid depth
	char* depth=strtok(0, " ");
	if (!depth) {
	  cerr<<me<<":  \"gdepth\" requires a grid depth"<<endl;
	  return 1;
	}

	gdepth=atoi(depth);
      } else if (strcmp(token, "cmap:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"cmap\" outside of a valid timestep"<<endl;
	  return 1;
	}
      
	// Check for a filename
	char* fname=strtok(0, " ");
	if (!fname) {
	  cerr<<me<<":  \"cmap\" requires a filename"<<endl;
	  return 1;
	}

	cmap_fname=strdup(fname);
      } else if (strcmp(token, "display:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"display\" outside of a valid timestep"<<endl;
	  return 1;
	}

	// Check for a truth value
	char* toggle=strtok(0, " ");
	if (!toggle) {
	  cerr<<me<<":  \"display\" requires a truth value"<<endl;
	  return 1;
	}

	if (strcmp(toggle, "true")==0)
	  display=true;
	else if (strcmp(toggle, "false")==0)
	  display=false;
	else {
	  cerr<<me<<":  expecting either \"true\" or \"false\", \""
	      <<toggle<<"\" found instead:  resetting to true"<<endl;
	  display=true;
	}
      } else if (strcmp(token, "sphere_file:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"sphere_file\" outside of a valid timestep"
	      <<endl;
	  return 1;
	}
      
	// Check for a sphere data file
	char* fname=strtok(0, " ");
	if (!fname) {
	  cerr<<me<<":  \"sphere_file\" requires a filename"<<endl;
	  return 1;
	}

	s_fname=strdup(fname);
      } else if (strcmp(token, "index_file:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"index_file\" outside of a valid timestep"
	      <<endl;
	  return 1;
	}
      
	// Check for a index data file
	char* fname=strtok(0, " ");
	if (!fname) {
	  cerr<<me<<":  \"index_file\" requires a filename"<<endl;
	  return 1;
	}

	i_fname=strdup(fname);
      } else if (strcmp(token, "texture_file:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"texture_file\" outside of a valid timestep"
	      <<endl;
	  return 1;
	}
      
	// Check for a texture data file
	char* fname=strtok(0, " ");
	if (!fname) {
	  cerr<<me<<":  \"texture_file\" requires a filename"<<endl;
	  return 1;
	}

	t_fname=strdup(fname);
      } else if (strcmp(token, "basis_file:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"basis_file\" outside of a valid timestep"
	      <<endl;
	  return 1;
	}
      
	// Check for a basis data file
	char* fname=strtok(0, " ");
	if (!fname) {
	  cerr<<me<<":  \"basis_file\" requires a filename"<<endl;
	  return 1;
	}

	b_fname=strdup(fname);
      } else if (strcmp(token, "mean_file:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"mean_file\" outside of a valid timestep"<<endl;
	  return 1;
	}
      
	// Check for a mean data file
	char* fname=strtok(0, " ");
	if (!fname) {
	  cerr<<me<<":  \"mean_file\" requires a filename"<<endl;
	  return 1;
	}

	m_fname=strdup(fname);
      } else if (strcmp(token, "coeff_file:")==0) {
	if (!ts_flag) {
	  cerr<<me<<":  encountered \"coeff_file\" outside of a valid timestep"
	      <<endl;
	  return 1;
	}
      
	// Check for a coefficient data file
	char* fname=strtok(0, " ");
	if (!fname) {
	  cerr<<me<<":  \"coeff_file\" requires a filename"<<endl;
	  return 1;
	}

	c_fname=strdup(fname);
      } else if (strcmp(token, "}")==0) {
	if (ts_flag) {
	  ts_flag=false;

	  if (ts_inc>1) {
	    // Position stream for reading next requested timestep
	    int ts_skipped=0;
	    infile.getline(line, MAX_LINE_LEN);
	    while (!infile.eof()) {
	      char* token=strtok(line, " ");
	      if (strcmp(token, "timestep")==0) {
		token=strtok(0, " ");
		if (!token) {
		  cerr<<me<<":  expecting \"{\", '\\n' found instead"<<endl;
		  return 1;
		} else if (strcmp(token, "{")==0)
		  ts_flag=true;
		else {
		  cerr<<me<<":  expecting \"{\", \""<<token
		      <<"\" found instead"<<endl;
		  return 1;
		}
	      } else if (strcmp(token, "}")==0) {
		if (ts_flag) {
		  ts_flag=false;
		  ts_skipped++;
		  if (ts_skipped==ts_inc-1)
		    break;
		} else {
		  cerr<<me<<":  encountered \"}\" outside of a valid timestep"<<endl;
		  return 1;
		}
	      }

	      // Read next line
	      infile.getline(line, MAX_LINE_LEN);
	    }
	  }

	  // Add current timestep
	  break;
	} else {
	  cerr<<me<<":  encountered \"}\" outside of a valid timestep"<<endl;
	  return 1;
	}
      }

      // Read next line
      infile.getline(line, MAX_LINE_LEN);
    }

    // Validate timestep
    if (!s_fname) {
      cerr<<me<<":  no sphere file specified"<<endl;
      return 1;
    }

    bool loadPCA=true;
    if (t_fname) {
      if (b_fname || m_fname || c_fname)
	cerr<<me<<"  warning:  texture file specified, ignoring PCA files"<<endl;
      loadPCA=false;
    } else if (b_fname) {
      bool error=false;
      if (!m_fname) {
	cerr<<me<<":  no mean file specified"<<endl;
	error=true;
      }

      if (!c_fname) {
	cerr<<me<<":  no coeff file specified"<<endl;
	error=true;
      }

      if (error)
	return 1;
    } else {
      cerr<<me<<":  no texture or basis file specified"<<endl;
      return 1;
    }

    if (radius<=0) {
      cerr<<me<<":  warning:  invalid radius ("
	  <<radius<<"), resetting to "<<DEFAULT_RADIUS<<endl;
      radius=DEFAULT_RADIUS;
    }

    if (radius_index<0) {
      cerr<<me<<":  warning:  invalid radius_index("
          <<radius_index<<"), resetting to -1"<<endl;
      radius_index=-1;
    }

    if (tex_res<=0) {
      cerr<<me<<":  warning:  invalid texture resolution ("
	  <<tex_res<<"), resetting to "<<DEFAULT_TEXRES<<endl;
      tex_res=DEFAULT_TEXRES;
    }
  
    if (nsides<=0) {
      cerr<<me<<":  warning:  invalid number of sides ("
	  <<nsides<<"), resetting to "<<DEFAULT_NSIDES<<endl;
      nsides=DEFAULT_NSIDES;
    }
  
    if (gdepth<=0) {
      cerr<<me<<":  warning:  invalid grid depth ("
	  <<nsides<<"), resetting to "<<DEFAULT_GDEPTH<<endl;
      gdepth=DEFAULT_GDEPTH;
    }

    // Load nrrd files
    char* err=0;
    Nrrd* sphereNrrd=nrrdNew();
    if (nrrdLoad(sphereNrrd, s_fname, 0)) {
      err=biffGet(NRRD);
      cerr<<me<<":  error loading particle data:  "<<err<<endl;
      free(err);
      biffDone(NRRD);
      return 1;
    }

    Nrrd* indexNrrd=0;
    if (i_fname) {
      indexNrrd=nrrdNew();
      if (nrrdLoad(indexNrrd, i_fname, 0)) {
	err=biffGet(NRRD);
	cerr<<me<<":  error loading particle indices:  "<<err<<endl;
	free(err);
	biffDone(NRRD);
	return 1;
      }
    }

    Nrrd* texNrrd=0;
    Nrrd* meanNrrd=0;
    Nrrd* coeffNrrd=0;
    if (loadPCA) {
      texNrrd=nrrdNew();
      if (nrrdLoad(texNrrd, b_fname, 0)) {
	err=biffGet(NRRD);
	cerr<<me<<":  error loading basis textures:  "<<err<<endl;
	free(err);
	biffDone(NRRD);
	return 1;
      }

      meanNrrd=nrrdNew();
      if (nrrdLoad(meanNrrd, m_fname, 0)) {
	err=biffGet(NRRD);
	cerr<<me<<":  error loading mean values:  "<<err<<endl;
	free(err);
	biffDone(NRRD);
	return 1;
      }

      coeffNrrd=nrrdNew();
      if (nrrdLoad(coeffNrrd, c_fname, 0)) {
	err=biffGet(NRRD);
	cerr<<me<<":  error loading PCA coefficients:  "<<err<<endl;
	free(err);
	biffDone(NRRD);
	return 1;
      }
    } else {
      texNrrd=nrrdNew();
      if (nrrdLoad(texNrrd, t_fname, 0)) {
	err=biffGet(NRRD);
	cerr<<me<<":  error loading textures:  "<<err<<endl;
	free(err);
	biffDone(NRRD);
	return 1;
      }
    }

    // Initialize variables
    float* sphere_data=(float*)(sphereNrrd->data);
    int nvars=sphereNrrd->axis[0].size;
    int nspheres=sphereNrrd->axis[1].size;
    sphereNrrd=nrrdNix(sphereNrrd);

    int* index_data=0;
    if (i_fname) {
      index_data=(int*)(indexNrrd->data);
      indexNrrd=nrrdNix(indexNrrd);
    }

    unsigned char* tex_data=0;
    int ntextures=0;
    float b_min=FLT_MAX;
    float b_max=-FLT_MAX;
    unsigned char* mean_data=0;
    unsigned char* coeff_data=0;
    int nvecs=0;
    float coeff_min=FLT_MAX;
    float coeff_max=-FLT_MAX;
    if (loadPCA) {
      tex_data=(unsigned char*)(texNrrd->data);
      ntextures=texNrrd->axis[1].size;
      b_min=texNrrd->oldMin;
      b_max=texNrrd->oldMax;
      texNrrd=nrrdNix(texNrrd);

      if (b_min>=FLT_MAX) {
	cerr<<me<<":  error reading basis minimum from \""<<b_fname<<"\""<<endl;
	return 1;
      } else if (b_max<=-FLT_MAX) {
	cerr<<me<<":  error reading basis maximum from \""<<b_fname<<"\""<<endl;
	return 1;
      }

      mean_data=(unsigned char*)(meanNrrd->data);
      meanNrrd=nrrdNix(meanNrrd);

      coeff_data=(unsigned char*)(coeffNrrd->data);
      nvecs=coeffNrrd->axis[1].size;
      coeff_min=coeffNrrd->oldMin;
      coeff_max=coeffNrrd->oldMax;
      coeffNrrd=nrrdNix(coeffNrrd);

      if (coeff_min>=FLT_MAX) {
	cerr<<me<<":  error reading coefficient minimum from \""
	    <<c_fname<<"\""<<endl;
	return 1;
      } else if (coeff_max<=-FLT_MAX) {
	cerr<<me<<":  error reading coefficient maximum from \""
	    <<c_fname<<"\""<<endl;
	return 1;
      }
    } else {
      tex_data=(unsigned char*)(texNrrd->data);
      ntextures=texNrrd->axis[2].size;
      texNrrd=nrrdNix(texNrrd);
    }

    cout<<"-----------------------------"<<endl;
    cout<<"  Timestep "<<ts_start+ts*ts_inc<<endl;
    cout<<"-----------------------------"<<endl;
    cout<<"Loaded "<<nspheres<<" particles ("<<nvars
	<<" variables/particle) from \""<<s_fname<<"\""<<endl;
    if (loadPCA) {
      cout<<"Loaded "<<ntextures<<" basis textures from \""<<b_fname<<"\""<<endl;
      cout<<"Loaded "<<nvecs<<"x"<<ntextures<<" coefficient matrix from \""
	  <<c_fname<<"\""<<endl;
      cout<<"Loaded mean values from \""<<m_fname<<"\""<<endl;
      cout<<"Unquantized basis min/max:  "<<b_min<<", "<<b_max<<endl;
      cout<<"Unquantized coefficient min/max:  "<<coeff_min<<", "<<coeff_max<<endl;
    } else
      cout<<"Loaded "<<ntextures<<" textures from \""<<t_fname<<"\""<<endl;
    cout<<endl;
  
    // Create color map
    RegularColorMap* cmap=0;
    if (cmap_fname)
      cmap=new RegularColorMap(cmap_fname);
    else {
      int cmap_type_index = RegularColorMap::parseType(cmap_type);
      cmap=new RegularColorMap(cmap_type_index);
    }

    if (!cmap) {
      cerr<<me<<":  error creating regular color map"<<endl;
      return 1;
    }
    
    // Create the appropriate grid structure
    TextureGridSpheres* tex_grid;
    if (loadPCA) {
      tex_grid=new PCAGridSpheres(sphere_data, nspheres, nvars, radius, index_data,
				  tex_data, ntextures, tex_res, coeff_data,
				  mean_data, nvecs, b_min, b_max, coeff_min,
				  coeff_max, nsides, gdepth, cmap, color);
    } else {
      tex_grid=new TextureGridSpheres(sphere_data, nspheres, nvars, radius,
				      index_data, tex_data, ntextures, tex_res,
				      nsides, gdepth, cmap, color);
    }

    if (!tex_grid) {
      cerr<<me<<":  error creating texture grid spheres"<<endl;
      return 1;
    }
    
    // Add current timestep
    dpy->attach(tex_grid);
    timesteps->add((Object*)tex_grid);

    nloaded++;
  }

  // Close the input file
  infile.close();

  if (nloaded==ntsteps)
    cout<<"Loaded "<<nloaded<<" of the "<<ntsteps<<" requested timesteps"<<endl;
  else {
    cout<<me<<":  warning:  only loaded "<<nloaded<<" of the "
	<<ntsteps<<" requested timesteps (start="<<ts_start<<", end="
	<<ts_start+ntsteps<<", increment="<<ts_inc<<")"<<endl;
  }
  
  return 0;
}
