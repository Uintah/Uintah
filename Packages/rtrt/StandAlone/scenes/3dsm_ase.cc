#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Array1.cc>
#include <Packages/rtrt/Core/Array3.cc>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Plane.h>
#include <Packages/rtrt/Core/Light.h>
#include <fstream>
#include <iostream>

using namespace SCIRun;
using namespace rtrt;
using namespace std;

Material *default_material = (Material*) new Phong(Color(0,0,0),
						   Color(.6,1,.4),
						   Color(0,0,0), 100, 0);
Array1<Material*> ase_matls;

Material* get_material(ifstream &infile) {
  Material *result = 0;
  string token("");
  int curley = 0;
  // eat the top curley brace and increment the curley brace counter
  infile >> token;
  curley++;
  Color ambient, diffuse, specular;
  double specpow = 20;
  double reflectance = 0;
  double shinestrength = 1;
  // loop until you get to the closing curley brace
  while (curley > 0) {
    // read the next token
    infile >> token;
    // parse it out
    if (token == "{") {
      curley++;
    } else if (token == "}") {
      curley--;
    } else if (token == "*MATERIAL_AMBIENT") {
      cout << "Found *MATERIAL_AMBIENT\n";
      double r, g, b;
      infile >> r >> g >> b;
      ambient = Color(r,g,b);
    } else if (token == "*MATERIAL_DIFFUSE") {
      cout << "Found *MATERIAL_DIFFUSE\n";
      double r, g, b;
      infile >> r >> g >> b;
      diffuse = Color(r,g,b);
    } else if (token == "*MATERIAL_SPECULAR") {
      cout << "Found *MATERIAL_SPECULAR\n";
      double r, g, b;
      infile >> r >> g >> b;
      specular = Color(r,g,b);
    } else if (token == "*MATERIAL_SHINE") {
      infile >> specpow;
    } else if (token == "*MATERIAL_SHINESTRENGTH") {
      infile >> shinestrength;
    } else if (token == "*MATERIAL_TRANSPARENCY") {
    } else if (token == "*MAP_DIFFUSE") {
      cout << "Found *MAP_DIFFUSE, but not parsing it.\n";
#if 0
    } else if (token == "") {
#endif
    }
  }
  cout << "ambient("<<ambient<<"), diffuse("<<diffuse<<"), specular("<<specular<<"), specpow("<<specpow<<"), reflectance("<<reflectance<<")\n";
  if (!result)
    result = (Material*) new Phong(ambient, diffuse, specular*shinestrength,
				   specpow, reflectance);
  return result;
}

bool get_materials(ifstream &infile) {
  string token("");
  int curley = 0;
  // eat the top curley brace and increment the curley brace counter
  infile >> token;
  curley++;
  // loop until you get to the closing curley brace
  while (curley > 0) {
    // read the next token
    infile >> token;
    // parse it out
    if (token == "{") {
      curley++;
    } else if (token == "}") {
      curley--;
    } else if (token == "*MATERIAL_COUNT") {
      // grab how many materials you have
      int num_materials = 0;
      infile >> num_materials;
      ase_matls.resize(num_materials);
      cout << "Reading "<<num_materials<<" materials.\n";
    } else if (token == "*MATERIAL") {
      // read which material it is
      int mat_index;
      infile >> mat_index;
      if (mat_index >= 0) {
	if (mat_index >= ase_matls.size())
	  ase_matls.resize(mat_index+1);
	ase_matls[mat_index] = get_material(infile);
      } else {
	cerr << "ase_load::get_materials::Bad material index:"<<mat_index<<
	  endl;
      }
    }
  }
  return true;
}

Object *get_object(ifstream &infile) {
  Object *result = 0;
  string token("");
  int curley = 0;
  // eat the top curley brace and increment the curley brace counter
  infile >> token;
  curley++;
  int matl_index = 0;
  Array1<Object*> faces;
  while (curley > 0) {
    // read the next token
    infile >> token;
    // parse it out
    if (token == "{") {
      curley++;
    } else if (token == "}") {
      curley--;
    } else if (token == "*NODE_NAME") {
      infile >> token;
      cout << "Parsing node named "<< token<<endl;
    } else if (token == "*NODE_TM") {
      cout << "Found *NODE_TM.\n";
      // eat the open curley brace
      infile >> token;
      curley++;
      while (curley > 1) {
	infile >> token;
	if (token == "{") {
	  curley++;
	} else if (token == "}") {
	  curley--;
	}
      }
    } else if (token == "*MESH") {
      cout << "Found *MESH.\n";
      // eat the open curley brace
      infile >> token;
      curley++;
      Array1<Point> verticies;
      while (curley > 1) {
	infile >> token;
	if (token == "{") {
	  curley++;
	} else if (token == "}") {
	  curley--;
	} else if (token == "*TIMEVALUE") {
	  cout << "Found *TIMEVALUE.\n";
	} else if (token == "*MESH_NUMVERTEX") {
	  cout << "Found *MESH_NUMVERTEX.\n";
	  int num_vertex;
	  infile >> num_vertex;
	  cout << "Reading "<<num_vertex<<" verticies\n";
	  verticies.resize(num_vertex);
	} else if (token == "*MESH_NUMFACES") {
	  cout << "Found *MESH_NUMFACES.\n";
	  int num_faces;
	  infile >> num_faces;
	  cout << "Reading "<<num_faces<<" faces.\n";
	  faces.resize(num_faces);
	} else if (token == "*MESH_VERTEX_LIST") {
	  cout << "Found *MESH_VERTEX_LIST.\n";
	  // eat the curley brace
	  infile >> token;
	  curley++;
	  for(unsigned int i = 0; i < verticies.size(); i++) {
	    // grab the vertex
	    int index;
	    double x, y, z;
	    infile >> token >> index >> x >> y >> z;
	    verticies[index] = Point(x,y,z);
	  }
	  // finish off the end curley
	  while( curley > 2 ) {
	    infile >> token;
	    if ( token == "}" )
	      curley--;
	  }
	} else if (token == "*MESH_FACE_LIST") {
	  cout << "Found *MESH_FACE_LIST.\n";
	  // eat the curley brace
	  infile >> token;
	  curley++;
	  for(unsigned int i = 0; i < faces.size(); i++) {
	    // grab the vertex
	    int index, p1, p2, p3;
	    infile >> token >> index >> token >> token >> p1 >> token >> p2
		   >> token >> p3;
	    // grab the rest of the line which is junk to us
	    char rest[512];
	    infile.get(rest,512);
	    //cout << "index("<<index<<")p1("<<p1<<")p2("<<p2<<")p3("<<p3<<")";
	    // set the material pointer to 0 for now.  Wait until we get the
	    // material to set it.
	    faces[index] = new Tri(0, verticies[p1], verticies[p2],
				   verticies[p3]);
	  }
	  // finish off the end curley
	  while( curley > 2 ) {
	    infile >> token;
	    //	    cout << "token("<<token<<")";
	    if ( token == "}" )
	      curley--;
	  }
	} else if (token == "*MESH_NUMTVERTEX") {
	  cout << "Found *MESH_NUMTVERTEX.\n";
	} else if (token == "*MESH_TVERTLIST") {
	  cout << "Found *MESH_TVERTLIST.\n";
	} else if (token == "*MESH_NUMTVFACES") {
	  cout << "Found *MESH_NUMTVFACES.\n";
	} else if (token == "*MESH_TFACELIST") {
	  cout << "Found *MESH_TFACELIST.\n";
	} else if (token == "*MESH_NUMCVERTEX") {
	  cout << "Found *MESH_NUMCVERTEX.\n";
	} else if (token == "*MESH_NORMALS") {
	  cout << "Found *MESH_NORMALS.\n";
	}
      }
    } else if (token == "*PROP_MOTIONBLUR") {
      cout << "Found *PROP_MOTIONBLUR.\n";
    } else if (token == "*PROP_CASTSHADOW") {
      cout << "Found *PROP_CASTSHADOW.\n";
    } else if (token == "*PROP_RECVSHADOW") {
      cout << "Found *PROP_RECVSHADOW.\n";
    } else if (token == "*MATERIAL_REF") {
      cout << "Found *MATERIAL_REF.\n";
      infile >> matl_index;
    }
  }
  // Now lets set the materials and add them to a group
  result = (Object*) new Group();
  for(unsigned int i = 0; i < faces.size(); i++) {
    Object *face = faces[i];
    face->set_matl(ase_matls[matl_index]);
    ((Group*)result)->add(face);
  }
  //  cerr << "ase_load::get_objects::No objects found.\n";
#if 0
  if(((Group*)result)->numObjects() > 20)
    result = new Grid(result, 30);
  else if(((Group*)result)->numObjects()>2)
    result = new BV1(result);
#endif
  return result;
}

Object *get_ase(char *file_name) {
  // open the file
  ifstream infile(file_name);
  if(infile){
    cout << "Reading file:"<<file_name<<endl;
  } else {
    cerr << "ase_load::get_ase::Error opening file: " << file_name << '\n';
    return 0;
  }

  // contains all the objects parsed in the scene
  Group *g = new Group(); 
  while (infile) {
    string token("");
    infile >> token;
    //    cout << "token("<<token<<")\n";
    if (token == "*3DSMAX_ASCIIEXPORT") {
      // read the number
      int number;
      infile >> number;
      cout << "Found *3DSMAX_ASCIIEXPORT "<<number<<endl;
    } else if (token == "*COMMENT") {
      // eat up the rest of the line
      cout << "Found comment:"<<endl;
    } else if (token == "*SCENE") {
      // read in the scene stuff
      cout << "Found *SCENE.\n";
    } else if (token == "*MATERIAL_LIST") {
      // get the materials
      cout << "Found *MATERIAL_LIST\n";
      get_materials(infile);
    } else if (token == "*GEOMOBJECT") {
      // read in a material
      cout << "Found *GEOMOBJECT\n";
      Object *obj = get_object(infile);
      if (obj)
	g->add(obj);
    } else if (token == "*LIGHTOBJECT") {
      cout << "Found *LIGHTOBJECT\n";
      // read in a light
      //    } else if (token == "") {
    }
  }
  infile.close();
  return g;
}

extern "C" 
Scene* make_scene(int argc, char* argv[], int) {
  char *in_file = 0;

  if (argc < 2) {
    cerr << "ase_load <file.ase>\n";
    return 0;
  }
  
  for(int i = 1; i < argc; i++) {
    if (in_file == 0)
      in_file = argv[i];
  }

  if (in_file == 0) {
    cerr << "You need to specify the name of a file to load.\n";
    return 0;
  }

  Object *all;
  if ((all = get_ase(in_file)) != 0) {
    cout << "File read successfully.\n";
  } else {
    cout << "File not read successfully.  Check above for erros.\n";
    return 0;
  }

  Camera cam(Point(1,0,0), Point(0,0,0),
	     Vector(0,0,1), 40);

  Color groundcolor(0,0,0);
  //Color averagelight(1,1,1);
  double ambient_scale=.5;

  Color bgcolor(0,0,0);

  Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
  Scene* scene=new Scene(all, cam,
			 bgcolor, groundcolor*bgcolor, bgcolor, groundplane,
			 ambient_scale);
  scene->add_light(new Light(Point(500,-300,300), Color(.8,.8,.8), 0));
  scene->shadow_mode=1;
  return scene;
}
