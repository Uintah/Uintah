#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array3.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Plane.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Background.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <fstream>
#include <iostream>

using namespace rtrt;
using namespace std;

Material *default_material = (Material*) new Phong(Color(.6,1,.4),
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
  double transparency = 0;
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
//      shinestrength *= 100;
    } else if (token == "*MATERIAL_TRANSPARENCY") {
      infile >> transparency;
    } else if (token == "*MAP_DIFFUSE") {
      cout << "Found *MAP_DIFFUSE, but not parsing it.\n";
#if 0
    } else if (token == "") {
#endif
    }
  }
  cout << "ambient("<<ambient<<"), diffuse("<<diffuse<<"), specular("<<specular<<"), specpow("<<specpow<<"), reflectance("<<reflectance<<")\n";
  if (!result)
    if (transparency == 0) {
      result = (Material*) new Phong( diffuse, specular*shinestrength,
				     specpow, reflectance);
    } else {
      result = (Material*) new DielectricMaterial(1.0, 1.0, 0.3, 400.0, Color(1,1,1), Color(1,1,1), false);
//      result = (Material*) new Phong( Color(1,0,0), Color(1,0,0), 10, 0);
    }
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
  int nm=ase_matls.size();
  ase_matls.resize(nm+1);
  ase_matls[nm] = new CycleMaterial;
  return true;
}

Group *Split_Data(Group* g)
{
    // Eqn of a line = n dot x + b = 0
    // Therefore if p is one of the pts on my split grid
    // b = -n dot p
    // Therefore the eqn is
    // n dot x -n dot p = 0 = n dot (x-p) = 0

    double xmin = -2430;
    double ymin = -1174;
    double xmax = 2430;
    double ymax = 1174;

    
    Group *regions[3][3];

    Group* result = new Group();

    for (int i=0; i<3; i++)
	for (int j=0; j<3; j++)
		regions[i][j] = new Group();

    for (int i=0; i<g->numObjects(); i++)
	{
	    Object* obj = g->objs[i];
	    Tri *tri;

	    if (tri = dynamic_cast<Tri*>(obj))
		{
		    int x_index;
		    int y_index;

		    Point centroid = tri->centroid();
		    
		    if (centroid.x() < xmin)
			x_index = 0;
		    else if (centroid.x() > xmax)
			x_index = 2;
		    else 
			x_index = 1;

		    if (centroid.y() < ymin)
			y_index = 0;
		    else if (centroid.y() > ymax)
			y_index = 2;
		    else 
			y_index = 1;
		    (regions[x_index][y_index])->add(tri);
		}
	}

    for (int i=0; i<3; i++)
	for (int j=0; j<3; j++)
	    {
		int num_objs = regions[i][j]->numObjects();

//  		if (num_objs > 0 && (i%2!=0 || j%2!=0))
		if (num_objs > 0)
		    {
			int nside = ceil(pow(num_objs, 1./3.));
			result->add(new Grid(regions[i][j],nside));
//  			result->add(new BV1(regions[i][j]));
		    }
	    }	
    return result;
}

void SplitTrisX(Group* g, double xval, double tol)
{
    int n_items = g->numObjects();
    enum {left, middle, right};

    for (int i=0; i<n_items; i++)
	{
	    Object* obj = g->objs[i];
	    Tri *t;

	    bool lt[3]={false,false,false};
	    int side[3] = {0,0,0};

	    if (t = dynamic_cast<Tri*>(obj))
		{
		    if (t->isbad())
			continue;
		    Point P0 = t->pt(0);
		    Point P1 = t->pt(1);
		    Point P2 = t->pt(2);

		    Tri *tri0, *tri1, *tri2;

		    for (int j=0; j<3; j++) {
			if (t->pt(j).x() < xval)
			    lt[j] = true;
			if (t->pt(j).x() < xval-tol)
			    side[j] = left;
			else if (t->pt(j).x() > xval+tol)
			    side[j] = right;
			else
			    side[j] = middle;
		    }
		    if (lt[0] == lt[1] &&
			lt[1] != lt[2] &&
			side[2] != middle &&
			(side[0] != middle || side[1] != middle))
			{
			    // break edges 02 and 12

			    double t0 = (xval-P0.x())/(P2.x()-P0.x());
			    double t1 = (xval-P1.x())/(P2.x()-P1.x());

			    Point P02 = AffineCombination(P0,(1-t0),
							  P2,t0);
			    Point P12 = AffineCombination(P1,(1-t1),
							  P2,t1);

			    tri0 = new Tri(t->get_matl(),
					   P12, P2, P02);
			    tri1 = new Tri(t->get_matl(),
					   P1, P12, P02);
			    tri2 = new Tri(t->get_matl(),
					   P0, P1, P02);

			}
		    else if (lt[0] != lt[1] &&
			     lt[1] == lt[2] &&
			     side[0] != middle &&
			     (side[1] != middle || side[2] != middle))
			{
			    // break edges 01 and 02
			    double t0 = (xval-P0.x())/(P1.x()-P0.x());
			    double t1 = (xval-P0.x())/(P2.x()-P0.x());

			    Point P01 = AffineCombination(P0,(1-t0),
							  P1,t0);
			    Point P02 = AffineCombination(P0,(1-t1),
							  P2,t1);

			    tri0 = new Tri(t->get_matl(),
					   P0, P01, P02);
			    tri1 = new Tri(t->get_matl(),
					   P1, P2, P01);
			    tri2 = new Tri(t->get_matl(),
					   P01, P2, P02);
			}
		    else if (lt[0] == lt[2] &&
			     lt[1] != lt[2] &&
			     side[1] != middle &&
			     (side[0] != middle || side[2] != middle))
			{
			    // break edges 01 and 12
			    double t0 = (xval-P0.x())/(P1.x()-P0.x());
			    double t1 = (xval-P1.x())/(P2.x()-P1.x());

			    Point P01 = AffineCombination(P0,(1-t0),
							  P1,t0);
			    Point P12 = AffineCombination(P1,(1-t1),
							  P2,t1);

			    tri0 = new Tri(t->get_matl(),
					   P01, P1, P12);
			    tri1 = new Tri(t->get_matl(),
					   P0, P01, P2);
			    tri2 = new Tri(t->get_matl(),
					   P01, P12, P2);
			}
		    else
			continue;
		    g->objs[i] = tri0;
		    g->add(tri1);
		    g->add(tri2);
		}
	}
}    

void SplitTrisY(Group* g, double yval, double tol)
{
    int n_items = g->numObjects();

    enum {left, middle, right};

    for (int i=0; i<n_items; i++)
	{
	    Object* obj = g->objs[i];
	    Tri *t;

	    bool lt[3]={false,false,false};
	    int side[3] = {0,0,0};

	    if (t = dynamic_cast<Tri*>(obj))
		{
		    if (t->isbad())
			continue;
		    Point P0 = t->pt(0);
		    Point P1 = t->pt(1);
		    Point P2 = t->pt(2);

		    Tri *tri0, *tri1, *tri2;

		    for (int j=0; j<3; j++) {
			if (t->pt(j).y() < yval)
			    lt[j] = true;
			// UGH -- hard coded for now
			if (t->pt(j).y() < yval-tol)
			    side[j] = left;
			else if (t->pt(j).y() > yval+tol)
			    side[j] = right;
			else
			    side[j] = middle;
		    }
		    if (lt[0] == lt[1] &&
			lt[1] != lt[2] &&
			side[2] != middle &&
			(side[0] != middle || side[1] != middle))
			{
			    // break edges 02 and 12

			    double t0 = (yval-P0.y())/(P2.y()-P0.y());
			    double t1 = (yval-P1.y())/(P2.y()-P1.y());

			    Point P02 = AffineCombination(P0,(1-t0),
							  P2,t0);
			    Point P12 = AffineCombination(P1,(1-t1),
							  P2,t1);

			    tri0 = new Tri(t->get_matl(),
					   P12, P2, P02);
			    tri1 = new Tri(t->get_matl(),
					   P1, P12, P02);
			    tri2 = new Tri(t->get_matl(),
					   P0, P1, P02);

			}
		    else if (lt[0] != lt[1] &&
			     lt[1] == lt[2] &&
			     side[0] != middle &&
			     (side[1] != middle || side[2] != middle))
			{
			    // break edges 01 and 02
			    double t0 = (yval-P0.y())/(P1.y()-P0.y());
			    double t1 = (yval-P0.y())/(P2.y()-P0.y());

			    Point P01 = AffineCombination(P0,(1-t0),
							  P1,t0);
			    Point P02 = AffineCombination(P0,(1-t1),
							  P2,t1);

			    tri0 = new Tri(t->get_matl(),
					   P0, P01, P02);
			    tri1 = new Tri(t->get_matl(),
					   P1, P2, P01);
			    tri2 = new Tri(t->get_matl(),
					   P01, P2, P02);
			}
		    else if (lt[0] == lt[2] &&
			     lt[1] != lt[2] &&
			     side[1] != middle &&
			     (side[0] != middle || side[2] != middle))
			{
			    // break edges 01 and 12
			    double t0 = (yval-P0.y())/(P1.y()-P0.y());
			    double t1 = (yval-P1.y())/(P2.y()-P1.y());

			    Point P01 = AffineCombination(P0,(1-t0),
							  P1,t0);
			    Point P12 = AffineCombination(P1,(1-t1),
							  P2,t1);

			    tri0 = new Tri(t->get_matl(),
					   P01, P1, P12);
			    tri1 = new Tri(t->get_matl(),
					   P0, P01, P2);
			    tri2 = new Tri(t->get_matl(),
					   P01, P12, P2);
			}
		    else
			continue;
		    g->objs[i] = tri0;
		    g->add(tri1);
		    g->add(tri2);
		}
	}
}    
		    
Group *Split_Data2(Group* g)
{
    printf("IN SPLIT DATA2****************************************\n");

    double xmin = -2430;
    double ymin = -1174;
    double xmax = 2430;
    double ymax = 1174;

    double ytol = (ymax-ymin)*.001;
    double xtol = (xmax-xmin)*.001;

    SplitTrisX(g,xmin,xtol);
    SplitTrisX(g,xmax,xtol);
    SplitTrisY(g,ymin,ytol);
    SplitTrisY(g,ymax,ytol);

    return Split_Data(g);
}

Group *get_object(ifstream &infile) {
  Group *result = 0;
  int parsing_roof=0;
  string token("");
  int curley = 0;
  // eat the top curley brace and increment the curley brace counter
  infile >> token;
  curley++;
  int matl_index = 0;
  int obj_too_small=0;
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
      string subtoken = token.substr(1, 10);
      if (subtoken == "ROOF_SHELL") {
	parsing_roof=1;
      } else parsing_roof=0;
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
	  if (num_faces == 0) {
	    cerr << "*** Note - object had no faces, disregarding it.\n";
	    obj_too_small=1;
	  }
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
  result = new Group();
  if (parsing_roof) {
    CycleMaterial *cm = dynamic_cast<CycleMaterial*>(ase_matls[ase_matls.size()-1]);
    if (cm->members.size() == 0) {
      cm->members.add(ase_matls[matl_index]);
      cm->members.add(new InvisibleMaterial);
      cm->members.add(new DielectricMaterial(1.0, 1.0, 0.3, 400.0, Color(1,1,1), Color(1,1,1), false));
    }
    matl_index=ase_matls.size()-1;
  }
  for(unsigned int i = 0; i < faces.size(); i++) {
    Object *face = faces[i];
    face->set_matl(ase_matls[matl_index]);
    result->add(face);
  }
  //  cerr << "ase_load::get_objects::No objects found.\n";
#if 0
  if(result->numObjects() > 20)
    result = new Grid(result, 30);
  else if (result->numObjects()>2)
    result = new BV1(result);
#endif

  if (obj_too_small) { delete result; return 0; }

//    return Split_Data2(result);

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

  rtrt::Plane groundplane ( rtrt::Point(0, 0, 0), Vector(1, 0, 0) );
  Scene* scene=new Scene(all, cam,
			 bgcolor, groundcolor*bgcolor, bgcolor, groundplane,
			 ambient_scale);
  scene->add_light(new Light(Point(500,-300,300), Color(.8,.8,.8), 0));
  scene->set_background_ptr(new EnvironmentMapBackground("/home/sci/dmw/stadium/SKY.ppm"));
  scene->select_shadow_mode(Hard_Shadows);
  scene->set_materials(ase_matls);
  cerr << "num materials="<<ase_matls.size()<<"\n";
  CycleMaterial *cm;
  if (cm = dynamic_cast<CycleMaterial*>(scene->get_material(ase_matls.size()-1))) {
    cerr << " CYCLE! -- nmembers="<<cm->members.size()<<"\n";
  } else {
    cerr << " not a cycle material.\n";
  }

  printf("LEAVING MAKE SCENE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

  return scene;
}

// usage:
// ./rtrt -np 16 -no_shadows -bv 3 -gridcellsize 80 -scene scenes/3dsm_ase /home/sci/dmw/stadium/Ford\ SGI\ model\ 003.ASE
//
// ./rtrt -np 16 -no_shadows -bv 4 -hgridcellsize 20 15 10 -minobjs 20 20 -scene scenes/3dsm_ase /home/sci/dmw/stadium/Ford\ SGI\ model\ 003.ASE
