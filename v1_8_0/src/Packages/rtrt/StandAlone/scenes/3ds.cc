
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/HashTable.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Core/Math/MiscMath.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#ifndef __linux
#include <bstring.h>
#endif

#ifdef __sgi
#define SWAP
#endif
#ifdef __i386__
#define NOSWAP
#endif

#ifndef SWAP
#ifndef NOSWAP
#error "Architecture bytesex not defined..."
#endif
#endif

#define CH_VERSION 0x0002
#define CH_COLOR_F 0x0010
#define CH_COLOR_24 0x0011
#define CH_INT_PERCENTAGE 0x0030
#define CH_FLOAT_PERCENTAGE 0x0031
#define CH_MASTER_SCALE 0x0100

#define CH_BACKGROUND 0x1200

#define CH_AMBIENT 0x2100

#define CH_EDIT3DS 0x3d3d
#define CH_MESH_VERSION 0x3d3e

#define CH_NAMED_OBJECT 0x4000
#define CH_NTRI_OBJECT 0x4100
#define CH_POINT_ARRAY 0x4110
#define CH_FACE_ARRAY 0x4120
#define CH_FACE_MATLS 0x4130
#define CH_MAIN3DS 0x4d4d
#define CH_LIGHT 0x4600
#define CH_CAMERA 0x4700

#define CH_VIEW1 0x7001

#define CH_MAT_NAME 0xa000
#define CH_MAT_AMBIENT 0xa010
#define CH_MAT_DIFFUSE 0xa020
#define CH_MAT_SPECULAR 0xa030
#define CH_MAT_SHININESS 0xa040
#define CH_MAT_TRANSPARENCY 0xa050
#define CH_MAT_XPFALL 0xa052
#define CH_MAT_REFBLUR 0xa053
#define CH_MAT_TWO_SIDE 0xa081
#define CH_MAT_SHADING 0xa100
#define CH_MAT_ENTRY 0xafff

#define CH_KEYFRAME 0xb000

using namespace rtrt;
using namespace std;

struct Face {
    int v1, v2, v3;
    int flags;
    Material* matl;
    Face();
    Face(int v1, int v2, int v3, int flags);
};

Face::Face()
{
    matl=0;
}

Face::Face(int v1, int v2, int v3, int flags)
    : v1(v1), v2(v2), v3(v3), flags(flags)
{
    matl=0;
}

struct State3D {
    int ntri;
    int maxtri;
    bool read_camera;
    Camera* camera;
    Scene* scene;
    double light_radius;
    float master_scale;
    HashTable<string, Material*> matls;
    State3D();
    Color ambient;
    Array1<Object*> nomatl;
};

State3D::State3D()
{
    ntri=0;
    maxtri=100000000;
    read_camera=false;
    master_scale=1;
    ambient=Color(0,0,0);
}

struct Mat {
    Mat();
    char* name;
    Color ambient;
    Color diffuse;
    Color specular;
    float shininess;
    float transp;
    float xpfall;
    float refblur;
    bool twosided;
    int shading_value;
};

Mat::Mat()
{
    name="unknown";
    ambient=diffuse=specular=Color(0,0,0);
    shininess=0;
    transp=0;
    xpfall=0;
    refblur=0;
    twosided=false;
    shading_value=0;
}

class Chunk {
    char* buf;
    char* cur_chunk;
    char* end_chunk;

    int get_long(char* buf);
    int get_short(char* buf);
    int get_byte(char* buf);
    float get_float(char* buf);
public:
    Chunk(void* addr);
    int type();
    int length();
    void basic_length(int length);
    Chunk* next_chunk();

    int get_long(int offset);
    int get_short(int offset);
    int get_byte(int offset);
    float get_float(int offset);
    char* get_cstr(int offset);
    void assert_length(int l, char* msg);
};

Chunk::Chunk(void* addr)
    : buf((char*)addr)
{
    cur_chunk=end_chunk=0;
}

int Chunk::type()
{
    return get_short(buf);
}

int Chunk::length()
{
    return get_long(buf+2);
}

void Chunk::basic_length(int l)
{
    cur_chunk=buf+l+6;
    end_chunk=buf+length();
}

void Chunk::assert_length(int l, char* msg)
{
    if(length()-6 != l){
	cerr << "Expecting a chunk length of " << l+6 << ", but have one of " << length() << '\n';
	cerr << msg << '\n';
	//exit(1);
    }
}

Chunk* Chunk::next_chunk()
{
    if(cur_chunk==0){
	cerr << "next_chunk called, but length not set!\n";
	exit(1);
    }
    if(cur_chunk == end_chunk){
	return 0;
    }
    if(cur_chunk > end_chunk){
	cerr << "Chunk overshoot!!!\n";
	exit(1);
    }
    Chunk* ret=new Chunk(cur_chunk);
    cur_chunk+=ret->length();
    return ret;
}

int Chunk::get_byte(char* p)
{
    return *(unsigned char*)p;
}

int Chunk::get_short(char* p)
{
#ifdef SWAP
    char b[2];
    b[0]=p[1];
    b[1]=p[0];
    return *(unsigned short*)b;
#else
    return *(unsigned short*)p;
#endif
}

int Chunk::get_long(char* p)
{
#ifdef SWAP
    char b[4];
    b[0]=p[3];
    b[1]=p[2];
    b[2]=p[1];
    b[3]=p[0];
    return *(int*)b;
#else
    return *(int*)p;
#endif
}

float Chunk::get_float(char* p)
{
#ifdef SWAP
    char b[4];
    b[0]=p[3];
    b[1]=p[2];
    b[2]=p[1];
    b[3]=p[0];
    return *(float*)b;
#else
    return *(float*)p;
#endif
}

char* Chunk::get_cstr(int offset)
{
    return buf+6+offset;
}

int Chunk::get_long(int offset)
{
    return get_long(buf+6+offset);
}

int Chunk::get_short(int offset)
{
    return get_short(buf+6+offset);
}

int Chunk::get_byte(int offset)
{
    return get_byte(buf+6+offset);
}

float Chunk::get_float(int offset)
{
    return get_float(buf+6+offset);
}

#if 0
void read_tmpl(Chunk* ch, Group* world, State3D& state)
{
    ch->basic_length(0);
    Chunk* chunk;
    while(chunk=ch->next_chunk()){
	switch(chunk->type()){
	default:
	    cerr << "read_tmpl: Unknown chunk type: " << hex << chunk->type() << dec << '\n';
	    break;
	}
	delete chunk;
    }
}
#endif

void read_camera(Chunk* ch, Group*, State3D& state)
{
    ch->basic_length(32);
    float eyex=ch->get_float(0);
    float eyey=ch->get_float(4);
    float eyez=ch->get_float(8);
    float lookatx=ch->get_float(12);
    float lookaty=ch->get_float(16);
    float lookatz=ch->get_float(20);
    float rotate=ch->get_float(24);
    float lens=ch->get_float(28);
    cerr << "rotate=" << rotate << '\n';
    cerr << "lens=" << lens << '\n';
    Vector up(0,0,1);
    Chunk* chunk;
    while(chunk=ch->next_chunk()){
	switch(chunk->type()){
	default:
	    cerr << "read_camera: Unknown chunk type: " << hex << chunk->type() << dec << '\n';
	    break;
	}
	delete chunk;
    }
    Point eye(eyex, eyey, eyez);
    Point lookat(lookatx, lookaty, lookatz);
    double fov=lens;
    *state.camera=Camera(eye, lookat, up, fov);
    state.read_camera=true;
}

void read_light(Chunk* ch, Group*, State3D& state)
{
    ch->basic_length(12);
    float lx=ch->get_float(0);
    float ly=ch->get_float(1);
    float lz=ch->get_float(2);
    Color lightcolor(0,0,0);
    Chunk* chunk;
    while(chunk=ch->next_chunk()){
	switch(chunk->type()){
	case CH_COLOR_24:
	    {
		chunk->assert_length(3, "CH_COLOR_24");
		unsigned char r=chunk->get_byte(0);
		unsigned char g=chunk->get_byte(1);
		unsigned char b=chunk->get_byte(2);
		lightcolor=Color(r/255., g/255., b/255.);
	    }
	break;
	case CH_COLOR_F:
	    {
		chunk->assert_length(12, "CH_COLOR_F");
		float r=chunk->get_float(0);
		float g=chunk->get_float(4);
		float b=chunk->get_float(8);
		lightcolor=Color(r, g, b);
	    }
	break;
	default:
	    cerr << "read_light: Unknown chunk type: " << hex << chunk->type() << dec << '\n';
	    break;
	}
	delete chunk;
    }
    state.scene->add_light(new Light(Point(lx,ly,lz), lightcolor, state.light_radius));
}

void read_view(Chunk* ch, Group*, State3D&)
{
    ch->basic_length(0);
    Chunk* chunk;
    while(chunk=ch->next_chunk()){
	switch(chunk->type()){
	default:
	    cerr << "read_view: Unknown chunk type: " << hex << chunk->type() << dec << '\n';
	    break;
	}
	delete chunk;
    }
}
void read_facedata(Chunk* ch, State3D& state, Array1<Face>& faces)
{
    Chunk* chunk;
    while(chunk=ch->next_chunk()){
	switch(chunk->type()){
	case CH_FACE_MATLS:
	    {
		char* matl_name=chunk->get_cstr(0);
		int p=(int)strlen(matl_name)+1;
		int nmatl=chunk->get_short(p);
		p+=2;
		Material* matl;
		if(!state.matls.lookup(matl_name, matl)){
		    cerr << "ERROR: Couldn't find material: " << matl_name << '\n';
		    exit(1);
		}
		//cerr << "Applying material " << matl_name << " to " << nmatl << " faces\n";
		for(int i=0;i<nmatl;i++){
		    int face=chunk->get_short(p);
		    faces[face].matl=matl;
		    p+=2;
		}
	    }
	    break;
	default:
	    cerr << "read_facedata: Unknown chunk type: " << hex << chunk->type() << dec << " (length=" << chunk->length() << ")\n";
	    break;
	}
	delete chunk;
    }
}
void read_ntri_object(Chunk* ch, Group* world, State3D& state)
{
    ch->basic_length(0);
    Chunk* chunk;
    Array1<Point> points;
    Array1<Face> faces;
    while(chunk=ch->next_chunk()){
	switch(chunk->type()){
	case CH_POINT_ARRAY:
	    {
		int npoints=chunk->get_short(0);
		points.resize(npoints);
		//cerr << "Reading " << npoints << " points\n";
		int p=2;
		for(int i=0;i<npoints;i++){
		    float x=chunk->get_float(p+0);
		    float y=chunk->get_float(p+4);
		    float z=chunk->get_float(p+8);
		    points[i]=Point(x,y,z);
		    p+=12;
		}
		chunk->assert_length(p, "POINT_ARRAY");
	    }
	    break;
	case CH_FACE_ARRAY:
	    {
		int nfaces=chunk->get_short(0);
		//cerr << "Reading " << nfaces << " faces\n";
		faces.resize(nfaces);
		int p=2;
		for(int i=0;i<nfaces;i++){
		    int v1=chunk->get_short(p+0);
		    int v2=chunk->get_short(p+2);
		    int v3=chunk->get_short(p+4);
		    int flags=chunk->get_short(p+6);
		    faces[i]=Face(v1, v2, v3, flags);
		    p+=8;
		}
		chunk->basic_length(p);
		read_facedata(chunk, state, faces);
	    }
	    break;
	default:
	    cerr << "read_ntri_object: Unknown chunk type: " << hex << chunk->type() << dec << '\n';
	    break;
	}
	delete chunk;
    }
    //cerr << "Adding " << faces.size() << " faces\n";
    for(int i=0;i<faces.size();i++){
	Face& f=faces[i];
	Point p1(points[f.v1]);
	Point p2(points[f.v2]);
	Point p3(points[f.v3]);
	Tri* tri=new Tri(f.matl, p1, p2, p3);
	state.ntri++;
	if(state.ntri > state.maxtri || tri->isbad()){
	    delete tri;
	} else {
	    world->add(tri);
	    if(!f.matl){
		state.nomatl.add(tri);
	    }
	}
    }
}

void read_named_object(Chunk* ch, Group* world, State3D& state)
{
    Chunk* chunk;
    while(chunk=ch->next_chunk()){
	switch(chunk->type()){
	case CH_NTRI_OBJECT:
	    read_ntri_object(chunk, world, state);
	    break;
	case CH_LIGHT:
	    read_light(chunk, world, state);
	    break;
	case CH_CAMERA:
	    read_camera(chunk, world, state);
	    break;
	default:
	    cerr << "read_named_object: Unknown chunk type: " << hex << chunk->type() << dec << '\n';
	    break;
	}
	delete chunk;
    }
}

Color read_color(Chunk* ch)
{
    ch->basic_length(0);
    Chunk* chunk=ch->next_chunk();
    if(chunk->type() == CH_COLOR_24){
	chunk->assert_length(3, "CH_COLOR_24");
	unsigned char r=chunk->get_byte(0);
	unsigned char g=chunk->get_byte(1);
	unsigned char b=chunk->get_byte(2);
	return Color(r/255., g/255., b/255.);
    } else if(chunk->type() == CH_COLOR_F){
	chunk->assert_length(12, "CH_COLOR_F");
	float r=chunk->get_float(0);
	float g=chunk->get_float(4);
	float b=chunk->get_float(8);
	return Color(r, g, b);
    } else {
	cerr << "Unknown color chunk: " << chunk->type() << '\n';
	return Color(0,0,0);
    }
}

float read_percent(Chunk* ch)
{
    ch->basic_length(0);
    Chunk* chunk=ch->next_chunk();
    if(chunk->type() == CH_INT_PERCENTAGE){
	chunk->assert_length(2, "CH_INT_PERCENTAGE");
	unsigned short p=chunk->get_short(0);
	return p/65535.;
    } else if(chunk->type() == CH_COLOR_F){
	chunk->assert_length(4, "CH_FLOAT_PERCENTAGE");
	float p=chunk->get_float(0);
	return p;
    } else {
	cerr << "Unknown precentage chunk: " << chunk->type() << '\n';
	return 0;
    }
}

void read_matentry(Chunk* ch, State3D& state)
{
    Mat mat;
    ch->basic_length(0);
    Chunk* chunk;
    while(chunk=ch->next_chunk()){
	switch(chunk->type()){
	case CH_MAT_NAME:
	    mat.name=chunk->get_cstr(0);
	    //cerr << "Reading material: " << mat.name << '\n';
	    break;
	case CH_MAT_AMBIENT:
	    mat.ambient=read_color(chunk);
	    break;
	case CH_MAT_DIFFUSE:
	    mat.diffuse=read_color(chunk);
	    break;
	case CH_MAT_SPECULAR:
	    mat.specular=read_color(chunk);
	    break;
	case CH_MAT_SHININESS:
	    mat.shininess=read_percent(chunk);
	    break;
	case CH_MAT_TRANSPARENCY:
	    mat.transp=read_percent(chunk);
	    break;
	case CH_MAT_XPFALL:
	    mat.xpfall=read_percent(chunk);
	    break;
	case CH_MAT_REFBLUR:
	    mat.refblur=read_percent(chunk);
	    break;
	case CH_MAT_TWO_SIDE:
	    chunk->assert_length(0, "MAT_TWO_SIDE");
	    mat.twosided=true;
	    break;
	case CH_MAT_SHADING:
	    chunk->assert_length(2, "MAT_SHADING");
	    mat.shading_value=chunk->get_short(0);
	    cerr << "? shading_value=" << mat.shading_value << '\n';
	    break;
	default:
	    cerr << "read_matentry: Unknown chunk type: " << hex << chunk->type() << dec << '\n';
	    break;
	}
	delete chunk;
    }
    mat.shininess=100;
    Material* matl=new Phong(mat.diffuse, mat.specular,
			     mat.shininess);
    //cerr << "Material: " << mat.name << " " << mat.ambient << " " << mat.diffuse << " " << mat.specular << " " << mat.shininess << '\n';
    state.matls.insert(mat.name, matl);
}

void read_edit3ds(Chunk* ch, Group* world, State3D& state)
{
    ch->basic_length(0);

    Chunk* chunk;
    while(chunk=ch->next_chunk()){
	switch(chunk->type()){
	case CH_MESH_VERSION:
	    {
		chunk->assert_length(4, "MESH_VERSION");
		int vers=chunk->get_long(0);
		if(vers != 2){
		    cerr << "Mesh Version is " << vers << ", but expected 2\n";
		    cerr << "proceeding anyway, but don't expect to get far...\n";
		}
	    }
	    break;
	case CH_MAT_ENTRY:
	    {
		read_matentry(chunk, state);
	    }
	    break;
	case CH_MASTER_SCALE:
	    {
		state.master_scale=chunk->get_float(0);
		chunk->assert_length(4, "MASTER_SCALE");
	    }
	    break;
	case CH_NAMED_OBJECT:
	    {
		char* name=chunk->get_cstr(0);
		cerr << "Reading object: " << name << '\n';
		chunk->basic_length((int)(strlen(name)+1));
		read_named_object(chunk, world, state);
	    }
#if 0
	case CH_VIEW1:
	    {
		read_view(chunk, world, state);
	    }
#endif
	    break;
	case CH_AMBIENT:
	    {
		state.ambient=read_color(chunk);
	    }
	    break;
	case CH_BACKGROUND:
	    {
		state.scene->set_bgcolor(read_color(chunk));
	    }
	    break;
	default:
	    cerr << "read_edit3ds: Unknown chunk type: " << hex << chunk->type() << dec << " (length=" << chunk->length() << ")\n";
	    break;
	}
	delete chunk;
    }
}

void read_3ds(char* filename, Scene* scene, double light_radius, int maxtri)
{
    Camera* camera=scene->get_camera(0);
    Group* world=new Group;
    scene->set_object(world);
    int fd=open(filename, O_RDONLY);
    struct stat buf;
    if(fstat(fd, &buf) != 0){
	perror("fstat");
	exit(-1);
    }
    long len=(long)buf.st_size;
    void* addr=mmap(0, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if((long)addr == -1){
	perror("mmap");
	exit(-1);
    }
    Chunk* mc=new Chunk(addr);
    if(mc->type() != CH_MAIN3DS){
	cerr << "Not a 3DS file!\n";
	exit(1);
    }
    mc->basic_length(0);

    State3D state;
    state.camera=camera;
    state.scene=scene;
    state.light_radius=light_radius;
    state.maxtri=maxtri;

    Chunk* chunk;
    while(chunk=mc->next_chunk()){
	switch(chunk->type()){
	case CH_VERSION:
	    {
		int vers=chunk->get_long(0);
		if(vers != 3){
		    cerr << "Version is " << vers << ", but expected 3\n";
		    cerr << "proceeding anyway, but don't expect to get far...\n";
		}
		chunk->assert_length(4, "VERSION");
	    }
	    break;
	case CH_EDIT3DS:
	    {
		read_edit3ds(chunk, world, state);
	    }
	    break;
	case CH_KEYFRAME:
	    {
		cerr << "Skipping keyframe config\n";
	    }
	    break;
	default:
	    cerr << "Unknown chunk type: " << hex << chunk->type() << dec << '\n';
	    break;
	}
	delete chunk;
    }
    cerr << "Done parsing file\n";
    if(!state.read_camera){
	BBox bbox;
	world->compute_bounds(bbox, 0);
	if(bbox.valid()){
	    cerr << "No valid camera - autoviewing\n";
	    Point min(bbox.min());
	    Point max(bbox.max());
	    Vector diag(max-min);
	    Point lookat(min+diag*0.5);
	    Point eye(lookat+Vector(0,0,diag.length()));
	    Vector up(0,1,0);
	    double fov=45;
	    *state.camera=Camera(eye, lookat, up, fov);
	}
    }
    if(state.nomatl.size()>0){
	cerr << "Applying a default material to " << state.nomatl.size() << " objects\n";
	Material* default_matl=new Phong(Color(.5,.5,.5),
					 Color(.4,.4,.4),
					 100, 0);
	for(int i=0;i<state.nomatl.size();i++){
	    Object* obj=state.nomatl[i];
	    obj->set_matl(default_matl);
	}
    }
    if(scene->nlights()==0){
	cerr << "No lights - making one at the eye\n";
	Camera* cam=scene->get_camera(0);
	scene->add_light(new Light(cam->get_eye(), Color(1,1,1), light_radius));
    }
		  
    scene->copy_camera(0);
    if(strcmp(filename, "ROACH.3DS")==0){
	Material* ground=new Checker(new Phong(Color(.95,.95,.95), Color(.1,.1,.1), 0, .05),
				     new Phong(Color(.7,.3,.3), Color(.1,.1,.1), 0, .05),
				     Vector(.02,.02,0), Vector(-.02,.02,0));
	world->add(new Rect(ground, Point(0,0,-40), Vector(0,8000,-500), Vector(8000,0,0)));
	scene->add_light(new Light(Point(300,300,300), Color(1,1,1), light_radius));
	//world->add(new Sphere(ground, Point(500,500,500), 20, Vector(0,0,0)));
	scene->set_bgcolor(Color(.2,.2,.2));
	//Camera* cam=scene->get_camera(0);
	//*cam=Camera(Point(-613,-5432,685), Point(-580,-5404,.26), Vector(.878,.477,-.023), 44.23);
	scene->copy_camera(0);
    }
}

#ifdef __GNUG__

#include <Packages/rtrt/Core/Array1.h>
template class Array1<Face>;

#endif

extern "C" 
Scene* make_scene(int argc, char* argv[])
{
    int maxtri=100000000;
    char* file=0;

    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-maxtri")==0){
	    i++;
	    maxtri=atoi(argv[i]);
	} else {
	    if(file){
		cerr << "Unknown option: " << argv[i] << '\n';
		cerr << "Valid options for scene: " << argv[0] << '\n';
		cerr << " -maxtri    - limit number of triangles in scene\n";
		return 0;
	    }
	    file=argv[i];
	}
    }

    Camera cam(Point(1,0,0), Point(0,0,0),
	       Vector(0,0,1), 40);

    Color groundcolor(0,0,0);
    //Color averagelight(1,1,1);
    double ambient_scale=.5;

    Color bgcolor(0,0,0);

    rtrt::Plane groundplane ( rtrt::Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(0, cam,
			   bgcolor, groundcolor*bgcolor, bgcolor, groundplane,
			   ambient_scale);
    read_3ds(file, scene, 0, maxtri);
    scene->select_shadow_mode(Hard_Shadows);
    return scene;
}
