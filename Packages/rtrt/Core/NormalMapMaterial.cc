#include <Packages/rtrt/Core/NormalMapMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/UV.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/vec.h>

#include <math.h>

using namespace rtrt;
using namespace std;
using namespace SCIRun;

Persistent* normalMapMaterial_maker() {
  return new NormalMapMaterial();
}

// initialize the static member type_id
PersistentTypeID NormalMapMaterial::type_id("NormalMapMaterial", "Material", 
					    normalMapMaterial_maker);

NormalMapMaterial::NormalMapMaterial(Material *m, char *filename, 
				     double persist)
{
  persistence = persist;
  material = m;
  if(m == NULL)
    material = new LambertianMaterial(Color(1,0,0));
  //if(read_file(filename) == 0)
  if(readfromppm(filename) == 0)
  {
    cout << "FILE " << filename  <<" NOT READ - PROBLEMS IMMINENT" << endl;
    dimension_x = 1;
    dimension_y = 1;
    normalmapimage = new Vector[1];
    normalmapimage[0] = Vector(0,0,1);
  }
  cout << "Dimx=" << dimension_x << " Dimy=" << dimension_y << endl;
}

NormalMapMaterial::~NormalMapMaterial()
{
}

void NormalMapMaterial::perturbnormal(Vector &normal, 
	const Ray &ray,
	const HitInfo &hit)
{
  UVMapping *map = hit.hit_obj->get_uvmapping();
  UV uv_m;
  Point hitpos(ray.origin()+ray.direction()*hit.min_t);
  if(map != NULL)
  {
    float u,v;

    map->uv(uv_m,hitpos,hit);
    double persist = get_persistence();
    u = uv_m.u()*persist; 
    v = uv_m.v()*persist;     
    u -= (int) u;
    v -= (int) v;
    if(u < 0) u += 1;
    if(v < 0) v += 1; 
    normal = fval(u,v);
  }
}

void NormalMapMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth,
		  double a, const Color& c,
		  Context* cx)
{

  double nearest=hit.min_t;
  Object* obj=hit.hit_obj;
  Point hitpos(ray.origin()+ray.direction()*nearest);
  Vector normal(obj->normal(hitpos, hit));
  perturbnormal(normal,ray,hit); 
  BumpObject n = BumpObject(normal, obj->get_uvmapping());
  BumpObject *n2 = &n;
  Object *o2 = (Object *)n2;
  //Object *o2 = new BumpObject(normal);
  HitInfo h2 = hit;
  h2.hit_obj = o2;
  material->shade(result,ray,h2,depth,a,c,cx);
  //delete o2;
}


//all of the following content is needed to bump objects from a file

int NormalMapMaterial::read_file(char *filename)
{
  ifstream fin;
  int dimension;
  //char buf[256];
  int i,j;
  double x,y,z;
  fin.open(filename);
  if(!fin)
    { //couldn't open file
      return 0;
    }
  fin >> dimension;
  dimension_x = dimension_y = dimension;
  normalmapimage = new Vector[dimension*dimension];
  for(i = 0; i < dimension; i++)
    for(j = 0; j < dimension; j++)
      {
	if(fin.eof()) // endo f file too early
	  {
	    return 0;
	  }
	fin >> x >> y >> z;
	normalmapimage[i*dimension + j] = Vector(x,y,z);
      }
  evaru = 1./(double)(dimension);
  evarv = evaru;
  return 1;
}


//useful in derivs
Vector NormalMapMaterial::fval(double u, double v)
{
  Vector f1,f2,f3,f4;
  int iu,iv,iu1,iv1;
  Vector fu0,fu1;
  double du,dv;
  iu = ((double)dimension_x*u);
  iv = ((double)dimension_y*v);
  if(iu >= dimension_x || iu < 0) iu = 0;
  if(iv >= dimension_y || iv < 0) iv = 0;
  iu1 = iu+1; 
  if(iu1 >= dimension_x) iu1 = 0;
  iv1 = iv+1;
  if(iv1 >= dimension_y) iv1 = 0;
  f1 = normalmapimage[iv*dimension_x + iu];
  f2 = normalmapimage[iv*dimension_x +iu1];
  f3 = normalmapimage[iv1*dimension_x +iu];
  f4 = normalmapimage[iv1*dimension_x +iu1];
  du = dimension_x*u - (double)iu;
  dv = dimension_y*v - (double)iv;
  fu0 = f1+du*(f2-f1);
  fu1 = f3+du*(f4-f3);
  return fu0 + dv * (fu1-fu0);
}


FILE* NormalMapMaterial::readcomments(FILE *fin)
{
  char q;
  char buf[1024];
  fscanf(fin,"%c",&q);
  while(q == '#' || q == '\n' || q == ' ')
  {
    if(q == '#')
      fgets(buf,256,fin); // read the line - ignore the line
    q = fgetc(fin);
  }
  if(ungetc(q,fin) == EOF)
    printf("error in putc\n");
  return fin;
}

int NormalMapMaterial::readfromppm6(char *filename)
{
  FILE *fin;
  fin = fopen(filename,"r");
  if(!fin)
  {printf("Couldn't open file %s\n",filename); return 0;}
  char buf[256];
  fscanf(fin,"%s",buf);
  if(strcmp(buf,"P6") != 0)
  {
    printf("File is not a P6 file - rejected!\n");
    return 0;
  }
  int temp;
  fin = readcomments(fin);
  fscanf(fin,"%d", &dimension_x);
  printf("width=%d ",dimension_x);
  fin = readcomments(fin);
  fscanf(fin,"%d", &dimension_y);
  printf("height=%d \n",dimension_y);
  fin = readcomments(fin);  
  fscanf(fin,"%d\n",&temp);
  unsigned char ra,ga,ba;
  double r,g,b;
  double max = temp;
  normalmapimage = (Vector *)malloc(dimension_x*dimension_y*sizeof(Vector));
  printf("Converting from 0->%d (x3) to Vector 0->1 by division\n",temp);
  printf("Reading in File for a Normal Map=%s\n",filename);
  for(int j = dimension_y-1; j >= 0; j--)
    for(int i = 0; i < dimension_x; i++)
      { //ramsey
	fscanf(fin,"%c%c%c",&ra,&ga,&ba);
	r = ((double)ra / (max/2.0f))-1.0f; 
	g = ((double)ga / (max/2.0f))-1.0f; 
	b = ((double)ba / (max/2.0f))-1.0f; 
        normalmapimage[j*dimension_x+i] =  Vector(r,g,b);
        normalmapimage[j*dimension_x+i].normalize(); // need normalized Vectors
      }
  printf("File read\n");
  fclose(fin);
  return 1;
}


int NormalMapMaterial::readfromppm(char *filename)
{
  FILE *fin;
  fin = fopen(filename,"r");
  if(!fin)
  {printf("Couldn't open file %s\n",filename); return 0;}
  char buf[256];
  fscanf(fin,"%s",buf);
  if(strcmp(buf,"P3") != 0)
  {
    if(strcmp(buf,"P6") != 0)
    {
      printf("Don't have a P3 file nor a P6\n");
      return 0;
    }
    else
    {
      fclose(fin);
      return readfromppm6(filename);
    }
  }
  
  int temp;
  fin = readcomments(fin);
  fscanf(fin,"%d %d\n", &dimension_x, &dimension_y);
  fin = readcomments(fin);
  fscanf(fin,"%d",&temp);
  double max = temp;
  int r,g,b;
  double rv,gv,bv;
  normalmapimage = (Vector *)malloc(dimension_x*dimension_y*sizeof(Vector));
  printf("Reading in File for a Normal Map=%s\n",filename);
  for(int j = dimension_y-1; j >= 0; j--)
    for(int i = 0; i < dimension_x; i++)
    {
      fscanf(fin,"%d %d %d",&r,&g,&b);
      rv = (r / (max/2.0f))-1.0f; 
      gv = (g / (max/2.0f))-1.0f; 
      bv = (b / (max/2.0f))-1.0f; 
      normalmapimage[j*dimension_x+i] =  Vector(rv,gv,bv);
      normalmapimage[j*dimension_x+i].normalize(); // need normalized Vectors
    }   
  printf("File read\n");
  fclose(fin);
  return 1;
}

const int NORMALMAPMATERIAL_VERSION = 1;

void 
NormalMapMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("NormalMapMaterial", NORMALMAPMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, material);
  SCIRun::Pio(str, dimension_x);
  SCIRun::Pio(str, dimension_y);
  if (str.reading()) {
    normalmapimage = new Vector[dimension_x*dimension_y];
  }
  
  for (int i = 0; i < dimension_x*dimension_y; i++) {
    SCIRun::Pio(str, normalmapimage[i]);
  }

  SCIRun::Pio(str, evaru);
  SCIRun::Pio(str, evarv);
  SCIRun::Pio(str, persistence);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::NormalMapMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::NormalMapMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::NormalMapMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
