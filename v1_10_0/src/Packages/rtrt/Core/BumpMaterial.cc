#include <Packages/rtrt/Core/BumpMaterial.h>
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
using namespace SCIRun;
using namespace std;


void BumpMaterial::print()
{
printf("Enter BUMPMATERIAL print\n");
for(int j = 0; j < dimension_y; j++) 
  for(int i = 0; i < dimension_x; i++)
    printf("[%d,%d]=%d\n",i,j,bumpimage[j*dimension_x+i]);
printf("  End BUMPMATERIAL print\n");
}

//BumpMaterial::BumpMaterial(Material *m, char *filename, double ntiles, double bump_scale) : ntiles(ntiles), bump_scale(bump_scale)


Persistent* bumpMaterial_maker() {
  return new BumpMaterial();
}

// initialize the static member type_id
PersistentTypeID BumpMaterial::type_id("BumpMaterial", "Material", 
				       bumpMaterial_maker);

BumpMaterial::BumpMaterial(Material *m, char *filename, double ntiles, 
			   double bump_scale) : 
  ntiles(ntiles), 
  bump_scale(bump_scale)
{
  material = m;
  if(m == NULL)
    material = new LambertianMaterial(Color(1,0,0));
  if(read_file(filename) == 0) {
    cout << "FILE " << filename <<  " NOT READ - PROBLEMS IMMINENT" << endl;
    dimension_x = 1;
    dimension_y = 1;
    bumpimage = new int[1];
    bumpimage[0] = 0;
  }
  
  cout << "Dimx=" << dimension_x << " Dimy=" << dimension_y << endl;
}

BumpMaterial::~BumpMaterial()
{
}

void BumpMaterial::perturbnormal(Vector &normal, const Ray& ray,
	const HitInfo &hit)
{
  UVMapping *map = hit.hit_obj->get_uvmapping();
  UV uv_m;
  Point hitpos(ray.origin()+ray.direction()*hit.min_t);
  if(map != NULL) {
    float u,v;
    Vector pu,pv,d;
    double fu,fv;    
    map->uv(uv_m,hitpos,hit);
    u = uv_m.u()*ntiles;
    v = uv_m.v()*ntiles;
    u -= (int) u;
    v -= (int) v;
    if(u < 0) u += 1;
    if(v < 0) v += 1; 
    map->get_frame(hitpos, hit, normal,pu,pv);
    if(dimension_x == dimension_y && dimension_x == 1)
      return;
    fu = (fval(u+evaru,v)-fval(u-evaru,v))/255;//  / (2*evaru);
    fv = (fval(u,v+evarv)-fval(u,v-evarv))/255;//  / (2*evarv);
    d = (fu*pu - fv*pv)*bump_scale;
    normal += d;
    normal.safe_normalize();
  }
}

// this take the current bump map and writes it out to a normal map
// since this is taken out of the context of an object,
// the normal should be constant over the object (a plane)
// as such, the user can also define what the tangent vectors should be

void BumpMaterial::CreateNormalMap(char *bumpfile, char *filename, Vector nrml, Vector pu, Vector pv)
{
  int i,j; //i and j can also be called u and v
  Vector normal = nrml;
  double fu,fv;
  Vector *vlist = new Vector[dimension_x*dimension_y], d;
  printf("***CreateNormalMap\n");
  cout << "   bumpfile=" << bumpfile << endl << "   filename=" << filename << endl;
  for( j = 0; j < dimension_y; j++)
    for( i = 0; i < dimension_x; i++)
      {
	//normal.x(nrml.x());normal.y(nrml.y());normal.z(nrml.z());
	normal = nrml;
	
	int id = i+1, jd = j+1, ic = i-1, jc = j -1;
	if(id == dimension_x)	  id = 0;
	if(jd == dimension_y)	  jd = 0;
	if(ic == -1)              ic = dimension_x-1;
	if(jc == -1)              jc = dimension_y-1;
	fu = (bumpimage[j*dimension_x+id] - 
	      bumpimage[j*dimension_x+ic])/255.0f; // * dimension_x/2.f;
	fv = (bumpimage[jd*dimension_x+i] - 
	      bumpimage[jc*dimension_x+i])/255.0f; // *dimension_y/2.f;
	d = (fu*pu - fv*pv)*bump_scale;
	normal += d;
	normal.safe_normalize();
	//printf("normal = %lf, %lf, %lf, \n",normal.x(),normal.y(),normal.z());
	vlist[j*dimension_x+i] = normal;
      }
  writetoppm(filename,bumpfile,vlist);
}




void BumpMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth,
		  double a, const Color& c,
		  Context* cx) {

  double nearest=hit.min_t;
  Object* obj=hit.hit_obj;
  Point hitpos(ray.origin()+ray.direction()*nearest);
  Vector normal(obj->normal(hitpos, hit));
  perturbnormal(normal,ray, hit);
  BumpObject n(normal, obj->get_uvmapping());
  BumpObject *n2 = &n;
  //Object *o2 = new BumpObject(normal); // (Object *)n2; 
  Object *o2 = (Object *)n2;
  HitInfo h2 = hit;
  h2.hit_obj = o2;
  material->shade(result,ray,h2,depth,a,c,cx);
  //delete o2;
  
}


FILE * BumpMaterial::readcomments(FILE *fin) {
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

//all of the following content is needed to bump objects from a file

int BumpMaterial::readfromppm6(char *filename)
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
  int r,g,b;
  //double max = temp;
  bumpimage = (int *)malloc(dimension_x*dimension_y*sizeof(int));
  printf("Converting from 0->%d (x3) to Vector 0->1 by division\n",temp);
  printf("Reading in File for a Bump Map=%s\n",filename);
  for(int j = dimension_y-1; j >= 0; j--)
    for(int i = 0; i < dimension_x; i++)
    { //ramsey
      fscanf(fin,"%c%c%c",&ra,&ga,&ba);
      r = (int)ra; g = (int)ga; b = (int)ba; 
      bumpimage[j*dimension_x+i] = (r+g+b)/3;
    }
  evaru = 1.0f/(dimension_x);
  evarv = 1.0f/(dimension_y);
  printf("File read\n");
  fclose(fin);
  return 1;
}

int BumpMaterial::read_file(char *filename)
{
  FILE *fin;
  fin = fopen(filename,"r");
  if (!fin) {printf("Couldn't open file %s\n",filename); return 0;}
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

  //otherwise readfromppm3

    int temp;
    fin = readcomments(fin);
    fscanf(fin,"%d %d\n", &dimension_x, &dimension_y);
    fin = readcomments(fin);
    fscanf(fin,"%d",&temp);
    //double max = temp;
    int r,g,b;
    bumpimage = (int *)malloc(dimension_x*dimension_y*sizeof(int));
    printf("Reading in File for a Bump Map Open File=%s\n",filename);
    for(int j = dimension_y-1; j >= 0; j--)
      for(int i = 0; i < dimension_x; i++)
        {
          fscanf(fin,"%d %d %d",&r,&g,&b);
	  bumpimage[j*dimension_x+i] = (r+g+b)/3.0f;
        }   
    evaru = 1.0f/(dimension_x);
    evarv = 1.0f/(dimension_y);
	
    printf("\n***File %s read\n",filename);
    fclose(fin);
    return 1;

}


//useful in derivs
double BumpMaterial::fval(double u, double v)
{
  int f1,f2,f3,f4,iu,iv,iu1,iv1;
  double fu0,fu1,du,dv;
  iu = ((double)dimension_x*u);
  iv = ((double)dimension_y*v);
  du = dimension_x*u - (double)iu;
  dv = dimension_y*v - (double)iv;
  if(iu >= dimension_x || iu < 0) iu = 0;
  if(iv >= dimension_y || iv < 0) iv = 0;
  iu1 = iu+1; 
  if(iu1 >= dimension_x) iu1 = 0;
  //iv1 = iv-1;
  //if(iv1 <0) iv1 = dimension_y-1;
  iv1 = iv+1;
  if(iv1 >= dimension_y) iv1 = 0;
  f1 = bumpimage[iv*dimension_x + iu];
  f2 = bumpimage[iv*dimension_x +iu1];
  f3 = bumpimage[iv1*dimension_x +iu];
  f4 = bumpimage[iv1*dimension_x +iu1];
  fu0 = f1+du*(f2-f1);
  fu1 = f3+du*(f4-f3);
  return fu0 + dv * (fu1-fu0);
}





void BumpMaterial::writetoppm(char *filename,char *bumpfile,Vector *v)
{
  FILE *fout;
  if(!v)
    {printf("invalid vector list\n");
    return; }
  fout = fopen(filename,"w");
  if(!fout) //couldn't open file
    {printf("Couldn't open file %s\n",filename);
    return; }
  printf("Begin Output to file %s\n",filename);
  fprintf(fout,"P3\n");
  fprintf(fout,"# Ramsey: Normal Map created from Bump File %s\n",bumpfile);
  fprintf(fout,"\n%d %d 256\n",dimension_x, dimension_y);
  for(int j = dimension_y-1; j >=0; j--)
    for(int i = 0; i < dimension_x; i++)
      {
	Vector rgb = v[j*dimension_x+i];
	int r = (rgb.x()+1)*128,
	  g = (rgb.y()+1)*128,
	  b = (rgb.z()+1)*128;
	if(r > 256 || r < 0) {printf("warning [%d,%d] r=%d\n",i,j,r); r=256;}
	if(g > 256 || g < 0) {printf("warning [%d,%d] g=%d\n",i,j,g); g=256;}
	if(b > 256 || b < 0) {printf("warning [%d,%d] b=%d\n",i,j,b); b=256;}
	fprintf(fout,"%d %d %d \n",
		r,g,b); //else would output unsigned char

      }
  fclose(fout);
  printf("Output complete %s\n",filename);
}


const int BUMPMATERIAL_VERSION = 1;

void 
BumpMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("BumpMaterial", BUMPMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, material);
  SCIRun::Pio(str, dimension_x);
  SCIRun::Pio(str, dimension_y);
  SCIRun::Pio(str, evaru);
  SCIRun::Pio(str, evarv);
  SCIRun::Pio(str, ntiles);
  SCIRun::Pio(str, bump_scale);

  size_t size = dimension_x*dimension_y*sizeof(int);
  if (str.reading()) {
    // why malloc?
    bumpimage = (int *)malloc(size);
  }
  for (size_t i = 0; i < size; i++) {
    SCIRun::Pio(str, bumpimage[i]);    
  }
  str.end_class();
}


namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::BumpMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::BumpMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::BumpMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
