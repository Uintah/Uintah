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



BumpMaterial::BumpMaterial(Material *m, char *filename, double persist)
{
  persistence = persist;
  material = m;
  if(m == NULL)
    material = new LambertianMaterial(Color(1,0,0));
  if(read_file(filename) == 0)
    {
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
 if(map != NULL)
   {
     float u,v;
     Vector pu,pv,d;
     double fu,fv;

     map->uv(uv_m,hitpos,hit);
     double persist = get_persistence();
     u = uv_m.u()*persist; 
     v = uv_m.v()*persist;     
     u -= (int) u;
     v -= (int) v;
     if(u < 0) u += 1;
     if(v < 0) v += 1; 
     map->get_frame(hitpos, hit, normal,pu,pv);
     if(dimension_x == dimension_y && dimension_x == 1)
       return;
     //how to get things form a file
     fu = (fval(u+evaru,v)-fval(u-evaru,v))*dimension_x/2.f;//  / (2*evaru);
     fv = (fval(u,v+evarv)-fval(u,v-evarv))*dimension_y/2.f;//  / (2*evarv);
     // could use fwd diff too 
     //normal += sin(u*M_PI*2*10)*pv; //use this for a sphere - procedural
     //normal += sin(u/120)*pv; // use this for a plane - procedural
     d = (fu*pu - fv*pv);
     d.safe_normalize();
     normal += d;
     normal.safe_normalize();
   }
}

void BumpMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth,
		  double a, const Color& c,
		  Context* cx)
{

    double nearest=hit.min_t;
    Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*nearest);
    Vector normal(obj->normal(hitpos, hit));
    perturbnormal(normal,ray, hit);
    BumpObject n(normal);
    BumpObject *n2 = &n;
    //Object *o2 = new BumpObject(normal); // (Object *)n2; 
    Object *o2 = (Object *)n2;
    HitInfo h2 = hit;
    h2.hit_obj = o2;
    material->shade(result,ray,h2,depth,a,c,cx);
    //delete o2;
    
}


FILE * BumpMaterial::readcomments(FILE *fin)
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
  double max = temp;
  bumpimage = (int *)malloc(dimension_x*dimension_y*sizeof(int));
  printf("Converting from 0->%d (x3) to Vector 0->1 by division\n",temp);
  printf("Reading in File for a Bump Map%s\n",filename);
  for(int j = dimension_y-1; j >= 0; j--)
    for(int i = 0; i < dimension_x; i++)
      { //ramsey
	fscanf(fin,"%c%c%c",&ra,&ga,&ba);
	r = (int)ra; g = (int)ga; b = (int)ba; 
	bumpimage[j*dimension_x+i] = (r+g+b)/3;
      }
  evaru = 1.0f/dimension_x;
  evarv = 1.0f/dimension_y;
  printf("File read\n");
  fclose(fin);
  return 1;
}

int BumpMaterial::read_file(char *filename)
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
    bumpimage = (int *)malloc(dimension_x*dimension_y*sizeof(int));
    printf("Reading in File for a Bump Map%s\n",filename);
    for(int j = dimension_y-1; j >= 0; j--)
      for(int i = 0; i < dimension_x; i++)
        {
          fscanf(fin,"%d %d %d",&r,&g,&b);
	  bumpimage[j*dimension_x+i] = (r+g+b)/3.0f;
        }   
    evaru = 1.0f/dimension_x;
    evarv = 1.0f/dimension_y;
	
    printf("File read\n");
    fclose(fin);
    return 1;







  /*
  ifstream fin;
  int dimension;
  //char buf[256];
  int i,j;
  fin.open(filename);
  if(!fin)
    { //couldn't open file
      return 0;
    }
  fin >> dimension;
  dimension_x = dimension_y = dimension;
  bumpimage = new int[dimension*dimension];
  for(i = 0; i < dimension; i++)
    for(j = 0; j < dimension; j++)
      {
	if(fin.eof()) // endo f file too early
	  {
	    return 0;
	  }
	fin >> bumpimage[i*dimension + j];
      }
  evaru = 1./(double)(dimension);
  evarv = evaru;
  */
  return 1;
}


//useful in derivs
double BumpMaterial::fval(double u, double v)
{
  int f1,f2,f3,f4,iu,iv,iu1,iv1;
  double fu0,fu1,du,dv;
  iu = ((double)dimension_x*u);
  iv = dimension_y - (int)((double)dimension_y*v);
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
  du = dimension_x*u - (double)iu;
  dv = dimension_y*v - (double)iv;
  fu0 = f1+du*(f2-f1);
  fu1 = f3+du*(f4-f3);
  return fu0 + dv * (fu1-fu0);
}


