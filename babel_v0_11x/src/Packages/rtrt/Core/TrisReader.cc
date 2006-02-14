#include <Packages/rtrt/Core/TrisReader.h>

using namespace rtrt;
using namespace SCIRun;

Group* rtrt::readtris(char *fname, Material* matl)
{
  FILE *fp = fopen(fname,"r");
  Group* g = new Group();
  int nverts,nfaces;
  Array1<Point>verts;

  fscanf(fp,"%d %d",&nverts, &nfaces);
  for (int i=0; i<nverts; i++)
    {
      double x,y,z;
      fscanf(fp,"%lf %lf %lf",&x,&y,&z);
      verts.add(Point(x,y,z));
    }
  for (int i=0; i<nfaces; i++)
    {
      int idx1, idx2, idx3;
      
      fscanf(fp, "%d %d %d", &idx1, &idx2, &idx3);
      g->add(new Tri(matl,verts[idx1],verts[idx3],verts[idx2]));
    }
  return g;
}
