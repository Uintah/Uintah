#ifndef PLYREADER_H
#define PLYREADER_H 1

namespace rtrt {
  class Material;
  class Group;
  class GridTris;
  class TriMesh;
  void read_ply(char *fname, Material* matl, TriMesh* &tm, Group* &g);
  void read_ply(char *fname, GridTris*);
  void read_ply(char *fname, GridTris*, Transform *t);
}

#endif
