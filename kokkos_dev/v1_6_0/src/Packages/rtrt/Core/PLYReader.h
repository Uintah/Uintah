#ifndef PLYREADER_H
#define PLYREADER_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/TriMesh.h>
#include <Packages/rtrt/Core/Group.h>

namespace rtrt {
void read_ply(char *fname, Material* matl, TriMesh* &tm, Group* &g);
}

#endif
