//////////////////////////////////////////////////////////////////////
//
// Model.h - Represent an entire model.
//
// Copyright David K. McAllister July 1999.
//
//////////////////////////////////////////////////////////////////////

#ifndef _model_h
#define _model_h

#include <Rendering/ZTex/Object.h>
#include <vector>
using namespace std;

namespace SemotusVisum {
namespace Rendering {

struct TexInfo
{
  const char *TexFName;
  Image *Im;
  int TexID; // This is the OpenGL texture object ID.

  inline TexInfo(const char *n, Image *i, int t)
    {
      TexFName = n; Im = i; TexID = t;
    }

  inline TexInfo()
    {
      TexFName = NULL; Im = NULL; TexID = -1;
    }
};

class TextureDB
{
public:
  vector<TexInfo> TexList;

  // Returns -1 if not found.
  inline int FindByName(const char *name)
  {
    for(unsigned tind=0; tind<TexList.size(); tind++)
      {
	if(!strcmp(TexList[tind].TexFName, name))
	  return tind;
      }

    return -1;
  }

  inline void Dump()
  {
    for(unsigned i=0; i<TexList.size(); i++)
      cerr << i << ": " << TexList[i].TexID << " " << TexList[i].TexFName
	   << endl;
  }
};

class Model
{
public:
  static TextureDB TexDB;
  BBox Box;
  //Matrix44 viewMat;

  vector<Object> Objs;
  int ObjID;

  inline Model() {ObjID = -1;}

  inline Model(const Object &O)
  {
    Box = O.Box;
    ObjID = -1;
    Objs.push_back(O);
  }

  void Dump() const;

  // Flatten this model into one object.
  void Flatten();

  void RebuildBBox();

  void GenNormals(double CreaseAngle = -1, bool KeepIfExist = true);

  void RemoveTriangles(Vector& pos, double thresh=0.05);
};


} // namespace Tools
} // namespace Remote

#endif
