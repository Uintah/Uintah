//////////////////////////////////////////////////////////////////////
// Object.cpp - Process an Object.
//
// Copyright David K. McAllister July 1999.
//
//////////////////////////////////////////////////////////////////////

#include <Malloc/Allocator.h>

#include <Rendering/ZTex/Model.h>
#include <Rendering/ZTex/Mesh.h>

#include <Rendering/ZTex/macros.h>

namespace SemotusVisum {
namespace Rendering {


// Convert the object from all quads to all tris.
// KeepBad keeps zero area triangles.
void Object::MakeTriangles(bool KeepBad)
{
  int i;
  if(PrimType != L_QUADS)
    return;

  PrimType = L_TRIANGLES;

  ASSERTERR(verts.size() % 4 == 0, "Must have a multiple of 4 vertices.");

  cerr << "Converting from " << (verts.size()/4) << " quads.\n";

  vector<Vector> _verts, _normals, _texcoords, _dcolors;

  bool DoNormals=false, DoTexcoords=false, DoDColors=false;

  if(normals.size() == verts.size()) DoNormals = true;
  if(texcoords.size() == verts.size()) DoTexcoords = true;
  if(dcolors.size() == verts.size()) DoDColors = true;

  for(i=0; (unsigned)i < verts.size(); i+= 4)
    {
      bool DoFirst = KeepBad ||	!(verts[i] == verts[i+1] ||
	  verts[i] == verts[i+2] || verts[i+2] == verts[i+1]);
      bool DoSecond = KeepBad || !(verts[i] == verts[i+3] ||
	  verts[i] == verts[i+2] || verts[i+2] == verts[i+3]);
	
      if(DoFirst)
	{
	  _verts.push_back(verts[i]);
	  _verts.push_back(verts[i+1]);
	  _verts.push_back(verts[i+2]);
	}
      if(DoSecond)
	{
	  _verts.push_back(verts[i]);
	  _verts.push_back(verts[i+2]);
	  _verts.push_back(verts[i+3]);
	}

      if(DoNormals)
	{
	  if(DoFirst)
	    {
	      _normals.push_back(normals[i]);
	      _normals.push_back(normals[i+1]);
	      _normals.push_back(normals[i+2]);
	    }
	  if(DoSecond)
	    {
	      _normals.push_back(normals[i]);
	      _normals.push_back(normals[i+2]);
	      _normals.push_back(normals[i+3]);
	    }
	}

      if(DoTexcoords)
	{
	  if(DoFirst)
	    {
	      _texcoords.push_back(texcoords[i]);
	      _texcoords.push_back(texcoords[i+1]);
	      _texcoords.push_back(texcoords[i+2]);
	    }
	  if(DoSecond)
	    {
	      _texcoords.push_back(texcoords[i]);
	      _texcoords.push_back(texcoords[i+2]);
	      _texcoords.push_back(texcoords[i+3]);
	    }
	}

      if(DoDColors)
	{
	  if(DoFirst)
	    {
	      _dcolors.push_back(dcolors[i]);
	      _dcolors.push_back(dcolors[i+1]);
	      _dcolors.push_back(dcolors[i+2]);
	    }
	  if(DoSecond)
	    {
	      _dcolors.push_back(dcolors[i]);
	      _dcolors.push_back(dcolors[i+2]);
	      _dcolors.push_back(dcolors[i+3]);
	    }
	}
    }

  verts = _verts;
  if(DoNormals) normals = _normals;
  if(DoTexcoords) texcoords = _texcoords;
  if(DoDColors) dcolors = _dcolors;

  if(DoNormals) ASSERTERR(normals.size() == verts.size(), "Bad normal count.");
  if(DoTexcoords) ASSERTERR(texcoords.size() == verts.size(), "Bad texcoords count.");
  if(DoDColors) ASSERTERR(dcolors.size() == verts.size(), "Bad dcolors count.");

  cerr << DoNormals << DoTexcoords << DoDColors << "Converted to " << (verts.size()/3) << " triangles.\n";
}

//----------------------------------------------------------------------
void Model::RemoveTriangles(Vector& pos, double thresh) {
  int i;
  for (i = 0; (unsigned)i < Objs.size(); i++) {
    Objs[i].RemoveTriangles(pos, thresh);
  }
}

//----------------------------------------------------------------------
void Object::RemoveTriangles(Vector& pos, double thresh)
{
  int i, n = verts.size();

  if ((n%3) != 0) {
    cerr << "Must have only triangles" << endl;
    return;
  }
  
  bool hasnorms, hastexcoords, hasdcolors, hasalpha;
  hasnorms = (normals.size() == (unsigned)n);
  hastexcoords = (texcoords.size() == (unsigned)n);
  hasdcolors = (dcolors.size() == (unsigned)n);
  hasalpha = (alpha.size() == (unsigned)n);

  Vector normal;
  Vector dir;

  vector<Vector> _verts, _normals, _texcoords, _dcolors;
  vector<double> _alpha;

  for (i = 0; i < n; i += 3) {

    dir = (verts[i] - pos).normal();
    normal = Cross(verts[i+1] - verts[i], verts[i+2] - verts[i+1]).normal();
    
    if (Dot(dir, normal) < thresh) continue;

    _verts.push_back(verts[i]);
    _verts.push_back(verts[i+1]);
    _verts.push_back(verts[i+2]);

    if (hasnorms) {
      _normals.push_back(normals[i+0]);
      _normals.push_back(normals[i+1]);
      _normals.push_back(normals[i+2]);
    }
    if (hastexcoords) {
      _texcoords.push_back(texcoords[i+0]);
      _texcoords.push_back(texcoords[i+1]);
      _texcoords.push_back(texcoords[i+2]);
    }
    if (hasdcolors) {
      _dcolors.push_back(dcolors[i+0]);
      _dcolors.push_back(dcolors[i+1]);
      _dcolors.push_back(dcolors[i+2]);
    }
    if (hasalpha) {
      _alpha.push_back(alpha[i+0]);
      _alpha.push_back(alpha[i+1]);
      _alpha.push_back(alpha[i+2]);
    }

  }

  verts = _verts;
  if (hasnorms) normals = _normals;
  if (hastexcoords) texcoords = _texcoords;
  if (hasdcolors) dcolors = _dcolors;
  if (hasalpha) alpha = _alpha;
  
}

} // namespace Tools
} // namespace Remote

