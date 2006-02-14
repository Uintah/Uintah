//////////////////////////////////////////////////////////////////////
// Object.cpp - Process an Object.
// Copyright David K. McAllister July 1999.
//////////////////////////////////////////////////////////////////////

#include <Packages/Remote/Tools/Model/Model.h>
#include <Packages/Remote/Tools/Model/Mesh.h>

#include <Packages/Remote/Tools/macros.h>

namespace Remote {
// Generate normals for each vertex.
// The smaller the crease angle, the more smoothing.
void Object::GenNormals(double CreaseAngle)
{
  int i;
  
  ASSERTERR(PrimType == L_TRIANGLES, "Bad PrimType.");
  ASSERTERR(CreaseAngle >= 0 && CreaseAngle <= M_PI, "Bad creaseAngle.");
  creaseAngle = CreaseAngle;

  //cerr << "dcolors.size = " << dcolors.size() << endl;

  // Build a mesh.
  Mesh Me(*this);

  if (creaseAngle == 0)		// hack hack
    Me.FixFacing();

  normals.clear();
  verts.clear();

  // This ensures that duplicate faces don't screw us up.
  Object TmpOb = Me.ExportObject();
  verts = TmpOb.verts;
  if(texcoords.size() != 1)
    texcoords = TmpOb.texcoords;
  if(dcolors.size() != 1)
    dcolors = TmpOb.dcolors;

  //cerr << creaseAngle << " dcolors.size = " << dcolors.size() << endl;

  // First compute the facet normals.
  Face *F = Me.Faces;
  for(i=0; i<verts.size(); i+=3, F = F->next)
    {
      // i is currently the first vertex of the face.
      Vector P0 = verts[i] - verts[i+1];
      Vector P1 = verts[i+2] - verts[i+1];

      Vector *N = new Vector(Cross(P1, P0));
      N->normalize();

      // Store a pointer to this facet normal in F->e0.
      F->e0 = (Edge *)N;
    }

  // For all vertices that match this one, accumulate those with
  // an angle less than CreaseAngle into a smooth normal.
  double CosCrease = cos(CreaseAngle);

  for(i=0, F = Me.Faces; i<verts.size(); i+=3, F = F->next)
    {
      ASSERT1(F);

      Vector &FN = *((Vector *)F->e0);
      for(int j=0; j<3; j++)
	{
	  Vertex *V;
	  if(F->v0->V == verts[i+j])
	    V = F->v0;
	  else if(F->v1->V == verts[i+j])
	    V = F->v1;
	  else
	    {
	      ASSERT1(F->v2->V == verts[i+j]);
	      V = F->v2;
	    }
	  
	  int cnt = 1;
	  Vector Accum(FN);

	  // Loop on all the faces of this vertex.
	  for(int k=0; k<V->Faces.size(); k++)
	    {
	      if(V->Faces[k] != F)
		{
		  // Compute cos of their dihedral angle.
		  // Dot goes from 1 to -1 as ang goes from 0 to PI.
		  Vector &FNT = *((Vector *)(V->Faces[k]->e0));
		  double AngDot = Dot(FN, FNT);
		  
		  // If the angle < CreaseAngle then Ang > CosCrease and we smooth.
		  if(AngDot > CosCrease)
		    {
		      //cerr << creaseAngle << " " << CosCrease << " " << AngDot << endl;
		      // cerr << FN << FNT << endl;
		      // Smooth with this face.
		      Accum += FNT;
		      cnt++;
		    }
		}
	    }
	  
	  // Now store the normal.
	  Accum /= cnt;
	  
	  Accum.normalize();
	  normals.push_back(Accum);
	}
    }

  // Remove the facet normals I made.
  for(F = Me.Faces; F; F = F->next)
    delete (Vector *)F->e0;
}

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

  for(i=0; i < verts.size(); i+= 4)
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

// Returns false on success; true on error.
bool Model::Save(const char *fname)
{
  ASSERTERR(fname, "NULL filename");

  char *extp = strrchr(fname, '.');

  extp++;

  if(strlen(extp) != 3)
	{
	  cerr << "Can't grok filename " << fname << endl;
	  return true;
	}

  extp[0] |= 0x20;
  extp[1] |= 0x20;
  extp[2] |= 0x20;

  if(!strcmp(extp, "wrl"))
	 return SaveVRML(fname);
  else if(!strcmp(extp, "obj"))
	 return SaveOBJ(fname);
  else {
    cerr << "Can't grok filename " << fname << endl;
    return true;
  }
  
  return true;
}

// Returns false on success; true on error.
bool Model::Read(const char *fname)
{
  ASSERTERR(fname, "NULL filename");

  char *extp = strrchr(fname, '.');

  if(!extp || strlen(extp) != 4) {
    cerr << "Can't grok filename " << fname << endl;
    return true;
  }

  extp++;

  extp[0] |= 0x20;
  extp[1] |= 0x20;
  extp[2] |= 0x20;

  if(!strcmp(extp, "wrl"))
    return ReadVRML(fname);
  else if(!strcmp(extp, "obj"))
    return ReadOBJ(fname);
  else {
    cerr << "Can't grok filename " << fname << endl;
    return true;
  }

  return true;
}

// Return a duplicate model with one Object.
void Model::Flatten()
{
  if(Objs.size() < 2)
    return;

  Object *NOb = (Object *) &Objs[0];
  
  for(int i=1; i<Objs.size(); i++)
    {
      const Object *O = &Objs[i];

      if(O->verts.size() < 1)
	continue;
      
      if(O->dcolors.size() > 1)
	ASSERTERR(O->dcolors.size() == O->verts.size(), "Object Bad dcolors.size()");
      
      if(O->normals.size() > 1)
	ASSERTERR(O->normals.size() == O->verts.size(), "Object Bad normals.size()");
      
      if(O->texcoords.size() > 1)
	ASSERTERR(O->texcoords.size() == O->verts.size(), "Object Bad texcoords.size()");
      
      if(NOb->dcolors.size() > 1)
	ASSERTERR(NOb->dcolors.size() == NOb->verts.size(), "List Bad dcolors.size()");
      
      if(NOb->normals.size() > 1)
	ASSERTERR(NOb->normals.size() == NOb->verts.size(), "List Bad normals.size()");
      
      if(NOb->texcoords.size() > 1)
	ASSERTERR(NOb->texcoords.size() == NOb->verts.size(), "List Bad texcoords.size()");
      
      if(NOb->dcolors.size() == 1 && ((O->dcolors.size() == 1 && O->dcolors[0] != NOb->dcolors[0])
				      || O->dcolors.size() > 1))
	{
	  // Expand my colors.
	  Vector col = NOb->dcolors[0];
	  NOb->dcolors.clear();
	  NOb->dcolors.insert(NOb->dcolors.begin(), NOb->verts.size(), col);
	}
      
      if(NOb->dcolors.size() != 1)
	{
	  if(O->dcolors.size() == 1)
	    {
	      // Expand its colors.
	      Vector col = O->dcolors[0];
	      NOb->dcolors.insert(NOb->dcolors.end(), O->verts.size(), col);
	    }
	  else if(O->dcolors.size() == 0)
	    {
	      if(NOb->dcolors.size())
		{
		  // Have to synthesize a bunch of them.
		  Vector col(0,1,0);
		  NOb->dcolors.insert(NOb->dcolors.end(), O->verts.size(), col);
		}
	    }
	  else
	    {
	      // Copy its list to my list.
	      NOb->dcolors.insert(NOb->dcolors.end(), O->dcolors.begin(), O->dcolors.end());
	    }
	}

      if(NOb->normals.size() == 1 && ((O->normals.size() == 1 && O->normals[0] != NOb->normals[0])
				      || O->normals.size() > 1))
	{
	  // Expand my normals.
	  Vector col = NOb->normals[0];
	  NOb->normals.clear();
	  NOb->normals.insert(NOb->normals.begin(), NOb->verts.size(), col);
	}
	  
      if(NOb->normals.size() != 1)
	{
	  if(O->normals.size() == 1)
	    {
	      // Expand its normals.
	      Vector col = O->normals[0];
	      NOb->normals.insert(NOb->normals.end(), O->verts.size(), col);
	    }
	  else if(O->normals.size() == 0)
	    {
	      if(NOb->normals.size())
		{
		  // Have to synthesize a bunch of them.
		  Vector col(0,1,0);
		  NOb->normals.insert(NOb->normals.end(), O->verts.size(), col);
		}
	    }
	  else
	    {
	      // Copy its list to my list.
	      NOb->normals.insert(NOb->normals.end(), O->normals.begin(), O->normals.end());
	    }
	}

      if(NOb->texcoords.size() == 1 && ((O->texcoords.size() == 1
	 && O->texcoords[0] != NOb->texcoords[0]) || O->texcoords.size() > 1))
	{
	  // Expand my texcoords.
	  Vector col = NOb->texcoords[0];
	  NOb->texcoords.clear();
	  NOb->texcoords.insert(NOb->texcoords.begin(), NOb->verts.size(), col);
	}
	  
      if(NOb->texcoords.size() != 1)
	{
	  if(O->texcoords.size() == 1)
	    {
	      // Expand its texcoords.
	      Vector col = O->texcoords[0];
	      NOb->texcoords.insert(NOb->texcoords.end(), O->verts.size(), col);
	    }
	  else if(O->texcoords.size() == 0)
	    {
	      if(NOb->texcoords.size())
		{
		  // Have to synthesize a bunch of them.
		  Vector col(0,1,0);
		  NOb->texcoords.insert(NOb->texcoords.end(), O->verts.size(), col);
		}
	    }
	  else
	    {
	      // Copy its list to my list.
	      NOb->texcoords.insert(NOb->texcoords.end(), O->texcoords.begin(), O->texcoords.end());
	    }
	}

      NOb->verts.insert(NOb->verts.end(), O->verts.begin(), O->verts.end());
    }

  if(Objs.size() > 1)
    Objs.erase(Objs.begin()+1, Objs.end());

  ASSERTERR(Objs[0].verts.size() % 3 == 0, "Must have a multiple of three vertices in Object.");

  if(Objs[0].dcolors.size() > 1)
    ASSERTERR(Objs[0].dcolors.size() == Objs[0].verts.size(), "Bad dcolors.size()");

  if(Objs[0].normals.size() > 1)
    ASSERTERR(Objs[0].normals.size() == Objs[0].verts.size(), "Bad normals.size()");

  if(Objs[0].texcoords.size() > 1)
    ASSERTERR(Objs[0].texcoords.size() == Objs[0].verts.size(), "Bad texcoords.size()");
}

void Object::Dump() const
{
  cerr << "Name: " << Name << " ObjID: " << ObjID << endl;
  cerr << "Vertex count: " << verts.size() << endl;

  cerr << "Specular color: " << scolor << endl
       << "Emissive color: " << ecolor << endl
       << "Ambient color: " << acolor << endl
       << "Shininess: " << shininess << " PrimType: " << PrimType << endl
       << "Object BBox: " << Box << "\n\nVertex     \t\tNormal\t\tTexcoord\t\tDColor\n";

  for(int i=0; i<verts.size(); i++)
    {
      cerr << verts[i] << "\t";
      if(i<normals.size()) cerr << normals[i] << "\t"; else cerr << "xxxxxxxxxxxxxxxxxxx\t";
      if(i<texcoords.size()) cerr << texcoords[i] << "\t"; else cerr << "xxxxxxxxxxxxxxxxxxx\t";
      if(i<dcolors.size()) cerr << dcolors[i] << "\t"; else cerr << "xxxxxxxxxxxxxxxxxxx\t";
      if(i<alpha.size()) cerr << alpha[i];
      cerr << endl;
    }
  cerr << endl << endl;
}

void Model::Dump() const
{
  cerr << "Dumping Model ObjId: " << ObjID << "\nModel BBox: " << Box << endl
       << "NumObjects: " << Objs.size() << endl << endl;
  for(int i=0; i<Objs.size(); i++)
    {
      cerr << "Object index: " << i << endl;
      Objs[i].Dump();
    }
}

void Object::RebuildBBox()
{
  Box.Reset();
  
  for(int i=0; i<verts.size(); i++)
      Box += verts[i];
}

void Model::RebuildBBox()
{
  Box.Reset();

  for(int i=0; i<Objs.size(); i++)
    {
      Objs[i].RebuildBBox();
      Box += Objs[i].Box;
    }
}

void Model::GenNormals(double creaseAngle, bool KeepIfExist)
{
  for(int i=0; i<Objs.size(); i++)
    if(!KeepIfExist || Objs[i].normals.size() != Objs[i].verts.size())
      //Objs[i].GenNormals(creaseAngle < 0 ? Objs[i].creaseAngle :
      //creaseAngle);
      Objs[i].GenNormals(0);
}

//----------------------------------------------------------------------
void Model::RemoveTriangles(Vector& pos, double thresh) {
  int i;
  for (i = 0; i < Objs.size(); i++) {
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
  hasnorms = (normals.size() == n);
  hastexcoords = (texcoords.size() == n);
  hasdcolors = (dcolors.size() == n);
  hasalpha = (alpha.size() == n);

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
} // End namespace Remote


