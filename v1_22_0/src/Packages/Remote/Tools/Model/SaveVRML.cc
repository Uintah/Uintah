//////////////////////////////////////////////////////////////////////
// SaveVRML.cpp - Write a vector<Object> as a VRML 1.0 file.
// Copyright David K. McAllister July 1999.
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>

#include <Packages/Remote/Tools/Model/Model.h>
#include <Packages/Remote/Tools/Model/SaveVRML.h>

namespace Remote {
// Define this static dude.
TextureDB Model::TexDB;

// Assume an eight-space tab.
void WrObject::indent()
{
  int i;
  for(i=0; i<Ind-8; i+=8) fprintf(out, "\t");
  for(; i<Ind; i++) fprintf(out, " ");
}

void WrObject::writeVector(const Vector &V)
{
  fprintf(out, "%f %f %f", V.x, V.y, V.z);
}

void WrObject::writeMaterials()
{
  if(dcolors.size() < 1)
    return;

  indent(); fprintf(out, "Material\n");
  indent(); fprintf(out, "{\n");
  IncIndent();

  {
      indent(); fprintf(out, "diffuseColor [\n");
      IncIndent();
      for(int i=0; i<dcolors.size(); i++)
	{
	  indent(); writeVector(dcolors[i]);
	  if(i==dcolors.size()-1)
	    fprintf(out, "]\n\n");
	  else
	    fprintf(out, ",\n");
	}
      DecIndent();
  }

  if(AColorValid)
    {
      indent(); fprintf(out, "ambientColor [\n");
      IncIndent();
      for(int i=0; i<dcolors.size(); i++)
	{
	  indent(); writeVector(acolor);
	  if(i==dcolors.size()-1)
	    fprintf(out, "]\n\n");
	  else
	    fprintf(out, ",\n");
	}
      DecIndent();
    }

  if(SColorValid)
    {
      indent(); fprintf(out, "specularColor [\n");
      IncIndent();
      for(int i=0; i<dcolors.size(); i++)
	{
	  indent(); writeVector(scolor);
	  if(i==dcolors.size()-1)
	    fprintf(out, "]\n\n");
	  else
	    fprintf(out, ",\n");
	}
      DecIndent();
    }

  if(EColorValid)
    {
      indent(); fprintf(out, "emissiveColor [\n");
      IncIndent();
      for(int i=0; i<dcolors.size(); i++)
	{
	  indent(); writeVector(ecolor);
	  if(i==dcolors.size()-1)
	    fprintf(out, "]\n\n");
	  else
	    fprintf(out, ",\n");
	}
      DecIndent();
    }

  if(ShininessValid)
    {
      indent(); fprintf(out, "shininess [\n");
      IncIndent();
      for(int i=0; i<dcolors.size(); i++)
	{
	  indent();
	  if(i==dcolors.size()-1)
	    fprintf(out, "%f]\n\n", shininess / 128.0);
	  else
	    fprintf(out, "%f,\n", shininess / 128.0);
	}
      DecIndent();
    }

  indent(); fprintf(out, "transparency [\n");
  IncIndent();
  if(alpha.size() == dcolors.size())
    {
      // Alpha per vertex.
      for(int i=0; i<alpha.size(); i++)
	{
	  indent();
	  if(i==alpha.size()-1)
	    fprintf(out, "%f]\n\n", 1-alpha[i]);
	  else
	    fprintf(out, "%f,\n", 1-alpha[i]);
	}
    }
  else if(alpha.size() <= 1)
    {
      // Overall alpha
      float Alpha = 0; // This is lame.
      if(alpha.size() == 1)
	Alpha = 1 - alpha[0];

      for(int i=0; i<dcolors.size(); i++)
	{
	  indent();
	  if(i==dcolors.size()-1)
	    fprintf(out, "%f]\n\n", Alpha);
	  else
	    fprintf(out, "%f,\n", Alpha);
	}
    }
  
  DecIndent();

  DecIndent();
  indent(); fprintf(out, "}\n\n");

  indent(); fprintf(out, "MaterialBinding\n");
  indent(); fprintf(out, "{\n");
  IncIndent();
  indent();
  if(dcolors.size() <= 1)
    fprintf(out, "value OVERALL\n");
  else
    fprintf(out, "value PER_VERTEX\n");
  DecIndent();
  indent(); fprintf(out, "}\n\n");
}

void WrObject::writeNormals()
{
  if(normals.size())
    {
      indent(); fprintf(out, "Normal\n");
      indent(); fprintf(out, "{\n");
      IncIndent();

      indent(); fprintf(out, "vector [\n");
      IncIndent();
      for(int i=0; i<normals.size(); i++)
	{
	  indent(); writeVector(normals[i]);
	  if(i==normals.size()-1)
	    fprintf(out, "]\n\n");
	  else
	    fprintf(out, ",\n");
	}
      DecIndent();

      DecIndent();
      indent(); fprintf(out, "}\n\n");
      
      indent(); fprintf(out, "NormalBinding\n");
      indent(); fprintf(out, "{\n");
      IncIndent();
      indent();
      if(normals.size() <= 1)
	fprintf(out, "value OVERALL\n");
      else
	fprintf(out, "value PER_VERTEX\n");
      DecIndent();
      indent(); fprintf(out, "}\n\n");
    }
}

void WrObject::writeTexCoords()
{
  if(texcoords.size())
    {
      // Even if there is no texture loaded we will put this node in
      // the file to make it easy to specify one later.
      indent(); fprintf(out, "Texture2\n");
      indent(); fprintf(out, "{\n");
      IncIndent();
      indent(); fprintf(out, "filename \"%s\"\n", ((TexInd>=0) ? Model::TexDB.TexList[TexInd].TexFName : ""));
      DecIndent();
      indent(); fprintf(out, "}\n\n");
  
      indent(); fprintf(out, "TextureCoordinate2\n");
      indent(); fprintf(out, "{\n");
      IncIndent();

      indent(); fprintf(out, "point [\n");
      IncIndent();
      for(int i=0; i<texcoords.size(); i++)
	{
	  indent(); fprintf(out, "%f %f", texcoords[i].x, texcoords[i].y);
	  if(i==texcoords.size()-1)
	    fprintf(out, "]\n\n");
	  else
	    fprintf(out, ",\n");
	}
      DecIndent();
      
      DecIndent();
      indent(); fprintf(out, "}\n\n");
    }
}

void WrObject::writeVertices()
{
  if(verts.size())
    {
      indent(); fprintf(out, "Coordinate3\n");
      indent(); fprintf(out, "{\n");
      IncIndent();

      indent(); fprintf(out, "point [\n");
      IncIndent();
      for(int i=0; i<verts.size(); i++)
	{
	  indent(); writeVector(verts[i]);
	  if(i==verts.size()-1)
	    fprintf(out, "]\n\n");
	  else
	    fprintf(out, ",\n");
	}
      DecIndent();

      DecIndent();
      indent(); fprintf(out, "}\n\n");
    }
}

void WrObject::writeIndices()
{
  if(verts.size())
    {
      indent(); fprintf(out, "IndexedFaceSet\n");
      indent(); fprintf(out, "{\n");
      IncIndent();

      indent(); fprintf(out, "coordIndex [\n");
      IncIndent();
      for(int i=0; i<verts.size();)
	{
	  indent();
	  for(int j=0; j<((PrimType==L_TRIANGLES)?3:4); j++)
	    fprintf(out, "%d, ", i++);
	  if(i==verts.size())
	    fprintf(out, "-1]\n\n");
	  else
	    fprintf(out, "-1,\n");
	}
      DecIndent();

      if(dcolors.size() > 1)
	{
	  // Have a material per vertex.
	  indent(); fprintf(out, "materialIndex [\n");
	  IncIndent();
	  for(int i=0; i<verts.size();)
	    {
	      indent();
	      for(int j=0; j<((PrimType==L_TRIANGLES)?3:4); j++)
		fprintf(out, "%d, ", i++);
	      if(i==verts.size())
		fprintf(out, "-1]\n\n");
	      else
		fprintf(out, "-1,\n");
	    }
	  DecIndent();
	}

      if(normals.size() > 1)
	{
	  // Have a normal per vertex.
	  indent(); fprintf(out, "normalIndex [\n");
	  IncIndent();
	  for(int i=0; i<verts.size();)
	    {
	      indent();
	      for(int j=0; j<((PrimType==L_TRIANGLES)?3:4); j++)
		fprintf(out, "%d, ", i++);
	      if(i==verts.size())
		fprintf(out, "-1]\n\n");
	      else
		fprintf(out, "-1,\n");
	    }
	  DecIndent();
	}

      if(texcoords.size() > 1)
	{
	  // Have a texcoord per vertex.
	  indent(); fprintf(out, "textureCoordIndex [\n");
	  IncIndent();
	  for(int i=0; i<verts.size();)
	    {
	      indent();
	      for(int j=0; j<((PrimType==L_TRIANGLES)?3:4); j++)
		fprintf(out, "%d, ", i++);
	      if(i==verts.size())
		fprintf(out, "-1]\n\n");
	      else
		fprintf(out, "-1,\n");
	    }
	  DecIndent();
	}

      DecIndent();
      indent(); fprintf(out, "}\n\n");
    }
}

void WrObject::Write(FILE *_out, int ind)
{
  out = _out;
  Ind = ind;

  indent(); fprintf(out, "Separator\n");
  indent(); fprintf(out, "{\n");
  IncIndent();

  writeMaterials();
  writeNormals();
  writeTexCoords();
  writeVertices();
  writeIndices();
  DecIndent();
  
  indent(); fprintf(out, "}\n");
}

// Returns false on success.
bool Model::SaveVRML(const char *fname)
{
  FILE *out = fopen(fname, "w");
  ASSERTERR(out, "Couldn't open file to save VRML file.");

  fprintf(out, "#VRML V1.0 ascii\n\n");

  fprintf(out, "Separator\n{\n  ShapeHints\n  {\n    vertexOrdering  COUNTERCLOCKWISE\n    creaseAngle %f\n  }\n\n",
	  Objs[0].creaseAngle);

  for(int i=0; i<Objs.size(); i++)
    {
      cerr << "Saving Object: " << i << endl;
      WrObject Wo(Objs[i]); 
      Wo.Write(out, 2);
    }

  fprintf(out, "}\n");

  fclose(out);

  return false;
}
} // End namespace Remote


