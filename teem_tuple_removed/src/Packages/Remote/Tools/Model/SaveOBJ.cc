// SaveOBJ.cpp - Save an OBJ model.
// David McAllister, Aug. 1999.

#include <Packages/Remote/Tools/Model/Model.h>

#include <stdio.h>

namespace Remote {
bool Model::SaveOBJ(const char *fname)
{
  int i;
  if(Objs.size() > 1)
    {
      // XXX We should be able to handle multiple groups.
      // XXX What should I do about materials?
      cerr << "Flattening your model and saving it as .OBJ\n";
      Model M(*this);

      M.Flatten();

      return M.SaveOBJ(fname);
    }

  FILE *out = fopen(fname, "w");
  ASSERTERR(out, "Couldn't open file to save WRL file.");

  fprintf(out, "#Wavefront .OBJ format\n# Saved by DaveMc's OBJ code.\n\ng\n");

  Object *Ob = &Objs[0];

  bool DoNormals = Ob->normals.size() == Ob->verts.size();
  bool DoTexcoords = Ob->texcoords.size() == Ob->verts.size();

  for(i=0; i<Ob->verts.size(); i++)
    fprintf(out, "v %f %f %f\n", Ob->verts[i].x, Ob->verts[i].y, Ob->verts[i].z);
  fprintf(out, "# %d vertices.\n\n", Ob->verts.size());

  if(DoTexcoords)
    for(i=0; i<Ob->texcoords.size(); i++)
      fprintf(out, "vt %f %f\n", Ob->texcoords[i].x, Ob->texcoords[i].y);
  fprintf(out, "# %d texcoords.\n\n", Ob->texcoords.size());

  if(DoNormals)
    for(i=0; i<Ob->normals.size(); i++)
      fprintf(out, "vn %f %f %f\n", Ob->normals[i].x, Ob->normals[i].y, Ob->normals[i].z);
  fprintf(out, "# %d normals.\n\n", Ob->normals.size());

  for(i=0; i<Ob->verts.size(); i+=3)
    {
      fprintf(out, "f ");
      for(int j=0; j<3; j++)
	{
	  fprintf(out, "%d", i+j+1);
	  if(DoTexcoords)
	    fprintf(out, "/%d", i+j+1);
	  else if(DoNormals)
	    fprintf(out, "/");
	  if(DoNormals)
	    fprintf(out, "/%d", i+j+1);
	  fprintf(out, " ");
	}
      fprintf(out, "\n");
    }

  fclose(out);

  return false;
}
} // End namespace Remote


