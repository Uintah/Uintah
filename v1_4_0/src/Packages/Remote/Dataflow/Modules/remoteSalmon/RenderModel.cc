
//=======================
// RenderModel.cc
// OpenGL draw and Socket send/receive
// functions for the Model class
// Copyright 2000
// Scientific Computing and Imaging,
// University of Utah
//=======================

#include <GL/glu.h>

#include <Core/OS/sock.h>

#include <Packages/Remote/Tools/macros.h>
#include <Packages/Remote/Tools/Model/Model.h>
#include <Packages/Remote/Tools/Util/Assert.h>

#include <Packages/Remote/Dataflow/Modules/remoteSalmon/RenderModel.h>

using namespace SCIRun;
using namespace Remote::Tools;

//----------------------------------------------------------------------
namespace Remote {
  }
}

namespace Remote {
//----------------------------------------------------------------------
struct GLObject : public Object
{
  inline GLObject(const Object &O) : Object(O) {}
  void RenderObject();
};

//----------------------------------------------------------------------
void GLObject::RenderObject()
{
  // Set up the per-object state.
  float duh[4] = {0, 0, 0, 1};

  if(ShininessValid) glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
  if(SColorValid) {
    duh[0]=scolor.x; duh[1]=scolor.y; duh[2]=scolor.z;
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, duh);
  }
  if(EColorValid) {
    duh[0]=ecolor.x; duh[1]=ecolor.y; duh[2]=ecolor.z;
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, duh);
  }
  if(AColorValid) {
    duh[0]=acolor.x; duh[1]=acolor.y; duh[2]=acolor.z;
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, duh);
  }
  
  if(TexInd >= 0) {
      int TexID = Model::TexDB.TexList[TexInd].TexID;

      ASSERT0(glIsTexture(TexID));

      //glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, TexID);
      
				// Assume texcoords are actually 2D
				// and ignore the third component.
      if(texcoords.size() == 1)
	glTexCoord2dv((GLdouble *) &texcoords[0]);
      else if(texcoords.size() > 0)
	{  
	  ASSERT0(texcoords.size() == verts.size());
	  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	  glTexCoordPointer(2, GL_DOUBLE, 3*sizeof(double), &texcoords[0]);
	}
    }
  else {
    //glDisable(GL_TEXTURE_2D);
  }

  if(normals.size() == 1) {
    glNormal3dv((GLdouble *) &normals[0]);
  }
  else if(normals.size() > 0) {
    ASSERT0(normals.size() == verts.size());
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_DOUBLE, 0, &normals[0]);
  }
  
  GL_ASSERT();

  if(dcolors.size() == 1) {
    glColor3dv((GLdouble *) &dcolors[0]);
    if(alpha.size() == 1)
      glColor4f(dcolors[0].x, dcolors[0].y, dcolors[0].z, alpha[0]);
  }
  else if(dcolors.size() > 0) {
    ASSERT0(dcolors.size() == verts.size());
    glEnableClientState(GL_COLOR_ARRAY);
    if(alpha.size() == verts.size()) {
				// Alpha is defined separately. Must
				// fix it.
      float *Colors = new float[dcolors.size()*4];
      for(int i=0,j=0; i<dcolors.size(); i++) {
	Colors[j++] = dcolors[i].x;
	Colors[j++] = dcolors[i].y;
	Colors[j++] = dcolors[i].z;
	Colors[j++] = alpha[i];
      }
      glColorPointer(4, GL_FLOAT, 0, Colors);
      delete [] Colors;
    }
    else {
      glColorPointer(3, GL_DOUBLE, 0, &dcolors[0]);
    }
  }
  else {    
    // cerr << "No colors.\n";
  }
  
  // First define the vertex arrays.
  glVertexPointer(3, GL_DOUBLE, 0, &verts[0]);

  // Use the vertices in order.
  glDrawArrays((GLenum)PrimType, 0, verts.size());
  
  GL_ASSERT();
}


//----------------------------------------------------------------------
// Render the given model in OpenGL.
void RenderModel(Model &M)
{
  glEnableClientState(GL_VERTEX_ARRAY);
  for(int i=0; i<M.Objs.size(); i++) {
    GLObject(M.Objs[i]).RenderObject();
  }
}

//----------------------------------------------------------------------
void LoadTexture(int TexInd)
{
  if(TexInd < 0)
    // This object has no texture.
    return;

  // Must get a texid, load the texture, and store the ID
  // in the texture DB.

  unsigned int tid=-1;
  glGenTextures(1, &tid);
  glBindTexture(GL_TEXTURE_2D, tid);
  
  // Store the tid in the TexDB.
  Model::TexDB.TexList[TexInd].TexID = tid;
  
  Image *TexIm = Model::TexDB.TexList[TexInd].Im;
  cerr << "Texture size is " << TexIm->wid<<"x"<<TexIm->hgt<<" channels="
       << TexIm->chan<<" bytes=" << TexIm->dsize << " texid="<< tid << endl;
  
  // Convert it to 3 or 4 channels if it was 1 or 2.
  if(TexIm->chan <= 2)
    TexIm->Set(-1, -1, TexIm->chan + 2);
  
  // XXX Could this ever happen twice with the same image?
  TexIm->VFlip();
  
  //#if 1
  if (gluBuild2DMipmaps(GL_TEXTURE_2D, TexIm->chan,
    TexIm->wid, TexIm->hgt, (TexIm->chan == 3) ? GL_RGB : GL_RGBA,
    GL_UNSIGNED_BYTE, TexIm->Pix) != 0) {
    cerr << "glubuild2dmipmap error" << endl;
  }
  /*
    #else
    glTexImage2D(GL_TEXTURE_2D, 0, TexIm->chan, TexIm->wid,
    TexIm->hgt, 0, (TexIm->chan == 3) ? GL_RGB : GL_RGBA,
    GL_UNSIGNED_BYTE, TexIm->Pix) != 0);
    #endif
    */
  
  ASSERT0(glIsTexture(tid));
}

//----------------------------------------------------------------------
// Load the specified textures into OpenGL.
void LoadTextures(Model &M)
{
  M.TexDB.Dump();
  for(int i=0; i<M.Objs.size(); i++)
      LoadTexture(M.Objs[i].TexInd);

  M.TexDB.Dump();
}

//----------------------------------------------------------------------
void sendModel(Model& M, Socket* sock) {
  int i, n;
  if (sock->Write((int)M.Objs.size()) == SOCKET_ERROR) {
    cerr << "error writing model #objs" << endl;
    return;
  }

  for (i = 0; i < M.Objs.size(); i++) {

				// write verts
    n = M.Objs[i].verts.size();
    if (sock->Write(n) == SOCKET_ERROR) {
      cerr << "error writing model #verts" << endl;
      return;
    }
    if (sock->Write((char*)&(M.Objs[i].verts[0]),
      n * sizeof(Vector)) == SOCKET_ERROR) {
      cerr << "error writing model vertices" << endl;
      return;
    }
				// write texcoords
    n = M.Objs[i].texcoords.size();
    if (sock->Write(n) == SOCKET_ERROR) {
      cerr << "error writing model #texcoords" << endl;
      return;
    }
    if (sock->Write((char*)&(M.Objs[i].texcoords[0]),
      n * sizeof(Vector)) == SOCKET_ERROR) {
      cerr << "error writing model texcoords" << endl;
      return;
    }
  }
}

//----------------------------------------------------------------------
Model* receiveModel(Socket* sock) {
  int i, j;
  Model* M = new Model();
  int numobjs, numverts, numtexcoords;
  
  if (sock->Read(numobjs) == SOCKET_ERROR) {
    delete M;
    cerr << "error reading model #objs" << endl;
    return NULL;
  }
  for (i = 0; i < numobjs; i++) {
    M->Objs.push_back();
				// read verts
    if (sock->Read(numverts) == SOCKET_ERROR) {
      cerr << "error reading model #verts" << endl;
      delete M;
      return NULL;
    }
    for (j = 0; j < numverts; j++) {
      M->Objs[i].verts.push_back();
    }
    if (sock->Read((char*)&(M->Objs[i].verts[0]),
      numverts * sizeof(Vector)) == SOCKET_ERROR) {
      cerr << "error reading model vertices" << endl;
      delete M;
      return NULL;
    }

				// read texcoords
    if (sock->Read(numtexcoords) == SOCKET_ERROR) {
      cerr << "error reading model #texcoords" << endl;
      delete M;
      return NULL;
    }
    for (j = 0; j < numtexcoords; j++) {
      M->Objs[i].texcoords.push_back();
    }
    if (sock->Read((char*)&(M->Objs[i].texcoords[0]),
      numtexcoords * sizeof(Vector)) == SOCKET_ERROR) {
      cerr << "error reading model texcoords" << endl;
      delete M;
      return NULL;
    }
  }
  M->RebuildBBox();
  //M->GenNormals();
  return M;
}
} // End namespace Remote


