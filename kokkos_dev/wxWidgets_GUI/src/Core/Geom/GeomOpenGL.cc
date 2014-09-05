/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/


/*
 *  GeomOpenGL.cc: Rendering for OpenGL windows
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <sci_defs/bits_defs.h>

#include <sci_defs/ogl_defs.h>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/Util/NotFinished.h>
#include <Core/Util/Environment.h>

#include <Core/Geom/DrawInfoOpenGL.h>

#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomArrows.h>
#include <Core/Geom/BBoxCache.h>
#include <Core/Geom/DirectionalLight.h>
#include <Core/Geom/GeomBillboard.h>
#include <Core/Geom/GeomBox.h>
#include <Core/Geom/GeomColorMap.h>
#include <Core/Geom/GeomCone.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomDisc.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomColormapInterface.h>
#include <Core/Geom/GeomGrid.h>
#include <Core/Geom/GeomQMesh.h>
#include <Core/Geom/tGrid.h>
#include <Core/Geom/TimeGrid.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomTimeGroup.h>
#include <Core/Geom/HeadLight.h>
#include <Core/Geom/IndexedGroup.h>
#include <Core/Geom/Light.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/PointLight.h>
#include <Core/Geom/SpotLight.h>
#include <Core/Geom/GeomPolyline.h>
#include <Core/Geom/GeomPoint.h>
#include <Core/Geom/GeomRenderMode.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomEllipsoid.h>
#include <Core/Geom/GeomDL.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomTetra.h>
#include <Core/Geom/GeomTexSlices.h>
#include <Core/Geom/TexSquare.h>
#include <Core/Geom/GeomTexRectangle.h>
#include <Core/Geom/ColorMapTex.h>
#include <Core/Geom/HistogramTex.h>
#include <Core/Geom/GeomStippleOccluded.h>
#include <Core/Geom/GeomTorus.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomQuads.h>
#include <Core/Geom/GeomTube.h>
#include <Core/Geom/GeomTriStrip.h>
#include <Core/Geom/View.h>
#include <Core/Geom/GeomSticky.h>
#include <Core/Datatypes/Color.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Math/Trig.h>
#include <Core/Math/TrigTable.h>
#include <Core/Geometry/Plane.h>

#include <iostream>
#include <algorithm>
using std::cerr;
using std::endl;


#if !defined(__linux) && !defined(_WIN32) && !defined(__digital__) && !defined(_AIX) && !defined(__APPLE__)
#include <GL/gls.h>
#endif

#ifndef _WIN32
#include <X11/X.h>
#include <X11/Xlib.h>
#else
#include <windows.h>
#endif

#include <stdio.h>


namespace SCIRun {

int
GeomObj::pre_draw(DrawInfoOpenGL* di, Material* matl, int lit)
{
  /* All primitives that get drawn must check the return value of this
     function to determine if they get drawn or not */
  if ((!di->pickmode_)||(di->pickmode_&&di->pickchild_))
  {
    if (lit && di->lighting_ && !di->currently_lit_)
    {
      di->currently_lit_=1;
      glEnable(GL_LIGHTING);
      switch(di->get_drawtype())
      {
      case DrawInfoOpenGL::WireFrame:
        gluQuadricNormals(di->qobj_, (GLenum)GLU_SMOOTH);
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
        break;
      case DrawInfoOpenGL::Flat:
        gluQuadricNormals(di->qobj_, (GLenum)GLU_FLAT);
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        break;
      case DrawInfoOpenGL::Gouraud:
        gluQuadricNormals(di->qobj_, (GLenum)GLU_SMOOTH);
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        break;
      }

    }
    if ((!lit || !di->lighting_) && di->currently_lit_)
    {
      di->currently_lit_=0;
      glDisable(GL_LIGHTING);
      gluQuadricNormals(di->qobj_, (GLenum)GLU_NONE);
      switch(di->get_drawtype())
      {
      case DrawInfoOpenGL::WireFrame:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
        break;
      case DrawInfoOpenGL::Flat:
      case DrawInfoOpenGL::Gouraud:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        break;
      }
    }
    di->set_material(matl);
#ifdef SCI_64BITS
    unsigned long o=(unsigned long)this;
    unsigned int o1=(o>>32)&0xffffffff;
    unsigned int o2=o&0xffffffff;
    glPushName(o1);
    glPushName(o2);
#else
    glPushName((GLuint)this);
#endif
    glPushName(0x12345678);
    return 1;
  }
  else return 0;
}


int
GeomObj::post_draw(DrawInfoOpenGL* di)
{
  if (di->pickmode_ && di->pickchild_)
  {
#ifdef SCI_64BITS
    glPopName();
    glPopName();
#else
    glPopName();//pops the face index once the obj is rendered
#endif
    glPopName();
  }
  return 1;  // needed to quiet visual c++
}




// WARNING - doesn''t respond to lighting correctly yet!

void
GeomArrows::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  int n = positions.size();
  di->polycount_ += 6 * n;

  // Draw shafts - they are the same for all draw types.
  double shaft_scale = headlength;

  // if we're not drawing cylinders, draw lines
  if ( drawcylinders == 0 )
  {
    if (di->get_drawtype() == DrawInfoOpenGL::WireFrame)
      shaft_scale=1.0;
    if (shaft_matls.size() == 1)
    {
      if (!pre_draw(di, shaft_matls[0].get_rep(), 0)) return;
      glBegin(GL_LINES);
      for (int i=0;i<n;i++)
      {
        Point from(positions[i]);
        Point to(from+directions[i]*shaft_scale);
        glVertex3d(from.x(), from.y(), from.z());
        glVertex3d(to.x(), to.y(), to.z());
      }
      glEnd();
    }
    else
    {
      if (!pre_draw(di, matl, 0)) return;
      glBegin(GL_LINES);
      for (int i=0;i<n;i++)
      {
        di->set_material(shaft_matls[i+1].get_rep());
        Point from(positions[i]);
        Point to(from+directions[i]*shaft_scale);
        glVertex3d(from.x(), from.y(), from.z());
        glVertex3d(to.x(), to.y(), to.z());
      }
      glEnd();
    }

  }
  else
  {
    // number of subdivisions
    int nu = 4;
    int nv = 1;
        
    // drawing cylinders
    if ( shaft_matls.size() == 1)
    {
      if (!pre_draw(di, shaft_matls[0].get_rep(), 1)) return;
      for ( int i =0; i < n; i++ )
      {
        Point from(positions[i]);
        Point to(from+directions[i]*shaft_scale);
        // create cylinder along axis with endpoints from and to
        Vector axis = to - from;
        Vector z(0,0,1);
        Vector zrotaxis;
        double zrotangle;
        if ( Abs(axis.y())+Abs(axis.x()) < 1.e-5)
        {
          // Only in x-z plane.
          zrotaxis=Vector(0,-1,0);
        }
        else
        {
          zrotaxis=Cross(axis, z);
          zrotaxis.normalize();
        }
        double cangle=Dot(z, axis)/axis.length();
        zrotangle=-Acos(cangle);

        // draw cylinder
        glPushMatrix();
        glTranslated( from.x(), from.y(), from.z() );
        glRotated( RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
        di->polycount_ += 2*(nu-1)*(nv-1);
        gluCylinder(di->qobj_, rad, rad, axis.length(), nu, nv);
        glPopMatrix();
      }
    }
    else
    {
      if (!pre_draw(di, matl, 1)) return;
      for ( int i =0; i < n; i++ )
      {
        di->set_material(shaft_matls[i+1].get_rep());
        Point from(positions[i]);
        Point to(from+directions[i]*shaft_scale);
        // create cylinder along axis with endpoints from and to
        Vector axis = to - from;
        Vector z(0,0,1);
        Vector zrotaxis;
        double zrotangle;
        if ( Abs(axis.y())+Abs(axis.x()) < 1.e-5)
        {
          // Only in x-z plane.
          zrotaxis=Vector(0,-1,0);
        }
        else
        {
          zrotaxis=Cross(axis, z);
          zrotaxis.normalize();
        }
        double cangle=Dot(z, axis)/axis.length();
        zrotangle=-Acos(cangle);

        // draw cylinder
        glPushMatrix();
        glTranslated( from.x(), from.y(), from.z() );
        glRotated( RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
        di->polycount_ += 2*(nu-1)*(nv-1);
        gluCylinder(di->qobj_, rad, rad, axis.length(), nu, nv);
        glPopMatrix();
      }
    }
  }

  if (headwidth == 0 || headlength == 0) { post_draw(di); return; }

  // Draw back and head
  switch(di->get_drawtype())
  {
  case DrawInfoOpenGL::WireFrame:
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    break;
  case DrawInfoOpenGL::Flat:
  case DrawInfoOpenGL::Gouraud:
    break;
  }

  int do_normals=1;
  if (di->get_drawtype() == DrawInfoOpenGL::Flat ||
      di->get_drawtype() == DrawInfoOpenGL::WireFrame)
    do_normals=0;

  if (drawcylinders == 0)
  {
    // Draw the back of the arrow
    if (back_matls.size() == 1)
    {
      if (!pre_draw(di, back_matls[0].get_rep(), 1)) return;
      glBegin(GL_QUADS);
      if (do_normals)
      {
        for (int i=0;i<n;i++)
        {
          glNormal3d(directions[i].x(), directions[i].y(), directions[i].z());
          Point from(positions[i]+directions[i]*headlength);
          Point p1(from+v1[i]);
          glVertex3d(p1.x(), p1.y(), p1.z());
          Point p2(from+v2[i]);
          glVertex3d(p2.x(), p2.y(), p2.z());
          Point p3(from-v1[i]);
          glVertex3d(p3.x(), p3.y(), p3.z());
          Point p4(from-v2[i]);
          glVertex3d(p4.x(), p4.y(), p4.z());
        }
      }
      else
      {
        for (int i=0;i<n;i++)
        {
          Point from(positions[i]+directions[i]*headlength);
          Point p1(from+v1[i]);
          glVertex3d(p1.x(), p1.y(), p1.z());
          Point p2(from+v2[i]);
          glVertex3d(p2.x(), p2.y(), p2.z());
          Point p3(from-v1[i]);
          glVertex3d(p3.x(), p3.y(), p3.z());
          Point p4(from-v2[i]);
          glVertex3d(p4.x(), p4.y(), p4.z());
        }
      }
      glEnd();
    }
    else
    {
      if (!pre_draw(di, matl, 1)) return;
      glBegin(GL_QUADS);
      if (do_normals)
      {
        for (int i=0;i<n;i++)
        {
          di->set_material(back_matls[i+1].get_rep());
          glNormal3d(directions[i].x(), directions[i].y(), directions[i].z());
          Point from(positions[i]+directions[i]*headlength);
          Point p1(from+v1[i]);
          glVertex3d(p1.x(), p1.y(), p1.z());
          Point p2(from+v2[i]);
          glVertex3d(p2.x(), p2.y(), p2.z());
          Point p3(from-v1[i]);
          glVertex3d(p3.x(), p3.y(), p3.z());
          Point p4(from-v2[i]);
          glVertex3d(p4.x(), p4.y(), p4.z());
        }
      }
      else
      {
        for (int i=0;i<n;i++)
        {
          di->set_material(back_matls[i+1].get_rep());
          Point from(positions[i]+directions[i]*headlength);
          Point p1(from+v1[i]);
          glVertex3d(p1.x(), p1.y(), p1.z());
          Point p2(from+v2[i]);
          glVertex3d(p2.x(), p2.y(), p2.z());
          Point p3(from-v1[i]);
          glVertex3d(p3.x(), p3.y(), p3.z());
          Point p4(from-v2[i]);
          glVertex3d(p4.x(), p4.y(), p4.z());
        }
      }
      glEnd();
    }

    // Draw the head of the arrow
    if (head_matls.size() == 1)
    {
      if (!pre_draw(di, head_matls[0].get_rep(), 1)) return;
      if (do_normals)
      {
        double w=headwidth;
        double h=1.0-headlength;
        double w2h2=w*w/h;
        for (int i=0;i<n;i++)
        {
          glBegin(GL_TRIANGLES);
          Vector dn(directions[i]*w2h2);
          Vector n(dn+v1[i]+v2[i]);
          glNormal3d(n.x(), n.y(), n.z());
          Point top(positions[i]+directions[i]);
          Point from;
          if (normalize_headsize == 0 )
          {
            from = top - directions[i] * h;
          }
          else
          {
            Vector dir( directions[i] );
            dir.normalize();
            from = top - dir * h;
          }
          Point p1(from+v1[i]);
          Point p2(from+v2[i]);
          glVertex3d(top.x(), top.y(), top.z());
          glVertex3d(p1.x(), p1.y(), p1.z());
          glVertex3d(p2.x(), p2.y(), p2.z()); // 1st tri
          n=dn-v1[i]+v2[i];
          glNormal3d(n.x(), n.y(), n.z());
          Point p3(from-v1[i]);
          glVertex3d(top.x(), top.y(), top.z());
          glVertex3d(p2.x(), p2.y(), p2.z());
          glVertex3d(p3.x(), p3.y(), p3.z()); // 2nd tri
          n=dn-v1[i]-v2[i];
          glNormal3d(n.x(), n.y(), n.z());
          Point p4(from-v2[i]);
          glVertex3d(top.x(), top.y(), top.z());
          glVertex3d(p3.x(), p3.y(), p3.z());
          glVertex3d(p4.x(), p4.y(), p4.z()); // 3rd tri
          n=dn+v1[i]-v2[i];
          glNormal3d(n.x(), n.y(), n.z());
          glVertex3d(top.x(), top.y(), top.z());
          glVertex3d(p4.x(), p4.y(), p4.z());
          glVertex3d(p1.x(), p1.y(), p1.z()); // 4th tri
          glEnd();
        }
      }
      else
      {
        for (int i=0;i<n;i++)
        {
          glBegin(GL_TRIANGLE_FAN);
          Point from(positions[i]+directions[i]);
          glVertex3d(from.x(), from.y(), from.z());
          if (normalize_headsize == 0 ) from-=directions[i]*(1.0-headlength);
          else
          {
            Vector dir( directions[i] );
            dir.normalize();
            from -= dir*(1.0-headlength);
          }
          Point p1(from+v1[i]);
          glVertex3d(p1.x(), p1.y(), p1.z());
          Point p2(from+v2[i]);
          glVertex3d(p2.x(), p2.y(), p2.z());
          Point p3(from-v1[i]);
          glVertex3d(p3.x(), p3.y(), p3.z());
          Point p4(from-v2[i]);
          glVertex3d(p4.x(), p4.y(), p4.z());
          glVertex3d(p1.x(), p1.y(), p1.z());
          glEnd();
        }
      }
    }
    else
    {
      if (!pre_draw(di, matl, 1)) return;
      if (do_normals)
      {
        double w=headwidth;
        double h=1.0-headlength;
        double w2h2=w*w/h;
        for (int i=0;i<n;i++)
        {
          glBegin(GL_TRIANGLES);
          di->set_material(head_matls[i+1].get_rep());
          Vector dn(directions[i]*w2h2);
          Vector n(dn+v1[i]+v2[i]);
          glNormal3d(n.x(), n.y(), n.z());
          Point top(positions[i]+directions[i]);
          Point from;
          if (normalize_headsize == 0 )
          {
            from = top - directions[i] * h;
          }
          else
          {
            Vector dir( directions[i] );
            dir.normalize();
            from = top - dir * h;
          }
          Point p1(from+v1[i]);
          Point p2(from+v2[i]);
          glVertex3d(top.x(), top.y(), top.z());
          glVertex3d(p1.x(), p1.y(), p1.z());
          glVertex3d(p2.x(), p2.y(), p2.z()); // 1st tri
          n=dn-v1[i]+v2[i];
          glNormal3d(n.x(), n.y(), n.z());
          Point p3(from-v1[i]);
          glVertex3d(top.x(), top.y(), top.z());
          glVertex3d(p2.x(), p2.y(), p2.z());
          glVertex3d(p3.x(), p3.y(), p3.z()); // 2nd tri
          n=dn-v1[i]-v2[i];
          glNormal3d(n.x(), n.y(), n.z());
          Point p4(from-v2[i]);
          glVertex3d(top.x(), top.y(), top.z());
          glVertex3d(p3.x(), p3.y(), p3.z());
          glVertex3d(p4.x(), p4.y(), p4.z()); // 3rd tri
          n=dn+v1[i]-v2[i];
          glNormal3d(n.x(), n.y(), n.z());
          glVertex3d(top.x(), top.y(), top.z());
          glVertex3d(p4.x(), p4.y(), p4.z());
          glVertex3d(p1.x(), p1.y(), p1.z()); // 4th tri
          glEnd();
        }
      }
      else
      {
        for (int i=0;i<n;i++)
        {
          glBegin(GL_TRIANGLE_FAN);
          di->set_material(head_matls[i+1].get_rep());
          Point from(positions[i]+directions[i]);
          glVertex3d(from.x(), from.y(), from.z());
          from-=directions[i]*(1.0-headlength);
          if (normalize_headsize == 0 )
          {
            from -= directions[i] * (1.0 - headlength);
          }
          else
          {
            Vector dir( directions[i] );
            dir.normalize();
            from -= dir*(1.0-headlength);
          }
          Point p1(from+v1[i]);
          glVertex3d(p1.x(), p1.y(), p1.z());
          Point p2(from+v2[i]);
          glVertex3d(p2.x(), p2.y(), p2.z());
          Point p3(from-v1[i]);
          glVertex3d(p3.x(), p3.y(), p3.z());
          Point p4(from-v2[i]);
          glVertex3d(p4.x(), p4.y(), p4.z());
          glVertex3d(p1.x(), p1.y(), p1.z());
          glEnd();
        }
      }
    }

    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      break;
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      break;
    }
    post_draw(di);

  }
  else
  {
    // Draw the head as a capped cone
    int nu = 4;
    int nv = 1;
    int nvdisc = 1;

    if (head_matls.size() == 1)
    {
      if (!pre_draw(di, head_matls[0].get_rep(), 1)) return;
      for (int i=0;i<n;i++)
      {
        glPushMatrix();
        Point top(positions[i]+directions[i]);
        Point bottom = top - directions[i]*(1-headlength);
        glTranslated(bottom.x(), bottom.y(), bottom.z());
        Vector axis = top - bottom;
        Vector z(0,0,1);
        Vector zrotaxis;
        double zrotangle;
        if ( Abs(axis.y())+Abs(axis.x()) < 1.e-5)
        {
          // Only in x-z plane.
          zrotaxis=Vector(0,-1,0);
        }
        else
        {
          zrotaxis=Cross(axis, z);
          zrotaxis.normalize();
        }
        const double cangle = Dot(z, axis)/axis.length();
        zrotangle = -Acos(cangle);
        glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
        double bot_rad = 0.5 * headwidth;
        double top_rad = 0.01 * headwidth;
        di->polycount_ += 2 * (nu - 1) * (nv - 1);
        gluCylinder(di->qobj_, bot_rad, top_rad, headlength, nu, nv);
        if (bot_rad > 1.e-6)
        {
          // Bottom endcap
          di->polycount_ += 2*(nu-1)*(nvdisc-1);
          gluDisk(di->qobj_, 0, bot_rad, nu, nvdisc);
        }
        if (top_rad > 1.e-6)
        {
          // Top endcap
          glTranslated(0, 0, headlength);
          di->polycount_ += 2 * (nu - 1) * (nvdisc - 1);
          gluDisk(di->qobj_, 0, top_rad, nu, nvdisc);
        }
        glPopMatrix();
      }
    }
    else
    {
      if (!pre_draw(di, matl, 1)) return;
      for (int i=0;i<n;i++)
      {
        di->set_material(head_matls[i+1].get_rep());
        glPushMatrix();
        Point top(positions[i]+directions[i]);
        Point bottom = top - directions[i]*(1-headlength);
        glTranslated(bottom.x(), bottom.y(), bottom.z());
        Vector axis = top - bottom;
        Vector z(0,0,1);
        Vector zrotaxis;
        double zrotangle;
        if ( Abs(axis.y())+Abs(axis.x()) < 1.e-5)
        {
          // Only in x-z plane.
          zrotaxis=Vector(0,-1,0);
        }
        else
        {
          zrotaxis=Cross(axis, z);
          zrotaxis.normalize();
        }
        double cangle=Dot(z, axis)/axis.length();
        zrotangle=-Acos(cangle);
        glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
        double bot_rad = 0.5*headwidth;
        double top_rad = 0.01*headwidth;
        di->polycount_+=2*(nu-1)*(nv-1);
        gluCylinder(di->qobj_, bot_rad, top_rad, headlength, nu, nv);
        if (bot_rad > 1.e-6)
        {
          // Bottom endcap
          di->polycount_+=2*(nu-1)*(nvdisc-1);
          gluDisk(di->qobj_, 0, bot_rad, nu, nvdisc);
        }
        if (top_rad > 1.e-6)
        {
          // Top endcap
          glTranslated(0, 0, headlength);
          di->polycount_+=2*(nu-1)*(nvdisc-1);
          gluDisk(di->qobj_, 0, top_rad, nu, nvdisc);
        }
        glPopMatrix();
      }
    }
    post_draw(di);
  }
}


GeomDL::~GeomDL()
{
  list<DrawInfoOpenGL *>::iterator itr = drawinfo_.begin();
  while (itr != drawinfo_.end())
  {
    (*itr)->dl_remove(this);
    ++itr;
  }
}


void
GeomDL::reset_bbox()
{
  GeomContainer::reset_bbox();

  // Update all display lists to bad state, forces redraw.
  list<DrawInfoOpenGL *>::iterator itr = drawinfo_.begin();
  while (itr != drawinfo_.end())
  {
    (*itr)->dl_update(this, 0xffffffff);
    ++itr;
  }
}


void
GeomDL::draw(DrawInfoOpenGL* di, Material *m, double time)
{
  if ( !child_.get_rep() ) return;

  if ( !pre_draw(di, m, 0) ) return;

  if ( !di->display_list_p_ )
  {
    child_->draw(di,m,time);  // do not use display list
  }
  else
  {
    // Compute current state.
    const unsigned int current_state =
      (di->get_drawtype() << 1) | di->lighting_;

    unsigned int state, display_list;
    if (di->dl_lookup(this, state, display_list))
    {
      if (state != current_state)
      {
        di->dl_update(this, current_state);

        const int pre_polycount_ = di->polycount_; // remember poly count
        
        // Fill in the display list.
        // Don't use COMPILE_AND_EXECUTE as it is slower (NVidia linux).
        glNewList(display_list, GL_COMPILE);
        child_->draw(di,m,time);
        glEndList();
        glCallList(display_list);

        // Update poly count;
        polygons_ = di->polycount_ - pre_polycount_;
      }
      else
      {
        // Display the child using the display_list.
        glCallList(display_list);
        di->polycount_ += polygons_;
      }
    }
    else if (di->dl_addnew(this, current_state, display_list))
    {
      const int pre_polycount_ = di->polycount_; // remember poly count
        
      // Fill in the display list.
      // Don't use COMPILE_AND_EXECUTE as it is slower (NVidia linux).
      glNewList(display_list, GL_COMPILE);
      child_->draw(di,m,time);
      glEndList();
      glCallList(display_list);

      // Update poly count;
      polygons_ = di->polycount_ - pre_polycount_;
    }
    else
    {
      child_->draw(di,m,time);  // Do not use display list.
    }
  }

  post_draw(di);
}


void
GeomColorMap::draw(DrawInfoOpenGL* di, Material *m, double time)
{
  if ( !pre_draw(di, m, 0) ) return;

  if (!cmap_.get_rep())
  {
    child_->draw(di, m, time);
  }
  else
  {
    // Set up and draw 1d texture.
    if (di->cmtexture_ == 0)
    {
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glGenTextures(1, &(di->cmtexture_));
    }

    // Send Cmap
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
    glTexImage1D(GL_TEXTURE_1D, 0, 4, 256, 0, GL_RGBA, GL_FLOAT,
                 cmap_->rawRGBA_);

#ifdef GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
#else
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
#endif
    if (cmap_->resolution_ == 256)
    {
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    else
    {
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    // Do Cmap transform to min-max
    glMatrixMode(GL_TEXTURE);
    glPushMatrix();

    const double r = cmap_->resolution_ / 256.0;
    glScaled(r / (cmap_->getMax() - cmap_->getMin()), 1.0, 1.0);
    glTranslated(-cmap_->getMin(), 0.0, 0.0);

    glMatrixMode(GL_MODELVIEW);

    // Draw child
    di->using_cmtexture_++;
    child_->draw(di,m,time);
    di->using_cmtexture_--;

    glMatrixMode(GL_TEXTURE);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
  }

  post_draw(di);
}


void
GeomBillboard::draw(DrawInfoOpenGL* di, Material* m, double time)
{
  double mat[16];

  glGetDoublev(GL_MODELVIEW_MATRIX, mat);
  glPushMatrix();
  glTranslated( at.x(), at.y(), at.z() );


  Vector u( mat[0], mat[4], mat[8] );
  Vector v( 0, 0, 1 );
  Vector w = Cross(u,v);
  w.normalize();
  mat[0] = u.x();
  mat[1] = u.y();
  mat[2] = u.z();
  mat[3] = 0;
  mat[4] = v.x();
  mat[5] = v.y();
  mat[6] = v.z();
  mat[7] = 0;
  mat[8] = w.x();
  mat[9] = w.y();
  mat[10]= w.z();
  mat[11]= 0;
  mat[12] = mat[13] = mat[14] = 0;
  mat[15] = 1;
  glMultMatrixd(mat);

  child_->draw(di,m,time);

  glPopMatrix();
}


void
GeomCappedCylinder::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (height < 1.e-6 || rad < 1.e-6)return;
  if (!pre_draw(di, matl, 1)) return;
  glPushMatrix();
  glTranslated(bottom.x(), bottom.y(), bottom.z());
  glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
  di->polycount_+=2*(nu-1)*(nv-1);
  gluCylinder(di->qobj_, rad, rad, height, nu, nv);
  // Bottom endcap
  di->polycount_+=2*(nu-1)*(nvdisc-1);
  gluDisk(di->qobj_, 0, rad, nu, nvdisc);
  // Top endcap
  glTranslated(0, 0, height);
  di->polycount_+=2*(nu-1)*(nvdisc-1);
  gluDisk(di->qobj_, 0, rad, nu, nvdisc);
  glPopMatrix();
  post_draw(di);
}


void
GeomCone::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (height < 1.e-6 || (bot_rad < 1.e-6 && top_rad < 1.e-6))return;
  if (!pre_draw(di, matl, 1)) return;
  glPushMatrix();
  glTranslated(bottom.x(), bottom.y(), bottom.z());
  glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
  di->polycount_+=2*(nu-1)*(nv-1);
  gluCylinder(di->qobj_, bot_rad, top_rad, height, nu, nv);
  glPopMatrix();
  post_draw(di);
}


void
GeomCappedCone::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (height < 1.e-6 || (bot_rad < 1.e-6 && top_rad < 1.e-6))return;
  if (!pre_draw(di, matl, 1)) return;
  glPushMatrix();
  glTranslated(bottom.x(), bottom.y(), bottom.z());
  glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
  di->polycount_+=2*(nu-1)*(nv-1);
  gluCylinder(di->qobj_, bot_rad, top_rad, height, nu, nv);
  if (bot_rad > 1.e-6)
  {
    // Bottom endcap
    di->polycount_+=2*(nu-1)*(nvdisc1-1);
    gluDisk(di->qobj_, 0, bot_rad, nu, nvdisc1);
  }
  if (top_rad > 1.e-6)
  {
    // Top endcap
    glTranslated(0, 0, height);
    di->polycount_+=2*(nu-1)*(nvdisc2-1);
    gluDisk(di->qobj_, 0, top_rad, nu, nvdisc2);
  }
  glPopMatrix();
  post_draw(di);
}


void
GeomCones::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;

  di->polycount_ += points_.size() * nu_ / 2;

  const bool texturing =
    di->using_cmtexture_ && indices_.size() == points_.size() / 2;
  if (texturing)
  {
    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }

  const bool coloring = colors_.size() == points_.size() * 2;
  const bool use_local_radii = radii_.size() == points_.size()/2;

  const float nz0 = 1.0/6.0;
  const float nzm = 1.0/sqrt(1.0 + 1.0 + nz0*nz0);
  const float nz = nz0 * nzm;
  float tabx[41];
  float taby[41];
  for (int j=0; j<nu_; j++)
  {
    tabx[j] = sin(2.0 * M_PI * j / nu_);
    taby[j] = cos(2.0 * M_PI * j / nu_);
  }
  tabx[nu_] = tabx[0];
  taby[nu_] = taby[0];

  for (unsigned int i=0; i < points_.size(); i+=2)
  {
    Vector v0(points_[i+1] - points_[i+0]);
    Vector v1, v2;
    v0.find_orthogonal(v1, v2);
    if (use_local_radii)
    {
      v1 *= radii_[i/2];
      v2 *= radii_[i/2];
    }
    else
    {
      v1 *= radius_;
      v2 *= radius_;
    }

    float matrix[16];
    matrix[0] = v1.x();
    matrix[1] = v1.y();
    matrix[2] = v1.z();
    matrix[3] = 0.0;
    matrix[4] = v2.x();
    matrix[5] = v2.y();
    matrix[6] = v2.z();
    matrix[7] = 0.0;
    matrix[8] = v0.x();
    matrix[9] = v0.y();
    matrix[10] = v0.z();
    matrix[11] = 0.0;
    matrix[12] = points_[i].x();
    matrix[13] = points_[i].y();
    matrix[14] = points_[i].z();
    matrix[15] = 1.0;

    glPushMatrix();
    glMultMatrixf(matrix);

    if (coloring) { glColor3ubv(&(colors_[i*2])); }
    if (texturing) { glTexCoord1f(indices_[i/2]); }

    glBegin(GL_QUAD_STRIP);
    for (int k = 0; k <= nu_; k++)
    {
      glNormal3f(tabx[k]*nzm, taby[k]*nzm, nz);
      glVertex3f(tabx[k], taby[k], 0.0);
      glVertex3f(0.0, 0.0, 1.0);
    }
    glEnd();

    glPopMatrix();
  }

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomContainer::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  if (child_.get_rep())
  {
    child_->draw(di, matl, time);
  }
}


void
ColorMapTex::draw(DrawInfoOpenGL *di, Material *matl, double time)
{
  if (child_.get_rep())
  {
    const bool lit = di->lighting_;
    di->lighting_ = false;
    child_->draw(di, matl, time);
    di->lighting_ = lit;
  }
}


void
GeomCylinder::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (height < 1.e-6 || rad < 1.e-6)return;
  if (!pre_draw(di, matl, 1)) return;
  glPushMatrix();
  glTranslated(bottom.x(), bottom.y(), bottom.z());
  glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
  di->polycount_+=2*(nu-1)*(nv-1);
  gluCylinder(di->qobj_, rad, rad, height, nu, nv);
  glPopMatrix();
  post_draw(di);
}


void
GeomCylinders::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;

  di->polycount_+=points_.size() * nu_ * 2;

  const bool texturing =
    di->using_cmtexture_ && indices_.size() == points_.size();
  if (texturing)
  {
    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }

  const bool coloring = colors_.size() == points_.size() * 4;

  float tabx[41];
  float taby[41];
  for (int j=0; j<nu_; j++)
  {
    tabx[j] = sin(2.0 * M_PI * j / nu_);
    taby[j] = cos(2.0 * M_PI * j / nu_);
  }
  tabx[nu_] = tabx[0];
  taby[nu_] = taby[0];

  for (unsigned int i=0; i < points_.size(); i+=2)
  {
    Vector v0(points_[i+1] - points_[i+0]);

    Vector v1, v2;
    v0.find_orthogonal(v1, v2);
    if (v0.length2() < 1e-5) //numeric_limits<float>::epsilon())
      continue;
    v1 *= radius_;
    v2 *= radius_;

    float matrix[16];
    matrix[0] = v1.x();
    matrix[1] = v1.y();
    matrix[2] = v1.z();
    matrix[3] = 0.0;
    matrix[4] = v2.x();
    matrix[5] = v2.y();
    matrix[6] = v2.z();
    matrix[7] = 0.0;
    matrix[8] = v0.x();
    matrix[9] = v0.y();
    matrix[10] = v0.z();
    matrix[11] = 0.0;
    matrix[12] = points_[i].x();
    matrix[13] = points_[i].y();
    matrix[14] = points_[i].z();
    matrix[15] = 1.0;

    glPushMatrix();
    glMultMatrixf(matrix);

    glBegin(GL_QUAD_STRIP);
    for (int k=0; k<nu_+1; k++)
    {
      glNormal3f(tabx[k], taby[k], 0.0);

      if (coloring) { glColor3ubv(&(colors_[i*4])); }
      if (texturing) { glTexCoord1f(indices_[i]); }
      glVertex3f(tabx[k], taby[k], 0.0);

      if (coloring) { glColor3ubv(&(colors_[(i+1)*4])); }
      if (texturing) { glTexCoord1f(indices_[i+1]); }
      glVertex3f(tabx[k], taby[k], 1.0);
    }
    glEnd();

    glPopMatrix();
  }

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomCappedCylinders::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;

  di->polycount_+=points_.size() * nu_ * 2;

  const bool texturing =
    di->using_cmtexture_ && indices_.size() == points_.size();
  if (texturing)
  {
    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }

  const bool coloring = colors_.size() == points_.size() * 4;
  const bool use_local_radii = radii_.size() == points_.size()/2;

  float tabx[41];
  float taby[41];
  for (int j=0; j<nu_; j++)
  {
    tabx[j] = sin(2.0 * M_PI * j / nu_);
    taby[j] = cos(2.0 * M_PI * j / nu_);
  }
  tabx[nu_] = tabx[0];
  taby[nu_] = taby[0];

  for (unsigned int i=0; i < points_.size(); i+=2)
  {
    int k;
    Vector v0(points_[i+1] - points_[i+0]);
    Vector v1, v2;
    v0.find_orthogonal(v1, v2);
    if (use_local_radii)
    {
      v1 *= radii_[i/2];
      v2 *= radii_[i/2];
    }
    else
    {
      v1 *= radius_;
      v2 *= radius_;
    }

    float matrix[16];
    matrix[0] = v1.x();
    matrix[1] = v1.y();
    matrix[2] = v1.z();
    matrix[3] = 0.0;
    matrix[4] = v2.x();
    matrix[5] = v2.y();
    matrix[6] = v2.z();
    matrix[7] = 0.0;
    matrix[8] = v0.x();
    matrix[9] = v0.y();
    matrix[10] = v0.z();
    matrix[11] = 0.0;
    matrix[12] = points_[i].x();
    matrix[13] = points_[i].y();
    matrix[14] = points_[i].z();
    matrix[15] = 1.0;

    glPushMatrix();
    glMultMatrixf(matrix);

    glBegin(GL_QUAD_STRIP);
    for (k=0; k<nu_+1; k++)
    {
      glNormal3f(tabx[k], taby[k], 0.0);

      if (coloring) { glColor3ubv(&(colors_[i*4])); }
      if (texturing) { glTexCoord1f(indices_[i]); }
      glVertex3f(tabx[k], taby[k], 0.0);

      if (coloring) { glColor3ubv(&(colors_[(i+1)*4])); }
      if (texturing) { glTexCoord1f(indices_[i+1]); }
      glVertex3f(tabx[k], taby[k], 1.0);
    }
    glEnd();

    // Bottom cap
    if (coloring) { glColor3ubv(&(colors_[i*4])); }
    if (texturing) { glTexCoord1f(indices_[i]); }
    glNormal3f(0.0, 0.0, -1.0);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0.0, 0.0, 0.0);
    for (k = 0; k < nu_+1; k++)
    {
      glVertex3f(tabx[k], taby[k], 0.0);
    }
    glEnd();

    // Top cap
    if (coloring) { glColor3ubv(&(colors_[(i+1)*4])); }
    if (texturing) { glTexCoord1f(indices_[i+1]); }
    glNormal3f(0.0, 0.0, 1.0);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0.0, 0.0, 1.0);
    for (k = nu_; k >= 0; k--)
    {
      glVertex3f(tabx[k], taby[k], 1.0);
    }
    glEnd();

    glPopMatrix();
  }

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomDisc::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (rad < 1.e-6)return;
  if (!pre_draw(di, matl, 1)) return;
  glPushMatrix();
  glTranslated(cen.x(), cen.y(), cen.z());
  glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
  di->polycount_+=2*(nu-1)*(nv-1);
  gluDisk(di->qobj_, 0, rad, nu, nv);
  glPopMatrix();
  post_draw(di);
}


// this is for alexandras volume render thing
// this should be changed to use texture objects

// deletes texture map - only works on single ViewWindow!

TexGeomGrid::~TexGeomGrid()
{
  if (tmapdata)
  {
    if (tmap_dlist != -1)
    {
      glDeleteLists(tmap_dlist,1);
    }
    delete tmapdata;
    tmapdata = 0;
  }
}


void
TexGeomGrid::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 0)) return;
  di->polycount_+=2;

  if (!convolve)
  {
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
      //      break;
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      {
        if (tmap_dlist == -1)
        {
          tmap_dlist = glGenLists(1);

          glNewList(tmap_dlist,GL_COMPILE_AND_EXECUTE);
          glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,
                    GL_MODULATE);

          glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,
                          GL_NEAREST);
          glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,
                          GL_NEAREST);

          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

          if (num_chan == 3)
          {
            glTexImage2D(GL_TEXTURE_2D,0,3,tmap_size,10,
                         0,GL_RGB,GL_UNSIGNED_INT,tmapdata);
          }
          else
          {
            glTexImage2D(GL_TEXTURE_2D,0,GL_LUMINANCE,tmap_size,tmap_size,
                         0,GL_LUMINANCE,GL_UNSIGNED_SHORT,tmapdata);
          }
          glEndList();
        }
        else
        {
          glCallList(tmap_dlist);
        }
        glColor4f(1.0,1.0,1.0,1.0);
        glEnable(GL_TEXTURE_2D);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0,0.0);
        glVertex3d(corner.x(),corner.y(),corner.z());

        glTexCoord2f(dimU/(1.0*tmap_size),0.0);
        glVertex3d(corner.x()+u.x(),corner.y()+u.y(),corner.z()+u.z());

        glTexCoord2f(dimU/(1.0*tmap_size),
                     dimV/(1.0*tmap_size));
        glVertex3d(corner.x()+v.x()+u.x(),
                   corner.y()+v.y()+u.y(),
                   corner.z()+v.z()+u.z());

        glTexCoord2f(0.0,dimV/(1.0*tmap_size));
        glVertex3d(corner.x()+v.x(),corner.y()+v.y(),corner.z()+v.z());

        glEnd();

        glDisable(GL_TEXTURE_2D);
        break;
      }
    }
  }
  else
  { // doing convolution
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
      break;
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      {
        if (tmap_dlist == -1 || kernal_change)
        {
          if (tmap_dlist == -1)
          {
            tmap_dlist = glGenLists(1);
          }
          glNewList(tmap_dlist,GL_COMPILE_AND_EXECUTE);
          glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,
                    GL_MODULATE);
          glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,
                          GL_NEAREST);
          glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,
                          GL_NEAREST);
          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

#if !defined(__linux)&&!defined(_WIN32) && !defined(__digital__) && !defined(_AIX) && !defined(__APPLE__)
          glConvolutionFilter2DEXT(GL_CONVOLUTION_2D_EXT,
                                   GL_INTENSITY_EXT,
                                   conv_dim,conv_dim,
                                   GL_FLOAT,GL_RED,conv_data);
        
          glTexImage2D(GL_TEXTURE_2D,0,GL_INTENSITY_EXT,
                       tmap_size,tmap_size,
                       0,GL_RED,GL_UNSIGNED_BYTE,tmapdata);
#endif
          glEndList();
        }
        else
        {
          glCallList(tmap_dlist);
        }
        glColor4f(1.0,1.0,1.0,1.0);
        glEnable(GL_TEXTURE_2D);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0,0.0);
        glVertex3d(corner.x(),corner.y(),corner.z());

        glTexCoord2f(dimU/(1.0*tmap_size),0.0);
        glVertex3d(corner.x()+u.x(),corner.y()+u.y(),corner.z()+u.z());

        glTexCoord2f(dimU/(1.0*tmap_size),
                     dimV/(1.0*tmap_size));
        glVertex3d(corner.x()+v.x()+u.x(),
                   corner.y()+v.y()+u.y(),
                   corner.z()+v.z()+u.z());

        glTexCoord2f(0.0,dimV/(1.0*tmap_size));
        glVertex3d(corner.x()+v.x(),corner.y()+v.y(),corner.z()+v.z());

        glEnd();

        glDisable(GL_TEXTURE_2D);
        break;
      }
    }
  }
  post_draw(di);
}


void
TimeGrid::draw(DrawInfoOpenGL* di, Material* matl, double t)
{
  if (!pre_draw(di, matl, 0)) return;
  di->polycount_ += 2;

  // First find which ones need to be drawn.

  int i;
  for (i=0;i<time.size() && (time[i] <= t);i++)
    ;

  int start,end;
  double dt;
  int last_frame=0;

  last_frame = (t >= time[time.size()-1]); // 1 if last time step.

  if (i) { i--; } // If it was zero, just keep it.

  start = i;
  end = i+1;

  if (last_frame)
  {
    start = time.size()-1; // just go to end.
    end = start;
  }

  dt = (t-time[start])/(time[1]-time[0]);
  cerr << time[start] << " " << t << " " << start << " ";
  cerr << dt << " Trying to do texture draw...\n";

  if (dt < 0.0) dt = 0.0;
  if (dt > 1.0) dt = 1.0;

  switch(di->get_drawtype())
  {
  case DrawInfoOpenGL::WireFrame:
    break;
  case DrawInfoOpenGL::Flat:
  case DrawInfoOpenGL::Gouraud:
    {
      // First blend two images together.

      float *startM=tmap[start],*endM=tmap[end];
      double bval = dt;
      double cdenom = 1.0/(map->getMax()-map->getMin()); // index

      if (last_frame)
      {
        for (int j=0;j<dimV;j++)
        {
          int bindex = j*tmap_size*3; // use RGB?
          int sindex = j*tmap_size;
        
          for (int i=0;i<dimU;i++)
          {
            float nval = startM[sindex + i];

            double rmapval = (nval-map->getMin())*cdenom;
            MaterialHandle hand = map->lookup2(rmapval);

            bmap[bindex + i*3 + 0] = hand->diffuse.r();
            bmap[bindex + i*3 + 1] = hand->diffuse.g();
            bmap[bindex + i*3 + 2] = hand->diffuse.b();
          }
        }
      }
      else
      {
        for (int j=0;j<dimV;j++)
        {
          int bindex = j*tmap_size*3; // use RGB?
          int sindex = j*tmap_size;
        
          for (int i=0;i<dimU;i++)
          {
            float nval = startM[sindex + i] +
              bval*(endM[sindex+i]-startM[sindex + i]);
            // now look this up in the color map.
        
            double rmapval = (nval-map->getMin())*cdenom;
            MaterialHandle hand = map->lookup2(rmapval);

            bmap[bindex + i*3 + 0] = hand->diffuse.r();
            bmap[bindex + i*3 + 1] = hand->diffuse.g();
            bmap[bindex + i*3 + 2] = hand->diffuse.b();
          }
        }
      }
      // now bmap contains the blended texture.
      glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,
                GL_MODULATE);
      glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,
                      GL_LINEAR);
      glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,
                      GL_LINEAR);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTexImage2D(GL_TEXTURE_2D,0,3,tmap_size,tmap_size,
                   0,GL_RGB,GL_FLOAT,bmap);

      glColor4f(1.0,1.0,1.0,1.0);

      glEnable(GL_TEXTURE_2D);

      glBegin(GL_QUADS);
      glTexCoord2f(0.0,0.0);
      glVertex3d(corner.x(),corner.y(),corner.z());

      glTexCoord2f(dimU/(1.0*tmap_size),0.0);
      glVertex3d(corner.x()+u.x(),corner.y()+u.y(),corner.z()+u.z());

      glTexCoord2f(dimU/(1.0*tmap_size),
                   dimV/(1.0*tmap_size));
      glVertex3d(corner.x()+v.x()+u.x(),
                 corner.y()+v.y()+u.y(),
                 corner.z()+v.z()+u.z());

      glTexCoord2f(0.0,dimV/(1.0*tmap_size));
      glVertex3d(corner.x()+v.x(),corner.y()+v.y(),corner.z()+v.z());

      glEnd();

      glDisable(GL_TEXTURE_2D);
      break;
    }
  }
  post_draw(di);
}


// WARNING doesn't respond to lighting correctly yet!
#ifdef BROKEN_BUT_FAST
void
GeomGrid::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  di->polycount_+=2*(nu-1)*(nv-1);
  Vector uu(u/(nu-1));
  Vector vv(v/(nv-1));
  glPushMatrix();
  glTranslated(-corner.x(), -corner.y(), -corner.z());
  double mat[16];
  mat[0]=uu.x(); mat[1]=uu.y(); mat[2]=uu.z(); mat[3]=0;
  mat[4]=vv.x(); mat[5]=vv.y(); mat[6]=vv.z(); mat[7]=0;
  mat[8]=w.x(); mat[9]=w.y(); mat[10]=w.z(); mat[11]=0;
  mat[12]=0; mat[13]=0; mat[14]=0; mat[15]=1.0;
  for (int i=0;i<16;i++)
  {
    cout << "mat[" << i << "]=" << mat[i] << endl;
  }
  glMultMatrixd(mat);
  switch(di->get_drawtype())
  {
  case DrawInfoOpenGL::WireFrame:
    {
      for (int i=0;i<nv;i++)
      {
        float* p=&data[vstride*i];
        glBegin(GL_LINE_STRIP);
        int j;
        switch(format)
        {
        case Regular:
          for (j=0;j<nv;j++)
          {
            glVertex3fv(p);
            p+=3;
          }
          break;
        case WithMaterials:
          for (j=0;j<nv;j++)
          {
            glColor4fv(p);
            glVertex3fv(p+4);
            p+=7;
          }
          break;
        case WithNormals:
          for (j=0;j<nv;j++)
          {
            glNormal3fv(p);
            glVertex3fv(p+3);
            p+=6;
          }
          break;
        case WithNormAndMatl:
          for (j=0;j<nv;j++)
          {
            glColor4fv(p);
            glNormal3fv(p+4);
            glVertex3fv(p+7);
            p+=10;
          }
          break;
        }
        glEnd();
      }
      for (i=0;i<nu;i++)
      {
        float* p=&data[stride*i];
        glBegin(GL_LINE_STRIP);
        int j;
        switch(format)
        {
        case Regular:
          for (j=0;j<nv;j++)
          {
            glVertex3fv(p);
            p+=vstride;
          }
          break;
        case WithMaterials:
          for (j=0;j<nv;j++)
          {
            glColor4fv(p);
            glVertex3fv(p+4);
            p+=vstride;
          }
          break;
        case WithNormals:
          for (j=0;j<nv;j++)
          {
            glNormal3fv(p);
            glVertex3fv(p+3);
            p+=vstride;
          }
          break;
        case WithNormAndMatl:
          for (j=0;j<nv;j++)
          {
            glColor4fv(p);
            glNormal3fv(p+4);
            glVertex3fv(p+7);
            p+=vstride;
          }
          break;
        }
        glEnd();
      }
    }
    break;
  case DrawInfoOpenGL::Flat:
  case DrawInfoOpenGL::Gouraud:
    {
      for (int i=1;i<nv;i++)
      {
        float* p1=&data[vstride*(i-1)];
        float* p2=&data[vstride*i];
        glBegin(GL_TRIANGLE_STRIP);
        int j;
        switch(format)
        {
        case Regular:
          for (j=0;j<nv;j++)
          {
            glVertex3fv(p1);
            glVertex3fv(p2);
            p1+=3;
            p2+=3;
          }
          break;
        case WithMaterials:
          for (j=0;j<nv;j++)
          {
            glColor4fv(p1);
            glVertex3fv(p1+4);
            glColor4fv(p2);
            glVertex3fv(p2+4);
            p1+=7;
            p2+=7;
          }
          break;
        case WithNormals:
          for (j=0;j<nv;j++)
          {
            glNormal3fv(p1);
            glVertex3fv(p1+3);
            glNormal3fv(p2);
            glVertex3fv(p2+3);
            p1+=6;
            p2+=6;
          }
          break;
        case WithNormAndMatl:
          for (j=0;j<nv;j++)
          {
            glColor4fv(p1);
            glNormal3fv(p1+4);
            glVertex3fv(p1+7);
            glColor4fv(p2);
            glNormal3fv(p2+4);
            glVertex3fv(p2+7);
            p1+=10;
            p2+=10;
          }
          break;
        }
        glEnd();
      }
    }
  }
  glPopMatrix();
  post_draw(di);
}
#else
void
GeomGrid::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  int nu=verts.dim1();
  int nv=verts.dim2();

  if (image)
  {
    di->polycount_+=2*nu*nv;
    Vector uu(u/nu);
    Vector vv(v/nv);
    if (!pre_draw(di,matl,0)) return;
    Point rstart(corner);
    glBegin(GL_QUADS);
    for (int i=0; i<nu; i++)
    {
      Point p1(rstart);
      Point p2(rstart+uu);
      Point p3(rstart+uu+vv);
      Point p4(rstart+vv);
      for (int j=0; j<nv; j++)
      {
        if (have_matls)
        {
          di->set_material(matls(i,j).get_rep());
        }
        glVertex3d(p1.x(), p1.y(), p1.z());
        glVertex3d(p2.x(), p2.y(), p2.z());
        glVertex3d(p3.x(), p3.y(), p3.z());
        glVertex3d(p4.x(), p4.y(), p4.z());
        p1+=vv;
        p2+=vv;
        p3+=vv;
        p4+=vv;
      }
      rstart+=uu;
    }
    glEnd();
    return;
  }

  if (!pre_draw(di, matl, 1)) return;
  di->polycount_+=2*(nu-1)*(nv-1);
  Vector uu(u/(nu-1));
  Vector vv(v/(nv-1));
  switch(di->get_drawtype())
  {
  case DrawInfoOpenGL::WireFrame:
    {
      Point rstart(corner);
      for (int i=0;i<nu;i++)
      {
        Point p1(rstart);
        glBegin(GL_LINE_STRIP);
        for (int j=0;j<nv;j++)
        {
          Point pp1(p1+w*verts(i, j));
          if (have_matls)
            di->set_material(matls(i, j).get_rep());
          if (have_normals)
          {
            Vector normal(normals(i, j));
            glNormal3d(normal.x(), normal.y(), normal.z());
          }
          glVertex3d(pp1.x(), pp1.y(), pp1.z());

          p1+=vv;
        }
        glEnd();
        rstart+=uu;
      }
      rstart=corner;
      for (int j=0;j<nv;j++)
      {
        Point p1(rstart);
        glBegin(GL_LINE_STRIP);
        for (int i=0;i<nu;i++)
        {
          Point pp1(p1+w*verts(i, j));
          if (have_matls)
            di->set_material(matls(i, j).get_rep());
          if (have_normals)
          {
            Vector normal(normals(i, j));
            glNormal3d(normal.x(), normal.y(), normal.z());
          }
          glVertex3d(pp1.x(), pp1.y(), pp1.z());

          p1+=uu;
        }
        glEnd();
        rstart+=vv;
      }
    }
    break;
  case DrawInfoOpenGL::Flat:
  case DrawInfoOpenGL::Gouraud:
    {
#if 0
      if (!have_normals)
        glNormal3d(w.x(), w.y(), w.z());
      Point rstart(corner);
      for (int i=0;i<nu-1;i++)
      {
        Point p1(rstart);
        Point p2(rstart+uu);
        rstart=p2;
        glBegin(GL_TRIANGLE_STRIP);
        for (int j=0;j<nv;j++)
        {
          Point pp1(p1+w*verts(i, j));
          Point pp2(p2+w*verts(i+1, j));
          if (have_matls)
            di->set_material(matls(i, j).get_rep());
          if (have_normals)
          {
            Vector normal(normals(i, j));
            glNormal3d(normal.x(), normal.y(), normal.z());
          }
          glVertex3d(pp1.x(), pp1.y(), pp1.z());

          if (have_matls)
            di->set_material(matls(i+1, j).get_rep());
          if (have_normals)
          {
            Vector normal(normals(i+1, j));
            glNormal3d(normal.x(), normal.y(), normal.z());
          }
          glVertex3d(pp2.x(), pp2.y(), pp2.z());
          p1+=vv;
          p2+=vv;
        }
        glEnd();
      }
#endif
      if (have_matls)
        di->set_material(matls(0,0).get_rep());
      Point rstart(corner);
      if (have_normals && have_matls)
      {
        for (int i=0;i<nu-1;i++)
        {
          Point p1(rstart);
          Point p2(rstart+uu);
          rstart=p2;
          glBegin(GL_QUAD_STRIP);
          for (int j=0;j<nv;j++)
          {
            Point pp1(p1+w*verts(i, j));
            Point pp2(p2+w*verts(i+1, j));
            float c[4];
            matls(i,j)->diffuse.get_color(c);
            glColor3fv(c);
            Vector& normal = normals(i, j);
            glNormal3d(normal.x(), normal.y(), normal.z());
            glVertex3d(pp1.x(), pp1.y(), pp1.z());

            matls(i+1, j)->diffuse.get_color(c);
            glColor3fv(c);
            Vector& normal2 = normals(i+1, j);
            glNormal3d(normal2.x(), normal2.y(), normal2.z());
            glVertex3d(pp2.x(), pp2.y(), pp2.z());
            p1+=vv;
            p2+=vv;
          }
          glEnd();
        }
      }
      else if (have_matls)
      {
        glNormal3d(w.x(), w.y(), w.z());
        for (int i=0;i<nu-1;i++)
        {
          Point p1(rstart);
          Point p2(rstart+uu);
          rstart=p2;
          glBegin(GL_QUAD_STRIP);
          for (int j=0;j<nv;j++)
          {
            Point pp1(p1+w*verts(i, j));
            Point pp2(p2+w*verts(i+1, j));
            float c[4];
            matls(i,j)->diffuse.get_color(c);
            glColor3fv(c);
            glVertex3d(pp1.x(), pp1.y(), pp1.z());

            matls(i+1, j)->diffuse.get_color(c);
            glColor3fv(c);
            glVertex3d(pp2.x(), pp2.y(), pp2.z());
            p1+=vv;
            p2+=vv;
          }
          glEnd();
        }
      }
      else if (have_normals)
      {
        for (int i=0;i<nu-1;i++)
        {
          Point p1(rstart);
          Point p2(rstart+uu);
          rstart=p2;
          glBegin(GL_QUAD_STRIP);
          for (int j=0;j<nv;j++)
          {
            Point pp1(p1+w*verts(i, j));
            Point pp2(p2+w*verts(i+1, j));
            Vector& normal = normals(i, j);
            glNormal3d(normal.x(), normal.y(), normal.z());
            glVertex3d(pp1.x(), pp1.y(), pp1.z());

            Vector& normal2 = normals(i+1, j);
            glNormal3d(normal2.x(), normal2.y(), normal2.z());
            glVertex3d(pp2.x(), pp2.y(), pp2.z());
            p1+=vv;
            p2+=vv;
          }
          glEnd();
        }
      }
      else
      {
        glNormal3d(w.x(), w.y(), w.z());
        for (int i=0;i<nu-1;i++)
        {
          Point p1(rstart);
          Point p2(rstart+uu);
          rstart=p2;
          glBegin(GL_QUAD_STRIP);
          for (int j=0;j<nv;j++)
          {
            Point pp1(p1+w*verts(i, j));
            Point pp2(p2+w*verts(i+1, j));
            glVertex3d(pp1.x(), pp1.y(), pp1.z());

            glVertex3d(pp2.x(), pp2.y(), pp2.z());
            p1+=vv;
            p2+=vv;
          }
          glEnd();
        }
      }
    }
    break;
  }
  post_draw(di);
}
#endif


void
GeomQMesh::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;

  di->polycount_ += (nrows-1)*(ncols-1)*2;

  if (di->currently_lit_)
  {
    for (int j=0;j<ncols-1;j++)
    {
      glBegin(GL_QUAD_STRIP);
      float *rpts=&pts[j*nrows*3];
      float *nrm=&nrmls[j*nrows*3];
      for (int i=0;i<nrows;i++)
      {
        glNormal3fv(nrm);
        glColor3ub(clrs[j*nrows+i].r(),clrs[j*nrows+i].g(),
                   clrs[j*nrows+i].b());
        glVertex3fv(rpts);

        glNormal3fv(nrm+nrows*3);
        glColor3ub(clrs[(j+1)*nrows+i].r(),clrs[(j+1)*nrows+i].g(),
                   clrs[(j+1)*nrows+i].b());
        glVertex3fv(rpts+nrows*3);
        rpts += 3;
        nrm += 3; // bump stuff along
      }
      glEnd();
    }
  }
  else
  {
    for (int j=0;j<ncols-1;j++)
    {
      glBegin(GL_QUAD_STRIP);
      float *rpts=&pts[j*nrows*3];
      for (int i=0;i<nrows;i++)
      {
        glColor3ub(clrs[j*nrows+i].r(),clrs[j*nrows+i].g(),
                   clrs[j*nrows+i].b());
        glVertex3fv(rpts);

        glColor3ub(clrs[(j+1)*nrows+i].r(),clrs[(j+1)*nrows+i].g(),
                   clrs[(j+1)*nrows+i].b());
        glVertex3fv(rpts+nrows*3);
        rpts += 3;
      }
      glEnd();
    }
  }
  post_draw(di);
}


void
GeomGroup::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  if (di->pickmode_)
  {
    for (unsigned int i=0; i<objs.size(); i++)
    {
      objs[i]->draw(di, matl, time);
    }
  }
  else
  {
    for (unsigned int i=0; i<objs.size(); i++)
    {
      objs[i]->draw(di, matl, time);
    }
  }
}


void
GeomTimeGroup::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  int i;
  for (i=0; i<(int)(objs.size()) && (start_times[i] <= time); i++)
    ;
  if (i)
  { // you can go.
    if (i > (int)(objs.size()-1)) { i = (int)(objs.size()-1); }

    if (start_times[i] > time) { --i; }

    if (i < 0) { i = 0; }

    objs[i]->draw(di,matl,time); // run with it.
  }
}


void
GeomLine::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 0)) return;
  di->polycount_++;
  // Set line width. Set it
  GLfloat lw;
  glGetFloatv(GL_LINE_WIDTH, &lw);
  glLineWidth(lineWidth_);
  glBegin(GL_LINE_STRIP);
  glVertex3d(p1.x(), p1.y(), p1.z());
  glVertex3d(p2.x(), p2.y(), p2.z());
  glEnd();
  // HACK set line width back to default
  // our scenegraph needs more graceful control of such state.
  glLineWidth(lw);
  post_draw(di);
}


void
GeomLines::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 0)) return;

  di->polycount_+=points_.size()/6;

  glLineWidth(line_width_);

  glVertexPointer(3, GL_FLOAT, 0, &(points_.front()));
  glEnableClientState(GL_VERTEX_ARRAY);

  if (colors_.size())
  {
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, &(colors_.front()));
    glEnableClientState(GL_COLOR_ARRAY);
  }
  else
  {
    glDisableClientState(GL_COLOR_ARRAY);
  }

  if (di->using_cmtexture_ && indices_.size() == points_.size() / 3)
  {
    glTexCoordPointer(1, GL_FLOAT, 0, &(indices_[0]));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }
  else
  {
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

  if (sci_getenv_p("SCIRUN_DRAWARRAYS_DISABLE"))
  {
    glBegin(GL_LINES);
    for (unsigned int i = 0; i < points_.size()/3; i++)
    {
      glArrayElement(i);
    }
    glEnd();
  }
  else
  {
    glDrawArrays(GL_LINES, 0, points_.size()/3);
  }

  glLineWidth(di->line_width_);

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomCull::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  if (normal_)
  {
    double mat[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, mat);
    if (Dot(Vector(mat[2], mat[6], mat[10]), *normal_) < 0) return;
  }
  child_->draw(di,matl,time);
}


void
GeomTranspLines::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 0)) return;

  sort();

  GLdouble matrix[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, matrix);
  const double lvx = fabs(matrix[2]);
  const double lvy = fabs(matrix[6]);
  const double lvz = fabs(matrix[10]);
  if (lvx >= lvy && lvx >= lvz)
  {
    di->axis_ = 0;
    if (matrix[2] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }

  }
  else if (lvy >= lvx && lvy >= lvz)
  {
    di->axis_ = 1;
    if (matrix[6] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }
  }
  else if (lvz >= lvx && lvz >= lvy)
  {
    di->axis_ = 2;
    if (matrix[10] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }
  }

  vector<unsigned int> &clist =
    (di->axis_==0)?xindices_:((di->axis_==1)?yindices_:zindices_);

  bool &reverse =
    (di->axis_==0)?xreverse_:((di->axis_==1)?yreverse_:zreverse_);

  di->polycount_+=points_.size()/6;

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glLineWidth(line_width_);

  glVertexPointer(3, GL_FLOAT, 0, &(points_.front()));
  glEnableClientState(GL_VERTEX_ARRAY);

  if (colors_.size())
  {
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, &(colors_.front()));
    glEnableClientState(GL_COLOR_ARRAY);
  }
  else
  {
    glDisableClientState(GL_COLOR_ARRAY);
  }

  if (di->using_cmtexture_ && indices_.size() == points_.size() / 3)
  {
    glTexCoordPointer(1, GL_FLOAT, 0, &(indices_[0]));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }
  else
  {
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

  if (di->dir_ == 1 && reverse ||
      di->dir_ == -1 && !reverse)
  {
    std::reverse(clist.begin(), clist.end());
    reverse = !reverse;
  }

  glDrawElements(GL_LINES, clist.size(), GL_UNSIGNED_INT, &(clist.front()));

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glLineWidth(di->line_width_);

  glDisable(GL_BLEND);
  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomCLineStrips::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 0)) return;

  glLineWidth(line_width_);
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  const int n_strips = points_.size();
  for (int i = 0; i < n_strips; i++)
  {
    const int n_points = points_[i].size()/3;
    di->polycount_ += n_points-1;
    glVertexPointer(3, GL_FLOAT, 0, &(points_[i].front()));
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, &(colors_[i].front()));

    if (sci_getenv_p("SCIRUN_DRAWARRAYS_DISABLE"))
    {
      glBegin(GL_LINE_STRIP);
      for (int j = 0; j < n_points; j++)
      {
        glArrayElement(j);
      }
      glEnd();
    }
    else
    {
      glDrawArrays(GL_LINE_STRIP, 0, n_points);
    }
  }

  glLineWidth(di->line_width_);

  post_draw(di);
}


//const int OD_TEX_INIT = 4096; // 12tg bit of clip planes.

void
TexGeomLines::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 0)) return;  // lighting is turned off here.
  di->polycount_+=pts.size()/2;

  static Vector view;  // shared by all of them - should be in ViewWindow!

  // always assume you are not there.
  // can't mix them.

  // first set up the line size stuff.

  // here is where the texture stuff has to be done.
  // first setup the texture matrix.

  double model_mat[16]; // this is the modelview matrix

  glGetDoublev(GL_MODELVIEW_MATRIX,model_mat);
  glMatrixMode(GL_TEXTURE);
  glPushMatrix();

  // this is what you rip the view vector from
  // just use the "Z" axis, normalized

  view = Vector(model_mat[0*4+2],model_mat[1*4+2],model_mat[2*4+2]);

  view.normalize();

  for (int q=0;q<15;q++)
    model_mat[q] = 0.0;

  model_mat[0*4+0] = view.x()*0.5;
  model_mat[1*4+0] = view.y()*0.5;
  model_mat[2*4+0] = view.z()*0.5;
  model_mat[3*4+0] = 0.5;

  model_mat[15] = 1.0;

  // you might want to zero out the rest, but id doesn't matter for 1D
  // texture maps

  glLoadMatrixd(model_mat); // loads the matrix.

  if (!tmapid)
  { // has the texture been created?
    tmap1d.resize(256*3); // that is the size of the 1D texture.
    for (int i=0;i<256;i++)
    {
      double r,ks,LdotT;

      LdotT = i*2/(255.0)-1.0;
      ks =  0.3*(pow(2*LdotT*LdotT - 1,30));

      r = 0.05 + 0.6*(1-LdotT*LdotT) + ks;

      if (r>1.0)
        r = 1.0;

      if (r < 0.0)
        cerr << "Negative r!\n";

      if (r>1.0 || ks>1.0)
        cerr << r << " " << ks << " Error - out of range.\n";

      tmap1d[i*3+0] = (unsigned char)(r*255);
      tmap1d[i*3+1] = (unsigned char)(r*255);    // just have them be red for now.
      tmap1d[i*3+2] = (unsigned char)(r*255);
    }

    // Now set the end conditions.
    tmapid = glGenLists(1);
    glNewList(tmapid,GL_COMPILE_AND_EXECUTE);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    GLfloat brder[4];

    brder[3] = 1.0; // this is just the alpha component.

    brder[0] = (tmap1d[0] + tmap1d[255*3 + 0])/510.0;
    brder[0] = (tmap1d[1] + tmap1d[255*3 + 1])/510.0;
    brder[0] = (tmap1d[2] + tmap1d[255*3 + 2])/510.0;

    glTexParameterfv(GL_TEXTURE_1D,GL_TEXTURE_BORDER_COLOR,brder);

    glEnable(GL_TEXTURE_1D);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage1D(GL_TEXTURE_1D,0,3,
                 256,0,GL_RGB,GL_UNSIGNED_BYTE,
                 &tmap1d[0]);
    glEndList();
  }
  else
  {
    glCallList(tmapid);
  }

  glEnable(GL_TEXTURE_1D);

  // see if you need to create the sorted lists.
  if (!colors.size()) // set if you don't have colors.
    glColor4f(1.0,0.0,0.0,1.0);  // this state always needs to be set.

  if (alpha != 1.0)
  { // create sorted lists.
    if (!sorted.size())
      SortVecs(); // creates sorted lists.
  }

  mutex.lock();
  if (alpha == 1.0)
  {

    glBegin(GL_LINES);
    if (tex_per_seg)
    {
      if (colors.size())
      {
        for (int i=0;i<pts.size()/2;i++)
        {
          Point& pt=pts[i*2];
          Point& pt2=pts[i*2+1];
          glColor3ubv(colors[i].ptr());
          glTexCoord3d(tangents[i].x(),tangents[i].y(),tangents[i].z());
          glVertex3d(pt.x(), pt.y(), pt.z());
          glVertex3d(pt2.x(), pt2.y(), pt2.z());
        }
      }
      else
      {
        for (int i=0;i<pts.size()/2;i++)
        {
          Point& pt=pts[i*2];
          Point& pt2=pts[i*2+1];
          glTexCoord3d(tangents[i].x(),tangents[i].y(),tangents[i].z());
          glVertex3d(pt.x(), pt.y(), pt.z());
          glVertex3d(pt2.x(), pt2.y(), pt2.z());
        }
      }
    }
    else
    {
      if (colors.size())
      {
        for (int i=0;i<pts.size()/2;i++)
        {
          Point& pt=pts[i*2];
          Point& pt2=pts[i*2+1];
          glColor3ubv(colors[i*2].ptr());
          glTexCoord3d(tangents[i*2].x(),tangents[i*2].y(),
                       tangents[i*2].z());
          glVertex3d(pt.x(), pt.y(), pt.z());
          glColor3ubv(colors[i*2 + 1].ptr());
          glTexCoord3d(tangents[i*2+1].x(),tangents[i*2+1].y(),
                       tangents[i*2+1].z());
          glVertex3d(pt2.x(), pt2.y(), pt2.z());
        }
      }
      else
      {
        for (int i=0;i<pts.size()/2;i++)
        {
          Point& pt=pts[i*2];
          Point& pt2=pts[i*2+1];
          glTexCoord3d(tangents[i*2].x(),tangents[i*2].y(),
                       tangents[i*2].z());
          glVertex3d(pt.x(), pt.y(), pt.z());
          glTexCoord3d(tangents[i*2+1].x(),tangents[i*2+1].y(),
                       tangents[i*2+1].z());
          glVertex3d(pt2.x(), pt2.y(), pt2.z());
        }
      }
    }
    glEnd();
  }
  else
  {
    // render with transparency.

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    if (!colors.size())
      glColor4f(1,0,0,alpha); // make sure it is used.

    int sort_start=0;
    int sort_dir=1; // positive direction

    //char which;

    if (Abs(view.x()) > Abs(view.y()))
    {
      if (Abs(view.x()) > Abs(view.z()))
      { // use x dir
        if (view.x() < 0)
        {
          sort_dir=-1; sort_start=pts.size()/2-1;
        }
        else
        {
          sort_start = 0;
        }
      }
      else
      { // use z dir
        if (view.z() < 0)
        {
          sort_dir=-1;sort_start = 2*pts.size()/2-1;
        } else
          sort_start = pts.size()/2;
      }
    }
    else if (Abs(view.y()) > Abs(view.z()))
    { // y greates
      if (view.y() < 0)
      {
        sort_dir=-1;sort_start = 3*(pts.size()/2)-1;
      }
      else
      {
        sort_start = 2*pts.size()/2-1;
      }
    }
    else
    { // z is the one
      if (view.z() < 0)
      {
        sort_dir=-1;sort_start = 2*pts.size()/2-1;
      }
      else
      {
        sort_start = pts.size()/2;
      }
    }

    glBegin(GL_LINES);
    int i = sort_start;
    if (tex_per_seg)
    {
      for (int p=0;p<pts.size()/2;p++)
      {
        Point& pt=pts[sorted[i]];
        Point& pt2=pts[sorted[i]+1]; // already times2.
        glTexCoord3d(tangents[sorted[i]/2].x(),
                     tangents[sorted[i]/2].y(),
                     tangents[sorted[i]/2].z());
        
        glVertex3d(pt.x(), pt.y(), pt.z());
        glVertex3d(pt2.x(), pt2.y(), pt2.z());
        i += sort_dir; // increment i.
      }
    }
    else
    { // this is from the stream line data.
      if (colors.size())
      {
        unsigned char aval = (unsigned char)(alpha*255); // quantize this.
        for (int p=0;p<pts.size()/2;p++)
        {
          Point& pt=pts[sorted[i]];
          Point& pt2=pts[sorted[i]+1]; // already times2.
          glColor4ub(colors[sorted[i]].r(),
                     colors[sorted[i]].g(),
                     colors[sorted[i]].b(),
                     aval);
          glTexCoord3d(tangents[sorted[i]].x(),
                       tangents[sorted[i]].y(),
                       tangents[sorted[i]].z());
          glVertex3d(pt.x(), pt.y(), pt.z());
          glColor4ub(colors[sorted[i]+1].r(),
                     colors[sorted[i]+1].g(),
                     colors[sorted[i]+1].b(),
                     aval);
          glTexCoord3d(tangents[sorted[i]+1].x(),
                       tangents[sorted[i]+1].y(),
                       tangents[sorted[i]+1].z());
          glVertex3d(pt2.x(), pt2.y(), pt2.z());
          i += sort_dir; // increment i.
        }
      }
      else
      {
        for (int p=0;p<pts.size()/2;p++)
        {
          Point& pt=pts[sorted[i]];
          Point& pt2=pts[sorted[i]+1]; // already times2.
          glTexCoord3d(tangents[sorted[i]].x(),
                       tangents[sorted[i]].y(),
                       tangents[sorted[i]].z());
          glVertex3d(pt.x(), pt.y(), pt.z());
          glTexCoord3d(tangents[sorted[i]+1].x(),
                       tangents[sorted[i]+1].y(),
                       tangents[sorted[i]+1].z());
          glVertex3d(pt2.x(), pt2.y(), pt2.z());
          i += sort_dir; // increment i.
        }
      }
    }
    glEnd();
    glDisable(GL_BLEND);
  }
  mutex.unlock();

  glDisable(GL_TEXTURE_1D);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  post_draw(di);
}


void
GeomMaterial::draw(DrawInfoOpenGL* di, Material* /* old_matl */, double time)
{
  child_->draw(di, matl.get_rep(), time);
}


void
GeomPick::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  if (draw_only_on_pick_ && !di->pickmode_) return;
  if (di->pickmode_)
  {
    ++di->npicks_;
#ifdef SCI_64BITS
    unsigned long o=(unsigned long)this;
    unsigned int o1=(o>>32)&0xffffffff;
    unsigned int o2=o&0xffffffff;
    glPushName(o1);
    glPushName(o2);
#else
    glPushName((GLuint)this);
#endif
    di->pickchild_ =1;
  }
  if (selected_ && highlight_.get_rep())
  {
    di->set_material(highlight_.get_rep());
    int old_ignore=di->ignore_matl_;
    di->ignore_matl_=1;
    child_->draw(di, highlight_.get_rep(), time);
    di->ignore_matl_=old_ignore;
  }
  else
  {
    child_->draw(di, matl, time);
  }
  if (di->pickmode_)
  {
#ifdef SCI_64BITS
    glPopName();
    glPopName();
#else
    glPopName();
#endif
        
    if ((--di->npicks_)<1)
    { // Could have multiple picks in stack.
      di->pickchild_=0;
    }
  }
}


void
GeomPolyline::draw(DrawInfoOpenGL* di, Material* matl, double currenttime)
{
  if (!pre_draw(di, matl, 0)) return;
  di->polycount_+=verts.size()-1;
  glBegin(GL_LINE_STRIP);
  if (times.size() == verts.size())
  {
    for (int i=0;i<verts.size() && currenttime >= times[i];i++)
    {
      verts[i]->emit_all(di);
    }
  }
  else
  {
    for (int i=0;i<verts.size();i++)
    {
      verts[i]->emit_all(di);
    }
  }
  glEnd();
  post_draw(di);
}


void
GeomPolylineTC::draw(DrawInfoOpenGL* di, Material* matl, double currenttime)
{
  if (data.size() == 0)
    return;
  if (!pre_draw(di, matl, 0)) return;
  float* d=&data[0];
  float* dend=d+data.size();
  if (drawmode < 1 || drawmode > 3)
  {
    cerr << "Bad drawmode: " << drawmode << endl;
  }
  if (drawmode==1)
  {
    glBegin(GL_LINE_STRIP);
    while (d<dend && *d <= currenttime)
    {
      glColor3fv(d+1);
      glVertex3fv(d+4);
      d+=7;
    }
    di->polycount_+=(d-&data[0])/7-1;
    glEnd();
  }
  else
  {
    // Find the start and end points.
    int n=(dend-d)/7;
    int l=0;
    int h=n-1;
    while (l<h-1)
    {
      int m=(l+h)/2;
      if (currenttime < d[7*m])
      {
        h=m;
      }
      else
      {
        l=m;
      }
    }
    int iend=l;
    l=0;
    // Leave h - it still bounds us on the top
    double begtime=Max(0.0, currenttime-drawdist);
    while (l<h-1)
    {
      int m=(l+h)/2;
      if (begtime < d[7*m])
      {
        h=m;
      }
      else
      {
        l=m;
      }
    }
    int istart=l;
    if (istart==iend)
      return;
    d=&data[7*istart];
    dend=&data[7*iend]+7;
    di->polycount_+=(dend-d)/7-1;
    glBegin(GL_LINE_STRIP);
    if (drawmode == 2)
    {
      while (d<dend)
      {
        glColor3fv(d+1);
        glVertex3fv(d+4);
        d+=7;
      }
    }
    else if (drawmode == 3)
    {
      while (d<dend)
      {
        float s=(*d-begtime)/drawdist;
        glColor3f(d[1]*s, d[2]*s, d[3]*s);
        glVertex3fv(d+4);
        d+=7;
      }
    }
    glEnd();
  }
  post_draw(di);
}


void
GeomPoints::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 0)) { return; }

  di->polycount_+=points_.size()/3;

  if (di->pickmode_)
  {
    if (pickable)
    {
      glLoadName(0);
      float* p=&points_[0];
      for (unsigned int i=0; i<points_.size(); i+=3)
      {
        glLoadName(i/3);
        glBegin(GL_POINTS);
        glVertex3fv(p);
        glEnd();
        p+=3;
      }
    }
  }
  else
  {
    glVertexPointer(3, GL_FLOAT, 0, &(points_[0]));
    glEnableClientState(GL_VERTEX_ARRAY);

    if (colors_.size())
    {
      glColorPointer(4, GL_UNSIGNED_BYTE, 0, &(colors_[0]));
      glEnableClientState(GL_COLOR_ARRAY);
    }
    else
    {
      glDisableClientState(GL_COLOR_ARRAY);
    }

    if (di->using_cmtexture_ && indices_.size() == points_.size() / 3)
    {
      glTexCoordPointer(1, GL_FLOAT, 0, &(indices_[0]));
      glEnableClientState(GL_TEXTURE_COORD_ARRAY);

      glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

      glEnable(GL_TEXTURE_1D);
      glDisable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
    }
    else
    {
      glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    if (sci_getenv_p("SCIRUN_DRAWARRAYS_DISABLE"))
    {
      glBegin(GL_POINTS);
      for (unsigned int i = 0; i < points_.size()/3; i++)
      {
        glArrayElement(i);
      }
      glEnd();
    }
    else
    {
      glDrawArrays(GL_POINTS, 0, points_.size()/3);
    }
  }

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomTranspPoints::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 0)) { return; }

  di->polycount_+=points_.size()/3;

  sort();

  GLdouble matrix[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, matrix);
  const double lvx = fabs(matrix[2]);
  const double lvy = fabs(matrix[6]);
  const double lvz = fabs(matrix[10]);
  if (lvx >= lvy && lvx >= lvz)
  {
    di->axis_ = 0;
    if (matrix[2] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }

  }
  else if (lvy >= lvx && lvy >= lvz)
  {
    di->axis_ = 1;
    if (matrix[6] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }
  }
  else if (lvz >= lvx && lvz >= lvy)
  {
    di->axis_ = 2;
    if (matrix[10] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }
  }

  vector<unsigned int> &clist =
    (di->axis_==0)?xindices_:((di->axis_==1)?yindices_:zindices_);

  bool &reverse =
    (di->axis_==0)?xreverse_:((di->axis_==1)?yreverse_:zreverse_);

  glVertexPointer(3, GL_FLOAT, 0, &(points_[0]));
  glEnableClientState(GL_VERTEX_ARRAY);

  if (colors_.size())
  {
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, &(colors_[0]));
    glEnableClientState(GL_COLOR_ARRAY);
  }
  else
  {
    glDisableClientState(GL_COLOR_ARRAY);
  }

  if (di->using_cmtexture_ && indices_.size() == points_.size() / 3)
  {
    glTexCoordPointer(1, GL_FLOAT, 0, &(indices_[0]));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }
  else
  {
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (di->dir_ == 1 && reverse ||
      di->dir_ == -1 && !reverse)
  {
    std::reverse(clist.begin(), clist.end());
    reverse = !reverse;
  }

  glDrawElements(GL_POINTS, clist.size(), GL_UNSIGNED_INT, &(clist[0]));

  glDisable(GL_BLEND);
  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomTexSlices::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 0)) return;
  if (!have_drawn)
  {
    have_drawn=1;
  }
  GLdouble model_mat[16]; // this is the modelview matrix

  glGetDoublev(GL_MODELVIEW_MATRIX,model_mat);

  // this is what you rip the view vector from
  // just use the "Z" axis, normalized

  Vector view = Vector(model_mat[0*4+2],model_mat[1*4+2],model_mat[2*4+2]);
  int sort_start=0;
  int sort_end=0;
  int sort_dir=1; // positive direction

  char which;

  if (Abs(view.x()) > Abs(view.y()))
  {
    if (Abs(view.x()) > Abs(view.z()))
    { // use x dir
      which = 0;
      if (view.x() < 0)
      {
        sort_dir=-1; sort_start=nx-1; sort_end=-1;
      }
      else
      {
        sort_start=0; sort_end=nx;
      }
    }
    else
    { // use z dir
      which = 2;
      if (view.z() < 0)
      {
        sort_dir=-1;sort_start =nz-1; sort_end=-1;
      }
      else
      {
        sort_start =0; sort_end=nz;
      }
    }
  }
  else if (Abs(view.y()) > Abs(view.z()))
  { // y greates
    which = 1;
    if (view.y() < 0)
    {
      sort_dir=-1;sort_start =ny-1; sort_end=-1;
    }
    else
    {
      sort_start =0; sort_end=ny;
    }
  }
  else
  { // z is the one
    which = 2;
    if (view.z() < 0)
    {
      sort_dir=-1;sort_start = nz-1; sort_end=-1;
    }
    else
    {
      sort_start = 0; sort_end = nz;
    }
  }

  Point pts[4];
  Vector v(0, 0, 0);
  switch (which)
  {
  case 0:       // x
    pts[0] = min;
    pts[1] = Point (min.x(), max.y(), min.z());
    pts[2] = Point (min.x(), max.y(), max.z());
    pts[3] = Point (min.x(), min.y(), max.z());
    v = Vector (max.x()-min.x(),0,0);
    break;
  case 1:       // y
    pts[0] = min;
    pts[1] = Point (max.x(), min.y(), min.z());
    pts[2] = Point (max.x(), min.y(), max.z());
    pts[3] = Point (min.x(), min.y(), max.z());
    v = Vector (0,max.y()-min.y(),0);
    break;
  case 2:
    pts[0] = min;
    pts[1] = Point (max.x(), min.y(), min.z());
    pts[2] = Point (max.x(), max.y(), min.z());
    pts[3] = Point (min.x(), max.y(), min.z());
    v = Vector (0,0,max.z()-min.z());
    break;
  }

  for (int i=0; i<4; i++)
  {
    if (sort_start)
    {
      pts[i] += v;
    }
  }

  glEnable(GL_TEXTURE_2D);
  glPixelStorei(GL_UNPACK_ALIGNMENT,1);
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,
            GL_MODULATE);
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,
                  GL_LINEAR);

  // Make vector be scaled based on slice distance.
  switch (which)
  {
  case 0: // x
    v = v*sort_dir*1.0/nx;
    break;
  case 1: // y
    v = v*sort_dir*1.0/ny;
    break;
  case 2: // z
    v = v*sort_dir*1.0/nz;
    break;
  }

  // Get GL stuff set up.
  glColor4f(1,1,1,bright);

  glAlphaFunc(GL_GEQUAL,accum);  // This might be to large.
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);
  glEnable(GL_ALPHA_TEST);

  for (int outer=sort_start; outer != sort_end; outer += sort_dir)
  {
    switch (which)
    {
    case 0:     // x
      glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY, ny, nz, 0,
                   GL_LUMINANCE, GL_UNSIGNED_BYTE, &(Xmajor(outer,0,0)));
      break;
    case 1: // y
      glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY, nx, nz, 0,
                   GL_LUMINANCE, GL_UNSIGNED_BYTE, &(Ymajor(outer,0,0)));
      break;
    case 2: // y
      glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY, nx, ny, 0,
                   GL_LUMINANCE, GL_UNSIGNED_BYTE, &(Zmajor(outer,0,0)));
      break;
    }
        
    // now draw the quad for this texture.

    for (int j=0;j<4;j++)  // v has been rescaled based on dir
      pts[j] += v;      

    glBegin(GL_QUADS);
    glTexCoord2f(0,0);
    glVertex3f(pts[0].x(),pts[0].y(),pts[0].z());

    glTexCoord2f(0,1);
    glVertex3f(pts[1].x(),pts[1].y(),pts[1].z());

    glTexCoord2f(1,1);
    glVertex3f(pts[2].x(),pts[2].y(),pts[2].z());

    glTexCoord2f(1,0);
    glVertex3f(pts[3].x(),pts[3].y(),pts[3].z());
    glEnd();
  }

  glDisable(GL_BLEND);
  glDisable(GL_ALPHA_TEST);
  glDisable(GL_TEXTURE_2D);
  post_draw(di);
}


void
GeomTube::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  di->polycount_+=(verts.size()-1)*2*20;
  Array1<Point> circle1;
  Array1<Point> circle2;
  Array1<Point>* p1=&circle1;
  Array1<Point>* p2=&circle2;
  SinCosTable tab(nu+1, 0, 2*Pi);
  make_circle(0, *p1, tab);
  if (di->get_drawtype() == DrawInfoOpenGL::WireFrame)
  {
    glBegin(GL_LINE_LOOP);
    for (int j=0;j<p1->size();j++)
    {
      Point pt2((*p1)[j]);
      glVertex3d(pt2.x(), pt2.y(), pt2.z());
    }
    glEnd();
  }
  for (int i=0; i<verts.size()-1; i++)
  {
    make_circle(i+1, *p2, tab);
    Array1<Point>& pp1=*p1;
    Array1<Point>& pp2=*p2;
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
      {
        // Draw lines
        glBegin(GL_LINES);
        int j;
        for (j=0;j<nu;j++)
        {
          Point pt1(pp1[j]);
          glVertex3d(pt1.x(), pt1.y(), pt1.z());
          Point pt2(pp2[j]);
          glVertex3d(pt2.x(), pt2.y(), pt2.z());
        }
        glEnd();
        glBegin(GL_LINE_LOOP);
        for (j=0;j<nu;j++)
        {
          Point pt2(pp2[j]);
          glVertex3d(pt2.x(), pt2.y(), pt2.z());
        }
        glEnd();
      }
      break;
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      {
        // draw triangle strips
        glBegin(GL_TRIANGLE_STRIP);
        Point cen1(verts[i]->p);
        Point cen2(verts[i+1]->p);
        for (int j=0;j<=nu;j++)
        {
          Point pt1(pp1[j]);
          Vector n1(pt1-cen1);
          verts[i]->emit_material(di);
          glNormal3d(n1.x(), n1.y(), n1.z());
          glVertex3d(pt1.x(), pt1.y(), pt1.z());

          Point pt2(pp2[j]);
          Vector n2(pt2-cen2);
          verts[i+1]->emit_material(di);
          glNormal3d(n2.x(), n2.y(), n2.z());
          glVertex3d(pt2.x(), pt2.y(), pt2.z());
        }
        glEnd();
      }
    }
    // Swith p1 and p2 pointers
    Array1<Point>* tmp=p1;
    p1=p2;
    p2=tmp;
  }
  post_draw(di);
}


// --------------------------------------------------

void
GeomRenderMode::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  NOT_FINISHED("GeomRenderMode");
  if (child_.get_rep())
  {
    child_->draw(di, matl, time);
  }
}


void
GeomSphere::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (rad < 1.e-6)return;
  if (!pre_draw(di, matl, 1)) return;
  glPushMatrix();

  glTranslated(cen.x(), cen.y(), cen.z());
  di->polycount_+=2*(nu-1)*(nv-1);

  gluSphere(di->qobj_, rad, nu, nv);

  glPopMatrix();
  post_draw(di);
}


void
GeomSuperquadric::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;

  di->polycount_ += nu_ * nv_;

  glVertexPointer(3, GL_FLOAT, 0, &(points_.front()));
  glEnableClientState(GL_VERTEX_ARRAY);

  glNormalPointer(GL_FLOAT, 0, &(normals_.front()));
  glEnableClientState(GL_NORMAL_ARRAY);

  glDisableClientState(GL_COLOR_ARRAY);

  glDrawElements(GL_TRIANGLE_FAN, nu_ + 2, GL_UNSIGNED_SHORT,
                 &(tindices_[0]));

  glDrawElements(GL_TRIANGLE_FAN, nu_ + 2, GL_UNSIGNED_SHORT,
                 &(tindices_[nu_+2]));

  for (int pi = 0; pi < nv_-2; pi++)
  {
    glDrawElements(GL_QUAD_STRIP, (nu_+1)*2, GL_UNSIGNED_SHORT,
                   &(qindices_[pi * (nu_+1) * 2]));
  }

  post_draw(di);
}


void
GeomSpheres::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  const bool ulr = radii_.size() == centers_.size();
  if (!ulr && global_radius_ < 1.0e-6) { return; }

  if (!pre_draw(di, matl, 1)) return;

  di->polycount_ += 2 * (nu_-1) * (nv_-1) * centers_.size();

  const bool using_texture =
    di->using_cmtexture_ && indices_.size() == centers_.size();
  if (using_texture)
  {
    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }

  const bool using_color = centers_.size() == colors_.size() / 4;

  glMatrixMode(GL_MODELVIEW);
  for (unsigned int i=0; i < centers_.size(); i++)
  {
    if (using_texture) { glTexCoord1f(indices_[i]); }
    if (using_color) { glColor3ubv(&(colors_[i*4])); }

    glPushMatrix();

    glTranslated(centers_[i].x(), centers_[i].y(), centers_[i].z());
    gluSphere(di->qobj_, ulr?radii_[i]:global_radius_, nu_, nv_);
        
    glPopMatrix();
  }

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomEllipsoid::draw(DrawInfoOpenGL* di, Material* matl, double)
{

  // no pre_draw, done in GeomSphere::draw
  glPushMatrix();
  glTranslated(cen.x(), cen.y(), cen.z());
  glMultMatrixd(m_tensor_matrix);
  glTranslated(-cen.x(), -cen.y(), -cen.z());
  GeomSphere::draw(di, matl, 1);
  glPopMatrix();
  // no post_draw, done in GeomSphere::draw

}


void
GeomSwitch::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  if (state && child_.get_rep())
  {
    child_->draw(di, matl, time);
  }
}


const GLubyte stipple_pattern[] = { 
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
  0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55 };


void
GeomStippleOccluded::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  if (state && child_.get_rep()) {
    child_->draw(di, matl, time);
    glDepthRange(0.05, 0.08);
    glEnable(GL_POLYGON_STIPPLE);
    glPolygonStipple(stipple_pattern);
    child_->draw(di, matl, time);
    glDepthRange(0.0, 1.0);
    glDisable(GL_POLYGON_STIPPLE);
  }
}


void
GeomTetra::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  di->polycount_+=4;

  switch(di->get_drawtype())
  {
  case DrawInfoOpenGL::WireFrame:
    if (di->currently_lit_)
    {
      Vector n1(Plane(p1, p2, p3).normal());
      glBegin(GL_LINE_STRIP);
      glNormal3d(n1.x(),n1.y(),n1.z());
      glVertex3d(p1.x(), p1.y(), p1.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p3.x(), p3.y(), p3.z());
      glVertex3d(p1.x(), p1.y(), p1.z());
      glVertex3d(p4.x(), p4.y(), p4.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p3.x(), p3.y(), p3.z());
      glVertex3d(p4.x(), p4.y(), p4.z());
      glEnd();
    }
    else
    {
      glBegin(GL_LINE_STRIP);
      glVertex3d(p1.x(), p1.y(), p1.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p3.x(), p3.y(), p3.z());
      glVertex3d(p1.x(), p1.y(), p1.z());
      glVertex3d(p4.x(), p4.y(), p4.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p3.x(), p3.y(), p3.z());
      glVertex3d(p4.x(), p4.y(), p4.z());
      glEnd();
    }
    break;
  case DrawInfoOpenGL::Flat:
    // this should be made into a tri-strip, but I couldn;t remember how.

    /* this can be done as a triangle strip using 8 vertices, or
     * as a triangle fan with 5 and 1 single triangle (8 verts)
     * I am doing the fan now (ordering is wierd with a tri-strip), but
     * will switch to the tri-strip when I can test it, if it's faster
     */ 
  case DrawInfoOpenGL::Gouraud:
    // this should be made into a tri-strip, but I couldn;t remember how.

    /*
     * These are actualy just flat shaded, to get "gourad" shading
     * you could average the facet normals for all the faces touching
     * a given vertex.  I don't think there is a faster way to do this
     * using flat shading.
     */
    if (di->currently_lit_)
    {
      Vector n1(Plane(p1, p2, p3).normal());
      Vector n2(Plane(p1, p2, p4).normal());
      Vector n3(Plane(p4, p2, p3).normal());
      Vector n4(Plane(p1, p4, p3).normal());

      glBegin(GL_TRIANGLES);
      glNormal3d(n1.x(), n1.y(), n1.z());
      glVertex3d(p1.x(), p1.y(), p1.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p3.x(), p3.y(), p3.z());
        
      glNormal3d(n2.x(), n2.y(), n2.z());
      glVertex3d(p1.x(), p1.y(), p1.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p4.x(), p4.y(), p4.z());

      glNormal3d(n3.x(), n3.y(), n3.z());
      glVertex3d(p4.x(), p4.y(), p4.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p3.x(), p3.y(), p3.z());

      glNormal3d(n4.x(), n4.y(), n4.z());
      glVertex3d(p1.x(), p1.y(), p1.z());
      glVertex3d(p4.x(), p4.y(), p4.z());
      glVertex3d(p3.x(), p3.y(), p3.z());
      glEnd();
    }
    else
    {
      glBegin(GL_TRIANGLE_FAN);
      glVertex3d(p1.x(), p1.y(), p1.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p3.x(), p3.y(), p3.z());
      glVertex3d(p4.x(), p4.y(), p4.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glEnd();
      glBegin(GL_TRIANGLES);
      glVertex3d(p4.x(), p4.y(), p4.z());
      glVertex3d(p2.x(), p2.y(), p2.z());
      glVertex3d(p3.x(), p3.y(), p3.z());
      glEnd();
    }
#if 0
    /*
     * This has inconsistant ordering.
     */
    glBegin(GL_TRIANGLES);
    glVertex3d(p1.x(), p1.y(), p1.z());
    glVertex3d(p2.x(), p2.y(), p2.z());
    glVertex3d(p3.x(), p3.y(), p3.z());
    glEnd();
    glBegin(GL_TRIANGLES);
    glVertex3d(p1.x(), p1.y(), p1.z());
    glVertex3d(p2.x(), p2.y(), p2.z());
    glVertex3d(p4.x(), p4.y(), p4.z());
    glEnd();
    glBegin(GL_TRIANGLES);
    glVertex3d(p4.x(), p4.y(), p4.z());
    glVertex3d(p2.x(), p2.y(), p2.z());
    glVertex3d(p3.x(), p3.y(), p3.z());
    glEnd();
    glBegin(GL_TRIANGLES);
    glVertex3d(p1.x(), p1.y(), p1.z());
    glVertex3d(p4.x(), p4.y(), p4.z());
    glVertex3d(p3.x(), p3.y(), p3.z());
    glEnd();
#endif
    break;
  }
  post_draw(di);
}


void
GeomTimeSwitch::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  if (time >= tbeg && time < tend)
  {
    child_->draw(di, matl, time);
  }
}


// WARNING not fixed for lighting correctly yet!

void
GeomTorus::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  glPushMatrix();
  glTranslated(cen.x(), cen.y(), cen.z());
  glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
  di->polycount_+=2*(nu-1)*(nv-1);

  // Draw the torus
  SinCosTable tab1(nu, 0, 2*Pi);
  SinCosTable tab2(nv, 0, 2*Pi, rad2);
  SinCosTable tab2n(nv, 0, 2*Pi, rad2);
  int u,v;
  switch(di->get_drawtype())
  {
  case DrawInfoOpenGL::WireFrame:
    for (u=0;u<nu;u++)
    {
      double rx=tab1.sin(u);
      double ry=tab1.cos(u);
      glBegin(GL_LINE_LOOP);
      for (v=1;v<nv;v++)
      {
        double z=tab2.cos(v);
        double rad=rad1+tab2.sin(v);
        double x=rx*rad;
        double y=ry*rad;
        glVertex3d(x, y, z);
      }
      glEnd();
    }
    for (v=1;v<nv;v++)
    {
      double z=tab2.cos(v);
      double rr=tab2.sin(v);
      glBegin(GL_LINE_LOOP);
      for (u=1;u<nu;u++)
      {
        double rad=rad1+rr;
        double x=tab1.sin(u)*rad;
        double y=tab1.cos(u)*rad;
        glVertex3d(x, y, z);
      }
      glEnd();
    }
    break;
#if 0
    for (v=0;v<nv-1;v++)
    {
      double z1=tab2.cos(v);
      double rr1=tab2.sin(v);
      double z2=tab2.cos(v+1);
      double rr2=tab2.sin(v+1);
      glBegin(GL_TRIANGLE_STRIP);
      for (u=0;u<nu;u++)
      {
        double r1=rad1+rr1;
        double r2=rad1+rr2;
        double xx=tab1.sin(u);
        double yy=tab1.cos(u);
        double x1=xx*r1;
        double y1=yy*r1;
        double x2=xx*r2;
        double y2=yy*r2;
        glVertex3d(x1, y1, z1);
        glVertex3d(x2, y2, z2);
      }
      glEnd();
    }
    break;
#endif
  case DrawInfoOpenGL::Flat:
    for (v=0;v<nv-1;v++)
    {
      double z1=tab2.cos(v);
      double rr1=tab2.sin(v);
      double z2=tab2.cos(v+1);
      double rr2=tab2.sin(v+1);
      double nr=-tab2n.sin(v);
      double nz=-tab2n.cos(v);
      glBegin(GL_TRIANGLE_STRIP);
      for (u=0;u<nu;u++)
      {
        double r1=rad1+rr1;
        double r2=rad1+rr2;
        double xx=tab1.sin(u);
        double yy=tab1.cos(u);
        double x1=xx*r1;
        double y1=yy*r1;
        double x2=xx*r2;
        double y2=yy*r2;
        glNormal3d(nr*xx, nr*yy, nz);
        glVertex3d(x1, y1, z1);
        glVertex3d(x2, y2, z2);
      }
      glEnd();
    }
    break;
  case DrawInfoOpenGL::Gouraud:
    for (v=0;v<nv-1;v++)
    {
      double z1=tab2.cos(v);
      double rr1=tab2.sin(v);
      double z2=tab2.cos(v+1);
      double rr2=tab2.sin(v+1);
      double nr1=-tab2n.sin(v);
      double nr2=-tab2n.sin(v+1);
      double nz1=-tab2n.cos(v);
      double nz2=-tab2n.cos(v+1);
      glBegin(GL_TRIANGLE_STRIP);
      for (u=0;u<nu;u++)
      {
        double r1=rad1+rr1;
        double r2=rad1+rr2;
        double xx=tab1.sin(u);
        double yy=tab1.cos(u);
        double x1=xx*r1;
        double y1=yy*r1;
        double x2=xx*r2;
        double y2=yy*r2;
        glNormal3d(nr1*xx, nr1*yy, nz1);
        glVertex3d(x1, y1, z1);
        glNormal3d(nr2*xx, nr2*yy, nz2);
        glVertex3d(x2, y2, z2);
      }
      glEnd();
    }
    break;      
  }
  glPopMatrix();
  post_draw(di);
}


void
GeomTransform::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  glPushMatrix();
  double mat[16];
  trans.get_trans(mat);
  glMultMatrixd(mat);
  child_->draw(di, matl, time);
  glPopMatrix();
}


void
GeomTri::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  di->polycount_++;
  if (di->currently_lit_)
  {
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
      glBegin(GL_LINE_LOOP);
      verts[0]->emit_all(di);
      verts[1]->emit_all(di);
      verts[2]->emit_all(di);
      glEnd();
      break;
    case DrawInfoOpenGL::Flat:
      glBegin(GL_TRIANGLES);
      glNormal3d(-n.x(), -n.y(), -n.z());
      verts[0]->emit_point(di);
      verts[1]->emit_point(di);
      verts[2]->emit_all(di);
      glEnd();
      break;
    case DrawInfoOpenGL::Gouraud:
      glBegin(GL_TRIANGLES);
      glNormal3d(-n.x(), -n.y(), -n.z());
      verts[0]->emit_all(di);
      verts[1]->emit_all(di);
      verts[2]->emit_all(di);
      glEnd();
      break;
    }
  }
  else
  {
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
      glBegin(GL_LINE_LOOP);
      verts[0]->emit_point(di);
      verts[1]->emit_point(di);
      verts[2]->emit_point(di);
      glEnd();
      break;
    case DrawInfoOpenGL::Flat:
      glBegin(GL_TRIANGLES);
      verts[0]->emit_point(di);
      verts[1]->emit_point(di);
      verts[2]->emit_material(di);
      verts[2]->emit_point(di);
      glEnd();
      break;
    case DrawInfoOpenGL::Gouraud:
      // posible change to just material and point.
      glBegin(GL_TRIANGLES);
      verts[0]->emit_all(di);
      verts[1]->emit_all(di);
      verts[2]->emit_all(di);
      glEnd();
      break;
    }
  }
  post_draw(di);
}


void
GeomTriangles::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  if (verts.size() <= 2)
    return;
  di->polycount_+=verts.size()/3;
  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#else
    glDisable(GL_NORMALIZE);
#endif
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
      {
        for (int i=0;i<verts.size();i+=3)
        {
          glBegin(GL_LINE_LOOP);
          glNormal3d(normals[i/3].x(), normals[i/3].y(),
                     normals[i/3].z());
          verts[i]->emit_all(di);
          verts[i+1]->emit_all(di);
          verts[i+2]->emit_all(di);
          glEnd();
        }
      }
      break;
    case DrawInfoOpenGL::Flat:
      {
        glBegin(GL_TRIANGLES);
        for (int i=0;i<verts.size();i+=3)
        {
          glNormal3d(normals[i/3].x(), normals[i/3].y(),
                     normals[i/3].z());
          verts[i]->emit_point(di);
          verts[i+1]->emit_point(di);
          verts[i+2]->emit_all(di);
        }
        glEnd();
      }
      break;
    case DrawInfoOpenGL::Gouraud:
      {
        glBegin(GL_TRIANGLES);
        for (int i=0;i<verts.size();i+=3)
        {
          glNormal3d(normals[i/3].x(), normals[i/3].y(),
                     normals[i/3].z());
          verts[i]->emit_all(di);
          verts[i+1]->emit_all(di);
          verts[i+2]->emit_all(di);
        }
        glEnd();
      }
      break;
    }
    glEnable(GL_NORMALIZE);
  }
  else {
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
      {
        for (int i=0;i<verts.size();i+=3)
        {
          glBegin(GL_LINE_LOOP);
          verts[i]->emit_all(di);
          verts[i+1]->emit_all(di);
          verts[i+2]->emit_all(di);
          glEnd();
        }
      }
      break;
    case DrawInfoOpenGL::Flat:
      {
        glBegin(GL_TRIANGLES);
        for (int i=0;i<verts.size();i+=3)
        {
          verts[i]->emit_point(di);
          verts[i+1]->emit_point(di);
          verts[i+2]->emit_all(di);
        }
        glEnd();
      }
      break;
    case DrawInfoOpenGL::Gouraud:
      {
        glDisable(GL_NORMALIZE);
        glBegin(GL_TRIANGLES);
        for (int i=0;i<verts.size();i+=3)
        {
          verts[i]->emit_all(di);
          verts[i+1]->emit_all(di);
          verts[i+2]->emit_all(di);
        }
        glEnd();
        glEnable(GL_NORMALIZE);
      }
      break;
    }
  }
  post_draw(di);
}


void
GeomFastTriangles::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  di->polycount_ += size();

  glShadeModel(GL_FLAT);
  glDisable(GL_NORMALIZE);
  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#endif
    if (di->get_drawtype() == DrawInfoOpenGL::Flat ||
        normals_.size() < face_normals_.size())
    {
      glNormalPointer(GL_FLOAT, 0, &(face_normals_.front()));
    }
    else
    {
      glNormalPointer(GL_FLOAT, 0, &(normals_.front()));
    }
    glEnableClientState(GL_NORMAL_ARRAY);
    if (di->get_drawtype() != DrawInfoOpenGL::Flat)
    {
      glShadeModel(GL_SMOOTH);
    }
  }
  else
  {
    glDisableClientState(GL_NORMAL_ARRAY);
  }

  if (material_.get_rep()) { di->set_material(material_.get_rep()); }

  if (colors_.size())
  {
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, &(colors_.front()));
    glEnableClientState(GL_COLOR_ARRAY);
  }
  else
  {
    glDisableClientState(GL_COLOR_ARRAY);
  }

  if (di->using_cmtexture_ && indices_.size() == points_.size() / 3)
  {
    glTexCoordPointer(1, GL_FLOAT, 0, &(indices_[0]));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }
  else
  {
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

  glVertexPointer(3, GL_FLOAT, 0, &(points_.front()));
  glEnableClientState(GL_VERTEX_ARRAY);

  if (sci_getenv_p("SCIRUN_DRAWARRAYS_DISABLE"))
  {
    glBegin(GL_TRIANGLES);
    for (unsigned int i = 0; i < points_.size()/3; i++)
    {
      glArrayElement(i);
    }
    glEnd();
  }
  else
  {
    glDrawArrays(GL_TRIANGLES, 0, points_.size()/3);
  }

  glDisableClientState(GL_NORMAL_ARRAY);
  glEnable(GL_NORMALIZE);
  glShadeModel(GL_SMOOTH);
  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomTranspTriangles::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  di->polycount_ += size();

  SortPolys();

  GLdouble matrix[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, matrix);
  const double lvx = fabs(matrix[2]);
  const double lvy = fabs(matrix[6]);
  const double lvz = fabs(matrix[10]);
  if (lvx >= lvy && lvx >= lvz)
  {
    di->axis_ = 0;
    if (matrix[2] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }

  }
  else if (lvy >= lvx && lvy >= lvz)
  {
    di->axis_ = 1;
    if (matrix[6] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }
  }
  else if (lvz >= lvx && lvz >= lvy)
  {
    di->axis_ = 2;
    if (matrix[10] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }
  }

  vector<unsigned int> &clist =
    (di->axis_==0)?xlist_:((di->axis_==1)?ylist_:zlist_);

  bool &reverse =
    (di->axis_==0)?xreverse_:((di->axis_==1)?yreverse_:zreverse_);

  glShadeModel(GL_FLAT);
  glDisable(GL_NORMALIZE);
  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#endif
    if (di->get_drawtype() == DrawInfoOpenGL::Flat ||
        normals_.size() < face_normals_.size())
    {
      glNormalPointer(GL_FLOAT, 0, &(face_normals_.front()));
    }
    else
    {
      glNormalPointer(GL_FLOAT, 0, &(normals_.front()));
    }
    glEnableClientState(GL_NORMAL_ARRAY);
    if (di->get_drawtype() != DrawInfoOpenGL::Flat)
    {
      glShadeModel(GL_SMOOTH);
    }
  }
  else
  {
    glDisableClientState(GL_NORMAL_ARRAY);
  }

  if (material_.get_rep()) { di->set_material(material_.get_rep()); }

  if (colors_.size())
  {
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, &(colors_.front()));
    glEnableClientState(GL_COLOR_ARRAY);
  }
  else
  {
    glDisableClientState(GL_COLOR_ARRAY);
  }

  if (di->using_cmtexture_ && indices_.size() == points_.size() / 3)
  {
    glTexCoordPointer(1, GL_FLOAT, 0, &(indices_[0]));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }
  else
  {
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

  glVertexPointer(3, GL_FLOAT, 0, &(points_.front()));
  glEnableClientState(GL_VERTEX_ARRAY);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (di->dir_ == 1 && reverse ||
      di->dir_ == -1 && !reverse)
  {
    std::reverse(clist.begin(), clist.end());
    reverse = !reverse;
  }

  glFrontFace(reverse?GL_CW:GL_CCW);

  glDrawElements(GL_TRIANGLES, clist.size(), GL_UNSIGNED_INT, &(clist[0]));

  glFrontFace(GL_CCW);

  glDisableClientState(GL_NORMAL_ARRAY);

  glDisable(GL_BLEND);
  glEnable(GL_NORMALIZE);

  if (di->get_drawtype() == DrawInfoOpenGL::Flat)
  {
    glShadeModel(GL_SMOOTH);
  }

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomFastQuads::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  di->polycount_ += size();

  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#else
    glDisable(GL_NORMALIZE);
#endif
    glNormalPointer(GL_FLOAT, 0, &(normals_.front()));
    glEnableClientState(GL_NORMAL_ARRAY);
  }
  else
  {
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisable(GL_NORMALIZE);
  }

  if (material_.get_rep()) { di->set_material(material_.get_rep()); }

  if (colors_.size())
  {
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, &(colors_.front()));
    glEnableClientState(GL_COLOR_ARRAY);
  }
  else
  {
    glDisableClientState(GL_COLOR_ARRAY);
  }

  if (di->using_cmtexture_ && indices_.size() == points_.size() / 3)
  {
    glTexCoordPointer(1, GL_FLOAT, 0, &(indices_[0]));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }
  else
  {
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

  glVertexPointer(3, GL_FLOAT, 0, &(points_.front()));
  glEnableClientState(GL_VERTEX_ARRAY);


  if (di->get_drawtype() == DrawInfoOpenGL::Flat)
  {
    glShadeModel(GL_FLAT);
  }
  else
  {
    glShadeModel(GL_SMOOTH);
  }

  if (sci_getenv_p("SCIRUN_DRAWARRAYS_DISABLE"))
  {
    glBegin(GL_QUADS);
    for (unsigned int i = 0; i < points_.size()/3; i++)
    {
      glArrayElement(i);
    }
    glEnd();
  }
  else
  {
    glDrawArrays(GL_QUADS, 0, points_.size()/3);
  }

  glDisableClientState(GL_NORMAL_ARRAY);

  glEnable(GL_NORMALIZE);
  if (di->get_drawtype() == DrawInfoOpenGL::Flat)
  {
    glShadeModel(GL_SMOOTH);
  }

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomTranspQuads::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  di->polycount_ += size();

  SortPolys();

  GLdouble matrix[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, matrix);
  const double lvx = fabs(matrix[2]);
  const double lvy = fabs(matrix[6]);
  const double lvz = fabs(matrix[10]);
  if (lvx >= lvy && lvx >= lvz)
  {
    di->axis_ = 0;
    if (matrix[2] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }

  }
  else if (lvy >= lvx && lvy >= lvz)
  {
    di->axis_ = 1;
    if (matrix[6] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }
  }
  else if (lvz >= lvx && lvz >= lvy)
  {
    di->axis_ = 2;
    if (matrix[10] > 0) { di->dir_ = 1; }
    else { di->dir_ = -1; }
  }

  vector<unsigned int> &clist =
    (di->axis_==0)?xlist_:((di->axis_==1)?ylist_:zlist_);

  bool &reverse =
    (di->axis_==0)?xreverse_:((di->axis_==1)?yreverse_:zreverse_);

  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#else
    glDisable(GL_NORMALIZE);
#endif
    glNormalPointer(GL_FLOAT, 0, &(normals_.front()));
    glEnableClientState(GL_NORMAL_ARRAY);
  }
  else
  {
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisable(GL_NORMALIZE);
  }

  if (material_.get_rep()) { di->set_material(material_.get_rep()); }

  if (colors_.size())
  {
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, &(colors_.front()));
    glEnableClientState(GL_COLOR_ARRAY);
  }
  else
  {
    glDisableClientState(GL_COLOR_ARRAY);
  }

  if (di->using_cmtexture_ && indices_.size() == points_.size() / 3)
  {
    glTexCoordPointer(1, GL_FLOAT, 0, &(indices_[0]));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }
  else
  {
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

  glVertexPointer(3, GL_FLOAT, 0, &(points_.front()));
  glEnableClientState(GL_VERTEX_ARRAY);

  if (di->get_drawtype() == DrawInfoOpenGL::Flat)
  {
    glShadeModel(GL_FLAT);
  }
  else
  {
    glShadeModel(GL_SMOOTH);
  }

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (di->dir_ == 1 && reverse ||
      di->dir_ == -1 && !reverse)
  {
    std::reverse(clist.begin(), clist.end());
    reverse = !reverse;
  }

  glFrontFace(reverse?GL_CW:GL_CCW);

  glDrawElements(GL_QUADS, clist.size(), GL_UNSIGNED_INT, &(clist.front()));

  glFrontFace(GL_CCW);

  glDisableClientState(GL_NORMAL_ARRAY);

  glDisable(GL_BLEND);
  glEnable(GL_NORMALIZE);

  if (di->get_drawtype() == DrawInfoOpenGL::Flat)
  {
    glShadeModel(GL_SMOOTH);
  }

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomTrianglesP::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (points.size() == 0)
    return;

  // DAVE: Hack for 3d texture mapping
  if (!pre_draw(di,matl,1)) return;

  di->polycount_ += size();

  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#else
    glDisable(GL_NORMALIZE);
#endif
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      { 
        float *pts = &points[0];
        float *nrmls = &normals[0];
        int niter = size();
        glBegin(GL_TRIANGLES);
        while (niter--)
        {
          glNormal3fv(nrmls);
          nrmls+=3;
          glVertex3fv(pts);
          pts += 3;
          glVertex3fv(pts);
          pts+=3;
          glVertex3fv(pts);
          pts+=3;
        }
        glEnd();
      }
                
      break;
    }
    glEnable(GL_NORMALIZE);
  }
  else
  { // lights are off, don't emit the normals
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      { 
        float *pts = &points[0];
        int niter = size();
        glBegin(GL_TRIANGLES);
        while (niter--)
        {
          glVertex3fv(pts);
          pts += 3;
          glVertex3fv(pts);
          pts+=3;
          glVertex3fv(pts);
          pts+=3;
        }
        glEnd();
      }
                
      break;
    }
  }
  post_draw(di);
}


void
GeomTrianglesPT1d::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  //  return;
  if (!pre_draw(di,matl,1)) return;
  di->polycount_ += size();

  if (cmap)
  { // use 1D texturing.
#if 1
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glEnable(GL_TEXTURE_1D);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage1D(GL_TEXTURE_1D,0,4,
                 256,0,GL_RGBA,GL_UNSIGNED_BYTE,
                 cmap);
    glColor4f(1,1,1,1);
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glAlphaFunc(GL_GREATER,0.0); // exactly 0 means draw nothing.
    glEnable(GL_ALPHA_TEST);
#endif
  }
  else
  {
    cerr << "No color map!\n";
    return; // don't draw if no color map.
  }

  if (di->currently_lit_)
  {
    float *pts=&points[0];
    float *nrmls=&normals[0];
    float *sclrs=&scalars[0];

    int niter=size();
    glBegin(GL_TRIANGLES);
    while (niter--)
    {
      glNormal3fv(nrmls);
      nrmls+=3;
      glTexCoord1fv(sclrs);
      sclrs++;
      glVertex3fv(pts);
      pts += 3;
      glVertex3fv(pts);
      pts+=3;
      glVertex3fv(pts);
      pts+=3;
    }
    glEnd();
  }
  else
  { // no normals.
    float *pts=&points[0];
    float *sclrs=&scalars[0];

    int niter=size();
    glBegin(GL_TRIANGLES);
    while (niter--)
    {
      glTexCoord1fv(sclrs);
      sclrs++;
      glVertex3fv(pts);
      pts += 3;
      glVertex3fv(pts);
      pts+=3;
      glVertex3fv(pts);
      pts+=3;
    }
    glEnd();

  }

  if (cmap)
  {
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_TEXTURE_1D);
  }
  post_draw(di);
}


void
GeomTranspTrianglesP::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!size())
  {
    return;
  }

  if (!pre_draw(di,matl,1)) return; // yes, this is lit.

  di->polycount_ += size();

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  if (!has_color)
    {
      glColor4f(0,1.0,0.0,alpha_);
    }
  else
    {
      glColor4f(r,g,b,alpha_);
    }

  if (!sorted_p_)
    {
      SortPolys(); // sort the iso-surface.
    }

  glDepthMask(GL_FALSE); // no zbuffering for now.
#ifdef SCI_NORM_OGL
  glEnable(GL_NORMALIZE);
#else
  glDisable(GL_NORMALIZE);
#endif

  GLdouble matrix[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, matrix);
  const double lvx = fabs(matrix[2]);
  const double lvy = fabs(matrix[6]);
  const double lvz = fabs(matrix[10]);
  if (lvx >= lvy && lvx >= lvz)
    {
      di->axis_ = 0;
      if (matrix[2] > 0) { di->dir_ = 1; }
      else { di->dir_ = -1; }

    }
  else if (lvy >= lvx && lvy >= lvz)
    {
      di->axis_ = 1;
      if (matrix[6] > 0) { di->dir_ = 1; }
      else { di->dir_ = -1; }
    }
  else if (lvz >= lvx && lvz >= lvy)
    {
      di->axis_ = 2;
      if (matrix[10] > 0) { di->dir_ = 1; }
      else { di->dir_ = -1; }
    }

  const vector<pair<float, unsigned int> >&cur_list =
    (di->axis_==0)?xlist_:((di->axis_==1)?ylist_:zlist_);

  const int sort_dir = (di->dir_<0)?-1:1;
  const unsigned int sort_start = (sort_dir>0)?0:(cur_list.size()-1);
  unsigned int i;
  unsigned int ndone = 0;
  glBegin(GL_TRIANGLES);
  if (di->currently_lit_)
    {
      for (i = sort_start ; ndone < cur_list.size(); ndone++, i += sort_dir)
        {
          const unsigned int nindex = cur_list[i].second * 3;
          const unsigned int pindex = nindex * 3;
          glNormal3fv(&normals[nindex]);
          glVertex3fv(&points[pindex+0]);
          glVertex3fv(&points[pindex+3]);
          glVertex3fv(&points[pindex+6]);
        }
    }
  else
    {
      for (i = sort_start; ndone < cur_list.size(); ndone++, i += sort_dir)
        {
          const int pindex = cur_list[i].second * 9;

          glVertex3fv(&points[pindex+0]);
          glVertex3fv(&points[pindex+3]);
          glVertex3fv(&points[pindex+6]);
        }
    }
  glEnd();

  glDepthMask(GL_TRUE); // turn zbuff back on.
  glDisable(GL_BLEND);
  glEnable(GL_NORMALIZE);
  post_draw(di);
}


void
GeomBox::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di,matl,1)) return;

  di->polycount_ += 6;

  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#else
    glDisable(GL_NORMALIZE);
#endif
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      { 
        glBegin(GL_QUADS);

        // top
        glNormal3f(0,0,1);
        glColor4f(0.0, 0.0, 1.0, 0.0);
        glVertex3d(min.x(),min.y(),max.z());
        glVertex3d(max.x(),min.y(),max.z());
        glVertex3d(max.x(),max.y(),max.z());
        glVertex3d(min.x(),max.y(),max.z());

        // bottom
        glNormal3f(0,0,-1);
        glColor4f(0.0, 0.0, 0.5, 0.0);
        glVertex3d(min.x(),min.y(),min.z());
        glVertex3d(min.x(),max.y(),min.z());
        glVertex3d(max.x(),max.y(),min.z());
        glVertex3d(max.x(),min.y(),min.z());
        
        // left
        glNormal3f(-1.0,0,0);
        glColor4f(0.5, 0.0, 0.0, 0.0);
        glVertex3d(min.x(),min.y(),min.z());
        glVertex3d(min.x(),min.y(),max.z());
        glVertex3d(min.x(),max.y(),max.z());
        glVertex3d(min.x(),max.y(),min.z());

        // right
        glNormal3f(1,0,0);
        glColor4f(1.0, 0.0, 0.0, 0.0);
        glVertex3d(max.x(),min.y(),min.z());
        glVertex3d(max.x(),max.y(),min.z());
        glVertex3d(max.x(),max.y(),max.z());
        glVertex3d(max.x(),min.y(),max.z());
                
        // top
        glNormal3f(0,1.0,0);
        glColor4f(0.0, 1.0, 0.0, 0.0);
        glVertex3d(min.x(),max.y(),min.z());
        glVertex3d(min.x(),max.y(),max.z());
        glVertex3d(max.x(),max.y(),max.z());
        glVertex3d(max.x(),max.y(),min.z());

        // back
        glNormal3f(0,-1,0);
        glColor4f(0.0, 0.5, 0.0, 0.0);
        glVertex3d(min.x(),min.y(),min.z());
        glVertex3d(max.x(),min.y(),min.z());
        glVertex3d(max.x(),min.y(),max.z());
        glVertex3d(min.x(),min.y(),max.z());

        glEnd();
      }
      break;
    }
    glEnable(GL_NORMALIZE);
  }
  else
  { // lights are off, don't emit the normals
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      { 
        glBegin(GL_QUADS);
        //front
        glVertex3d(max.x(),min.y(),max.z());
        glVertex3d(max.x(),max.y(),max.z());
        glColor4f(0.0,1.0,0.0,0.2);
        glVertex3d(min.x(),max.y(),max.z());
        glVertex3d(min.x(),min.y(),max.z());

        //back
        glVertex3d(max.x(),max.y(),min.z());
        glVertex3d(max.x(),min.y(),min.z());
        glVertex3d(min.x(),min.y(),min.z());
        glColor4f(0.0,1.0,0.0,0.2);
        glVertex3d(min.x(),max.y(),min.z());
        
        glColor4f(1.0,0.0,0.0,0.2);
        
        //left
        glVertex3d(min.x(),min.y(),max.z());
        glVertex3d(min.x(),max.y(),max.z());
        glVertex3d(min.x(),max.y(),min.z());
        glVertex3d(min.x(),min.y(),min.z());
        glColor4f(1.0,0.0,0.0,0.2);
        
        //right
        glVertex3d(max.x(),min.y(),min.z());
        glVertex3d(max.x(),max.y(),min.z());
        glVertex3d(max.x(),max.y(),max.z());
        glColor4f(1.0,0.0,0.0,0.2);
        glVertex3d(max.x(),min.y(),max.z());
        
        
        glColor4f(0.0,0.0,1.0,0.2);
        
        //top
        glVertex3d(min.x(),max.y(),max.z());
        glVertex3d(max.x(),max.y(),max.z());
        glColor4f(0.0,0.0,1.0,0.2);
        glVertex3d(max.x(),max.y(),min.z());
        glVertex3d(min.x(),max.y(),min.z());

        //bottom
        glVertex3d(min.x(),min.y(),min.z());
        glColor4f(0.0,0.0,1.0,0.2);
        glVertex3d(max.x(),min.y(),min.z());
        glVertex3d(max.x(),min.y(),max.z());
        glVertex3d(min.x(),min.y(),max.z());

        glEnd();
      }
      break;
    }
  }
  post_draw(di);
}


void
GeomSimpleBox::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di,matl,1)) return;

  di->polycount_ += 6;

#ifdef SCI_NORM_OGL
  glEnable(GL_NORMALIZE);
#else
  glDisable(GL_NORMALIZE);
#endif

  glBegin(GL_QUADS);

  // top
  glNormal3f(0,0,1);
  glVertex3d(min.x(),min.y(),max.z());
  glVertex3d(max.x(),min.y(),max.z());
  glVertex3d(max.x(),max.y(),max.z());
  glVertex3d(min.x(),max.y(),max.z());

  // bottom
  glNormal3f(0,0,-1);
  glVertex3d(min.x(),min.y(),min.z());
  glVertex3d(min.x(),max.y(),min.z());
  glVertex3d(max.x(),max.y(),min.z());
  glVertex3d(max.x(),min.y(),min.z());
        
  // left
  glNormal3f(-1.0,0,0);
  glVertex3d(min.x(),min.y(),min.z());
  glVertex3d(min.x(),min.y(),max.z());
  glVertex3d(min.x(),max.y(),max.z());
  glVertex3d(min.x(),max.y(),min.z());

  // right
  glNormal3f(1,0,0);
  glVertex3d(max.x(),min.y(),min.z());
  glVertex3d(max.x(),max.y(),min.z());
  glVertex3d(max.x(),max.y(),max.z());
  glVertex3d(max.x(),min.y(),max.z());
                
  // top
  glNormal3f(0,1.0,0);
  glVertex3d(min.x(),max.y(),min.z());
  glVertex3d(min.x(),max.y(),max.z());
  glVertex3d(max.x(),max.y(),max.z());
  glVertex3d(max.x(),max.y(),min.z());

  // back
  glNormal3f(0,-1,0);
  glVertex3d(min.x(),min.y(),min.z());
  glVertex3d(max.x(),min.y(),min.z());
  glVertex3d(max.x(),min.y(),max.z());
  glVertex3d(min.x(),min.y(),max.z());

  glEnd();

  glEnable(GL_NORMALIZE);

  post_draw(di);
}


void
GeomCBox::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di,matl,1)) return;

  di->polycount_ += 6;

#ifdef SCI_NORM_OGL
  glEnable(GL_NORMALIZE);
#else
  glDisable(GL_NORMALIZE);
#endif

  glBegin(GL_QUADS);

  // top
  glNormal3f(0,0,1);
  glColor4f(0.0, 0.0, 1.0, 0.0);
  glVertex3d(min.x(),min.y(),max.z());
  glVertex3d(max.x(),min.y(),max.z());
  glVertex3d(max.x(),max.y(),max.z());
  glVertex3d(min.x(),max.y(),max.z());

  // bottom
  glNormal3f(0,0,-1);
  glColor4f(0.0, 0.0, 0.5, 0.0);
  glVertex3d(min.x(),min.y(),min.z());
  glVertex3d(min.x(),max.y(),min.z());
  glVertex3d(max.x(),max.y(),min.z());
  glVertex3d(max.x(),min.y(),min.z());
        
  // left
  glNormal3f(-1.0,0,0);
  glColor4f(0.5, 0.0, 0.0, 0.0);
  glVertex3d(min.x(),min.y(),min.z());
  glVertex3d(min.x(),min.y(),max.z());
  glVertex3d(min.x(),max.y(),max.z());
  glVertex3d(min.x(),max.y(),min.z());

  // right
  glNormal3f(1,0,0);
  glColor4f(1.0, 0.0, 0.0, 0.0);
  glVertex3d(max.x(),min.y(),min.z());
  glVertex3d(max.x(),max.y(),min.z());
  glVertex3d(max.x(),max.y(),max.z());
  glVertex3d(max.x(),min.y(),max.z());
                
  // top
  glNormal3f(0,1.0,0);
  glColor4f(0.0, 1.0, 0.0, 0.0);
  glVertex3d(min.x(),max.y(),min.z());
  glVertex3d(min.x(),max.y(),max.z());
  glVertex3d(max.x(),max.y(),max.z());
  glVertex3d(max.x(),max.y(),min.z());

  // back
  glNormal3f(0,-1,0);
  glColor4f(0.0, 0.5, 0.0, 0.0);
  glVertex3d(min.x(),min.y(),min.z());
  glVertex3d(max.x(),min.y(),min.z());
  glVertex3d(max.x(),min.y(),max.z());
  glVertex3d(min.x(),min.y(),max.z());

  glEnd();

  glEnable(GL_NORMALIZE);

  post_draw(di);
}


void
GeomBoxes::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  const bool ulr = edges_.size() == centers_.size();
  if (!ulr && global_edge_ < 1.0e-6) { return; }

  if (!pre_draw(di, matl, 1)) return;

  const bool using_texture =
    di->using_cmtexture_ && indices_.size() == centers_.size();
  if (using_texture)
  {
    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }

  const bool using_color = centers_.size() == colors_.size() / 4;

  glMatrixMode(GL_MODELVIEW);
  for (unsigned int i=0; i < centers_.size(); i++)
  {
    if (using_texture) { glTexCoord1f(indices_[i]); }
    if (using_color) { glColor3ubv(&(colors_[i*4])); }

    glPushMatrix();

    double edge = ulr ? edges_[i] : global_edge_;

    glTranslated(centers_[i].x()-edge/2.0,
                 centers_[i].y()-edge/2.0,
                 centers_[i].z()-edge/2.0);

    di->polycount_ += 6;

#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#else
    glDisable(GL_NORMALIZE);
#endif

    glBegin(GL_QUADS);

    // top
    glNormal3f(0,0,1);
    glVertex3d(0,0,edge);
    glVertex3d(edge,0,edge);
    glVertex3d(edge,edge,edge);
    glVertex3d(0,edge,edge);

    // bottom
    glNormal3f(0,0,-1);
    glVertex3d(0,0,0);
    glVertex3d(0,edge,0);
    glVertex3d(edge,edge,0);
    glVertex3d(edge,0,0);
        
    // left
    glNormal3f(-1.0,0,0);
    glVertex3d(0,0,0);
    glVertex3d(0,0,edge);
    glVertex3d(0,edge,edge);
    glVertex3d(0,edge,0);

    // right
    glNormal3f(1,0,0);
    glVertex3d(edge,0,0);
    glVertex3d(edge,edge,0);
    glVertex3d(edge,edge,edge);
    glVertex3d(edge,0,edge);
                
    // top
    glNormal3f(0,1.0,0);
    glVertex3d(0,edge,0);
    glVertex3d(0,edge,edge);
    glVertex3d(edge,edge,edge);
    glVertex3d(edge,edge,0);

    // back
    glNormal3f(0,-1,0);
    glVertex3d(0,0,0);
    glVertex3d(edge,0,0);
    glVertex3d(edge,0,edge);
    glVertex3d(0,0,edge);

    glEnd();

    glEnable(GL_NORMALIZE);

    glPopMatrix();
  }

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomTrianglesPC::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (points.size() == 0)
    return;
  if (!pre_draw(di,matl,1)) return;

  di->polycount_ += size();

  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#else
    glDisable(GL_NORMALIZE);
#endif
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      { 
        float *pts = &points[0];
        float *nrmls = &normals[0];
        float *clrs = &colors[0];
        int niter = size();
        glBegin(GL_TRIANGLES);
        while (niter--)
        {
          glNormal3fv(nrmls);
          nrmls+=3;

          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts += 3;

          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts+=3;

          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts+=3;
        }
        glEnd();
      }
      break;
    }
    glEnable(GL_NORMALIZE);
  }
  else
  { // lights are off, don't emit the normals
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      { 
        float *pts = &points[0];
        float *clrs = &colors[0];
        int niter = size();
        glBegin(GL_TRIANGLES);
        while (niter--)
        {
          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts += 3;

          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts+=3;

          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts+=3;
        }
        glEnd();
      }
      break;
    }
  }
  post_draw(di);
}


void
GeomTrianglesVP::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (points.size() == 0)
    return;
  if (!pre_draw(di,matl,1)) return;

  di->polycount_ += size();

  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#else
    glDisable(GL_NORMALIZE);
#endif
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      { 
        float *pts = &points[0];
        float *nrmls = &normals[0];
        int niter = size();
        glBegin(GL_TRIANGLES);
        while (niter--)
        {
          glNormal3fv(nrmls);
          nrmls+=3;
          glVertex3fv(pts);
          pts += 3;
          glNormal3fv(nrmls);
          nrmls+=3;
          glVertex3fv(pts);
          pts+=3;
          glNormal3fv(nrmls);
          nrmls+=3;
          glVertex3fv(pts);
          pts+=3;
        }
        glEnd();
      }
                
      break;
    }
    glEnable(GL_NORMALIZE);
  }
  else { // lights are off, don't emit the normals
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      { 
        float *pts = &points[0];
        int niter = size();
        glBegin(GL_TRIANGLES);
        while (niter--)
        {
          glVertex3fv(pts);
          pts += 3;
          glVertex3fv(pts);
          pts+=3;
          glVertex3fv(pts);
          pts+=3;
        }
        glEnd();
      }
      break;
    }
  }
  post_draw(di);
}


void
GeomTrianglesVPC::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (points.size() == 0)
    return;
  if (!pre_draw(di,matl,1)) return;

  di->polycount_ += size();

  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#else
    glDisable(GL_NORMALIZE);
#endif
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      { 
        float *pts = &points[0];
        float *nrmls = &normals[0];
        float *clrs = &colors[0];
        int niter = size();
        glBegin(GL_TRIANGLES);
        while (niter--)
        {
          glNormal3fv(nrmls);
          nrmls+=3;
          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts += 3;

          glNormal3fv(nrmls);
          nrmls+=3;
          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts+=3;

          glNormal3fv(nrmls);
          nrmls+=3;
          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts+=3;
        }
        glEnd();
      }
                
      break;
    }
    glEnable(GL_NORMALIZE);
  }
  else { // lights are off, don't emit the normals
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      { 
        float *pts = &points[0];
        float *clrs = &colors[0];
        int niter = size();
        glBegin(GL_TRIANGLES);
        while (niter--)
        {
          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts += 3;

          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts+=3;

          glColor3fv(clrs);
          clrs+=3;
          glVertex3fv(pts);
          pts+=3;
        }
        glEnd();
      }
      break;
    }
  }
  post_draw(di);
}


// WARNING not fixed for lighting correctly yet!

void
GeomTorusArc::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  glPushMatrix();
  glTranslated(cen.x(), cen.y(), cen.z());
  double matrix[16];
  matrix[0]=zero.x(); matrix[1]=zero.y(); matrix[2]=zero.z(); matrix[3]=0;
  matrix[4]=yaxis.x();matrix[5]=yaxis.y();matrix[6]=yaxis.z();matrix[7]=0;
  matrix[8]=axis.x(); matrix[9]=axis.y(); matrix[10]=axis.z();matrix[11]=0;
  matrix[12]=0;       matrix[13]=0;       matrix[14]=0;       matrix[15]=1;
  glMultMatrixd(matrix);
  di->polycount_+=2*(nu-1)*(nv-1);

  // Draw the torus
  double a1=start_angle;
  double a2=start_angle-arc_angle;
  if (a1 > a2)
  {
    double tmp=a1;
    a1=a2;
    a2=tmp;
  }
  SinCosTable tab1(nu, a1, a2);
  SinCosTable tab2(nv, 0, 2*Pi, rad2);
  SinCosTable tab2n(nv, 0, 2*Pi, rad2);
  int u,v;
  switch(di->get_drawtype())
  {
  case DrawInfoOpenGL::WireFrame:
    {
      double srx=tab1.sin(0);
      double sry=tab1.cos(0);
      glBegin(GL_LINE_LOOP);
      for (v=1;v<nv;v++)
      {
        double sz=tab2.cos(v);
        double srad=rad1+tab2.sin(v);
        double sx=srx*srad;
        double sy=sry*srad;
        glVertex3d(sx, sy, sz);
        glVertex3d(srx*rad1, sry*rad1, 0);
      }
      glEnd();

      srx=tab1.sin(nu-1);
      sry=tab1.cos(nu-1);
      glBegin(GL_LINE_LOOP);
      for (v=1;v<nv;v++)
      {
        double sz=tab2.cos(v);
        double srad=rad1+tab2.sin(v);
        double sx=srx*srad;
        double sy=sry*srad;
        glVertex3d(sx, sy, sz);
        glVertex3d(srx*rad1, sry*rad1, 0);
      }
      glEnd();
        
      for (u=0;u<nu;u++)
      {
        double rx=tab1.sin(u);
        double ry=tab1.cos(u);
        glBegin(GL_LINE_LOOP);
        for (v=1;v<nv;v++)
        {
          double z=tab2.cos(v);
          double rad=rad1+tab2.sin(v);
          double x=rx*rad;
          double y=ry*rad;
          glVertex3d(x, y, z);
        }
        glEnd();
      }
      for (v=1;v<nv;v++)
      {
        double z=tab2.cos(v);
        double rr=tab2.sin(v);
        glBegin(GL_LINE_LOOP);
        for (u=1;u<nu;u++)
        {
          double rad=rad1+rr;
          double x=tab1.sin(u)*rad;
          double y=tab1.cos(u)*rad;
          glVertex3d(x, y, z);
        }
        glEnd();
      }
    }
    break;
  case DrawInfoOpenGL::Flat:
    for (v=0;v<nv-1;v++)
    {
      double z1=tab2.cos(v);
      double rr1=tab2.sin(v);
      double z2=tab2.cos(v+1);
      double rr2=tab2.sin(v+1);
      glBegin(GL_TRIANGLE_STRIP);
      for (u=0;u<nu;u++)
      {
        double r1=rad1+rr1;
        double r2=rad1+rr2;
        double xx=tab1.sin(u);
        double yy=tab1.cos(u);
        double x1=xx*r1;
        double y1=yy*r1;
        double x2=xx*r2;
        double y2=yy*r2;
        glVertex3d(x1, y1, z1);
        glVertex3d(x2, y2, z2);
      }
      glEnd();
    }
    break;
  case DrawInfoOpenGL::Gouraud:
    for (v=0;v<nv-1;v++)
    {
      double z1=tab2.cos(v);
      double rr1=tab2.sin(v);
      double z2=tab2.cos(v+1);
      double rr2=tab2.sin(v+1);
      double nr=-tab2n.sin(v);
      double nz=-tab2n.cos(v);
      glBegin(GL_TRIANGLE_STRIP);
      for (u=0;u<nu;u++)
      {
        double r1=rad1+rr1;
        double r2=rad1+rr2;
        double xx=tab1.sin(u);
        double yy=tab1.cos(u);
        double x1=xx*r1;
        double y1=yy*r1;
        double x2=xx*r2;
        double y2=yy*r2;
        glNormal3d(nr*xx, nr*yy, nz);
        glVertex3d(x1, y1, z1);
        glVertex3d(x2, y2, z2);
      }
      glEnd();
    }
    break;
  }
  glPopMatrix();
  post_draw(di);
}


void
GeomTriStrip::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  if (verts.size() <= 2)
    return;
  di->polycount_+=verts.size()-2;
  if (di->currently_lit_)
  {
    glDisable(GL_NORMALIZE);
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
      {
        verts[0]->emit_all(di);
        verts[1]->emit_all(di);
        for (int i=2;i<verts.size();i++)
        {
          glBegin(GL_LINE_LOOP);
          verts[i-2]->emit_all(di);
          verts[i-1]->emit_all(di);
          verts[i]->emit_all(di);
          glEnd();
        }
      }
      break;
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      {
        glBegin(GL_TRIANGLE_STRIP);
        for (int i=0;i<verts.size();i++)
        {
          verts[i]->emit_all(di);
        }
        glEnd();
      }
      break;
    }
    glEnable(GL_NORMALIZE);
  }
  else
  {
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
      {
        verts[0]->emit_material(di);
        verts[0]->emit_point(di);
        verts[1]->emit_material(di);
        verts[1]->emit_point(di);
        for (int i=2;i<verts.size();i++)
        {
          glBegin(GL_LINE_LOOP);
          verts[i-2]->emit_material(di);
          verts[i-2]->emit_point(di);
          verts[i-1]->emit_material(di);
          verts[i-1]->emit_point(di);
          verts[i]->emit_material(di);
          verts[i]->emit_point(di);
          glEnd();
        }
      }
      break;
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      {
        glBegin(GL_TRIANGLE_STRIP);
        for (int i=0;i<verts.size();i++)
        {
          verts[i]->emit_material(di);
          verts[i]->emit_point(di);
        }
        glEnd();
      }
      break;
    }
  }
  post_draw(di);
}


void
GeomTriStripList::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (pts.size() == 0)
    return;
  if (!pre_draw(di, matl, 1)) return;

  di->polycount_ += size();
  if (di->currently_lit_)
  {
#ifdef SCI_NORM_OGL     
    glEnable(GL_NORMALIZE);
#else
    glDisable(GL_NORMALIZE);
#endif
    switch(di->get_drawtype())
    {
    case DrawInfoOpenGL::WireFrame:
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
      {
        int nstrips = strips.size();
        int index=0;
        float *rpts = &pts[0];
        float *nrm = &nrmls[0];
        for (int ns = 0;ns < nstrips; ns++)
        {
          glBegin(GL_TRIANGLE_STRIP);
          glVertex3fv(rpts);
          rpts += 3;
          glVertex3fv(rpts);
          rpts += 3;
          index += 6;
          while (index < strips[ns])
          {
            glNormal3fv(nrm);
            nrm += 3;
            index += 3;
            glVertex3fv(rpts);
            rpts += 3;
          }
          glEnd();
        }
      }
      break;
    }
#ifndef SCI_NORM_OGL
    glEnable(GL_NORMALIZE);
#endif

  }
  else
  {
    int nstrips = strips.size();
    int index=0;
    float *rpts = &pts[0];
    for (int ns = 0;ns < nstrips; ns++)
    {
      glBegin(GL_TRIANGLE_STRIP);
      glVertex3fv(rpts);
      rpts += 3;
      glVertex3fv(rpts);
      rpts += 3;
      index += 6;
      while (index < strips[ns])
      {
        index += 3;
        glVertex3fv(rpts);
        rpts += 3;
      }
      glEnd();
    }
  }
  post_draw(di);
}


void
GeomVertex::emit_all(DrawInfoOpenGL*)
{
  glVertex3d(p.x(), p.y(), p.z());
}


void
GeomVertex::emit_point(DrawInfoOpenGL*)
{
  glVertex3d(p.x(), p.y(), p.z());
}


void
GeomVertex::emit_material(DrawInfoOpenGL*)
{
  // Do nothing
}


void
GeomVertex::emit_normal(DrawInfoOpenGL*)
{
  // Do nothing
}


void
GeomNVertex::emit_all(DrawInfoOpenGL*)
{
  glNormal3d(normal.x(), normal.y(), normal.z());
  glVertex3d(p.x(), p.y(), p.z());
}


void
GeomNVertex::emit_normal(DrawInfoOpenGL*)
{
  glNormal3d(normal.x(), normal.z(), normal.z());
}


void
GeomNMVertex::emit_all(DrawInfoOpenGL* di)
{
  di->set_material(matl.get_rep());
  glNormal3d(normal.x(), normal.y(), normal.z());
  glVertex3d(p.x(), p.y(), p.z());
}


void
GeomNMVertex::emit_material(DrawInfoOpenGL* di)
{
  di->set_material(matl.get_rep());
}


void
GeomMVertex::emit_all(DrawInfoOpenGL* di)
{
  di->set_material(matl.get_rep());
  glVertex3d(p.x(), p.y(), p.z());
}


void
GeomMVertex::emit_material(DrawInfoOpenGL* di)
{
  di->set_material(matl.get_rep());
}


void
GeomCVertex::emit_all(DrawInfoOpenGL* /*di*/)
{
  glColor3f(color.r(),color.g(),color.b());
  glVertex3d(p.x(), p.y(), p.z());
}


void
GeomCVertex::emit_material(DrawInfoOpenGL* /*di*/)
{
  glColor3f(color.r(),color.g(),color.b());
}


void
Light::opengl_reset_light(int i)
{
  float f[4];
  f[0]=0.0; f[1]=0.0; f[2]=0.0; f[3]=1.0;
  glLightfv((GLenum)(GL_LIGHT0+i), GL_AMBIENT, f);

  if ( i != 0 )
  {
    glLightfv((GLenum)(GL_LIGHT0+i), GL_DIFFUSE, f);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_SPECULAR, f);
  }
  else
  {
    f[0] = 1.0; f[1]=1.0; f[2]=1.0; f[3]=1.0;
    glLightfv((GLenum)(GL_LIGHT0+i), GL_DIFFUSE, f);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_SPECULAR, f);
  }
  f[0]=0.0; f[1]=0.0; f[2]=-1.0; f[3]=1.0;
  glLightfv((GLenum)(GL_LIGHT0+i), GL_POSITION, f);

  glLightfv((GLenum)(GL_LIGHT0+i), GL_SPOT_DIRECTION, f);
  f[0] = 180.0;
  glLightfv((GLenum)(GL_LIGHT0+i), GL_SPOT_CUTOFF, f);
  f[0] = 0.0;
  glLightfv((GLenum)(GL_LIGHT0+i), GL_SPOT_EXPONENT, f);
  glLightfv((GLenum)(GL_LIGHT0+i), GL_LINEAR_ATTENUATION, f);
  glLightfv((GLenum)(GL_LIGHT0+i), GL_QUADRATIC_ATTENUATION, f);
  f[0] = 1.0;
  glLightfv((GLenum)(GL_LIGHT0+i), GL_CONSTANT_ATTENUATION, f);

}


void
SpotLight::opengl_setup( const View&, DrawInfoOpenGL*, int& idx)
{
  if (on )
  {
    int i = idx++;
    float f[4];

    opengl_reset_light( i );

    if ( !transformed )
    {
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
    }

    f[0]=p.x(); f[1]=p.y(); f[2]=p.z(); f[3]=1.0;
    glLightfv((GLenum)(GL_LIGHT0+i), GL_POSITION, f);
    f[0]=v.x(); f[1]=v.y(); f[2]=v.z(); f[3]=0.0;
    glLightfv((GLenum)(GL_LIGHT0+i), GL_SPOT_DIRECTION, f);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_SPOT_CUTOFF, &cutoff);
    c.get_color(f);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_DIFFUSE, f);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_SPECULAR, f);

    if ( !transformed )
    {
      glPopMatrix();
    }
  }
}


void
DirectionalLight::opengl_setup( const View&, DrawInfoOpenGL*, int& idx)
{
  if (on )
  {
    const int i = idx++;
    float f[4];

    opengl_reset_light( i );

    if ( !transformed )
    {
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
    }

    f[0]=v.x(); f[1]=v.y(); f[2]=v.z(); f[3]=0.0;
    glLightfv((GLenum)(GL_LIGHT0+i), GL_POSITION, f);
    c.get_color(f);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_DIFFUSE, f);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_SPECULAR, f);

    if ( !transformed )
    {
      glPopMatrix();
    }
  }
}


void
PointLight::opengl_setup(const View&, DrawInfoOpenGL*, int& idx)
{
  if ( on )
  {
    const int i = idx++;
    float f[4];

    opengl_reset_light( i );

    if ( !transformed )
    {
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
    }
    f[0]=p.x(); f[1]=p.y(); f[2]=p.z(); f[3]=1.0;
    glLightfv((GLenum)(GL_LIGHT0+i), GL_POSITION, f);
    c.get_color(f);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_DIFFUSE, f);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_SPECULAR, f);

    if ( !transformed )
    {
      glPopMatrix();
    }
  }
}


void
HeadLight::opengl_setup(const View& /*view*/, DrawInfoOpenGL*, int& idx)
{
  if ( on )
  {
    const int i = idx++;
    float f[4];

    opengl_reset_light(i);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity(); // Never transformed: its a headlight afterall--Kurt

    f[0] = f[1] = f[3] = 0.0;
    f[2] = 1.0;
    glLightfv((GLenum)(GL_LIGHT0+i), GL_POSITION, f);
    c.get_color(f);     
    glLightfv((GLenum)(GL_LIGHT0+i), GL_DIFFUSE, f);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_SPECULAR, f);

    glPopMatrix();
  }
}


//----------------------------------------------------------------------
void
GeomIndexedGroup::draw(DrawInfoOpenGL* di, Material* m, double time)
{
  MapIntGeomObj::iterator iter;
  for (iter = objs.begin(); iter != objs.end(); iter++)
  {
    (*iter).second->draw(di, m, time);
  }
}




//----------------------------------------------------------------------
void
GeomText::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di,matl,0)) return;

  const int fontindex = 2;

  if (!di->init_font(fontindex))
  {
    post_draw(di);
    return;
  }

  glColor3f(c.r(), c.g(), c.b());
  glDisable(GL_LIGHTING);
  glRasterPos3d( at.x(), at.y(), at.z() );
  glPushAttrib (GL_LIST_BIT);
  glListBase(di->fontbase_[fontindex]);
  glCallLists(text.size(), GL_UNSIGNED_BYTE, (GLubyte *)text.c_str());
  glPopAttrib ();
  post_draw(di);
}


//----------------------------------------------------------------------
void
GeomTexts::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di,matl,0)) return;

  if (!di->init_font(fontindex_))
  {
    post_draw(di);
    return;
  }

  const bool coloring = color_.size() == location_.size();
  bool indexing = false;
  if (di->using_cmtexture_ && index_.size() == location_.size())
  {
    indexing = true;
    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }

  glDisable(GL_LIGHTING);
  glPushAttrib(GL_LIST_BIT);
  for (unsigned int i = 0; i < text_.size(); i++)
  {
    if (coloring) { glColor3f(color_[i].r(), color_[i].g(), color_[i].b()); }
    if (indexing) { glTexCoord1f(index_[i]); }

    glRasterPos3d( location_[i].x(), location_[i].y(), location_[i].z() );
    glListBase(di->fontbase_[fontindex_]);
    glCallLists(text_[i].size(), GL_UNSIGNED_BYTE,
                (GLubyte *)text_[i].c_str());
  }
  glPopAttrib ();

  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


//----------------------------------------------------------------------
void
GeomTextsCulled::draw(DrawInfoOpenGL* di, Material* matl, double)
{

  if (!pre_draw(di,matl,0)) return;

  if (!di->init_font(fontindex_))
  {
    post_draw(di);
    return;
  }

  const bool coloring = color_.size() == location_.size();
  bool indexing = false;
  if (di->using_cmtexture_ && index_.size() == location_.size())
  {
    indexing = true;
    glColor3d(di->diffuse_scale_, di->diffuse_scale_, di->diffuse_scale_);

    glEnable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_1D, di->cmtexture_);
  }

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glPushAttrib(GL_LIST_BIT);

  double mat[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, mat);
  const Vector view (mat[2], mat[6], mat[10]);
  for (unsigned int i = 0; i < text_.size(); i++)
  {
    if (Dot(view, normal_[i]) > 0)
    {
      if (coloring) { glColor3f(color_[i].r(), color_[i].g(), color_[i].b()); }
      if (indexing) { glTexCoord1f(index_[i]); }

      glRasterPos3d( location_[i].x(), location_[i].y(), location_[i].z() );
      glListBase(di->fontbase_[fontindex_]);
      glCallLists(text_[i].size(), GL_UNSIGNED_BYTE,
                  (GLubyte *)text_[i].c_str());
    }
  }

  glPopAttrib ();
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_TEXTURE_1D);

  post_draw(di);
}


void
GeomTextTexture::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di,matl,0)) return;

  glColor4d(1.0,1.0,1.0,1.0);
  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_TEXTURE_2D);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  double trans[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, trans);
  Transform modelview;
  modelview.set_trans(trans);
  build_transform(modelview);
  transform_.get_trans(trans);
  glMultMatrixd((GLdouble *)trans);
  render();
  glPopMatrix();
  glDepthMask(GL_TRUE);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_BLEND);
  post_draw(di);
}


void
HistogramTex::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 0)) return;
  static GLuint texName = 0;
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  if ( !glIsTexture( texName ) )
  {
    glGenTextures(1, &texName);
    glBindTexture(GL_TEXTURE_1D, texName);

    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    glTexImage1D( GL_TEXTURE_1D, 0, GL_RGB, 256, 0,
                  GL_RGBA, GL_FLOAT, texture );
  }
  else
  {
    glBindTexture(GL_TEXTURE_1D, texName);
  }

  int vp[4];
  glGetIntegerv( GL_VIEWPORT, vp );
  glClearColor(0.0, 0.0, 0.0, 0.0);

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);

  double dx = 1.0/nbuckets;
  double ddx = (b.x() - a.x())/nbuckets;
  double y_max = log(float(1+buckets[max]));

  glColor3f(0.5,0.5,0.5);
  glBegin( GL_QUADS );
  for (int i = 0; i <nbuckets; i++)
  {
    float bval = log(float(1+buckets[i]));
    glTexCoord2f(dx*i, 0.0);
    glVertex3f( a.x() + ddx*i, a.y(), a.z() );
    glVertex3f( a.x() + ddx*(i+1), b.y(), b.z() );
    glVertex3f( a.x() + ddx*(i+1), a.y() + (c.y()-a.y())*bval/y_max, c.z() );
    glVertex3f( a.x() + ddx*(i+1), b.y() + (d.y()-b.y())*bval/y_max, d.z() );
  }
  glEnd();
  glFlush();
  glDisable(GL_TEXTURE_1D);
  glEnable(GL_LIGHTING);
  post_draw(di);
}


void
TexSquare::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  double old_ambient = di->ambient_scale_;
  di->ambient_scale_ = 15;
  if (!pre_draw(di, matl, 1)) {
    di->ambient_scale_ = old_ambient;
    return;
  }
  glEnable(GL_TEXTURE_2D);
  bool bound = glIsTexture(texname_);
  if (!bound)
    glGenTextures(1, &texname_);
  glBindTexture(GL_TEXTURE_2D, texname_);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  if (!bound && texture)
  {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelTransferi(GL_MAP_COLOR,0);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, texture);
  }
  if (GL_NO_ERROR == glGetError())
  {
    glAlphaFunc(GL_GEQUAL, alpha_cutoff_);
    glEnable(GL_ALPHA_TEST);
    glDisable(GL_BLEND);
    glColor4d(1., 1., 1., 1.);
    glBegin( GL_QUADS );
    for (int i = 0; i < 4; i++)
    {
      glNormal3d(normal_.x(), normal_.y(), normal_.z());
      glTexCoord2fv(tex_coords_+i*2);
      glVertex3fv(pos_coords_+i*3);
    }
    glEnd();
    glDisable(GL_ALPHA_TEST);
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
  di->ambient_scale_ = old_ambient;
  post_draw(di);
}


#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#define FRAG \
"!!ARBfp1.0 \n" \
"ATTRIB t = fragment.texcoord[1]; \n" \
"TEMP c; \n" \
"ATTRIB cf = fragment.color; \n" \
"TEX c, t, texture[1], 2D; \n" \
"MUL c, c, cf; \n" \
"MOV result.color, c; \n" \
"END"

#define FOG \
"!!ARBfp1.0 \n" \
"PARAM fc = state.fog.color; \n" \
"PARAM fp = state.fog.params; \n" \
"TEMP fctmp; \n" \
"ATTRIB t = fragment.texcoord[1]; \n" \
"ATTRIB tf = fragment.texcoord[2]; \n" \
"TEMP c; \n" \
"TEMP v; \n" \
"ATTRIB cf = fragment.color; \n" \
"TEX c, t, texture[1], 2D; \n" \
"MUL c, c, cf; \n" \
"SUB v.x, fp.z, tf.x; \n" \
"MUL_SAT v.x, v.x, fp.w; \n" \
"MUL fctmp, c.w, fc; \n" \
"LRP c.xyz, v.x, c.xyzz, fctmp.xyzz; \n" \
"MOV result.color, c; \n" \
"END"
#endif


void
GeomTexRectangle::draw(DrawInfoOpenGL* di, Material* matl, double)
{
  if (!pre_draw(di, matl, 1)) return;
  GLboolean use_fog = glIsEnabled(GL_FOG);
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if ( !shader_ || !fog_shader_ )
  {
    shader_ = new FragmentProgramARB( FRAG );
    shader_->create();
    fog_shader_ = new FragmentProgramARB( FOG );
    fog_shader_->create();
  }

  if (use_fog)
  {
#ifdef _WIN32
    if (glActiveTexture)
#endif
    {
      // enable texture unit 2 for fog
      glActiveTexture(GL_TEXTURE2);
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      glEnable(GL_TEXTURE_3D);
    }
  }

#ifdef _WIN32
  if (glActiveTexture)
#endif
    glActiveTexture(GL_TEXTURE1);
#endif

  bool bound = glIsTexture(texname_);

  if (!bound)
  {
    glGenTextures(1, &texname_);
  }

  glBindTexture(GL_TEXTURE_2D, texname_);

  if (!bound)
  {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, texture_ );
  }

  if (GL_NO_ERROR == glGetError())
  {
    glEnable(GL_TEXTURE_2D);

    if (interp_)
    {
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    else
    {
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }

#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
    if (use_fog)
    {
      fog_shader_->bind();
    }
    else
    {
      shader_->bind();
    }
#endif

    GLfloat ones[4] = {1.0, 1.0, 1.0, 1.0};
    glColor4fv(ones);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glDepthMask(GL_TRUE);

    if (trans_)
    {
#if !defined(GL_ARB_fragment_program) && !defined(GL_ATI_fragment_shader)
      glAlphaFunc(GL_GREATER, alpha_cutoff_);
      glEnable(GL_ALPHA_TEST);
#endif
      glEnable(GL_BLEND);

      // Workaround for old bad nvidia headers.
#if defined(GL_FUNC_ADD)
      glBlendEquation(GL_FUNC_ADD);
#else
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
#endif
    }

    float mvmat[16];
    if (use_fog)
    {
      glGetFloatv(GL_MODELVIEW_MATRIX, mvmat);
    }

    glBegin( GL_QUADS );
    if ( use_normal_ )
    {
      glNormal3fv( normal_ );
    }

    for (int i = 0; i < 4; i++)
    {
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#  ifdef _WIN32
      if (glMultiTexCoord2fv && glMultiTexCoord3f)
      {
#  endif
        glMultiTexCoord2fv(GL_TEXTURE1,tex_coords_+i*2);
        if (use_fog)
        {
          float *pos = pos_coords_+i*3;
          float vz = mvmat[2]* pos[0]
            + mvmat[6]*pos[1]
            + mvmat[10]*pos[2] + mvmat[14];
          glMultiTexCoord3f(GL_TEXTURE2, -vz, 0.0, 0.0);
        }
#  ifdef _WIN32
      }
      else
      {
        glTexCoord2fv(tex_coords_+i*2);
      }
#  endif // _WIN32
#else
      glTexCoord2fv(tex_coords_+i*2);
#endif
      glVertex3fv(pos_coords_+i*3);
    }
    glEnd();

    glFlush();
    if ( trans_ )
    {
      glDisable(GL_ALPHA_TEST);
    }
    glDisable(GL_BLEND);
    glDisable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
    if ( use_fog )
    {
      fog_shader_->release();
#ifdef _WIN32
      if (glActiveTexture)
#endif
        glActiveTexture(GL_TEXTURE2);
      glDisable(GL_TEXTURE_3D);
    }
    else
    {
      shader_->release();
    }
#endif

  }
  else
  {
    cerr<<"Some sort of texturing error\n";
  }

#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#ifdef _WIN32
  if (glActiveTexture)
#endif
    glActiveTexture(GL_TEXTURE0);
#endif
  di->polycount_++;
  post_draw(di);
}


void
GeomSticky::draw(DrawInfoOpenGL* di, Material* matl, double t)
{
  if (!pre_draw(di, matl, 0)) return;

  int ii = 0;
  // Disable clipping planes for sticky objects.
  vector<bool> cliplist(6, false);
  for (ii = 0; ii < 6; ii++)
  {
    if (glIsEnabled((GLenum)(GL_CLIP_PLANE0+ii)))
    {
      glDisable((GLenum)(GL_CLIP_PLANE0+ii));
      cliplist[ii] = true;
    }
  }
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glDisable(GL_DEPTH_TEST);
  glRasterPos2d(0.55, -0.98);

  child_->draw(di,matl,t);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  // Reenable clipping planes.
  for (ii = 0; ii < 6; ii++)
  {
    if (cliplist[ii])
    {
      glEnable((GLenum)(GL_CLIP_PLANE0+ii));
    }
  }
  post_draw(di);
}


} // End namespace SCIRun
