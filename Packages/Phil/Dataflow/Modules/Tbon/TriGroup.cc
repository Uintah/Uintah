
/*
 * TriGroup.cc:  Group of triangles
 *    (stripped down to improve performance)
 *
 * Packages/Philip Sutton
 * April 1999

  Copyright (C) 2000 SCI Group, University of Utah
 */

#include "TriGroup.h"
#include <Core/Util/NotFinished.h>
#include <Core/Geom/Material.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <Core/Geom/GeomOpenGL.h>
#include <unistd.h>

namespace Phil {
using namespace std;

GeomTriGroup::GeomTriGroup( int n ) : GeomObj() {
  tris = new Triangle[n];
  size = 0;
  nalloc = n;
}

GeomTriGroup::GeomTriGroup() : GeomObj() {
  size = 0;
  nalloc = 0;

#if 0
  tris = new Triangle[96];
#endif
}

void
GeomTriGroup::reserve_clear( int n ) {
#if 1
  if( n > nalloc ) {
    if( nalloc > 0 ) 
      delete [] tris;
    int fudge = (int)(0.1 * n);
    tris = new Triangle[n+fudge];
    nalloc = n+fudge;
  }
#endif
  size = 0;
}


GeomTriGroup::GeomTriGroup( const GeomTriGroup& copy ) : GeomObj() {
  size = copy.size;
  tris = new Triangle[size];
  for( int i = 0; i < size; i++ )
    tris[i] = copy.tris[i];
}

GeomTriGroup::~GeomTriGroup() {
  if( size > 0 ) {
    delete [] tris;
  }
}

void
GeomTriGroup::add( const pPoint& p1, const pPoint& p2, const pPoint& p3 ) {
  Triangle& tri = tris[size];
  tri.vertices[0] = p1;
  tri.vertices[1] = p2;
  tri.vertices[2] = p3;
  tri.normal = Cross(p2-p1, p3-p1);
  size++;
#if 0
  if( size == 96 ) {
    // write to disk if needed, then
    size = 0;
  }
#endif
}


void
GeomTriGroup::write( const char* filename ) {
  int i;
  if( size == 0 )
    return;
  FILE* file = fopen( filename, "w" );

  // write in Wavefront OBJ format (ASCII)
  // write vertices
  for( i = 0; i < size; i++ ) {
    float x, y, z;
    x = (float)tris[i].vertices[0].x();
    y = (float)tris[i].vertices[0].y();
    z = (float)tris[i].vertices[0].z();
    fprintf( file, "v  %f %f %f\n", x, y, z );
    x = (float)tris[i].vertices[1].x();
    y = (float)tris[i].vertices[1].y();
    z = (float)tris[i].vertices[1].z();
    fprintf( file, "v  %f %f %f\n", x, y, z );
    x = (float)tris[i].vertices[2].x();
    y = (float)tris[i].vertices[2].y();
    z = (float)tris[i].vertices[2].z();
    fprintf( file, "v  %f %f %f\n", x, y, z );
  }

  // write normals
  for( i = 0; i < size; i++ ) {
    fprintf( file, "vn  %f %f %f\n", tris[i].normal.x(), tris[i].normal.y(), 
	     tris[i].normal.z() );
  }

  // write faces
  int last = 1;
  for( i = 1; i <= size; i++ ) {
    fprintf( file, "f %d%c%c%d %d%c%c%d %d%c%c%d\n", last, '/', '/', i, 
	     last+1, '/', '/', i, last+2, '/', '/', i );
    last += 3;
  }

  // write in OFF format
  /*
  // write header info
  fprintf( file, "OFF BINARY\n" );
  int val = size*3;
  fwrite( &val, sizeof(int), 1, file );
  val = size;
  fwrite( &val, sizeof(int), 1, file );
  val = 0;
  fwrite( &val, sizeof(int), 1, file );

  // write points
  for( i = 0; i < size; i++ ) {
    float pt = (float)tris[i].vertices[0].x();
    fwrite( &pt, sizeof(float), 1, file );
    pt = (float)tris[i].vertices[0].y();
    fwrite( &pt, sizeof(float), 1, file );
    pt = (float)tris[i].vertices[0].z();
    fwrite( &pt, sizeof(float), 1, file );
    pt = (float)tris[i].vertices[1].x();
    fwrite( &pt, sizeof(float), 1, file );
    pt = (float)tris[i].vertices[1].y();
    fwrite( &pt, sizeof(float), 1, file );
    pt = (float)tris[i].vertices[1].z();
    fwrite( &pt, sizeof(float), 1, file );
    pt = (float)tris[i].vertices[2].x();
    fwrite( &pt, sizeof(float), 1, file );
    pt = (float)tris[i].vertices[2].y();
    fwrite( &pt, sizeof(float), 1, file );
    pt = (float)tris[i].vertices[2].z();
    fwrite( &pt, sizeof(float), 1, file );
  }
  
  // write triangles
  for( i = 0; i < size; i++ ) {
    val = 3;
    fwrite( &val, sizeof(int), 1, file );
    val = i*3;
    fwrite( &val, sizeof(int), 1, file );
    val = i*3+1;
    fwrite( &val, sizeof(int), 1, file );
    val = i*3+2;
    fwrite( &val, sizeof(int), 1, file );
    val = 0;
    fwrite( &val, sizeof(int), 1, file );
  }
  */
  fclose(file);
}

void
GeomTriGroup::shift( double x, double y, double z ) {
  int i;
  for( i = 0; i < size; i++ ) {
    tris[i].vertices[0] = tris[i].vertices[0] + pPoint(x, y, z);
    tris[i].vertices[1] = tris[i].vertices[1] + pPoint(x, y, z);
    tris[i].vertices[2] = tris[i].vertices[2] + pPoint(x, y, z);
  }

}

void 
GeomTriGroup::get_bounds(BBox& box) {
  if( size > 0 ) {
    for( int i = 0; i < size; i++ ) {
      box.extend( Point( (double)tris[i].vertices[0].x(), 
			 (double)tris[i].vertices[0].y(),
			 (double)tris[i].vertices[0].z() ) );
      box.extend( Point( (double)tris[i].vertices[1].x(), 
			 (double)tris[i].vertices[1].y(),
			 (double)tris[i].vertices[1].z() ) );
      box.extend( Point( (double)tris[i].vertices[2].x(), 
			 (double)tris[i].vertices[2].y(),
			 (double)tris[i].vertices[2].z() ) );
    } 
  } else {
    box.extend( Point(-1,-1,-1) );
    box.extend( Point(1,1,1) );
  }
}


void 
GeomTriGroup::draw(DrawInfoOpenGL* di, Material* matl, double) {
  pre_draw(di, matl, 1);
  di->polycount += size;
  if( matl->transparency == 0.0 ) {
    glEnable(GL_NORMALIZE);
    //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
    glBegin(GL_TRIANGLES);
    for( int i = 0; i < size; i++ ) {
      glNormal3f( tris[i].normal.x(), tris[i].normal.y(), tris[i].normal.z() );
      glVertex3f( tris[i].vertices[0].x(), tris[i].vertices[0].y(),
		  tris[i].vertices[0].z() );
      glVertex3f( tris[i].vertices[1].x(), tris[i].vertices[1].y(),
		  tris[i].vertices[1].z() );
      glVertex3f( tris[i].vertices[2].x(), tris[i].vertices[2].y(),
		  tris[i].vertices[2].z() );
    }
    glEnd();
  } else {
    double model_mat[16]; // this is the modelview matrix
    glGetDoublev(GL_MODELVIEW_MATRIX,model_mat);
    
    // this is what you rip the view vector from
    // just use the "Z" axis, normalized
    Vector view = Vector(model_mat[0*4+2],model_mat[1*4+2],model_mat[2*4+2]);
    
    view.normalize();
    cout << "view = " << view << endl;
    
    glEnable(GL_NORMALIZE);
    //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
    glBegin(GL_TRIANGLES);

    float color[4];
    matl->diffuse.get_color(color);
    color[3] = matl->transparency;
    //      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    //    glClearColor(0.0, 0.0, 0.0, 0.0);
    //    glClear(GL_COLOR_BUFFER_BIT);

    for( int i = 0; i < size; i++ ) {
      glColor4fv(color);
      glNormal3f( tris[i].normal.x(), tris[i].normal.y(), tris[i].normal.z() );
      glVertex3f( tris[i].vertices[0].x(), tris[i].vertices[0].y(),
		  tris[i].vertices[0].z() );
      glVertex3f( tris[i].vertices[1].x(), tris[i].vertices[1].y(),
		  tris[i].vertices[1].z() );
      glVertex3f( tris[i].vertices[2].x(), tris[i].vertices[2].y(),
		  tris[i].vertices[2].z() );
    }
    
    glDisable(GL_BLEND);
    glEnd();
  }
}


static Persistent* make_GeomTriGroup() {
  return new GeomTriGroup();
}

PersistentTypeID GeomTriGroup::type_id("GeomTriGroup", "GeomObj", make_GeomTriGroup);

GeomObj*
GeomTriGroup::clone() {
  return new GeomTriGroup(*this);
}

#define GEOMTRIGROUP_VERSION 1

void
GeomTriGroup::io(Piostream& stream) {
  NOT_FINISHED("GeomTriGroup::io");
}

bool
GeomTriGroup::saveobj(std::ostream& out, const clString& format,
			GeomSave* saveinfo)
{
  NOT_FINISHED("GeomTriGroup::saveobj");
  return 0;
}
} // End namespace Phil


