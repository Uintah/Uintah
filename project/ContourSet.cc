/*
 *  ContourSet.cc: The ContourSet Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <ContourSet.h>
#include <Surface.h>
#include <Geom.h>
#include <MtXEventLoop.h>
#include <GL/glu.h>

#define Sqr(x) ((x)*(x))

extern MtXEventLoop* evl;

Vector mmult(double *m, const Vector &v) {
    double x[3], y[3];
    x[0]=v.x();x[1]=v.y();x[2]=v.z();
    for (int i=0; i<3; i++) {
	y[i]=0;
	for (int j=0; j<3; j++) {
	    y[i]+=m[j*4+i]*x[j];
	}
    }
    return Vector(y[0],y[1],y[2]);
}

ContourSet::ContourSet()
{
    basis[0]=Vector(1,0,0);
    basis[1]=Vector(0,1,0);
    basis[2]=Vector(0,0,1);
    origin=Vector(0,0,0);
    space=1;
};

ContourSet::ContourSet(const ContourSet &copy)
: space(copy.space), origin(copy.origin)
{
    basis[0]=copy.basis[0];
    basis[1]=copy.basis[1];
    basis[2]=copy.basis[2];
}

ContourSet::~ContourSet() {
}

void ContourSet::translate(const Vector &v) {
    origin=origin+v;
}

void ContourSet::scale(double sc) {
    basis[0]=basis[0]*sc;
    basis[1]=basis[1]*sc;
    basis[2]=basis[2]*sc;
}

// just takes the (dx, dy, dz) vector as input -- read off dials...
void ContourSet::rotate(const Vector &rot) {
    evl->lock();
    glMatrixMode(GL_MODELVIEW_MATRIX);
    glPushMatrix();
    glLoadIdentity();
    glRotated(rot.x(), 1, 0, 0);
    glRotated(rot.y(), 0, 1, 0);
    glRotated(rot.z(), 0, 0, 1);
    double mm[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, mm);
    glPopMatrix();
    basis[0]=mmult(mm, basis[0]);
    basis[1]=mmult(mm, basis[1]);
    basis[2]=mmult(mm, basis[2]);
}

void ContourSet::io(Piostream& stream) 
{
}
