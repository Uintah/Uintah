#include <gfx/std.h>
#include <gfx/math/random.h>
#include <gfx/geom/3D.h>

Vec3 randomPoint(const Vec3& v1, const Vec3& v2)
{
    real a = random1();

    return a*v1 + (1-a)*v2;
}

Vec3 randomPoint(const Vec3& v1, const Vec3& v2, const Vec3& v3)
{
    real b1 = 1 - sqrt( 1-random1() );
    real b2 = (1-b1) * random1();
    real b3 = 1 - b1 - b2;

    return b1*v1 + b2*v2 + b3*v3;
}

real triangleArea(const Vec3& v1, const Vec3& v2, const Vec3& v3)
{
    Vec3 a = v2 - v1;
    Vec3 b = v3 - v1;

    return 0.5 * length(a ^ b);
}

#define ROOT3 1.732050807568877
#define FOUR_ROOT3 6.928203230275509

real triangleCompactness(const Vec3& v1, const Vec3& v2, const Vec3& v3)
{
    real L1 = norm2(v2-v1);
    real L2 = norm2(v3-v2);
    real L3 = norm2(v1-v3);

    return FOUR_ROOT3 * triangleArea(v1,v2,v3) / (L1+L2+L3);
}




////////////////////////////////////////////////////////////////////////.
//
// class: Bounds
//

void Bounds::reset()
{
    min[X] = min[Y] = min[Z] = HUGE;
    max[X] = max[Y] = max[Z] = -HUGE;

    center[X] = center[Y] = center[Z] = 0.0;
    radius = 0.0;

    points = 0;
}

void Bounds::addPoint(const Vec3& v)
{
    if( v[X] < min[X] ) min[X] = v[X];
    if( v[Y] < min[Y] ) min[Y] = v[Y];
    if( v[Z] < min[Z] ) min[Z] = v[Z];

    if( v[X] > max[X] ) max[X] = v[X];
    if( v[Y] > max[Y] ) max[Y] = v[Y];
    if( v[Z] > max[Z] ) max[Z] = v[Z];


    center += v;

    points++;
}

void Bounds::complete()
{
    center /= (real)points;

    Vec3 R1 = max-center;
    Vec3 R2 = min-center;

    radius = MAX(length(R1), length(R2));
}



////////////////////////////////////////////////////////////////////////
//
// class: Plane
//

void Plane::calcFrom(const Vec3& p1, const Vec3& p2, const Vec3& p3)
{
    Vec3 v1 = p2-p1;
    Vec3 v2 = p3-p1;

    n = v1 ^ v2;
    unitize(n);

    d = -n*p1;
}

void Plane::calcFrom(const array<Vec3>& verts)
{
    n[X] = n[Y] = n[Z] = 0.0;

    int i;
    for(i=0; i<verts.length()-1; i++)
    {
        const Vec3& cur = verts[i];
        const Vec3& next = verts[i+1];

        n[X] += (cur[Y] - next[Y]) * (cur[Z] + next[Z]);
        n[Y] += (cur[Z] - next[Z]) * (cur[X] + next[X]);
        n[Z] += (cur[X] - next[X]) * (cur[Y] + next[Y]);
    }

    const Vec3& cur = verts[verts.length()-1];
    const Vec3& next = verts[0];
    n[X] += (cur[Y] - next[Y]) * (cur[Z] + next[Z]);
    n[Y] += (cur[Z] - next[Z]) * (cur[X] + next[X]);
    n[Z] += (cur[X] - next[X]) * (cur[Y] + next[Y]);

    unitize(n);

    d = -n*verts[0];
}



////////////////////////////////////////////////////////////////////////
//
// class: Face3
//

real Face3::area()
{
    return triangleArea(vertexPos(0),vertexPos(1),vertexPos(2));
}


//
// Use Anoop's distance code
//

extern real __gfx_hoppe_dist(const Face3& f, const Vec3& v);

real Face3::distTo(const Vec3& v) const
{
    return __gfx_hoppe_dist(*this, v);
}
