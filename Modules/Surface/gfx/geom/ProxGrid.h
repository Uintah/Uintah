#ifndef GFXGEOM_PROXGRID_INCLUDED // -*- C++ -*-
#define GFXGEOM_PROXGRID_INCLUDED

#include <gfx/math/Vec3.h>
#include <gfx/tools/Buffer.h>
#include <gfx/tools/Array3.h>


class ProxGrid_Cell : public buffer<Vec3 *>
{

public:

    ProxGrid_Cell()
	: buffer<Vec3 *>(8)
    { }
};





class ProxGrid
{
    array3<ProxGrid_Cell> cells;
    int xdiv, ydiv, zdiv;
    real cellsize;
    real cellsize2;

    Vec3 min, max;

    void cell_for_point(const Vec3&, int *i, int *j, int *k);
    void maybe_collect_points(Vec3 *v, buffer<Vec3 *>& close,
			      ProxGrid_Cell& cell);

public:

    ProxGrid(const Vec3& min, const Vec3& max, real dist);
    ~ProxGrid() { cells.free(); }

    void addPoint(Vec3 *);
    void removePoint(Vec3 *);
    void proximalPoints(Vec3 *, buffer<Vec3 *>&);

};







// GFXGEOM_PROXGRID_INCLUDED
#endif
