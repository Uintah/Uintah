#include <iostream.h>
#include <gfx/std.h>
#include <gfx/math/Vec3.h>
#include <gfx/tools/Buffer.h>
#include <gfx/tools/Array3.h>
#include <gfx/geom/ProxGrid.h>



ProxGrid::ProxGrid(const Vec3& lo, const Vec3& hi, real dist)
{
    cellsize = dist;
    cellsize2 = dist*dist;

    min = lo;
    max = hi;

    xdiv = (int)ceil((max[X] - min[X])/dist);
    ydiv = (int)ceil((max[Y] - min[Y])/dist);
    zdiv = (int)ceil((max[Z] - min[Z])/dist);

    cells.init(xdiv, ydiv, zdiv);
}


void ProxGrid::cell_for_point(const Vec3& v,int *i_out,int *j_out,int *k_out)
{
    int i = (int)floor((v[X] - min[X]) / cellsize);
    int j = (int)floor((v[Y] - min[Y]) / cellsize);
    int k = (int)floor((v[Z] - min[Z]) / cellsize);
   
    // In case we hit the max bounds
    if( i==xdiv ) i--;
    if( j==ydiv ) j--;
    if( k==zdiv ) k--;
  

    *i_out = i;
    *j_out = j;
    *k_out = k;
}

void ProxGrid::addPoint(Vec3 *v)
{
    int i, j, k;
    cell_for_point(*v, &i, &j, &k);
    ProxGrid_Cell& cell = cells(i,j,k);

    cell.add(v);
}

void ProxGrid::removePoint(Vec3 *v)
{
    int i, j, k;
    cell_for_point(*v, &i, &j, &k);
    ProxGrid_Cell& cell = cells(i,j,k);


    int index = cell.find(v);
    if( index >= 0 )
	cell.remove(index);
    else
	cerr << "WARNING: ProxGrid -- removing non-member point." << endl;
}



void ProxGrid::maybe_collect_points(Vec3 *v, buffer<Vec3 *>& close,
				    ProxGrid_Cell& cell)
{
    for(int i=0; i<cell.length(); i++)
    {
	Vec3 *u = cell(i);

	if( u!=v && norm2(*u - *v) < cellsize2 )
	    close.add(u);
    }
}



void ProxGrid::proximalPoints(Vec3 *v, buffer<Vec3 *>& close)
{
    int i, j, k;
    cell_for_point(*v, &i, &j, &k);

    for(int dk=-1; dk<2; dk++)
	for(int dj=-1; dj<2; dj++)
	    for(int di=-1; di<2; di++)
	    {
		if( i+di>=0 && j+dj>=0 && k+dk>=0
		    && i+di<cells.width()
		    && j+dj<cells.height()
		    && k+dk<cells.depth() )
		{
		    maybe_collect_points(v, close, cells(i+di, j+dj, k+dk));
		}
	    }
}
