#ifndef __GEOMETRY_GRID_H__
#define __GEOMETRY_GRID_H__


#include <string>

class GeometryGrid {
private:
  //ofstream gridcel;
  //ofstream gridpos;

public:
  GeometryGrid();
  ~GeometryGrid();
  
  void buildGeometryGrid(std::string posname, std::string celname,
			 const double probbnds[7], const double dx[4]);

};

#endif /* __GEOMETRY_GRID_H__ */


// $Log$
// Revision 1.3  2000/03/20 17:17:14  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.2  2000/03/15 21:58:24  jas
// Added logging and put guards in.
//
// Revision 1.1  2000/03/14 22:10:49  jas
// Initial creation of the geometry specification directory with the legacy
// problem setup.
//
// Revision 1.1  2000/02/24 06:11:56  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:50  sparker
// Stuff may actually work someday...
//
// Revision 1.2  1999/07/22 20:30:48  jas
// Added namespace std.
//
// Revision 1.1  1999/06/14 06:23:40  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.3  1999/01/26 21:53:33  campbell
// Added logging capabilities
//
