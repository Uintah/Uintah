#ifndef _MATERIAL_H_
#define _MATERIAL_H_
#include <iosfwd>

class Material {
private:
	double	Density;
	int	MatType;
        double mp[10];

public:
	Material();
	~Material();

	void addMaterial(std::ifstream &filename);
	double getDensity() const;
	int getMaterialType() const;
	void getMatProps(double matprop[10]) const;
};

// $Log$
// Revision 1.1  2000/02/27 07:48:42  sparker
// Homebrew code all compiles now
// First step toward PSE integration
// Added a "Standalone Uintah Simulation" (sus) executable
// MPM does NOT run yet
//
// Revision 1.1  2000/02/24 06:12:18  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:53  sparker
// Stuff may actually work someday...
//
// Revision 1.1  1999/06/14 06:23:41  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.6  1999/05/24 20:56:49  guilkey
// Added a new constitutive model.
//
// Revision 1.5  1999/02/25 22:32:33  guilkey
// Fixed some functions associated with the HyperElastic constitutive model.
//
// Revision 1.4  1999/01/26 21:53:34  campbell
// Added logging capabilities
//
#endif /* _MATERIAL_H_ */
