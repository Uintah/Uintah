#include "Problem.h"
#include "GeometryGrid.h"
#include <Uintah/Components/MPM/BoundCond.h>
#include "Material.h"
#include <iostream>
#include <fstream>
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

#ifdef WONT_COMPILE_YET

Problem::Problem()
  : NumMaterial(0),
    NumObjects(0),
    numbcs(0)
{
}

Problem::~Problem()
{
  int i;
  for(i = 0; i < NumMaterial; i++)
    {
      delete materials[i];
    }
}

void Problem::preProcessor(string filename)
{
  int n;
  int obj, piece, surf;
  double   force[4];
  BoundCond BCTemp;
  string stuff;		//stuff is a throwaway string

  ifstream infile;
  
  infile.open(filename.c_str());
  if (infile.bad()) {
    cerr << "Error opening input file" << endl;
    exit(1);
  }
  
  infile >> stuff;
  if(stuff != string("Material_Property_Info"))
    {
      cerr << "Bad input file: " << filename << endl;
      cerr << "Input file needs:  Material_Property_Info" << endl;
      exit(1);
    }
  infile >> stuff;
  if(stuff != string("Number_of_Materials"))
    {
      cerr << "Bad input file: " << filename << endl;
      cerr << "Input file needs: Number_of_Materials" << endl;	
      exit(1);
    }
  infile >> NumMaterial;
  for(n = 1; n <= NumMaterial; n++)
    {
      infile >> stuff;
      char tmpnum[5];
      sprintf(tmpnum, "%d", n);
      string tmp = string("Density&CONSTANTS_for_material_") + string(tmpnum);
      if(stuff != tmp)
	{
	  cerr << "Bad input file: " << filename << endl;
	  cerr << "Input file needs: Density&CONSTANTS_for_material_#" << endl;
	  exit(1);
	}
      materials.push_back( new Material );
      materials[n-1]->addMaterial(infile);
    }
  infile >> stuff;
  if(stuff != string("Problem_Boundaries"))
    {
      cerr << "Bad input file: " << filename << endl;
      cerr << "Input file needs: Problem_Boundaries" <<  endl;
      exit(1);
    }
  infile >> bnds[1] >> bnds[2] >> bnds[3] >> bnds[4] >> bnds[5] >> bnds[6];
  infile >> stuff;
  if(stuff != string("Grid_Spacing"))
    {
      cerr << "Bad input file: " << filename << endl;
      cerr << "Input file needs: Grid_Spacing" << endl;
      exit(1);
    }
  infile >> dx[1] >> dx[2] >> dx[3];
  infile >> stuff;
  if(stuff != string("Number_of_Particles/Cell"))
    {
      cerr << "Bad input file: " << filename << endl;
      cerr << "Input file needs: Number_of_Particles/Cell" << endl;
      exit(1);
    }
  infile >> nppcel[1] >> nppcel[2] >> nppcel[3];
  infile >> stuff;
  if(stuff != string("Number_of_Objects"))
    {
      cerr << "Bad input file: " << filename << endl;
      cerr << "Input file needs: Number_of_Objects" << endl;
      exit(1);
    }
  infile >> NumObjects;
  
  for(n = 1; n <= NumObjects; n++)
    {
      infile >> stuff;
      char tmpnum[5];
      sprintf(tmpnum, "%d", n);
      string tmp = string("Number_of_Pieces_this_Object") + string(tmpnum);
      if(stuff != tmp)
	{
	  cerr << "Bad input file: " << filename << endl;
	  cerr << "Input file needs Number_of_Pieces_this_Object" << endl;
	  exit(1);
	}
      int tmpnpieces;
      infile >> tmpnpieces;
      npieces.push_back(tmpnpieces);
      
      GeomObject tempObject;
      
      tempObject.setObjInfo(n, bnds, npieces[n-1], dx, nppcel);
      tempObject.addPieces(infile);
      Objects.push_back(tempObject);
    }
  
  infile >>stuff;
  if(stuff != string("Number_of_Force_Boundary_Conditions"))
    {
      cerr << "Bad input file: " << filename << endl;
      cerr << "Input file needs: Number_of_Force_Boundary_Conditions" << endl;
      cerr << "You have probably misspelled Conditions i.e. Condtions" << endl;
      exit(1);
    }
  infile >> numbcs;
  
  for(n = 0; n < numbcs; n++)
    {
      infile >> stuff;
      infile >> obj >> piece >> surf >> force[1] >> force[2] >> force[3];
      
      BCTemp.setBC(obj, piece, surf, force);
      BC.push_back(BCTemp);
    }
}

void Problem::createParticles(const Region* region, DataWarehouseP& dw)
{
  for (int i = 0; i < NumObjects; i++)
    {
      Objects[i].FillWParticles(materials, BC, region, dw);
    }
}

void Problem::writeGridFiles(string gridposname, string gridcellname)
{
  GeometryGrid ProblemGrid;
  ProblemGrid.buildGeometryGrid(gridposname, gridcellname, bnds, dx);
}

int Problem::getNumMaterial() const
{
  return(NumMaterial);
}


int Problem::getNumObjects() const
{
   return(NumObjects);
}

vector<GeomObject> * Problem::getObjects()
{
    return(&Objects);
}


void Problem::writeSAMRAIGridFile(string gridname)
{
  ofstream f(gridname.c_str());
  if(!f)
    {
      cerr << "Can't open " << gridname << endl;
      exit(1);
    }
  
  int ncelx,ncely,ncelz;  /* # of cells in each dir.      */
  
  ncelx = (int)((bnds[2]-bnds[1])/dx[1]+.000001);
  ncely = (int)((bnds[4]-bnds[3])/dx[2]+.000001);
  ncelz = (int)((bnds[6]-bnds[5])/dx[3]+.000001);
  
  f << "CartesianGeometry" << endl;
  f << endl;

  // Write out info for 1 box     
  
  f << "  domain_boxes     1" << endl;
  f << "    box1  ";
  f << "1 1 1  "
    << ncelx << " " << ncely << " " << ncelz << endl;
  f << "  end_domain_boxes" << endl;
  f << endl;

 

  f << "  cartesian_grid_data" << endl;
  f << "    x_lo  " << bnds[1] << " " << bnds[3] << " " << bnds[5] << endl;
  f << "    x_up  " << bnds[2] << " " << bnds[4] << " " << bnds[6] << endl;
  f << "  end_cartesian_grid_data" << endl;
  f << endl;
  f << "end" << endl;

  // Specify info about Time Refinement Integrator

  f << endl;	
  f << endl;	
  f << endl;	
  f << "TimeRefinementIntegrator" << endl;
  f << "    start_time         0.e0" << endl;
  f << "    end_time            10.e0" << endl;
  f << "    grow_dt	       1.0e0" << endl;
  f << "    max_integrator_steps  80" << endl;
  f << "end" << endl;
  f << endl;
  f << endl;

  // Specify info about the Gridding Algorithm

  f << "GriddingAlgorithm" << endl;
  f <<  "   max_levels        1" << endl;
  f << endl;
  f << endl;

  f << "   ratio_to_coarser"  << endl;
  f << "      level1    4 4 4" << endl;
  f << "      level2    4 4 4" << endl;
  f << "      level3    4 4 4" << endl;
  f << "   end_ratio_to_coarser" << endl;

  f << endl;
  f << endl;

  f << "   regrid_interval" << endl;
  f << "     level0         2" << endl;
  f << "     level1         2" << endl;
  f << "     level2         2" << endl;
  f << "   end_regrid_interval" << endl;

  f << endl; 
  f << endl; 

  f << "   smallest_patch_size  1 1 1" << endl;
  f << "   largest_patch_size " << ncelx << " " << ncely << " " << ncelz  << endl;

  f << endl;
  f << endl;

  f << "   efficiency_tolerance    0.85e0" << endl;
  f << "   combine_efficiency      0.95e0" << endl;

  f << endl;
  f << endl;

  f << "end" << endl;

  // Specify infor about the Material Point Level Integrator

  f << "MaterialPointLevelIntegrator" << endl;
  f << "   cfl            1.0e0" << endl;
  f << "   cfl_init       1.0e0" << endl;
  f << "   lag_dt_computation     1" << endl;
  f << "   use_ghosts_to_compute_dt  1" << endl;
  f << "   use_ghosts_for_cons_diff  1" << endl;
  f << "end" << endl; 
 


}

#endif


// $Log$
// Revision 1.2  2000/03/20 17:17:15  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:36:06  jas
// Readded geometry specification source files.
//
// Revision 1.1  2000/02/24 06:11:59  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:56  sparker
// Stuff may actually work someday...
//
// Revision 1.8  1999/10/28 23:24:48  jas
// Eliminated the -n option for manually creating multiple boxes.  SAMRAI does
// this automatically now.
//
// Revision 1.7  1999/09/04 23:00:32  jas
// Changed from Particle * to Particle to fix a memory leak.
//
// Revision 1.6  1999/08/18 19:23:09  jas
// Now can return information about the problem specification and object
// bounds.  This is necessary for the creation of particles on the processor
// that they reside.
//
// Revision 1.5  1999/08/06 21:14:45  jas
// Fixed largest patch size to use as big as patch as possible
//
// Revision 1.4  1999/07/28 16:54:55  cgl
// - Multiple Velocity fields in smpm
// - stop creating extra particles
//
// Revision 1.3  1999/07/22 20:17:20  jas
// grid.amr contains all the input needed by SAMRAI to run a problem.  We
// no longer need the test.amr file.
//
// Revision 1.2  1999/06/18 05:44:53  cgl
// - Major work on the make environment for smpm.  See doc/smpm.make
// - fixed getSize(), (un)packStream() for all constitutive models
//   and Particle so that size reported and packed amount are the same.
// - Added infomation to Particle.packStream().
// - fixed internal force summation equation to keep objects from exploding.
// - speed up interpolateParticlesToPatchData()
// - Changed lists of Particles to lists of Particle*s.
// - Added a command line option for smpm `-c npatch'.  Valid values are 1 2 4
//
// Revision 1.1  1999/06/14 06:23:42  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.6  1999/02/10 20:53:10  guilkey
// Updated to release 2-0
//
// Revision 1.5  1999/01/26 21:53:34  campbell
// Added logging capabilities
//
