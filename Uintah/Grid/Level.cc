/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Exceptions/InvalidGrid.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
#include <PSECore/XMLUtil/XMLUtil.h>
using namespace Uintah;
using namespace SCICore::Geometry;
using namespace std;
using namespace PSECore::XMLUtil;
#include <map>
#include <Uintah/Grid/BoundCondFactory.h>
#include <Uintah/Grid/KinematicBoundCond.h>
#include <Uintah/Grid/FluxThermalBoundCond.h>
#include <Uintah/Grid/TempThermalBoundCond.h>
#include <Uintah/Grid/SymmetryBoundCond.h>

Level::Level(Grid* grid, const Point& anchor, const Vector& dcell)
   : grid(grid), d_anchor(anchor), d_dcell(dcell)
{
   d_finalized=false;
}

Level::~Level()
{
  // Delete all of the patches managed by this level
  for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); iter++)
    delete *iter;
}

Level::const_patchIterator Level::patchesBegin() const
{
    return d_patches.begin();
}

Level::const_patchIterator Level::patchesEnd() const
{
    return d_patches.end();
}

Level::patchIterator Level::patchesBegin()
{
    return d_patches.begin();
}

Level::patchIterator Level::patchesEnd()
{
    return d_patches.end();
}

Patch* Level::addPatch(const IntVector& lowIndex, const IntVector& highIndex)
{
    Patch* r = scinew Patch(this, lowIndex, highIndex);
    d_patches.push_back(r);
    return r;
}

Patch* Level::addPatch(const IntVector& lowIndex, const IntVector& highIndex,
		       int ID)
{
    Patch* r = scinew Patch(this, lowIndex, highIndex, ID);
    d_patches.push_back(r);
    return r;
}

int Level::numPatches() const
{
  return (int)d_patches.size();
}

void Level::performConsistencyCheck() const
{
   if(!d_finalized)
      throw InvalidGrid("Consistency check cannot be performed until Level is finalized");
  for(int i=0;i<d_patches.size();i++){
    Patch* r = d_patches[i];
    r->performConsistencyCheck();
  }

  // This is O(n^2) - we should fix it someday if it ever matters
  for(int i=0;i<d_patches.size();i++){
    Patch* r1 = d_patches[i];
    for(int j=i+1;j<d_patches.size();j++){
      Patch* r2 = d_patches[j];
      if(r1->getBox().overlaps(r2->getBox())){
	cerr << "r1: " << r1 << '\n';
	cerr << "r2: " << r2 << '\n';
	throw InvalidGrid("Two patches overlap");
      }
    }
  }

  // See if abutting boxes have consistent bounds
}

void Level::getIndexRange(BBox& b)
{
  for(int i=0;i<d_patches.size();i++){
    Patch* r = d_patches[i];
    IntVector l( r->getNodeLowIndex() );
    IntVector u( r->getNodeHighIndex() );
    Point lower( l.x(), l.y(), l.z() );
    Point upper( u.x(), u.y(), u.z() );
    b.extend(lower);
    b.extend(upper);
  }
}

void Level::getSpatialRange(BBox& b)
{
  for(int i=0;i<d_patches.size();i++){
    Patch* r = d_patches[i];
    b.extend(r->getBox().lower());
    b.extend(r->getBox().upper());
  }
}

long Level::totalCells() const
{
  long total=0;
  for(int i=0;i<d_patches.size();i++)
    total+=d_patches[i]->totalCells();
  return total;
}

GridP Level::getGrid() const
{
   return grid;
}

Point Level::getNodePosition(const IntVector& v) const
{
   return d_anchor+d_dcell*v;
}

IntVector Level::getCellIndex(const Point& p) const
{
   Vector v((p-d_anchor)/d_dcell);
   return IntVector((int)v.x(), (int)v.y(), (int)v.z());
}

Point Level::positionToIndex(const Point& p) const
{
   return Point((p-d_anchor)/d_dcell);
}

void Level::selectPatches(const IntVector& low, const IntVector& high,
			  std::vector<const Patch*>& neighbors) const
{
   // This sucks - it should be made faster.  -Steve
   for(const_patchIterator iter=d_patches.begin();
       iter != d_patches.end(); iter++){
      const Patch* patch = *iter;
      IntVector l=SCICore::Geometry::Max(patch->getCellLowIndex(), low);
      IntVector u=SCICore::Geometry::Min(patch->getCellHighIndex(), high);
      if(u.x() > l.x() && u.y() > l.y() && u.z() > l.z())
	 neighbors.push_back(*iter);
   }
}

bool Level::containsPoint(const Point& p) const
{
   // This sucks - it should be made faster.  -Steve
   for(const_patchIterator iter=d_patches.begin();
       iter != d_patches.end(); iter++){
      const Patch* patch = *iter;
      if(patch->getBox().contains(p))
	 return true;
   }
   return false;
}

void Level::finalizeLevel()
{
  for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); iter++){
    Patch* patch = *iter;
    // See if there are any neighbors on the 6 faces
    for(Patch::FaceType face = Patch::startFace;
	face < Patch::endFace; face=Patch::nextFace(face)){
      IntVector l,h;
      patch->getFace(face, 1, l, h);
      std::vector<const Patch*> neighbors;
      selectPatches(l, h, neighbors);
      if(neighbors.size() == 0)
	patch->setBCType(face, Patch::None);
      else
	patch->setBCType(face, Patch::Neighbor);
    }
  }
  
  d_finalized=true;
}

void Level::assignBCS(const ProblemSpecP& grid_ps)
{

  // Read the bcs for the grid
  ProblemSpecP bc_ps = grid_ps->findBlock("BoundaryConditions");
  if (bc_ps == 0)
    return;
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");
       face_ps != 0; face_ps=face_ps->findNextBlock("Face")) {
    // 
    map<string,string> values;
    face_ps->getAttributes(values);
    //    std::cerr << "face side = " << values["side"] << std::endl;

    Patch::FaceType face_side;
    std::string fc = values["side"];
    if (fc == "x-")
      face_side = Patch::xminus;
    if (fc ==  "x+")
      face_side = Patch::xplus;
    if (fc ==  "y-")
      face_side = Patch::yminus;
    if (fc ==  "y+")
      face_side = Patch::yplus;
    if (fc ==  "z-")
      face_side = Patch::zminus;
    if (fc == "z+")
      face_side = Patch::zplus;

    
    //    std::cerr << "face_side = " << face_side << std::endl;
    vector<BoundCond *> bcs;
    BoundCondFactory::create(face_ps,bcs);

    for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); 
	iter++){
      Patch* patch = *iter;
      Patch::BCType bc_type = patch->getBCType(face_side);
      if (bc_type != Patch::None || bc_type != Patch::Neighbor ||
	  bc_type != Patch::Symmetry) {
	patch->setBCValues(face_side,bcs);
      }
      vector<BoundCond* > new_bcs;
      new_bcs = patch->getBCValues(face_side);
      //cout << "number of bcs on face " << face_side << " = " 
      //   << new_bcs.size() << endl;
    }  // end of patch iterator
  } // end of face_ps

#if 0
  Vector vel;
  double temp,h_f;
  bool symm_test = false;
  bool vel_test = false;
  bool temp_test = false;
  bool heat_fl_test = false;
#endif

#if 0
    for (ProblemSpecP bc_ps = face_ps->findBlock("BCType");
	 bc_ps != 0; bc_ps = bc_ps->findNextBlock("BCType")) {
      map<string,string> bc_attr;
      bc_ps->getAttributes(bc_attr);
      //      std::cerr << "bc type = " << bc_attr["label"] << std::endl;
      //      std::cerr << "bc var = " << bc_attr["var"] << std::endl;
      if (bc_attr["var"] == "velocity" ) {
	bc_ps->require("velocity",vel);
	//	std::cerr << "velocity = " << vel << std::endl;
	vel_test = true;
      }
      else if (bc_attr["var"] == "temperature" ) {
	bc_ps->require("temperature",temp);
	//	std::cerr << "temperature = " << temp << std::endl;
	temp_test = true;
      }
      else if (bc_attr["var"] == "heat_flux" ) {
	bc_ps->require("heat_flux",h_f);
	//	std::cerr << "heat_flux = " << h_f << std::endl;
	heat_fl_test = true;
      }
      else if (bc_attr["var"] == "symmetry" ) {
	// do nothing
	symm_test = true;
	//	std::cerr << "symmetry don't do anything" << std::endl;
      }
      else {
	// error
	//	std::cerr << "Don't know anything about this" << std::endl;
      }
    }  // end of looping through the bcs for a given face
    
#endif
    // Now loop through all the patches and store the bcs for a given
    // face

#if 0
    // Do the x- (xminus) face
  
    if (values["side"] == "x-" ) {
      Patch::FaceType f = Patch::xminus;
      for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); 
	  iter++){
	Patch* patch = *iter;
	if ( patch->getBCType(f) == Patch::None) {
	  // Assign it to the value given in the prob spec
	  //	  std::cerr << "xminus is none" << std::endl;
	  if (vel_test == true) {
	    patch->setBCType(f,Patch::Fixed);
	    KinematicBoundCond* bc = new KinematicBoundCond(vel);
	    // patch->setBCValue(f,bc);
	  }
	  else if (temp_test == true ) {
	    patch->setBCType(f,Patch::Fixed);
	    TempThermalBoundCond* bc = new TempThermalBoundCond(temp);
	    //patch->setBCValue(f,bc);
	  }
	  else if (heat_fl_test == true) {
	    patch->setBCType(f,Patch::Fixed);
	    FluxThermalBoundCond* bc = new FluxThermalBoundCond(h_f);
	    //patch->setBCValue(f,bc);
	  }
	  else if (symm_test == true) {
	    patch->setBCType(f,Patch::Symmetry);
	    SymmetryBoundCond* bc = new SymmetryBoundCond();
	    //patch->setBCValue(f,bc);
	  }
	  
	}
      }
    }
    // Do the x+ (xplus) face
  
    if (values["side"] == "x+" ) {
      Patch::FaceType f = Patch::xplus;
      for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); 
	  iter++){
	Patch* patch = *iter;
	if ( patch->getBCType(f) == Patch::None) {
	  // Assign it to the value given in the prob spec
	  //	  std::cerr << "xplus is none" << std::endl;
	  if (vel_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (temp_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (heat_fl_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (symm_test == true)
	    patch->setBCType(f,Patch::Symmetry);
	}
      }
    }
    // Do the y- (yminus) face
  
    if (values["side"] == "y-" ) {
      Patch::FaceType f = Patch::yminus;
      for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); 
	  iter++){
	Patch* patch = *iter;
	if ( patch->getBCType(f) == Patch::None) {
	  // Assign it to the value given in the prob spec
	  //	  std::cerr << "yminus is none" << std::endl;
	  if (vel_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (temp_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (heat_fl_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (symm_test == true)
	    patch->setBCType(f,Patch::Symmetry);
	}
      }
    }
    // Do the y+ (yplus) face
  
    if (values["side"] == "y+" ) {
      Patch::FaceType f = Patch::yplus;
      for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); 
	  iter++){
	Patch* patch = *iter;
	if ( patch->getBCType(f) == Patch::None) {
	  // Assign it to the value given in the prob spec
	  //	  std::cerr << "yplus is none" << std::endl;
	  if (vel_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (temp_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (heat_fl_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (symm_test == true)
	    patch->setBCType(f,Patch::Symmetry);
	}
      }
    }
    // Do the z- (zminus) face
  
    if (values["side"] == "z-" ) {
      Patch::FaceType f = Patch::zminus;
      for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); 
	  iter++){
	Patch* patch = *iter;
	if ( patch->getBCType(f) == Patch::None) {
	  // Assign it to the value given in the prob spec
	  //	  std::cerr << "zminus is none" << std::endl;
	  if (vel_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (temp_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (heat_fl_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (symm_test == true)
	    patch->setBCType(f,Patch::Symmetry);
	}
      }
    }
    // Do the z+ (zplus) face
  
    if (values["side"] == "z+" ) {
      Patch::FaceType f = Patch::zplus;
      for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); 
	  iter++){
	Patch* patch = *iter;
	if ( patch->getBCType(f) == Patch::None) {
	  // Assign it to the value given in the prob spec
	  //	  std::cerr << "zplus is none" << std::endl;
	  if (vel_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (temp_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (heat_fl_test == true)
	    patch->setBCType(f,Patch::Fixed);
	  else if (symm_test == true)
	    patch->setBCType(f,Patch::Symmetry);
	}
      }
    }
  } // End of looping through the face_ps
 #endif
}

//
// $Log$
// Revision 1.13  2000/06/27 22:49:03  jas
// Added grid boundary condition support.
//
// Revision 1.12  2000/06/23 19:20:19  jas
// Added in the early makings of Grid bcs.
//
// Revision 1.11  2000/06/15 21:57:16  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.10  2000/05/30 20:19:29  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.9  2000/05/20 08:09:21  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.8  2000/05/20 02:36:05  kuzimmer
// Multiple changes for new vis tools and DataArchive
//
// Revision 1.7  2000/05/15 19:39:47  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.6  2000/05/10 20:02:59  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.5  2000/04/26 06:48:49  sparker
// Streamlined namespaces
//
// Revision 1.4  2000/04/13 06:51:01  sparker
// More implementation to get this to work
//
// Revision 1.3  2000/04/12 23:00:47  sparker
// Starting problem setup code
// Other compilation fixes
//
// Revision 1.2  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
