#include <Packages/Uintah/Core/Grid/BCDataArray.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/Core/Grid/DifferenceBCData.h>
#include <Packages/Uintah/Core/Grid/SideBCData.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
#include <functional>
#include <sgi_stl_warnings_on.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

// export SCI_DEBUG="BCDA_DBG:+"
static DebugStream BCDA_dbg("BCDA_DBG",false);

BCDataArray::BCDataArray() 
{
}

BCDataArray::~BCDataArray()
{
  bcDataArrayType::const_iterator mat_id_itr;
  for (mat_id_itr=d_BCDataArray.begin(); mat_id_itr != d_BCDataArray.end();
       ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    vector<BCGeomBase*>& vec = d_BCDataArray[mat_id];
    vector<BCGeomBase*>::const_iterator bcd_itr;
    for (bcd_itr = vec.begin(); bcd_itr != vec.end(); ++bcd_itr) {
      delete *bcd_itr;
    }
    vec.clear();
  }
  d_BCDataArray.clear();
  
}

BCDataArray::BCDataArray(const BCDataArray& mybc)
{
  bcDataArrayType::const_iterator mat_id_itr;
  for (mat_id_itr = mybc.d_BCDataArray.begin(); 
       mat_id_itr != mybc.d_BCDataArray.end(); ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    const vector<BCGeomBase*>& mybc_vec = mat_id_itr->second;
    vector<BCGeomBase*>& d_BCDataArray_vec =  d_BCDataArray[mat_id];
    vector<BCGeomBase*>::const_iterator vec_itr;
    for (vec_itr = mybc_vec.begin(); vec_itr != mybc_vec.end(); ++vec_itr) {
      d_BCDataArray_vec.push_back((*vec_itr)->clone());
    }
  }
}

BCDataArray& BCDataArray::operator=(const BCDataArray& rhs)
{
  if (this == &rhs) 
    return *this;

  // Delete the lhs
  bcDataArrayType::const_iterator mat_id_itr;
  for (mat_id_itr=d_BCDataArray.begin(); mat_id_itr != d_BCDataArray.end();
       ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    vector<BCGeomBase*>& vec = d_BCDataArray[mat_id];
    vector<BCGeomBase*>::const_iterator bcd_itr;
    for (bcd_itr = vec.begin(); bcd_itr != vec.end(); ++bcd_itr)
	delete *bcd_itr;
    vec.clear();
  }
  d_BCDataArray.clear();
  // Copy the rhs to the lhs
  for (mat_id_itr = rhs.d_BCDataArray.begin(); 
       mat_id_itr != rhs.d_BCDataArray.end(); ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    vector<BCGeomBase*>& d_BCDataArray_vec = d_BCDataArray[mat_id];
    const vector<BCGeomBase*>& rhs_vec = mat_id_itr->second;
    vector<BCGeomBase*>::const_iterator vec_itr;
    for (vec_itr = rhs_vec.begin(); vec_itr != rhs_vec.end(); ++vec_itr) 
      d_BCDataArray_vec.push_back((*vec_itr)->clone());
  }
  return *this;
}

BCDataArray* BCDataArray::clone()
{
  return new BCDataArray(*this);

}

void BCDataArray::determineIteratorLimits(Patch::FaceType face,
					  const Patch* patch)
{
  IntVector lpts,hpts;
  patch->getFaceCells(face,-1,lpts,hpts);
  vector<Point> test_pts;

  for (CellIterator candidatePoints(lpts,hpts); !candidatePoints.done();
       candidatePoints++) {
    IntVector nodes[8];
    patch->findNodesFromCell(*candidatePoints,nodes);
    Point pts[8];
    Vector p;
    for (int i = 0; i < 8; i++)
      pts[i] = patch->getLevel()->getNodePosition(nodes[i]);
    if (face == Patch::xminus)
      p = (pts[0].asVector()+pts[1].asVector()+pts[2].asVector()
	   +pts[3].asVector())/4.;
    if (face == Patch::xplus)
      p = (pts[4].asVector()+pts[5].asVector()+pts[6].asVector()
	   +pts[7].asVector())/4.;
    if (face == Patch::yminus)
      p = (pts[0].asVector()+pts[1].asVector()+pts[4].asVector()
	   +pts[5].asVector())/4.;
    if (face == Patch::yplus)
      p = (pts[2].asVector()+pts[3].asVector()+pts[6].asVector()
	   +pts[7].asVector())/4.;
    if (face == Patch::zminus)
      p = (pts[0].asVector()+pts[2].asVector()+pts[4].asVector()
	   +pts[6].asVector())/4.;
    if (face == Patch::zplus)
      p = (pts[1].asVector()+pts[3].asVector()+pts[5].asVector()
	   +pts[7].asVector())/4.;

    test_pts.push_back(Point(p.x(),p.y(),p.z()));
  }
  
  BCDataArray::bcDataArrayType::iterator mat_id_itr;
  for (mat_id_itr = d_BCDataArray.begin();
       mat_id_itr != d_BCDataArray.end(); ++mat_id_itr) {
    vector<BCGeomBase*>& bc_objects = mat_id_itr->second;
    for (vector<BCGeomBase*>::iterator obj = bc_objects.begin();
	 obj != bc_objects.end(); ++obj) {
      (*obj)->determineIteratorLimits(face,patch,test_pts);
#if 0
      (*obj)->printLimits();
#endif
    }
  }

}

void BCDataArray::oldDetermineIteratorLimits(Patch::FaceType face, 
					     const Patch* patch)

{
#if 0
  cout << "Face = " << face << endl;
#endif
  IntVector l,h,li,hi,ln,hn,ln_ec,hn_ec,l_pts,h_pts;
  patch->getFaceCells(face,0,l,h);
  patch->getFaceCells(face,-1,li,hi);
  patch->getFaceNodes(face,0,ln,hn);

  l_pts = li;
  h_pts = hi;
  // Loop over the various material ids.

  set<IntVector,ltiv_x> boundary_set,all_face_nodes_set;

  for (CellIterator boundary(l,h);!boundary.done();boundary++) 
    boundary_set.insert(*boundary);

#if 0
  for (set<IntVector>::const_iterator it = boundary_set.begin(); 
       it != boundary_set.end(); ++it)
    cout << "boundary_set = " << *it << endl;
#endif

  for (NodeIterator bound_nodes(ln,hn);!bound_nodes.done();
       bound_nodes++) 
    all_face_nodes_set.insert(*bound_nodes);

#if 0
  for (set<IntVector>::const_iterator it = all_face_nodes_set.begin();
       it != all_face_nodes_set.end(); ++it)
    cout << "all_face_nodes_set = " << *it << endl;
#endif
  
  BCDataArray::bcDataArrayType::const_iterator mat_id_itr;
  for (mat_id_itr = d_BCDataArray.begin(); 
       mat_id_itr != d_BCDataArray.end(); ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    set<IntVector,ltiv_x>  diff_set,bc_set;
    for (int c = 0; c < getNumberChildren(mat_id); c++) {
      CellIterator interior(li,hi), candidatePoints(l_pts,h_pts);
      vector<IntVector> bound,inter,nbound,sfx,sfy,sfz;
      for (CellIterator boundary(l,h);!boundary.done();boundary++,interior++,
	     candidatePoints++) {
	IntVector nodes[8];
	patch->findNodesFromCell(*candidatePoints,nodes);
	Point pts[8];
	Vector p;
	for (int i = 0; i < 8; i++)
	  pts[i] = patch->getLevel()->getNodePosition(nodes[i]);
	if (face == Patch::xminus)
	  p = (pts[0].asVector()+pts[1].asVector()+pts[2].asVector()
	       +pts[3].asVector())/4.;
	if (face == Patch::xplus)
      	  p = (pts[4].asVector()+pts[5].asVector()+pts[6].asVector()
	       +pts[7].asVector())/4.;
	if (face == Patch::yminus)
      	  p = (pts[0].asVector()+pts[1].asVector()+pts[4].asVector()
	       +pts[5].asVector())/4.;
	if (face == Patch::yplus)
      	  p = (pts[2].asVector()+pts[3].asVector()+pts[6].asVector()
	       +pts[7].asVector())/4.;
	if (face == Patch::zminus)
      	  p = (pts[0].asVector()+pts[2].asVector()+pts[4].asVector()
	       +pts[6].asVector())/4.;
	if (face == Patch::zplus)
      	  p = (pts[1].asVector()+pts[3].asVector()+pts[5].asVector()
	       +pts[7].asVector())/4.;
	
	if ((getChild(mat_id,c))->inside(Point(p.x(),p.y(),p.z()))) {
	  bound.push_back(*boundary);

	  // Insert iterators and do a sanity check for all the possible
	  // iterators on a face and compare it to the set of iterators that
	  // the bc calculations performed.  
	  if (bc_set.find(*boundary) == bc_set.end())
	    bc_set.insert(*boundary);
	}
	
      }
      // Only find the sfx,sfy,sfz iterators for the non-side bcgeom types.
      // Once we get all of them, union them together and then do a difference
      // for the side type.  If the numberChildren > 1 for a side, then
      // look at the non-side types.  

      if ((getNumberChildren(mat_id) > 1 && 
	  typeid(getChild(mat_id,c)) != typeid(SideBCData)) ||
	  getNumberChildren(mat_id) == 1) {
#if 0
	for (vector<IntVector>::const_iterator it = bound.begin();
	     it != bound.end();++it)
	  cout << "bc = " << c << " bound = " << *it << endl;
#endif
	
	set<int> same_x,same_y,same_z;
	for (vector<IntVector>::const_iterator it = bound.begin();
	     it != bound.end(); ++it) {
	  same_x.insert((*it).x());
	  same_y.insert((*it).y());
	  same_z.insert((*it).z());
	}
	
	// For the fixed x face, we look at the extents for the y and z 
	// indices to determine what iterators need to be added for the
	// right and top extents for the bc region.  
	set<IntVector,ltiv_x> x_iterator;
	for (set<int>::const_iterator x = same_x.begin();x!=same_x.end();++x){
	  vector<IntVector> same_x_element;	
	  for (vector<IntVector>::const_iterator it = bound.begin();
	       it != bound.end(); ++it) {
	    if (*x == (*it).x())
	      same_x_element.push_back(*it);
	  }
#if 0
	  for (vector<IntVector>::const_iterator it = same_x_element.begin();
	       it != same_x_element.end(); ++it)
	    cout << "same_x_element = " << *it << endl;
#endif
	  IntVector max_x = same_x_element.back();
	  //	cout << "max_y = " << max_y << endl;
	  x_iterator.insert(max_x + IntVector(1,0,0));
	}
#if 0
	for (set<IntVector>::const_iterator it = x_iterator.begin();
	     it != x_iterator.end(); ++it)
	  cout << "x_iterator = " << *it << endl;
#endif
	
	set<IntVector,ltiv_y> y_iterator;
	for (set<int>::const_iterator z = same_z.begin(); z!=same_z.end();++z){
	  vector<IntVector> same_z_element;	
	  for (vector<IntVector>::const_iterator it = bound.begin();
	       it != bound.end(); ++it) {
	    if (*z == (*it).z())
	      same_z_element.push_back(*it);
	  }
#if 0
	  for (vector<IntVector>::const_iterator it = same_z_element.begin();
	       it != same_z_element.end(); ++it)
	    cout << "same_z_element = " << *it << endl;
#endif
	  IntVector max_y = same_z_element.back();
	  //	cout << "max_y = " << max_y << endl;
	  y_iterator.insert(max_y + IntVector(0,1,0));
	}
#if 0
	for (set<IntVector>::const_iterator it = y_iterator.begin();
	     it != y_iterator.end(); ++it)
	  cout << "y_iterator = " << *it << endl;
#endif
	set<IntVector,ltiv_z> z_iterator;
	for (set<int>::const_iterator y = same_y.begin(); y!=same_y.end();++y){
	  vector<IntVector> same_y_element;	
	  for (vector<IntVector>::const_iterator it = bound.begin();
	       it != bound.end(); ++it) {
	    if (*y == (*it).y())
	      same_y_element.push_back(*it);
	  }
#if 0
	  for (vector<IntVector>::const_iterator it = same_y_element.begin();
	       it != same_y_element.end(); ++it)
	    cout << "same_y_element = " << *it << endl;
#endif
	  IntVector max_z = same_y_element.back();
	  //	cout << "max_z = " << max_z << endl;
	  z_iterator.insert(max_z + IntVector(0,0,1));
	}
#if 0
	for (set<IntVector>::const_iterator it = z_iterator.begin();
	     it != z_iterator.end(); ++it)
	  cout << "z_iterator = " << *it << endl;
#endif
	
	if (face == Patch::xminus) {
	  transform(bound.begin(),bound.end(),back_inserter(sfx),
		    bind2nd(plus<IntVector>(),IntVector(1,0,0)));
	  copy(sfx.begin(),sfx.end(),back_inserter(sfy));
	  copy(y_iterator.begin(),y_iterator.end(),back_inserter(sfy));
	  copy(sfx.begin(),sfx.end(),back_inserter(sfz));
	  copy(z_iterator.begin(),z_iterator.end(),back_inserter(sfz));
	}
	
	if (face == Patch::xplus) {
	  transform(bound.begin(),bound.end(),back_inserter(sfx),
		    bind2nd(plus<IntVector>(),IntVector(-1,0,0)));
	  copy(sfx.begin(),sfx.end(),back_inserter(sfy));
	  copy(y_iterator.begin(),y_iterator.end(),back_inserter(sfy));
	  copy(sfx.begin(),sfx.end(),back_inserter(sfz));
	  copy(z_iterator.begin(),z_iterator.end(),back_inserter(sfz));
	}
	
	if (face == Patch::yminus) {
	  transform(bound.begin(),bound.end(),back_inserter(sfy),
		    bind2nd(plus<IntVector>(),IntVector(0,1,0)));
	  copy(sfy.begin(),sfy.end(),back_inserter(sfx));
	  copy(x_iterator.begin(),x_iterator.end(),back_inserter(sfx));
	  copy(sfy.begin(),sfy.end(),back_inserter(sfz));
	  copy(z_iterator.begin(),z_iterator.end(),back_inserter(sfz));
	}
	
	if (face == Patch::yplus) {
	  transform(bound.begin(),bound.end(),back_inserter(sfy),
		    bind2nd(plus<IntVector>(),IntVector(0,-1,0)));
	  copy(sfy.begin(),sfy.end(),back_inserter(sfx));
	  copy(x_iterator.begin(),x_iterator.end(),back_inserter(sfx));
	  copy(sfy.begin(),sfy.end(),back_inserter(sfz));
	  copy(z_iterator.begin(),z_iterator.end(),back_inserter(sfz));
	}
	
	if (face == Patch::zminus) {
	  transform(bound.begin(),bound.end(),back_inserter(sfz),
		    bind2nd(plus<IntVector>(),IntVector(0,0,1)));
	  copy(sfz.begin(),sfz.end(),back_inserter(sfx));
	  copy(x_iterator.begin(),x_iterator.end(),back_inserter(sfx));
	  copy(sfz.begin(),sfz.end(),back_inserter(sfy));
	  copy(y_iterator.begin(),y_iterator.end(),back_inserter(sfy));
	}
	
	if (face == Patch::zplus) {
	  transform(bound.begin(),bound.end(),back_inserter(sfz),
		    bind2nd(plus<IntVector>(),IntVector(0,0,-1)));
	  copy(sfz.begin(),sfz.end(),back_inserter(sfx));
	  copy(x_iterator.begin(),x_iterator.end(),back_inserter(sfx));
	  copy(sfz.begin(),sfz.end(),back_inserter(sfy));
	  copy(y_iterator.begin(),y_iterator.end(),back_inserter(sfy));
	}
	
      }
      
      for (NodeIterator boundary(ln,hn);!boundary.done();boundary++) {
	Point p = patch->getLevel()->getNodePosition(*boundary);
	if ((getChild(mat_id,c))->inside(p)) 
	  nbound.push_back(*boundary);
      }
      
      setBoundaryIterator(mat_id,bound,c);
      setNBoundaryIterator(mat_id,nbound,c);
      setSFCXIterator(mat_id,sfx,c);
      setSFCYIterator(mat_id,sfy,c);
      setSFCZIterator(mat_id,sfz,c);
    }

    // If there were children in the list, then we need to do get all of
    // the SFC iterators for the children and put those in a set.  Then
    // we need to take store in a set all of the possible SFC iterators
    // for a side.  Then to actually figure out the SFC iterators for the
    // Side with "holes" we can do a set difference operation of the
    // complete set of SFC iterators from the set of "hole" SFC iterators.

    if (getNumberChildren(mat_id) > 1) {
      set<IntVector,ltiv_x> sfx_set,sfy_set,sfz_set;
      for (int c = 0; c < getNumberChildren(mat_id); c++) {
	// Get all of the sfc iterators and store in sfx,sfy,sfz.
	vector<IntVector> sfx,sfy,sfz;
	getSFCXIterator(mat_id,sfx,c);
	getSFCYIterator(mat_id,sfy,c);
	getSFCZIterator(mat_id,sfz,c);
	copy(sfx.begin(),sfx.end(),inserter(sfx_set,sfx_set.begin()));
	copy(sfy.begin(),sfy.end(),inserter(sfy_set,sfy_set.begin()));
	copy(sfz.begin(),sfz.end(),inserter(sfz_set,sfz_set.begin()));
      }

      // Create all the SFC iterators for the ENTIRE side this is just
      // the face nodes.
      IntVector ln_i,hn_i;
      patch->getFaceNodes(face,1,ln_i,hn_i);

      set<IntVector,ltiv_x> all_side_sf_set;
      for (NodeIterator sf_itr(ln_i,hn_i); !sf_itr.done(); sf_itr++)
	all_side_sf_set.insert(*sf_itr);
	
      set<IntVector,ltiv_x> side_sfx_set,side_sfy_set,side_sfz_set;
#if 0
      for (set<IntVector>::const_iterator it = all_side_sf_set.begin();
	   it != all_side_sf_set.end(); ++it)
	cout << "all_side_sf_set = " << *it << endl;

      for (set<IntVector>::const_iterator it = sfx_set.begin(); 
	   it != sfx_set.end(); ++it)
	cout << "sfx_set = " << *it << endl;
#endif

      set_difference(all_side_sf_set.begin(),all_side_sf_set.end(),
		     sfx_set.begin(),sfx_set.end(),
		     inserter(side_sfx_set,side_sfx_set.begin()),ltiv_xyz());

      set_difference(all_side_sf_set.begin(),all_side_sf_set.end(),
		     sfy_set.begin(),sfy_set.end(),
		     inserter(side_sfy_set,side_sfy_set.begin()),ltiv_xyz());

      set_difference(all_side_sf_set.begin(),all_side_sf_set.end(),
		     sfz_set.begin(),sfz_set.end(),
		     inserter(side_sfz_set,side_sfz_set.begin()),ltiv_xyz());

      vector<IntVector> sfx,sfy,sfz;
      copy(side_sfx_set.begin(),side_sfx_set.end(),back_inserter(sfx));
      copy(side_sfy_set.begin(),side_sfy_set.end(),back_inserter(sfy));
      copy(side_sfz_set.begin(),side_sfz_set.end(),back_inserter(sfz));
      
      // Need to find the SideBCData so we can store the new sfx,sfy,sfz
      // iterators for the side with "holes"

      for (int child = 0; child < getNumberChildren(mat_id); child++) {
	if (typeid(getChild(mat_id,child)) == typeid(DifferenceBCData)) {
	  setSFCXIterator(mat_id,sfx,child);
	  setSFCYIterator(mat_id,sfy,child);
	  setSFCZIterator(mat_id,sfz,child);
	}
#if 0
	for (vector<IntVector>::const_iterator it = sfx.begin();
	     it != sfx.end(); ++it)
	  cout << "sfx = " << *it << endl;
	for (vector<IntVector>::const_iterator it = sfy.begin();
	     it != sfy.end(); ++it)
	  cout << "sfy = " << *it << endl;
	for (vector<IntVector>::const_iterator it = sfz.begin();
	     it != sfz.end(); ++it)
	  cout << "sfz = " << *it << endl;
#endif

      }
      

    }
    // Check if the set of all IntVectors that exist on a face -- stored
    // in boundary_set also exist in the boundary conditions -- stored
    // in bc_set.  Use a set difference operation to see if there are any
    // missing IntVectors which would show up as unitialized cells in the
    // ICE boundary condition set operation -- code will fail to converge.

#if 0    
    
    set_difference(boundary_set.begin(),boundary_set.end(),bc_set.begin(),
		   bc_set.end(),inserter(diff_set,diff_set.begin()),ltiv_x());
    
    if (!diff_set.empty()) {
      cout << "Size of boundary_set = " << boundary_set.size() << " " 
	   << "size of bc_set = " << bc_set.size() << endl;

      for (set<IntVector>::const_iterator it = boundary_set.begin(); 
	   it != boundary_set.end(); ++it)
	cout << "boundary_set = " << *it << endl;

      for (set<IntVector>::const_iterator it = bc_set.begin(); 
	   it != bc_set.end(); ++it)
	cout << "bc_set = " << *it << endl;
    
      set<IntVector,ltiv_x> new_diff_set;
      for (set<IntVector>::const_iterator it = diff_set.begin(); 
	   it != diff_set.end(); ++it) {
	cout << "diff_set = " << *it << endl;
	new_diff_set.insert(*it);
      }
      
      for (set<IntVector>::const_iterator it = new_diff_set.begin(); 
	   it != new_diff_set.end(); ++it) 
	cout << "new_diff_set = " << *it << endl;
    }
#endif
  }

#if 0
  for (mat_id_itr = d_BCDataArray.begin();
       mat_id_itr != d_BCDataArray.end(); ++mat_id_itr) {
    for (vector<BCGeomBase*>::const_iterator obj = mat_id_itr->second.begin();
	 obj !=mat_id_itr->second.end(); ++obj) {
      (*obj)->printLimits();
    }
  }
#endif

}



void BCDataArray::addBCData(int mat_id,BCGeomBase* bc)
{
  vector<BCGeomBase*>& d_BCDataArray_vec = d_BCDataArray[mat_id];
  d_BCDataArray_vec.push_back(bc);
}


const BoundCondBase* 
BCDataArray::getBoundCondData(int mat_id, string type, int i) const
{
  BCData new_bc,new_bc_all;
  // Need to check two scenarios -- the given mat_id and the all mat_id (-1)
  // Check the given mat_id
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  
  if (itr != d_BCDataArray.end()) {
    itr->second[i]->getBCData(new_bc);
    bool found_it = new_bc.find(type);
    if (found_it == true)
      return new_bc.getBCValues(type);
  }
  // Check the mat_id = "all" case
  itr = d_BCDataArray.find(-1);
  if (itr  != d_BCDataArray.end()) {
    if (i < (int)itr->second.size()) {
      itr->second[i]->getBCData(new_bc_all);
      bool found_it = new_bc_all.find(type);
      if (found_it == true)
	return new_bc_all.getBCValues(type);
      else
	return 0;
    }
  }
  return 0;
}


void BCDataArray::setBoundaryIterator(int mat_id,vector<IntVector>& b,int i)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[i]->setBoundaryIterator(b);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[i]->setBoundaryIterator(b);
  }
}

void BCDataArray::setNBoundaryIterator(int mat_id,vector<IntVector>& b,int i)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[i]->setNBoundaryIterator(b);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[i]->setNBoundaryIterator(b);
  }
}

void BCDataArray::setSFCXIterator(int mat_id,vector<IntVector>& i,int ii)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[ii]->setSFCXIterator(i);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->setSFCXIterator(i);
  }
}

void BCDataArray::setSFCYIterator(int mat_id,vector<IntVector>& i,int ii)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[ii]->setSFCYIterator(i);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->setSFCYIterator(i);
  }
}

void BCDataArray::setSFCZIterator(int mat_id,vector<IntVector>& i,int ii)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[ii]->setSFCZIterator(i);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->setSFCZIterator(i);
  }
}

void BCDataArray::getBoundaryIterator(int mat_id,vector<IntVector>& b,
				      int i) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[i]->getBoundaryIterator(b);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[i]->getBoundaryIterator(b);
  }
}

void BCDataArray::getNBoundaryIterator(int mat_id,vector<IntVector>& b,
				       int i) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[i]->getNBoundaryIterator(b);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[i]->getNBoundaryIterator(b);
  }
}

void BCDataArray::getSFCXIterator(int mat_id,vector<IntVector>& i,int ii) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[ii]->getSFCXIterator(i);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->getSFCXIterator(i);
  }
}

void BCDataArray::getSFCYIterator(int mat_id,vector<IntVector>& i,int ii) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[ii]->getSFCYIterator(i);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->getSFCYIterator(i);
  }
}

void BCDataArray::getSFCZIterator(int mat_id,vector<IntVector>& i,int ii) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[ii]->getSFCZIterator(i);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->getSFCZIterator(i);
  }
}

int BCDataArray::getNumberChildren(int mat_id) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    return itr->second.size();
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      return itr->second.size();
  }
  return 0;
}

BCGeomBase* BCDataArray::getChild(int mat_id,int i) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    return itr->second[i];
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      return itr->second[i];
  }
  return 0;
}

void BCDataArray::print()
{
  bcDataArrayType::const_iterator bcda_itr;
  for (bcda_itr = d_BCDataArray.begin(); bcda_itr != d_BCDataArray.end(); 
       bcda_itr++) {
    BCDA_dbg << endl << "mat_id = " << bcda_itr->first << endl;
    for (vector<BCGeomBase*>::const_iterator i = bcda_itr->second.begin();
	 i != bcda_itr->second.end(); ++i) {
      BCDA_dbg << "BCGeometry Type = " << typeid((*i)).name() <<  " "
	   << *i << endl;
      (*i)->print();
    }
  }
	
  
}
