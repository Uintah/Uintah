/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  IndexedGroup.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef SCI_Geom_IndexedGroup_h
#define SCI_Geom_IndexedGroup_h 1

#include <Core/Containers/Array1.h>
#include <map>

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

using namespace std;

class SCICORESHARE GeomIndexedGroup: public GeomObj {
    
public:
    typedef map< int, GeomHandle, less<int> > MapIntGeomObj;
    
    typedef pair< MapIntGeomObj::iterator,
                  MapIntGeomObj::iterator >
        IterIntGeomObj;

private:
    
    MapIntGeomObj objs;
    
public:

    GeomIndexedGroup( const GeomIndexedGroup& );
    GeomIndexedGroup();
    virtual ~GeomIndexedGroup();

    virtual GeomObj* clone();
    virtual void reset_bbox();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    void addObj(GeomHandle obj, int id);  // adds an object to the table
    GeomHandle getObj(int id);        // gets an object from the table
    void delObj(int id);       	 // removes object from table
    void delAll();		 // deletes everything

    IterIntGeomObj getIter(); // gets an iter 
    
    MapIntGeomObj* getHash(); // gets the table
};

} // End namespace SCIRun


#endif
