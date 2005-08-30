/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

class GeomIndexedGroup: public GeomObj {
    
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
