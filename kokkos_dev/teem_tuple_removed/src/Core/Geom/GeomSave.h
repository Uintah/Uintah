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
 *  GeomSave.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef GeomSave_h
#define GeomSave_h 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {


struct GeomSave {
    int nindent;
    void indent();
    void unindent();
    void indent(std::ostream&);

    // For VRML.
    void start_sep(std::ostream&);
    void end_sep(std::ostream&);
    void start_tsep(std::ostream&);
    void end_tsep(std::ostream&);
    void orient(std::ostream&, const Point& center, const Vector& up,
		const Vector& new_up=Vector(0,1,0));

    // For RIB.
    void start_attr(std::ostream&);
    void end_attr(std::ostream&);
    void start_trn(std::ostream&);
    void end_trn(std::ostream&);
    void rib_orient(std::ostream&, const Point& center, const Vector& up,
		const Vector& new_up=Vector(0,1,0));

    void translate(std::ostream&, const Point& p);
    void rotateup(std::ostream&, const Vector& up, const Vector& new_up);
    void start_node(std::ostream&, char*);
    void end_node(std::ostream&);
};

} // End namespace SCIRun


#endif
