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
 *  Surface.cc: The Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <FieldConverters/Core/Datatypes/Surface.h>
#include <Core/Util/NotFinished.h>

namespace FieldConverters {

PersistentTypeID Surface::type_id("Surface", "Datatype", 0);

Surface::Surface(Representation rep, int closed)
  : monitor("Surface crowd monitor"),
    rep(rep), closed(closed), grid(0), pntHash(0), boundary_type(BdryNone)
{
}

Surface::~Surface()
{
    destroy_grid();
    destroy_hash();
}

Surface::Surface(const Surface& copy)
  : monitor("Surface crowd monitor"), name(copy.name), rep(copy.rep),
    closed(copy.closed), grid(0), pntHash(0)
{
//    NOT_FINISHED("Surface::Surface");
}

void Surface::destroy_grid()
{
    if (grid) delete grid;
}

void Surface::destroy_hash() {
    if (pntHash) delete pntHash;
}

#define SURFACE_VERSION 4

void Surface::io(Piostream& stream) {


    int version=stream.begin_class("Surface", SURFACE_VERSION);
    Pio(stream, name);
    if (version >= 4){
	int* repp=(int*)&rep;
	Pio(stream, *repp);
	int* btp=(int*)&boundary_type;
	Pio(stream, *btp);
	Pio(stream, boundary_expr);
    }	
    if (version == 3){
	int* btp=(int*)&boundary_type;
	Pio(stream, *btp);
	Pio(stream, boundary_expr);
    }
    if (version == 2) {
	Array1<double> conductivity;
	Pio(stream, conductivity);
	int bt;
	Pio(stream, bt);
    }
    stream.end_class();
}

SurfTree* Surface::getSurfTree() {
    if (rep==STree)
	return (SurfTree*) this;
    else
	return 0;
}

TriSurfFieldace* Surface::getTriSurfFieldace()
{
    if(rep==TriSurfField)
	return (TriSurfFieldace*)this;
    else
	return 0;
}

PointsSurface* Surface::getPointsSurface() {
    if (rep==PointsSurf)
	return (PointsSurface*)this;
    else
	return 0;
}

void Surface::set_bc(const string& bc_expr)
{
    boundary_expr=bc_expr;
    boundary_type=DirichletExpression;
}

} // End namespace FieldConverters

