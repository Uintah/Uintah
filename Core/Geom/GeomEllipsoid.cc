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

#include "GeomEllipsoid.h"
#include <Core/Geometry/BBox.h>
//#include <Core/Geometry/BSphere.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


Persistent* make_GeomEllipsoid()
{
    return scinew GeomEllipsoid;
}

PersistentTypeID GeomEllipsoid::type_id("GeomEllipsoid", "GeomSphere", make_GeomEllipsoid);



GeomEllipsoid::GeomEllipsoid()
{
}

GeomEllipsoid::GeomEllipsoid(const Point& point, double radius,
			     int inu, int inv,
			     double* matrix, double mev) :
  GeomSphere(point, radius, inu, inv),
  mev(mev)
{
  for (int y = 0; y < 16; y++)
  {
    m_tensor_matrix[y] = matrix[y]; 
  }
}


GeomEllipsoid::~GeomEllipsoid()
{
}

void
GeomEllipsoid::get_bounds(BBox& bb)
{
    bb.extend(cen, mev);
}

// void GeomEllipsoid::get_bounds(BSphere& bs)
// {
//     bs.extend(cen, mev*1.000001);
// }
} // End namespace SCIRun
