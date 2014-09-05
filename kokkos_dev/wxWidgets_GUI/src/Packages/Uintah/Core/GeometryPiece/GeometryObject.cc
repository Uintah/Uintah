#include <Packages/Uintah/Core/GeometryPiece/GeometryObject.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;

GeometryObject::GeometryObject(GeometryPiece* piece, ProblemSpecP& ps,
                               list<string>& data)
   : d_piece(piece)
{
   ps->require("res", d_resolution);
   ps->require("velocity", d_initialVel);

   for (list<string>::iterator it = data.begin(); it != data.end();it++){
     double val;
     ps->require(*it,val);
     d_data[*it] = val;
   }

}

GeometryObject::~GeometryObject()
{
  delete d_piece;
}

void GeometryObject::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP geom_obj_ps = ps->appendChild("geom_object",true,3);
  d_piece->outputProblemSpec(geom_obj_ps);
  
  geom_obj_ps->appendElement("res", d_resolution,false,4);
  geom_obj_ps->appendElement("velocity", d_initialVel,false,4);
  for (map<string,double>::iterator it = d_data.begin(); 
       it != d_data.end(); it++) {
    geom_obj_ps->appendElement(it->first.c_str(),it->second,false,4);
  }

}

IntVector GeometryObject::getNumParticlesPerCell()
{
  return d_resolution;
}

