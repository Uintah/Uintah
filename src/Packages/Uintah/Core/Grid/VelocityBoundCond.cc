#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <iostream>
#include <stdlib.h>
#include <Core/Malloc/Allocator.h>
using namespace Uintah;

VelocityBoundCond::VelocityBoundCond(ProblemSpecP& ps,const std::string& kind)
  : BoundCond<Vector>(kind)
{
  d_type = "Velocity";
  if (kind != "Mixed")
    ps->require("value",d_value);
  else {
    for (ProblemSpecP child = ps->findBlock("value"); child != 0;
	 child = child->findNextBlock("value")) {
      map<string,string> bc_attr;
      child->getAttributes(bc_attr);
      d_comp_var[bc_attr["comp"]] = bc_attr["var"];
      if (bc_attr["comp"] == "x") {
	string v = bc_attr["val"];
	d_value.x(atof(v.c_str()));
      }
      if (bc_attr["comp"] == "y") {
	string v = bc_attr["val"];
	d_value.y(atof(v.c_str()));
      }
	
      if (bc_attr["comp"] == "z") {
	string v = bc_attr["val"];
	d_value.z(atof(v.c_str()));
      }

    }
  }
          
}


map<string,string> VelocityBoundCond::getMixed() const
{
  return d_comp_var;
}
VelocityBoundCond::~VelocityBoundCond()
{
}

VelocityBoundCond* VelocityBoundCond::clone()
{
  return scinew VelocityBoundCond(*this);

}

