
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

using namespace Uintah;

ExamplesLabel::ExamplesLabel()
{
  phi = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
  residual = VarLabel::create("residual", sum_vartype::getTypeDescription());

  //For Burger's
  u = VarLabel::create("u", NCVariable<double>::getTypeDescription());

  // For SimpleCFD
  bctype = VarLabel::create("bctype",
			    NCVariable<double>::getTypeDescription());
  xvelocity = VarLabel::create("xvelocity",
			       SFCXVariable<double>::getTypeDescription());
  yvelocity = VarLabel::create("yvelocity",
			       SFCYVariable<double>::getTypeDescription());
  zvelocity = VarLabel::create("zvelocity",
			       SFCZVariable<double>::getTypeDescription());
  density = VarLabel::create("density",
			     CCVariable<double>::getTypeDescription());
}

ExamplesLabel::~ExamplesLabel()
{
  VarLabel::destroy(phi);
  VarLabel::destroy(residual);
  VarLabel::destroy(u);
}

