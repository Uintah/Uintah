
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

using namespace Uintah;

ExamplesLabel::ExamplesLabel()
{
  phi = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
  residual = VarLabel::create("residual", sum_vartype::getTypeDescription());
}

ExamplesLabel::~ExamplesLabel()
{
  VarLabel::destroy(phi);
  VarLabel::destroy(residual);
}

