
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

using namespace Uintah;

ExamplesLabel::ExamplesLabel()
{
  phi = new VarLabel("phi", NCVariable<double>::getTypeDescription());
  residual = new VarLabel("residual", sum_vartype::getTypeDescription());
}

ExamplesLabel::~ExamplesLabel()
{
  delete phi;
  delete residual;
}

