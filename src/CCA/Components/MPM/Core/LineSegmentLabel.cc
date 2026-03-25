/*
 * Copyright © 2025 by Geocosm LLC                                   
 */
#include <CCA/Components/MPM/Core/LineSegmentLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using namespace Uintah;
using namespace std;

LineSegmentLabel::LineSegmentLabel()
{
  // Reduction variables
  lineSegmentCountLabel = VarLabel::create("lineSegmentCount",
                                   sumlong_vartype::getTypeDescription());

  linesegIDLabel = VarLabel::create("ls.linesegID",
			ParticleVariable<long64>::getTypeDescription() );

  linesegIDLabel_preReloc = VarLabel::create("ls.linesegID+",
			ParticleVariable<long64>::getTypeDescription() );

  lsMidToEndVectorLabel = VarLabel::create("ls.MidToEndVector",
			ParticleVariable<Vector>::getTypeDescription() );

  lsMidToEndVectorLabel_preReloc = VarLabel::create("ls.MidToEndVector+",
			ParticleVariable<Vector>::getTypeDescription() );
}

LineSegmentLabel::~LineSegmentLabel()
{
  VarLabel::destroy(lineSegmentCountLabel);
  VarLabel::destroy(linesegIDLabel);
  VarLabel::destroy(linesegIDLabel_preReloc);
  VarLabel::destroy(lsMidToEndVectorLabel);
  VarLabel::destroy(lsMidToEndVectorLabel_preReloc);
}
