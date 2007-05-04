#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometryFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometry.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/NullCrack.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/QuadCrack.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/CurvedQuadCrack.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/TriangularCrack.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/ArcCrack.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/EllipticCrack.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/PartialEllipticCrack.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>
using std::cerr;

using namespace Uintah;

CrackGeometry* CrackGeometryFactory::create(ProblemSpecP& ps)
{
  ProblemSpecP child = ps->findBlock("crack");
  if(!child)
    return scinew NullCrack(ps);

  for (ProblemSpecP crack_segment_ps = child->findBlock(); 
       crack_segment_ps != 0; 
       crack_segment_ps = crack_segment_ps->findNextBlock()) {
    string crack_type = crack_segment_ps->getNodeName();

    if (crack_type == "quad")
      return scinew QuadCrack(crack_segment_ps);

    else if (crack_type == "curved_quad")
      return scinew CurvedQuadCrack(crack_segment_ps);

    else if (crack_type == "triangle")
      return scinew TriangularCrack(crack_segment_ps);

    else if (crack_type == "arc")
      return scinew ArcCrack(crack_segment_ps);

    else if (crack_type == "ellipse")
      return scinew EllipticCrack(crack_segment_ps);

    else if (crack_type == "partial_ellipse")
      return scinew PartialEllipticCrack(crack_segment_ps);

    else 
      throw ProblemSetupException("Unknown Crack Segment Type R ("+crack_type+")", __FILE__, __LINE__);
  }

  return 0;
}


