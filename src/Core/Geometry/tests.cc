#include <Core/Tester/TestTable.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

using namespace SCIRun;

TestTable test_table[] = {
        {"Vector_Point", &Point::test_rigorous, 0},
	{0,0,0}
};

