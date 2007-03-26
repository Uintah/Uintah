#include <Packages/Wangxl/Core/Datatypes/Mesh/Utilities.h>

namespace Wangxl {

const char Utilities::tab_next_around_edge[4][4] = {
      {5, 2, 3, 1},
      {3, 5, 0, 2},
      {1, 3, 5, 0},
      {2, 0, 1, 5} };
//{{0,2,3,1},{3,0,0,2},{1,3,0,0},{2,0,1,0}};
unsigned int  Utilities::random_value = 0;
unsigned int  Utilities::count = 0;
unsigned int  Utilities::val = 0;
}



/*const char Triangulation_utils_3::tab_next_around_edge[4][4] = {
      {5, 2, 3, 1},
      {3, 5, 0, 2},
      {1, 3, 5, 0},
      {2, 0, 1, 5} };

*/
