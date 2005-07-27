#ifndef __INPUTS_H__
#define __INPUTS_H__

#include <vector>
#include <stdlib.h>

#include "mydriver.h"

using std::vector;

void initTest(Param& param);

double diffusion(const int problemType,
                 const Location& x);

#endif // __INPUTS_H__
