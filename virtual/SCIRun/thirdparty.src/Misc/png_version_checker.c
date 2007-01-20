#include <png.h>
#include <stdio.h>

int
main() {
  if (PNG_LIBPNG_VER < 10002) {
    return 1;
  } else {
    return 0;
  }	   
}

