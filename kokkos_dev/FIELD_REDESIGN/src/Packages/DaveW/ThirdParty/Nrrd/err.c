#include "include/nrrd.h"

static char errStr[NRRD_ERR_STRLEN] = "";

void
nrrdClearErr() {
  
  errStr[0] = 0;
}

int
nrrdSetErr(char *str) {
  int len;

  len = strlen(str);
  strncpy(errStr, str, NRRD_ERR_STRLEN-1);
  errStr[NRRD_ERR_STRLEN-1] = 0;
  if (len > NRRD_ERR_STRLEN-1) {
    return(1);
  }
  else {
    return(0);
  }
}

int
nrrdAddErr(char *str) {
  int len, newlen;
  
  len = strlen(errStr);
  newlen = NRRD_ERR_STRLEN-1-len;
  strncat(errStr, str, newlen);
  if (len > newlen) {
    return(1);
  }
  else {
    return(0);
  }
}

char *
nrrdStrdupErr() {

  return(strdup(errStr));
}

int
nrrdGetErr(char *str) {
  
  strcpy(str, errStr);
  return(0);
}
