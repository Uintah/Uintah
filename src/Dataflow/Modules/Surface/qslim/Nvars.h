#ifndef NAUTILUS_NVARS_INCLUDED // -*- C++ -*-
#define NAUTILUS_NVARS_INCLUDED

/************************************************************************

  $Id$

 ************************************************************************/

extern ostream *logfile;

#define OUTPUT_NONE         0x00
#define OUTPUT_CONTRACTIONS 0x01
#define OUTPUT_QUADRICS     0x02
#define OUTPUT_COST         0x04
#define OUTPUT_VERT_NOTES   0x08
#define OUTPUT_FACE_NOTES   0x10
#define OUTPUT_MODEL_DEFN   0x20
#define OUTPUT_ALL          0xFFFFFFFF

extern unsigned int selected_output;

extern char *global_cmdline_options;
extern char *nautilus_usage_string;
extern void nautilus_add_options(char *);
extern bool nautilus_parseopt(int opt, char *optarg);


// NAUTILUS_NVARS_INCLUDED
#endif
