#include "Nvars.h"
#include <gfx/SMF/smf.h>

extern ostream *outfile;

extern int face_target;
extern real error_tolerance;


extern bool will_use_plane_constraint;
extern bool will_use_vertex_constraint;

extern bool will_preserve_boundaries;
extern bool will_preserve_mesh_quality;
extern bool will_constrain_boundaries;
extern real boundary_constraint_weight;

extern bool will_weight_by_area;

#define PLACE_ENDPOINTS 0
#define PLACE_ENDORMID  1
#define PLACE_LINE      2
#define PLACE_OPTIMAL   3

extern int placement_policy;

extern real pair_selection_tolerance;


extern void process_cmdline(int argc, char **argv);

extern void read_model(SMF_Reader&);
