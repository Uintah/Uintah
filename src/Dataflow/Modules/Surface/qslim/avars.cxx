#include <stdlib.h>
#include <iostream.h>

#include <gfx/std.h>
#include "avars.h"
#include "AdjModel.h"

ostream *outfile=NULL;

int face_target = 0;
real error_tolerance = HUGE;


bool will_use_plane_constraint = true;
bool will_use_vertex_constraint = false;

bool will_preserve_boundaries = false;
bool will_preserve_mesh_quality = false;
bool will_constrain_boundaries = false;
real boundary_constraint_weight = 1.0;

bool will_weight_by_area = false;

int placement_policy = PLACE_OPTIMAL;

real pair_selection_tolerance = 0.0;


Model M0;


void read_model(SMF_Reader& reader)
{
    reader.read(&M0);
    M0.bounds.complete();
}
