// $Id$

#include "qslim.h"
#include "decimate.h"

#include <gfx/tools/Buffer.h>
#include <gfx/tools/Heap.h>
#include <gfx/geom/ProxGrid.h>

class pair_info : public Heapable
{
public:
    Vertex *v0, *v1;

    Vec3 candidate;
    real cost;

    pair_info(Vertex *a,Vertex *b) { v0=a; v1=b; cost=HUGE; }

    bool isValid() { return v0->isValid() && v1->isValid(); }
};

typedef buffer<pair_info *> pair_buffer;

class vert_info
{
public:

    pair_buffer pairs;

    Mat4 Q;
    real norm;

    vert_info() : Q(Mat4::zero) { pairs.init(2); norm=0.0; }
};

int will_draw_pairs = 0;

static Heap *heap;
static array<vert_info> vinfo;
static real proximity_limit;    // distance threshold squared



////////////////////////////////////////////////////////////////////////
//
// Low-level routines for manipulating pairs
//

static inline
vert_info& vertex_info(Vertex *v)
{
    return vinfo(v->validID());
}

static
bool check_for_pair(Vertex *v0, Vertex *v1)
{
    const pair_buffer& pairs = vertex_info(v0).pairs;

    for(int i=0; i<pairs.length(); i++)
    {
	if( pairs(i)->v0==v1 || pairs(i)->v1==v1 )
	    return true;
    }

    return false;
}

static
pair_info *new_pair(Vertex *v0, Vertex *v1)
{
    vert_info& v0_info = vertex_info(v0);
    vert_info& v1_info = vertex_info(v1);

    pair_info *pair = new pair_info(v0,v1);
    v0_info.pairs.add(pair);
    v1_info.pairs.add(pair);

    return pair;
}

static
void delete_pair(pair_info *pair)
{
    vert_info& v0_info = vertex_info(pair->v0);
    vert_info& v1_info = vertex_info(pair->v1);

    v0_info.pairs.remove(v0_info.pairs.find(pair));
    v1_info.pairs.remove(v1_info.pairs.find(pair));

    if( pair->isInHeap() )
	heap->kill(pair->getHeapPos());

    delete pair;
}



////////////////////////////////////////////////////////////////////////
//
// The actual guts of the algorithm:
//
//     - pair_is_valid
//     - compute_pair_info
//     - do_contract
//

static
bool pair_is_valid(Vertex *u, Vertex *v)
{
    return norm2(*u - *v) < proximity_limit;
}

static
int predict_face(Face& F, Vertex *v1, Vertex *v2, Vec3& vnew,
		 Vec3& f1, Vec3& f2, Vec3& f3)
{
    int nmapped = 0;

    if( F.vertex(0) == v1 || F.vertex(0) == v2 )
    { f1 = vnew;  nmapped++; }
    else f1 = *F.vertex(0);

    if( F.vertex(1) == v1 || F.vertex(1) == v2 )
    { f2 = vnew;  nmapped++; }
    else f2 = *F.vertex(1);

    if( F.vertex(2) == v1 || F.vertex(2) == v2 )
    { f3 = vnew;  nmapped++; }
    else f3 = *F.vertex(2);

    return nmapped;
}

#define MESH_INVERSION_PENALTY 1e9

static
real pair_mesh_penalty(Model& M, Vertex *v1, Vertex *v2, Vec3& vnew)
{
    static face_buffer changed;
    changed.reset();

    M.contractionRegion(v1, v2, changed);

    // real Nsum = 0;
    real Nmin = 0;

    for(int i=0; i<changed.length(); i++)
    {
	Face& F = *changed(i);
	Vec3 f1, f2, f3;

	int nmapped = predict_face(F, v1, v2, vnew, f1, f2, f3);

	//
	// Only consider non-degenerate faces
	if( nmapped < 2 )
	{
	    Plane Pnew(f1, f2, f3);
	    real delta =  Pnew.normal() * F.plane().normal();

	    if( Nmin > delta ) Nmin = delta;
	}
    }

    //return (-Nmin) * MESH_INVERSION_PENALTY;
    if( Nmin < 0.0 )
	return MESH_INVERSION_PENALTY;
    else
	return 0.0;
}

static
void compute_pair_info(pair_info *pair)
{
    Vertex *v0 = pair->v0;
    Vertex *v1 = pair->v1;

    vert_info& v0_info = vertex_info(v0);
    vert_info& v1_info = vertex_info(v1);

    Mat4 Q = v0_info.Q + v1_info.Q;
    real norm = v0_info.norm + v1_info.norm;

    pair->cost = quadrix_pair_target(Q, v0, v1, pair->candidate);

    if( will_weight_by_area )
 	pair->cost /= norm;

    if( will_preserve_mesh_quality )
	pair->cost += pair_mesh_penalty(M0, v0, v1, pair->candidate);


    //
    // NOTICE:  In the heap we use the negative cost.  That's because
    //          the heap is implemented as a MAX heap.
    //
    if( pair->isInHeap() )
    {
	heap->update(pair, (float)-pair->cost);
    }
    else
    {
	heap->insert(pair, (float)-pair->cost);
    }
}

static
void do_contract(Model* m, pair_info *pair)
{
    Vertex *v0 = pair->v0;  Vertex *v1 = pair->v1;
    vert_info& v0_info = vertex_info(v0);
    vert_info& v1_info = vertex_info(v1);
    Vec3 vnew = pair->candidate;
    int i;

    //
    // Make v0 be the new vertex
    v0_info.Q += v1_info.Q;
    v0_info.norm += v1_info.norm;

    //
    // Perform the actual contraction
    static face_buffer changed;
    changed.reset();
    m->contract(v0, v1, vnew, changed);

#ifdef SUPPORT_VCOLOR
    //
    // If the vertices are colored, color the new vertex
    // using the average of the old colors.
    v0->props->color += v1->props->color;
    v0->props->color /= 2;
#endif

    //
    // Remove the pair that we just contracted
    delete_pair(pair);

    //
    // Recalculate pairs associated with v0
    for(i=0; i<v0_info.pairs.length(); i++)
    {
	pair_info *p = v0_info.pairs(i);
	compute_pair_info(p);
    }

    //
    // Process pairs associated with now dead vertex

    static pair_buffer condemned(6); // collect condemned pairs for execution
    condemned.reset();

    for(i=0; i<v1_info.pairs.length(); i++)
    {
	pair_info *p = v1_info.pairs(i);

	Vertex *u;
	if( p->v0 == v1 )      u = p->v1;
	else if( p->v1 == v1)  u = p->v0;
	else cerr << "YOW!  This is a bogus pair." << endl;


	if( !check_for_pair(u, v0) )
	{
	    p->v0 = v0;
	    p->v1 = u;
	    v0_info.pairs.add(p);
	    compute_pair_info(p);
	}
	else
	    condemned.add(p);
    }

    for(i=0; i<condemned.length(); i++)
	// Do you have any last requests?
	delete_pair(condemned(i));
    v1_info.pairs.reset(); // safety precaution
}




////////////////////////////////////////////////////////////////////////
//
// External interface: setup and single step iteration
//

bool decimate_quadric(Vertex *v, Mat4& Q)
{
    if( vinfo.length() > 0 )
    {
	Q = vinfo(v->uniqID).Q;
	return true;
    }
    else
	return false;
}


void decimate_contract(Model* m)
{
    heap_node *top;
    pair_info *pair;

    for(;;)
    {
	top = heap->extract();
	if( !top ) return;
	pair = (pair_info *)top->obj;

	//
	// This may or may not be necessary.  I'm just not quite
	// willing to assume that all the junk has been removed from the
	// heap.
	if( pair->isValid() )
	    break;

	delete_pair(pair);
    }

    do_contract(m, pair);

    if( logfile && (selected_output&OUTPUT_COST) )
	*logfile << "#$cost " << m->validFaceCount << " "
		 << pair->cost << endl;

    //    M0.validVertCount--;  
    // Attempt to maintain valid vertex information
	
    m->validVertCount--;
}

real decimate_error(Vertex *v)
{
    vert_info& info = vertex_info(v);

    real err = quadrix_evaluate_vertex(*v, info.Q);

    if( will_weight_by_area )
	err /= info.norm;

    return err;
}

real decimate_min_error()
{
    heap_node *top;
    pair_info *pair;

    for(;;)
    {
	top = heap->top();
	if( !top ) return -1.0;
	pair = (pair_info *)top->obj;

	if( pair->isValid() )
	    break;

	top = heap->extract();
	delete_pair(pair);
    }

    return pair->cost;
}

real decimate_max_error(Model& m)
{
    real max_err = 0;

    for(int i=0; i<m.vertCount(); i++)
	if( m.vertex(i)->isValid() )
	{
	    max_err = MAX(max_err, decimate_error(m.vertex(i)));
	}

    return max_err;
}


void decimate_init(Model* m, real limit)
{
    int i,j;

    vinfo.init(m->vertCount());

    cout << "  Decimate:  Distributing shape constraints." << endl;
    
    if( will_use_vertex_constraint )
	for(i=0; i<m->vertCount(); i++)
	{
	    Vertex *v = m->vertex(i);
	    if( v->isValid() )
		vertex_info(v).Q = quadrix_vertex_constraint(*v);
	}

    for(i=0; i<m->faceCount(); i++)
	if( m->face(i)->isValid() )
	{
	    if( will_use_plane_constraint )
	    {
		Mat4 Q = quadrix_plane_constraint(*m->face(i));
		real norm = 0.0;

		if( will_weight_by_area )
		{
		    norm = m->face(i)->area();
		    Q *= norm;
		}

		for(j=0; j<3; j++)
		{
		    vertex_info(m->face(i)->vertex(j)).Q += Q;
		    vertex_info(m->face(i)->vertex(j)).norm += norm;
		    
		}
	    }
	}

    if( will_constrain_boundaries )
    {
	cout << "  Decimate:  Accumulating discontinuity constraints." << endl;
	for(i=0; i<m->edgeCount(); i++)
	    if( m->edge(i)->isValid() && check_for_discontinuity(m->edge(i)) )
	    {
		Mat4 B = quadrix_discontinuity_constraint(m->edge(i));
		real norm = 0.0;

		if( will_weight_by_area )
		{
		    norm = norm2(*m->edge(i)->org() - *m->edge(i)->dest());
		    B *= norm;
		}

		B *= boundary_constraint_weight;
		vertex_info(m->edge(i)->org()).Q += B;
		vertex_info(m->edge(i)->org()).norm += norm;
		vertex_info(m->edge(i)->dest()).Q += B;
		vertex_info(m->edge(i)->dest()).norm += norm;
	    }
    }

    cout << "  Decimate:  Allocating heap." <<m->validEdgeCount << endl;
    heap = new Heap(m->validEdgeCount);

    int pair_count = 0;

    cout << "  Decimate:  Collecting pairs [edges]." << endl;
    for(i=0; i<m->edgeCount(); i++)
	if( m->edge(i)->isValid() )
	{
	    pair_info *pair = new_pair(m->edge(i)->org(), m->edge(i)->dest());
	    compute_pair_info(pair);
	    pair_count++;
	}

    if( limit<0 )
    {
	limit = m->bounds.radius * 0.05;
	cout << "  Decimate:  Auto-limiting at 5% of model radius." << endl;
    }
    proximity_limit = limit * limit;
    if( proximity_limit > 0 )
    {
	cout << "  Decimate:  Collecting pairs [limit="<<limit<<"]." << endl;
	ProxGrid grid(m->bounds.min, m->bounds.max, limit);
	for(i=0; i<m->vertCount(); i++)
	    grid.addPoint(m->vertex(i));

	buffer<Vec3 *> nearby(32);
	for(i=0; i<m->vertCount(); i++)
	{
	    nearby.reset();
	    grid.proximalPoints(m->vertex(i), nearby);

	    for(j=0; j<nearby.length(); j++)
	    {
		Vertex *v1 = m->vertex(i);
		Vertex *v2 = (Vertex *)nearby(j);

		if( v1->isValid() && v2->isValid() )
		{
#ifdef SAFETY
		    assert(pair_is_valid(v1,v2));
#endif
		    if( !check_for_pair(v1,v2) )
		    {
			pair_info *pair = new_pair(v1,v2);
			compute_pair_info(pair);
			pair_count++;
		    }
		}

	    }
	}
    }
    else
	cout << "  Decimate:  Ignoring non-edge pairs [limit=0]." << endl;

    cout << "  Decimate:  Designated " << pair_count << " pairs." << endl;
}
