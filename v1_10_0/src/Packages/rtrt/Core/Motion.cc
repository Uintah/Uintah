#include <iostream>
#include <Packages/rtrt/Core/Motion.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>

using namespace rtrt;
using namespace SCIRun;
using SCIRun::Thread;
using SCIRun::Time;

Motion::Motion (Object *object, int x, int y, double iyres,
		const Vector& u, const Vector& v, 
		double depth)
  : object(object), x(x), y(y), iyres(iyres), u(u), v(v), depth(depth)
{
  continuous = false;
  selected = NOT_SELECTED;    // No object selected
  translation.x (0);          // Reset cumulative translation of object
  translation.y (0);
  translation.z (0);
}

Motion::Motion ()
{
  object = NULL;
}

Motion::~Motion ()
{
}

void Motion::set_scene (Scene *sceneptr)
{
  BBox bbox;
  scene = sceneptr;
  scene->get_object()->compute_bounds (bbox, 0);
  factor = bbox.diagonal().length() / 110.;
}

void Motion::set_continuous (bool is_continuous)
{
  continuous = is_continuous;
}

void Motion::write_stats ()
{
  if (total_updates)
  {
    cerr << "Time per update: " << total_time/total_updates << " (" << total_updates << " updates)\n";
    cerr << "Update rate:     " << total_updates/total_time << "\n";
  }
  else
    cerr << "No updates recorded\n";
}

void Motion::reset_timer ()
{
  total_time    = 0.;
  total_updates = 0;
}

void Motion::set (Object *nobject, int nx, int ny, double niyres, 
		  const Vector& nu, const Vector& nv, 
		  double ndepth)
{
  double offset = 0;             // Not sure what this is used for....

  object   = nobject;            // Animated object
  x        = nx;                 // Screen coordinates
  y        = ny;
  iyres    = niyres;             // Screen y resolution
  u        = nu;                 // Image plane vectors
  v        = nv;
  depth    = ndepth;             // Depth of object
  bbox.reset();
  object->disallow_animation ();
  object->compute_bounds (bbox, offset);
  if (continuous)
    selected = CONTINUOUS;       // Continuous updates
  else
  {
    selected = SEPARATE;         // No updates during motion
    translation.x (0);           // Reset cumulative translation
    translation.y (0);
    translation.z (0);
    scene->get_object()->remove (object, bbox);
    group_index = scene->get_group()->add2 (object);
    scene->group_count++;
  }
  object->print (cerr);
}

void Motion::unset ()
{
  if (selected & SEPARATE)
  {
    bbox.translate (translation);               // Update bounding box
    scene->get_object()->insert (object, bbox); // Reinsert object
    scene->group_count--;                       // One object less to render separately
    scene->get_group()->remove2 (group_index);  // Remove from outside group
  }
  object->allow_animation ();                   // If animated, continue animation
  object   = NULL;                              // Object no longer selected
  selected = false;                             // Object deselected
}

// Move object in viewingplane

void Motion::update (double nx, double ny)
{
  double dx = (nx - x)/iyres;
  double dy = (y - ny)/iyres;
  Vector sv (v*dx);
  Vector su (u*dy);
  x = nx;
  y = ny;
  object->update (su + sv);
  if (selected & CONTINUOUS)
  {
    double tnow = Time::currentSeconds();
    scene->get_object()->remove (object, bbox);
    bbox.translate (su + sv);
    scene->get_object()->insert (object, bbox);
    total_time += Time::currentSeconds()-tnow;
    total_updates++;
   }
  else
    translation += su + sv;
}

// Move object perpendicular to viewingplane

void Motion::zoom (double nx, double ny)
{
  // Compute ray direction

  Vector raydir = u.cross (v);
  raydir.normalize ();

  // Further or closer?

  double dx = nx - x;
  double dy = y - ny;
  double dom;
  x = nx;
  y = ny;

  dom =  (Abs (dx) > Abs (dy)) ? dx : dy;
  if (Abs (dom) < 1e-6)
    return;
  if (dom < 0)
    raydir = -raydir;
  raydir = raydir * factor;
  object->update (raydir);
  if (selected & CONTINUOUS)
  {
    double tnow = Time::currentSeconds();
    scene->get_object()->remove (object, bbox);
    bbox.translate (raydir);
    scene->get_object()->insert (object, bbox);
    total_time += Time::currentSeconds()-tnow;
    total_updates++;
  }
  else
    translation += raydir;
}
