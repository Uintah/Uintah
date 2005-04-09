/*
 *  Datatype.cc: The Datatype Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/Datatype.h>

Mutex generation_lock;
int current_generation;

Datatype::Datatype() {
    ref_cnt=0;
    generation_lock.lock();
    generation=++current_generation;
    generation_lock.unlock();
}

Datatype::~Datatype() {
}
