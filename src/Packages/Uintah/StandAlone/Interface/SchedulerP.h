
#ifndef UINTAH_HOMEBREW_SchedulerP_H
#define UINTAH_HOMEBREW_SchedulerP_H

namespace Uintah {
    namespace Grid {
	template<class T> class Handle;
    }
    namespace Interface {
	class Scheduler;
	typedef Uintah::Grid::Handle<Scheduler> SchedulerP;
    }
}

#endif
