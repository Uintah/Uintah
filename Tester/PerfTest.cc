
#include <Tester/PerfTest.h>
#include <iostream.h>
#include <limits.h>
#include <string.h>
#include <sys/times.h>
#include <unistd.h>

PerfTest::PerfTest(char* symname)
    : symname(symname)
{
    baseline=0;
}

PerfTest::~PerfTest()
{
}
void PerfTest::print_msg()
{
    cout << max;
    cout.flush();
}
void PerfTest::start_test(char* name)
{
    cout << "\nStarting test: " << name << endl;
    count=0;
    max=1;
    print_msg();
    start_time=time();
    if(strcmp(name, "baseline") == 0)
	is_baseline=true;
    else
	is_baseline=false;
    if(strcmp(name, "sanity") == 0)
	is_sanity=true;
    else
	is_sanity=false;
}
bool PerfTest::do_test()
{
    if(count++<max)
	return true;
    long stop_time=time();
    double dt=deltat(stop_time-start_time);
    cout << '(' << dt*1000 << " ms)";
    if(dt < MINTIME) {
	// Continue the tests...
	if(dt == 0)
	    max*=MULFACTOR;
	else if(MINTIME/dt > MULFACTOR)
	    max*=MULFACTOR;
	else
	    max*=MINTIME/dt*1.2;
	count=0;
	if(dt==0)
	    cout << '\r';
	else
	    cout << ' ';
	print_msg();
	start_time=time();
	return true;
    }
    dt/=max;
    if(is_baseline)
	baseline=dt;
    else
	dt-=baseline;
    cout << "\n ->" << dt*1000 << " ms/operation";
    if(!is_baseline)
	cout << " above baseline\n";
    else
	cout << " baseline\n";
    return false;
}

void PerfTest::finish()
{
}

long PerfTest::time()
{
    struct tms buf;
    return times(&buf);
}

double PerfTest::deltat(long ticks)
{
    static double tck=0;
    if(tck==0)
	tck=1./CLK_TCK;
    return ticks*tck;
}

