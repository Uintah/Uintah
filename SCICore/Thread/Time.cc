
#include <Thread/Time.h>
#include <Tester/RigorousTest.h>
#include <iostream.h>

void Time::test_rigorous(RigorousTest* __test)
{
    // Crude excercises of the timer functions...
    SysClock time=Time::currentTicks();
    SysClock time2=time;
    while(time == time2){
	time2=Time::currentTicks();
    }
    TEST(time2 > time);
    double stime=Time::currentSeconds();
    double stime2=stime;
    while(stime == stime2) {
	stime2=Time::currentSeconds();
    }
    
    double diff=stime2-time2*Time::secondsPerTick();
    TEST(diff < .01);
    TEST(diff > 0);

    double spt=Time::secondsPerTick();
    diff=0;
    time=Time::currentTicks();
    SysClock otime=time;
    double tick=1;
    while(diff < 99){
	SysClock time2=Time::currentTicks();
	SysClock dt=time2-time;
	if(dt < 0){
	    cerr << "dt=" << dt*Time::secondsPerTick() << '\n';
	    tick=1;
	    otime=time2;
	}
	time=time2;
	diff=(time-otime)*spt;
	if(diff > tick){
	    cerr << "diff=" << diff << ", ticks=" << time << '\n';
	    tick=tick+1;
	}
    }
    Time::waitUntil(stime2+.01);
    Time::waitUntil(time2+(int)(.02*Time::ticksPerSecond()));
    Time::waitUntil(stime2-1);
}
