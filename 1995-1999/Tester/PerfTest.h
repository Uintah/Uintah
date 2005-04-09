
#ifndef TESTER_PERFTEST_H
#define TESTER_PERFTEST_H 1

#define PERFTEST(testname) \
    __pt->start_test(testname); \
    while(__pt->do_test())

#define MINTIME 2.0
#define MULFACTOR 10

class PerfTest {
public:
    PerfTest(char* symname);
    ~PerfTest();
    void start_test(char* name);
    bool do_test();
    void finish();

    static long time();
    static double deltat(long ticks);
private:
    void print_msg();
    char* symname;
    int count;
    double max;
    long start_time;
    double baseline;
    double baseline_time;
    bool is_baseline;
    bool is_sanity;
};

#endif
