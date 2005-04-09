
#ifndef TESTER_RIGOROUSTEST_H
#define TESTER_RIGOROUSTEST_H 1

/*
 * Helper class for rigorous tests
 */

#define TEST(cond) __test->test(cond, #cond, __FILE__, __LINE__, __DATE__, __TIME__)


class RigorousTest {
public:
    RigorousTest(char* symname);
    ~RigorousTest();
    void test(bool condition, char* descr, char* file, int line, char* date, char* time);

    bool get_passed();
    int get_ntests();
    int get_nfailed();

private:
    char* symname;

    bool passed;
    int ntests;
    int nfailed;
    int nprint;
    char* old_descr;
    char* old_file;
    int old_line;
    char* old_date;
    char* old_time;
};

#endif
