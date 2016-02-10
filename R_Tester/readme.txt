Environmental Flags that control the regression tester:

* TEST_COMPONENTS: MPM, Wasatch, Arches… specifies which component to run.
* WHICH_TESTS: LOCALTESTS, etc…, specifies which suite of tests to run.
Note that the test suite is component-specific.

Example:
in a bash shell
export TEST_COMPONENTS=Wasatch
export WHICH_TESTS=GPUTESTS

make runLocalRT (or gold_standards) will run the Wasatch:GPUTESTS