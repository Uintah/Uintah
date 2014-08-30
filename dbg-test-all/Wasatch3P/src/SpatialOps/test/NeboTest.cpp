#include <spatialops/structured/FVStaggeredFieldTypes.h>

#include <spatialops/Nebo.h>
#include <test/TestHelper.h>
#include <spatialops/structured/FieldHelper.h>

#include <spatialops/structured/FieldComparisons.h>

#include <iostream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace SpatialOps;
using std::cout;
using std::endl;

typedef double (*EXPR)(int ii, int jj, int kk); ///< Explicit type for a function which can be evaluated at each point in field

/**
 *  @class three_functor
 *
 *  @brief abstract class for a functor which takes three int parameters
 *
 *  Abstract class which take three int parameters.  This class is primarily
 *  used as the base class for functors which will be evaluated at each point
 *  in a field.
 *
 */
class three_functor
{
  public:
  virtual double operator() (int ii, int jj, int kk) const = 0;
};

/**
 *  @class unary_lambda
 *
 *  @brief Functor which calls \c OP functions or functor on a value in given
 *  index of field
 *
 *  This class was designed to take a function or functor and call it on the
 *  data at a certain point in the passed field.  This class can then be passed
 *  to a function which will call it at every index in field.
 *
 *  @tparam OP -- The function or functor which takes a single parameter and
 *  returns field data type at point.
 *  @tparam FieldT -- Field type accessed.
 *
 */
template<typename OP, typename FieldT>
class unary_lambda : public three_functor
{

  OP op;
  FieldT const & p1;

  public:
  unary_lambda(OP op, FieldT const & p1) : op(op), p1(p1)
  {}
  virtual typename FieldT::value_type operator() (int ii, int jj, int kk) const
  {
    //pass value at (ii, jj, kk) through op and return it
    return op(p1(ii, jj, kk));
  }

};

/**
 *  @class binary_lambda
 *
 *  @brief Functor which calls \c OP functor on two values at a given index
 *
 *  This class was designed to take a functor and call it on the data at a
 *  certain point in the passed parameters.  The parameters can be
 *  \c three_functors or fields.  Note that one of the parameters may also be a
 *  double instead of a field or functor.  This class can then be passed to a
 *  function which will call it at every index in field.
 *
 *  @tparam OP -- The functor which takes two parameters and
 *  returns \c Type1 data type at point.
 *  @tparam Type1 -- Field or \c three_functor representing first parameter
 *  @tparam Type2 -- Field or \c three_functor representing second parameter
 *
 */
template<typename OP, typename Type1, typename Type2>
class binary_lambda : public three_functor
{
  Type1 const & p1;
  Type2 const & p2;

  public:
  typedef typename Type1::value_type value_type;

  binary_lambda(Type1 const & p1, Type2 const & p2) : p1(p1), p2(p2)
  {}
  virtual value_type operator() (int ii, int jj, int kk) const
  {
    return OP()(p1(ii, jj, kk), p2(ii, jj, kk));
  }

};
template<typename OP, typename Type>
class binary_lambda<OP, Type, double> : public three_functor
{
  Type const & p1;
  double p2;

  public:
  typedef typename Type::value_type value_type;

  binary_lambda(Type const & p1, double p2) : p1(p1), p2(p2)
  {}
  virtual value_type operator() (int ii, int jj, int kk) const
  {
    return OP()(p1(ii, jj, kk), p2);
  }

};
template<typename OP, typename Type>
class binary_lambda<OP, double, Type> : public three_functor
{
  double p1;
  Type const & p2;

  public:
  typedef typename Type::value_type value_type;

  binary_lambda(double p1, Type const & p2) : p1(p1), p2(p2)
  {}
  virtual value_type operator() (int ii, int jj, int kk) const
  {
    return OP()(p1, p2(ii, jj, kk));
  }

};

/**
 * @brief This function manually fills the contents of \c ref without using Nebo.
 *
 * This function assigns \c val at every point in \c ref.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param ref -- pointer to the field to assign to.
 * @param val -- the value to assign at each point of field.
 */
template<typename FieldT>
void manual(FieldT* const ref, typename FieldT::value_type val)
{
  const MemoryWindow window(ref->window_with_ghost());
  const int xLength = window.extent(0);
  const int yLength = window.extent(1);
  const int zLength = window.extent(2);

  for(int kk = 0; kk < zLength; kk++) {
    for(int jj = 0; jj < yLength; jj++) {
      for(int ii = 0; ii < xLength; ii++) {
        (*ref)(ii, jj, kk) = val;
      }
    }
  }
}

/**
 * @brief This function manually fills the contents of \c ref without using Nebo.
 *
 * This function assigns result of \c f(ii, jj, kk) at every point in \c ref
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param ref -- pointer to the field to assign to.
 * @param f -- function which takes current index and returns a value.
 */
template<typename FieldT>
void manual(FieldT* const ref, EXPR f)
{
  const MemoryWindow window(ref->window_with_ghost());
  const int xLength = window.extent(0);
  const int yLength = window.extent(1);
  const int zLength = window.extent(2);

  for(int kk = 0; kk < zLength; kk++) {
    for(int jj = 0; jj < yLength; jj++) {
      for(int ii = 0; ii < xLength; ii++) {
        (*ref)(ii, jj, kk) = f(ii, jj, kk);
      }
    }
  }
}

/**
 * @brief This function manually fills the contents of \c ref without using Nebo.
 *
 * This function assigns result of \c f(ii, jj, kk) at every point in \c ref
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param ref -- pointer to the field to assign to.
 * @param f -- functor which takes current index and returns a value.
 */
template<typename FieldT>
void manual(FieldT* const ref, three_functor const & f)
{
  const MemoryWindow window(ref->window_with_ghost());
  const int xLength = window.extent(0);
  const int yLength = window.extent(1);
  const int zLength = window.extent(2);

  for(int kk = 0; kk < zLength; kk++) {
    for(int jj = 0; jj < yLength; jj++) {
      for(int ii = 0; ii < xLength; ii++) {
        (*ref)(ii, jj, kk) = f(ii, jj, kk);
      }
    }
  }
}

/**
 *  @class NAN_Guard
 *
 *  @brief Functor which returns the value at a specified index or 0.0 if the
 *  value is NAN.
 *
 *  @tparam FieldT -- Field type accessed.
 *
 */
template<typename FieldT>
class NAN_Guard : public three_functor
{
  FieldT& f1;

  public:
    NAN_Guard(FieldT& f1) : f1(f1) {}
    virtual double operator()(int ii, int jj, int kk) const
    {
      //NAN is never equal to itself
      return f1(ii, jj, kk) != f1(ii, jj, kk) ? 0.0 : f1(ii, jj, kk);
    }
};

/**
 *  @class NAN_Guard
 *
 *  @brief Functor which returns the value at a specified index or 0.0 if the
 *  value is NAN.  This version is used when the template parameter is not a
 *  field but instead a functor.
 *
 */
template<>
class NAN_Guard<three_functor*> : public three_functor
{
  three_functor const * f1;

  public:
    NAN_Guard(three_functor const * f1) : f1(f1) {}
    virtual double operator()(int ii, int jj, int kk) const
    {
      //NAN is never equal to itself
      return (*f1)(ii, jj, kk) != (*f1)(ii, jj, kk) ? 0.0 : (*f1)(ii, jj, kk);
    }
};

//------------------------------------------------------------------
//                        Binary Functors
//------------------------------------------------------------------


/**
 *  @class Pow
 *
 *  @brief Functor which calls \c std::pow on \c a and \c b
 *
 */
class Pow
{
  public:
    double operator() (double a, double b) const
    {
      return std::pow(a, b);
    }
};

/**
 *  @class Max_Functor
 *
 *  @brief Functor which calls \c std::max on \c a and \c b
 *
 */
class Max_Functor
{
  public:
    double operator() (double a, double b) const
    {
      return std::max(a, b);
    }
};

/**
 *  @class Min_Functor
 *
 *  @brief Functor which calls \c std::min on \c a and \c b
 *
 */
class Min_Functor
{
  public:
    double operator() (double a, double b) const
    {
      return std::min(a, b);
    }
};

//------------------------------------------------------------------
//                            Tests
//------------------------------------------------------------------

/**
 * @brief Macro which inlines the tested operator in between parameters
 *
 * @param OP -- The binary operator to inline between arguments. An example is
 * the \c + operator.
 * @param MANUAL_FUNCTOR -- the functor which does the equivalent of \c OP.
 * @param TESTTYPE -- The character array of which to append the sub-test type
 * to.  An example is "summation" which would imply the \c + operator.
 */
#define BINARY_OP_TESTS(OP, MANUAL_FUNCTOR, TESTTYPE)                                                                                                                               \
{                                                                                                                                                                                   \
  /*General operator tests */                                                                                                                                                       \
  status(run_test(test <<= 50.3 OP 4.7, 50.3 OP 4.7), TESTTYPE" (Scalar x Scalar) test");                                                                                           \
  status(run_test(test <<= input1 OP 4.7, (binary_lambda<MANUAL_FUNCTOR, Field, double>(input1, 4.7))), TESTTYPE" (Field x Scalar) test");                                          \
  status(run_test(test <<= 50.3 OP input2, (binary_lambda<MANUAL_FUNCTOR, double, Field>(50.3, input2))), TESTTYPE" (Scalar x Field) test");                                        \
  status(run_test(test <<= input1 OP input2, (binary_lambda<MANUAL_FUNCTOR, Field, Field>(input1, input2))), TESTTYPE" (Field x Field) test");                                      \
  typedef binary_lambda<MANUAL_FUNCTOR, Field, Field> SubExpr;                                                                                                                      \
  status(run_test(test <<= input1 OP (input2 OP input3), (binary_lambda<MANUAL_FUNCTOR, Field, SubExpr>(input1, SubExpr(input2, input3)))), TESTTYPE" (Field x SubExpr) test");     \
  status(run_test(test <<= (input1 OP input2) OP input3, (binary_lambda<MANUAL_FUNCTOR, SubExpr, Field>(SubExpr(input1, input2), input3))), TESTTYPE" (SubExpr x Field) test");     \
  status(run_test(test <<= (input1 OP input2) OP (input3 OP input4),                                                                                                                \
         (binary_lambda<MANUAL_FUNCTOR, SubExpr, SubExpr>(SubExpr(input1, input2), SubExpr(input3, input4)))),                                                                      \
         TESTTYPE" (SubExpr x SubExpr) test");                                                                                                                                      \
  status(run_test(test <<= 50.3 OP (input2 OP input3), (binary_lambda<MANUAL_FUNCTOR, double, SubExpr>(50.3, SubExpr(input2, input3)))), TESTTYPE" (Scalar x SubExpr) test");       \
  status(run_test(test <<= (input1 OP input2) OP 4.7, (binary_lambda<MANUAL_FUNCTOR, SubExpr, double>(SubExpr(input1, input2), 4.7))), TESTTYPE" (SubExpr x Scalar) test");         \
                                                                                                                                                                                    \
  /*Single value field tests to a normal field */                                                                                                                                   \
  status(run_test(test <<= SVinput1 OP 4, SVinput1[0] OP 4), TESTTYPE" (SVField x integer) test");                                                                                  \
  status(run_test(test <<= SVinput1 OP 4.7, SVinput1[0] OP 4.7), TESTTYPE" (SVField x Scalar) test");                                                                               \
  status(run_test(test <<= SVinput1 OP input1, (binary_lambda<MANUAL_FUNCTOR, double, Field>(SVinput1[0], input1))), TESTTYPE" (SVField x Field) test");                            \
  status(run_test(test <<= SVinput1 OP SVinput2, SVinput1[0] OP SVinput2[0]), TESTTYPE" (SVField x SVField) test");                                                                 \
                                                                                                                                                                                    \
  /*Single value field tests to a single value field */                                                                                                                             \
  status(run_sv_test(SVtest <<= SVinput1 OP 4, SVinput1[0] OP 4), TESTTYPE" (SVField x integer) -> SVField test");                                                                  \
  status(run_sv_test(SVtest <<= SVinput1 OP 4.7, SVinput1[0] OP 4.7), TESTTYPE" (SVField x Scalar) -> SVField test");                                                               \
  status(run_sv_test(SVtest <<= SVinput1 OP SVinput2, SVinput1[0] OP SVinput2[0]), TESTTYPE" (SVField x SVField) -> SVField test");                                                 \
}

/**
 *  @class TestContext
 *
 *  @brief Class which contains all of the variables necessary to run the tests
 *  in this file.  To run the tests in this file, the \c run function should be
 *  called on this class.
 *
 */
class TestContext
{
  int nx, ny, nz;
  bool* bcplus;

  typedef SVolField Field;
  typedef SingleValueField SVField;

  const GhostData ghost;
  const GhostData SVghost;
  const BoundaryCellInfo bcinfo;
  const BoundaryCellInfo SVbcinfo;
  const MemoryWindow window;
  const MemoryWindow SVwindow;

  //Input fields for tests
  Field input1;
  Field input2;
  Field input3;
  Field input4; //This field is only positive
  SVField SVinput1;
  SVField SVinput2;

  //Testing fields used to compare final answer for each test
  Field test;
  Field ref;
  SVField SVtest;
  SVField SVref;

  TestHelper& status;

  public:

  /**
   * @brief Creates a context for the tests in this file
   *
   * Initializes variables used in the various testing functions for this file.
   *
   * @param nx -- number of points in x-dir for base mesh
   * @param ny -- number of points in y-dir for base mesh
   * @param nz -- number of points in z-dir for base mesh
   *
   * @param bcplus -- boolean array of size three representing if a physical
   * boundary appears on the x, y, or z side respectfully.
   *
   * @param status -- Pointer to \c TestHelper to send test statuses through.
   */
  TestContext(int nx, int ny, int nz, bool bcplus[], TestHelper* status)
    : nx(nx), ny(ny), nz(nz), bcplus(bcplus),
    ghost(1),
    SVghost(0),
    bcinfo(BoundaryCellInfo::build<Field>(bcplus[0], bcplus[1], bcplus[2])),
    SVbcinfo(BoundaryCellInfo::build<Field>(false,false,false)),
    window(get_window_with_ghost(IntVec(nx, ny, nz), ghost, bcinfo)),
    SVwindow(get_window_with_ghost(IntVec(1, 1, 1), SVghost, SVbcinfo)),
    input1( window, bcinfo, ghost, NULL ),
    input2( window, bcinfo, ghost, NULL ),
    input3( window, bcinfo, ghost, NULL ),
    input4( window, bcinfo, ghost, NULL ),
    test  ( window, bcinfo, ghost, NULL ),
    ref   ( window, bcinfo, ghost, NULL ),
    SVinput1( SVwindow, SVbcinfo, SVghost, NULL ),
    SVinput2( SVwindow, SVbcinfo, SVghost, NULL ),
    SVtest  ( SVwindow, SVbcinfo, SVghost, NULL ),
    SVref   ( SVwindow, SVbcinfo, SVghost, NULL ),
    status(*status)
  {}

  /**
   * @brief Run all tests in this file
   *
   * This function initializes the input fields with values and then runs the
   * suite of tests, reporting each success or failure to \c status.
   *
   */
  void run()
  {
    const int total = nx * ny * nz;

    //Initialize input fields with data
    initialize_field(input1, 0.0);
    initialize_field(input2, total);
    initialize_field(input3, 2 * total);
    initialize_field(input4, 3 * total);
    //make input4 positive only:
    input4 <<= abs(input4);

    //Same for single value fields
    initialize_field(SVinput1, 0.0);
    initialize_field(SVinput2, total);


    //Assignment tests
    status(run_test(test <<= 0.0, 0.0), "scalar assignment test"); cout << "\n";
    status(run_test(test <<= SVinput1, SVinput1[0]), "SingleValue assignment test"); cout << "\n";
    status(run_sv_test(SVtest <<= SVinput1, SVinput1[0]), "SingleValue -> SingleValue assignment test"); cout << "\n";

    //Binary operator tests
    BINARY_OP_TESTS(+, std::plus<double>, "summation"); cout << "\n";
    BINARY_OP_TESTS(-, std::minus<double>, "difference"); cout << "\n";
    BINARY_OP_TESTS(/, std::divides<double>, "division"); cout << "\n";
    BINARY_OP_TESTS(*, std::multiplies<double>, "product"); cout << "\n";

    //Trigonometric tests
    status(run_test_ulp(test <<= sin(input1), (unary_lambda<double(*)(double), Field>(std::sin, input1)), 1), "sin test");
    status(run_test_ulp(test <<= cos(input1), (unary_lambda<double(*)(double), Field>(std::cos, input1)), 1), "cos test");
    status(run_test_ulp(test <<= tan(input1), (unary_lambda<double(*)(double), Field>(std::tan, input1)), 2), "tan test");
    status(run_test_ulp(test <<= exp(input1), (unary_lambda<double(*)(double), Field>(std::exp, input1)), 1), "exp test");
    //NVidia documentation says with 1 ulp, empirically found to be 2
    status(run_test_ulp(test <<= tanh(input1), (unary_lambda<double(*)(double), Field>(std::tanh, input1)), 2), "tanh test");

    //Sign related functions
    status(run_test(test <<= abs(input1), (unary_lambda<double(*)(double), Field>(std::abs, input1))), "abs test");
    status(run_test(test <<= -input1, (unary_lambda<std::negate<double>, Field>(std::negate<double>(), input1))), "negation test");

    //Power related functions
    status(run_test_ignore_nan_ulp(test <<= pow(input1, input2), (binary_lambda<Pow, Field, Field>(input1, input2)), 2), "power test");
    status(run_test_ignore_nan_ulp(test <<= sqrt(input1), (unary_lambda<double(*)(double), Field>(std::sqrt, input1)), 0), "square root test");
    status(run_test_ignore_nan_ulp(test <<= log(input1), (unary_lambda<double(*)(double), Field>(std::log, input1)), 1), "log test");
    //documentation says with 1 ulp, empirically found to be 2
    status(run_test_ignore_nan_ulp(test <<= log10(input1), (unary_lambda<double(*)(double), Field>(std::log10, input1)), 2), "log10 test");

    //erf
    status(run_test_ulp(test <<= erf(input1), (unary_lambda<double(*)(double), Field>(boost::math::erf, input1)), 2), "erf test");
    status(run_test_ulp(test <<= erfc(input1),(unary_lambda<double(*)(double), Field>(boost::math::erfc, input1)), 4), "erfc test");
    status(run_test_ulp(test <<= inv_erf(input1), (unary_lambda<double(*)(double), Field>(boost::math::erf_inv, input1)), 5), "inv_erf test");
    //domain is (0.0, 1.0), so use input4:
    status(run_test_ulp(test <<= inv_erfc(input4), (unary_lambda<double(*)(double), Field>(boost::math::erfc_inv, input4)), 6), "inv_erfc test");

    //Equivalence operator tests
    status(run_test(test <<= cond(input1 == input2, true)(false), (binary_lambda<std::equal_to<double>, Field, Field>(input1, input2))), "equivalence test");
    status(run_test(test <<= cond(input1 != input2, true)(false), (binary_lambda<std::not_equal_to<double>, Field, Field>(input1, input2))), "non-equivalence test");

    //Other comparison operators
    status(run_test(test <<= cond(input1 < input2, true)(false), (binary_lambda<std::less<double>, Field, Field>(input1, input2))), "less than test");
    status(run_test(test <<= cond(input1 <= input2, true)(false), (binary_lambda<std::less_equal<double>, Field, Field>(input1, input2))), "less than or equal test");
    status(run_test(test <<= cond(input1 > input2, true)(false), (binary_lambda<std::greater<double>, Field, Field>(input1, input2))), "greater than test");
    status(run_test(test <<= cond(input1 >= input2, true)(false), (binary_lambda<std::greater_equal<double>, Field, Field>(input1, input2))), "greater than or equal test");

    //These tests are dependent on nebo less than working
    //Logical Operators
    typedef binary_lambda<std::less<double>, Field, Field> lessThanLambda;
    status(run_test(test <<= cond(input1 < input2 && input3 < input4, true)(false),
                   (binary_lambda<std::logical_and<bool>, lessThanLambda, lessThanLambda>(lessThanLambda(input1, input2), lessThanLambda(input3, input4)))),
          "boolean and test");
    status(run_test(test <<= cond(input1 < input2 || input3 < input4, true)(false),
                   (binary_lambda<std::logical_or<bool>, lessThanLambda, lessThanLambda>(lessThanLambda(input1, input2), lessThanLambda(input3, input4)))),
          "boolean or test");
    status(run_test(test <<= cond(!(input1 < input2), true)(false),
                   (unary_lambda<std::logical_not<bool>, lessThanLambda>(std::logical_not<bool>(), lessThanLambda(input1, input2)))),
          "boolean not test");

    //Min and Max
    status(run_test(test <<= max(input1, 0.0), (binary_lambda<Max_Functor, Field, double>(input1, 0.0))), "max test");
    status(run_test(test <<= min(input1, 0.0), (binary_lambda<Min_Functor, Field, double>(input1, 0.0))), "min test");
  }

  /**
   * @brief Compares the results of \c neboExpr with the results of the manual
   * version.
   *
   * This function fills the \c ref field with the result of \c expr, then
   * compares \c ref with the \c neboExpr passed.
   *
   * @tparam FieldT -- The type of field \c neboExpr is.
   * @tparam ExprT -- The type of the manual expression.  Usually data value or
   * function.
   *
   * @param neboExpr -- the result of running the Nebo expression to test.
   * @param expr -- the manual way to compute the Nebo expression to test.
   */
  template<typename FieldT, typename ExprT>
  bool run_test(FieldT const & neboExpr, ExprT expr)
  {
    manual(&ref, expr);
    return field_equal(ref, neboExpr, 0.0);
  }
  /**
   * @brief single value field version of \c run_test.
   *
   * Same as \c run_test but uses \c SVtest and \c SVref as compared fields.  Note
   * that only a scalar value may be assigned to a single value field.
   *
   * @tparam SVFieldT -- The type of single value field.
   *
   * @param neboExpr -- the result of running the Nebo expression to test.
   * @param expr -- the manual way to compute the Nebo expression to test.
   */
  template<typename SVFieldT>
  bool run_sv_test(SVFieldT const & neboExpr, typename SVFieldT::value_type expr)
  {
    //This tests that neboExpr is indeed a single value field (or at least can be
    //assigned to one)
    SVtest <<= neboExpr;
    SVref[0] = expr;
    return field_equal(SVref, SVtest, 0.0);
  }
  /**
   * @brief Same as \c run_test but passes if fields are within a certain number
   * of ulps.
   *
   * Same as \c run_test but compares the two fields with \c field_equal_ulp
   * instead of field_equal.
   *
   * @tparam FieldT -- The type of field \c neboExpr is.
   * @tparam ExprT -- The type of the manual expression.  Usually data value or
   * function.
   *
   * @param neboExpr -- the result of running the Nebo expression to test.
   * @param expr -- the manual way to compute the Nebo expression to test.
   * @param ulps -- The number of allowable ulps of error between Nebo and
   * manual version.
   */
  template<typename FieldT, typename ExprT>
  bool run_test_ulp(FieldT const & neboExpr, ExprT expr, int ulps)
  {
    manual(&ref, expr);
    return field_equal_ulp(ref, neboExpr, ulps);
  }
  /**
   * @brief version of \c run_test which ignores NANs in expressions and allows
   * a certain amount of ulps of error.
   *
   * Assigns zero to any NAN's in \c neboExpr or \c expr before calling \c
   * run_test_ulp.
   *
   * @tparam FieldT -- The type of field \c neboExpr is.
   * @tparam ExprT -- The type of the manual expression.  Usually data value or
   * function.
   *
   * @param neboExpr -- the result of running the Nebo expression to test.
   * @param expr -- the manual way to compute the Nebo expression to test.
   * @param ulps -- The number of allowable ulps of error between Nebo and
   * manual version.
   */
  template<typename FieldT, typename ExprT>
  bool run_test_ignore_nan_ulp(FieldT const & neboExpr, ExprT expr, int ulps)
  {
    FieldT tmp( window, bcinfo, ghost, NULL );
    tmp <<= cond(neboExpr != neboExpr, 0.0)(neboExpr);
    return run_test_ulp(tmp, NAN_Guard<three_functor*>(&expr), ulps);
  }
};


/**
 * @brief Sets up testing framework based off of user input and calls
 * \c TestContext's \c run function.
 *
 */
int main( int iarg, char* carg[] )
{
    int nx, ny, nz;
    bool bcplus[] = { false, false, false };

    po::options_description desc("Supported Options");
    desc.add_options()
        ( "help", "print help message\n" )
        ( "nx",   po::value<int>(&nx)->default_value(11), "number of points in x-dir for base mesh" )
        ( "ny",   po::value<int>(&ny)->default_value(11), "number of points in y-dir for base mesh" )
        ( "nz",   po::value<int>(&nz)->default_value(11), "number of points in z-dir for base mesh" )
        ( "bcx",  "physical boundary on +x side?" )
        ( "bcy",  "physical boundary on +y side?" )
        ( "bcz",  "physical boundary on +z side?" );

    po::variables_map args;
    po::store( po::parse_command_line(iarg,carg,desc), args );
    po::notify(args);

    if( args.count("bcx") ) bcplus[0] = true;
    if( args.count("bcy") ) bcplus[1] = true;
    if( args.count("bcz") ) bcplus[2] = true;

    if( args.count("help") ){
      cout << desc << endl
           << "Examples:" << endl
           << " test_nebo --nx 5 --ny 10 --nz 3 --bcx" << endl
           << " test_nebo --bcx --bcy --bcz" << endl
           << " test_nebo --nx 50 --bcz" << endl
           << endl;
      return -1;
    }

    TestHelper status(true);

    //run tests
    TestContext TC(nx, ny, nz, bcplus, &status);
    TC.run();


    if( status.ok() ) {
      cout << "ALL TESTS PASSED :)" << endl;
      return 0;
    }
    else {
        cout << "******************************" << endl
             << " At least one test FAILED! :(" << endl
             << "******************************" << endl;
        return -1;
    };
}
