import KokkosVerification as KV
import unittest
import os
import sys

def comparator(a,b,rel_tol=1e-3,abs_tol=1e-3,verbose=False): 
    """
        Compare absolute and relative tolerances
    """
    abs_diff = abs(a-b)
    rel_diff = abs(a-b)/b

    if verbose == True: 
        print('\n Comparing a=',a,' b=',b)
        print('     abs_diff = ', abs_diff)
        print('     rel_diff = ', rel_diff)

    value = True 
    if ( abs_diff > abs_tol ): 
        value=False
    if ( rel_diff > rel_tol ): 
        value=False
    return value

class ArchesKokkosVerification(unittest.TestCase): 

    mainsuspath='' #this is grabbed from the environment

    def get_suspath(self): 
        """
            Get suspath from environment
        """
        self.mainsuspath = os.environ.get('SUSPATH', self.mainsuspath) 
        if self.mainsuspath == '': 
            print('Error: please set SUSPATH environment variable.')
            sys.exit()

    #------------------------------------------------------------------------
    def test_almgrenConv(self): 
        """
            The almgren MMS conv only test
        """
        print('\n ------------------------------------------------------ ' )
        print('             Running MMS Conv test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        GOLD = {'m':1.99768228168, 'b':-8.14809906691, 'r':0.999999919229}

        self.get_suspath()

        class Args: 
            ups='mom/almgren-mms_conv.ups'
            levels=3
            nsteps=1
            tstep=1
            tsave=None
            suspath=self.mainsuspath
            vars='x-mom' 
            axis='x,y' 
            bc=None
            var_mms='x_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = comparator(results[args.vars]['m'], GOLD['m'],verbose=True)

        #assert if not the same
        self.assertTrue(m_check)

    #------------------------------------------------------------------------
    def test_xScalar(self): 
        """
            The x-scalar test
        """

        print('\n ------------------------------------------------------ ' )
        print('             Running x-scalar test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        GOLD = [{'m':0.99871222969, 'b':-6.58012529482, 'r':0.999999900223},
                {'m':1.50187839949, 'b':-5.76651614399, 'r':0.99999999083},
                {'m':1.48240819129, 'b':-5.40695603565, 'r':0.999986756338},
                {'m':1.99768228161, 'b':-5.84551397414, 'r':0.999999919229}]

        self.get_suspath()

        class Args: 
            ups='scalars/kokkos-x-scalar_mms.ups'
            levels=None
            nsteps=None
            tstep=None
            tsave=None
            suspath=self.mainsuspath
            vars='cc_phi_upwind,cc_phi_vanleer,cc_phi_superbee,cc_phi_central'
            axis='x' 
            bc=None
            var_mms='phi_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = [False, False, False, False]
        the_vars = args.vars.split(',')
        for i in range(len(GOLD)): 
            m_check[i] = comparator(results[the_vars[i]]['m'], GOLD[i]['m'],verbose=True)

        print('\n -- Checking output for each variable -- ') 
        for i, check_result in enumerate(m_check): 
            print('       '+the_vars[i]) 
            self.assertTrue(check_result)
        print(' -- End check --')

    #------------------------------------------------------------------------
    def test_almgrenDiff(self): 
        """
            The almgren MMS diff only test
        """
        print('\n ------------------------------------------------------ ' )
        print('             Running MMS Diff test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        GOLD = {'m':1.99768228168, 'b':-6.65330384082, 'r':0.999999964148}

        self.get_suspath()

        class Args: 
            ups='mom/almgren-mms_diff.ups'
            levels=3
            nsteps=1
            tstep=1
            tsave=None
            suspath=self.mainsuspath
            vars='x-mom' 
            axis='x,y' 
            bc=None
            var_mms='x_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = comparator(results[args.vars]['m'], GOLD['m'],verbose=True)

        #assert if not the same
        self.assertTrue(m_check)

    #------------------------------------------------------------------------
    def test_xScalarDiff(self): 
        """
            The x-scalar diff test
        """

        print('\n ------------------------------------------------------ ' )
        print('             Running x-scalar diff test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        GOLD =[ {'m':1.99845512229, 'b':-6.99987743108, 'r':0.999999964148},
                {'m':1.99845512229, 'b':-6.99987743108, 'r':0.999999964148},
                {'m':1.99845512229, 'b':-6.99987743108, 'r':0.999999964148},
                {'m':1.99845512229, 'b':-6.99987743108, 'r':0.999999964148} ]

        self.get_suspath()

        class Args: 
            ups='scalars/kokkos-x-scalar_mms_diff.ups'
            levels=None
            nsteps=None
            tstep=None
            tsave=None
            suspath=self.mainsuspath
            vars='cc_phi_upwind,cc_phi_vanleer,cc_phi_superbee,cc_phi_central'
            axis='x' 
            bc=None
            var_mms='phi_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = [False, False, False, False]
        the_vars = args.vars.split(',')
        for i in range(len(GOLD)): 
            m_check[i] = comparator(results[the_vars[i]]['m'], GOLD[i]['m'],verbose=True)

        print('\n -- Checking output for each variable -- ') 
        for i, check_result in enumerate(m_check): 
            print('       '+the_vars[i]) 
            self.assertTrue(check_result)
        print(' -- End check --')

    #------------------------------------------------------------------------
    def test_kokkosScalarRK1(self): 
        """
            The Scalar RK1 test
        """
        print('\n ------------------------------------------------------ ' )
        print('             Running Scalar RK1 Test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        GOLD = {'m':0.972405518523, 'b': -0.213437131928, 'r':0.999985294018}

        self.get_suspath()

        class Args: 
            ups='scalars/kokkos-x-scalar_mms_RK1.ups'
            levels=None
            nsteps=100
            tstep=None
            tsave=None
            suspath=self.mainsuspath
            vars='cc_phi_upwind' 
            axis='t' 
            bc=None
            var_mms='phi_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = comparator(results[args.vars]['m'], GOLD['m'],verbose=True)

        #assert if not the same
        self.assertTrue(m_check)

    #------------------------------------------------------------------------
    def test_kokkosScalarRK2(self): 
        """
            The Scalar RK2 test
        """
        print('\n ------------------------------------------------------ ' )
        print('             Running Scalar RK2 Test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        GOLD = {'m':2.0019948411, 'b':1.13206672282, 'r':0.999999846333}

        self.get_suspath()

        class Args: 
            ups='scalars/kokkos-x-scalar_mms_RK2.ups'
            levels=None
            nsteps=100
            tstep=None
            tsave=None
            suspath=self.mainsuspath
            vars='cc_phi_upwind' 
            axis='t' 
            bc=None
            var_mms='phi_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = comparator(results[args.vars]['m'], GOLD['m'],verbose=True)

        #assert if not the same
        self.assertTrue(m_check)

    #------------------------------------------------------------------------
    def test_kokkosScalarRK3(self): 
        """
            The Scalar RK3 test
        """
        print('\n ------------------------------------------------------ ' )
        print('             Running Scalar RK3 Test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        GOLD = {'m':4.00529516884, 'b':2.60577077694, 'r':0.9999998411}

        self.get_suspath()

        class Args: 
            ups='scalars/kokkos-x-scalar_mms_RK3.ups'
            levels=None
            nsteps=100
            tstep=None
            tsave=None
            suspath=self.mainsuspath
            vars='cc_phi_upwind' 
            axis='t' 
            bc=None
            var_mms='phi_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = comparator(results[args.vars]['m'], GOLD['m'],verbose=True)

        #assert if not the same
        self.assertTrue(m_check)

    #------------------------------------------------------------------------
    def test_xy2DScalar(self): 
        """
            The Kokkos XY 2D Scalar
        """
        print('\n ------------------------------------------------------ ' )
        print('             Running Kokkos XY 2D Scalar Test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        GOLD = {'m':1.99768228174, 'b':-8.14809906667, 'r':0.999999919229}

        self.get_suspath()

        class Args: 
            ups='scalars/2D/kokkos-xy-scalar.ups'
            levels=None
            nsteps=None
            tstep=None
            tsave=None
            suspath=self.mainsuspath
            vars='phi' 
            axis='x,y' 
            bc=None
            var_mms='phi_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = comparator(results[args.vars]['m'], GOLD['m'],verbose=True)

        #assert if not the same
        self.assertTrue(m_check)

    #------------------------------------------------------------------------
    def test_xy2DScalarHandoff(self): 
        """
            The Kokkos XY HANDOFF 2D Scalar
        """
        print('\n ------------------------------------------------------ ' )
        print('         Running Kokkos XY 2D Scalar Handoff Test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        GOLD = {'m':1.79009294855, 'b':-8.66898383031, 'r':0.999928371945}

        self.get_suspath()

        class Args: 
            ups='scalars/2D/kokkos-xy-scalar-handoff.ups'
            levels=None
            nsteps=None
            tstep=None
            tsave=None
            suspath=self.mainsuspath
            vars='phi' 
            axis='x,y' 
            bc='x,y'
            var_mms='phi_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = comparator(results[args.vars]['m'], GOLD['m'],verbose=True)

        #assert if not the same
        self.assertTrue(m_check)

    #------------------------------------------------------------------------
    def test_xy2DScalarMMSBC(self): 
        """
            The Kokkos XY 2D MMSBC Scalar
        """
        print('\n ------------------------------------------------------ ' )
        print('         Running Kokkos XY 2D Scalar MMSBC Test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        GOLD = {'m':1.99823263785, 'b':-8.14560473046, 'r':0.999999951634}

        self.get_suspath()

        class Args: 
            ups='scalars/2D/kokkos-xy-scalar-MMSBC.ups'
            levels=None
            nsteps=None
            tstep=None
            tsave=None
            suspath=self.mainsuspath
            vars='phi' 
            axis='x,y' 
            bc='x,y'
            var_mms='phi_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = comparator(results[args.vars]['m'], GOLD['m'],verbose=True)

        #assert if not the same
        self.assertTrue(m_check)

    #------------------------------------------------------------------------
    def test_almgrenMMSBC(self): 
        """
            The Almgren MMS BC
        """
        print('\n ------------------------------------------------------ ' )
        print('         Running Almgren MMS BC Test ')
        print(' ------------------------------------------------------ \n ' )
        #GOLD STANDARD: 
        #GOLD = {'m':2.00161276434, 'b':-7.6568819516, 'r':0.999999968556} #original from Oscar
        GOLD = {'m':1.9993602427515982, 'b':-8.333393691412919, 'r':0.99999997016195474} #updated 11/20/19

        self.get_suspath()

        class Args: 
            ups='mom/almgren-mmsBC.ups'
            levels=3
            nsteps=1
            tstep=None
            tsave=None
            suspath=self.mainsuspath
            vars='x-mom' 
            axis='x,y' 
            bc='x,y'
            var_mms='x_mms'
            keep_uda=None

        args = Args

        results = KV.run_test(args)

        #compare outputs
        m_check = comparator(results[args.vars]['m'], GOLD['m'],verbose=True)

        #assert if not the same
        self.assertTrue(m_check)

#------------------------------------------------                   
if __name__ == '__main__':

    unittest.main()
