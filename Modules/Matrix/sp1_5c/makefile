

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
#         ********   ***                                                */
#      *******  **  ***       ***      ***               SparseLib++    */
#       *****      ***     ******** ********                            */
#        *****    ***     ******** ********              R. Pozo        */
#   **  *******  ***   **   ***      ***                 K. Remington   */
#    ********   ********                                 A. Lumsdaine   */
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
#
#               S p a r s e L i b  ++     m a k e f i l e 
#
#             Edit makefile.def to reflect your environment
#-------------------------------------------------------------------------


################################################################
#
# Subdirectories for building
#
################################################################

error:
	@echo "+-----------------------------------------------------------------+"
	@echo "|                                                                 |"
	@echo "|              SparseLib++ Sparse Matrix Library                  |"	
	@echo "|                                                                 |"
	@echo "| Usage: make all              install and test SparseLib++       |"
	@echo "|        make sp               install Sparselib++  library       |"
	@echo "|        make test             run test suite                     |"
	@echo "|        make clean            clean *.o and test executables     |"
	@echo "|                                                                 |"
	@echo "|  Make sure the system-specific makefile.def has been edited     |"
	@echo "|  to reflect your system configuration.                          |"
	@echo "|                                                                 |"
	@echo "+-----------------------------------------------------------------+"
	
all: sp test done


sp:
	cd ./src; make;
	cd ./spblas; make;
	cd ./mv/src; make ;
	

test:
	cd ./testing; make;
	cd ./testing; sp_test > sp_test.out;
	@echo "  "
	@echo " +---------------------------------------------------------------+"
	@echo " |                                                               |"
	@echo " |     SparseLib++ test results are in ./testing/sp_test.out     |"
	@echo " |                                                               |"
	@echo " +---------------------------------------------------------------+"
	@echo "  "

clean:
	cd ./src; make clean;
	cd ./spblas; make clean;
	cd ./mv/src; make clean;
	cd ./testing; make clean;

wipe:
	cd ./src; make wipe;
	cd ./spblas; make wipe;
	cd ./mv/src; make wipe;
	cd ./testing; make wipe;

done:
	@echo "  "
	@echo " +---------------------------------------------------------------+"
	@echo " |                                                               |"
	@echo " |                   SparseLib++ installed.                      |"
	@echo " |                                                               |"
	@echo " +---------------------------------------------------------------+"
	@echo "  "
	@echo "  "
