G++ = /home/lucerolo/workspace/gcc/install/bin/g++

mgrid: mgrid.cc lagrange.h lobatto.h
	@$(G++) -I/home/lucerolo/workspace/eigen/install/include/eigen3 -I. -DEIGEN_STACK_ALLOCATION_LIMIT=0 -DNDEBUG -march=native -std=c++1z -Ofast -ftree-vectorizer-verbose=9 -fopt-info-optall=vec_report -Wa,-adhln -g mgrid.cc -o mgrid > mgrid.s

clean:
	@rm -rf mgrid vec_report mgrid.s *~ \#*

run: mgrid
	@ulimit -s unlimited
	@./mgrid
