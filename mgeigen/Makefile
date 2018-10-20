G++ = /home/lucerolo/workspace/gcc/install/bin/g++

mgrid: mgrid.cc
	@$(G++) -DEIGEN_STACK_ALLOCATION_LIMIT=0 -I/home/lucerolo/workspace/eigen/install/include/eigen3 -march=native -std=c++1z -Ofast -DNDEBUG -ftree-vectorizer-verbose=9 -fopt-info-optall=vec_report -Wa,-adhln -g mgrid.cc -o mgrid > mgrid.s

clean:
	@rm -rf mgrid vec_report mgrid.s *~ \#*

run: mgrid
	@ulimit -s unlimited
	@./mgrid
