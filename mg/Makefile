G++ = /home/lucerolo/workspace/gcc/install/bin/g++

mgrid: mgrid.cc LagrangeBasis.h LobattoQuadrature.h ArrayTools.h Laplace.h MFOperator.h Projections.h Multigrid.h
	@$(G++) -DPARALLEL -fopenmp -I/home/lucerolo/workspace/eigen/install/include/eigen3 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -DNDEBUG -I. -march=native -std=c++1z -fconcepts -Ofast  mgrid.cc -o mgrid

static: mgrid.cc LagrangeBasis.h LobattoQuadrature.h ArrayTools.h Laplace.h MFOperator.h Projections.h Multigrid.h
	@$(G++) -I/home/lucerolo/workspace/eigen/install/include/eigen3 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -DNDEBUG -I. -march=native -std=c++1z -fconcepts -Ofast -ftree-vectorizer-verbose=9 -fopt-info-optall=vec_report -Wa,-adhln -g -fno-var-tracking-assignments mgrid.cc -o mgrid > mgrid.s

clean:
	@rm -rf mgrid vec_report mgrid.s *~ \#*

run: mgrid
	@ulimit -s unlimited
	@./mgrid
