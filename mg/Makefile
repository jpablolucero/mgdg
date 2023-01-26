G++ = g++
EIGEN = /home/pablo/workspace/eigen/install/

mgrid: mgrid.cc LagrangeBasis.h LobattoQuadrature.h ArrayTools.h Laplace.h MFOperator.h Projections.h Multigrid.h Solvers.h
	@$(G++) -DPARALLEL -fopenmp -I/usr/lib/gcc/x86_64-linux-gnu/10/include/ -I$(EIGEN)/include/eigen3 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -DNDEBUG -I. -march=native -std=c++20 -fconcepts -Ofast  mgrid.cc -o mgrid

nopar: mgrid.cc LagrangeBasis.h LobattoQuadrature.h ArrayTools.h Laplace.h MFOperator.h Projections.h Multigrid.h
	@$(G++) -fopenmp -I$(EIGEN)include/eigen3 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -DNDEBUG -I. -march=native -std=c++20 -fconcepts -Ofast  mgrid.cc -o mgrid

static: mgrid.cc LagrangeBasis.h LobattoQuadrature.h ArrayTools.h Laplace.h MFOperator.h Projections.h Multigrid.h
	@$(G++) -I$(EIGEN)include/eigen3 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -DNDEBUG -I. -march=native -std=c++20 -fconcepts -Ofast -fconstexpr-ops-limit=999999999 -ftree-vectorizer-verbose=9 -fopt-info-optall=vec_report -Wa,-adhln -g -fno-var-tracking-assignments mgrid.cc -o mgrid > mgrid.s

clean:
	@rm -rf mgrid vec_report mgrid.s *~ \#*

run: mgrid
	@ulimit -s unlimited
	@./mgrid
