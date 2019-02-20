#ifndef SOLVERS_H
#define SOLVERS_H

#include <random>
#include <iostream>
#include <iomanip>
#include <ios>

#include <Multigrid.h>

template <std::size_t p=1,bool rnd=true,bool mf=false,bool prt=true,bool rndtest=false,
	  typename number=double>
constexpr auto richardson(const auto & A,const auto & rhs,const number accuracy)
{
  auto itm = 0. ;
  auto itt = 0u ;
  while(true)
    {
      typename std::remove_const<typename std::remove_reference<decltype(rhs)>::type>::type x{},res{};
      if constexpr (rnd)
  	{
  	  std::uniform_real_distribution<number> unif(0.,1.);
  	  std::random_device re;
  	  for (auto & el : x) el = unif(re);
  	}

      res = rhs - A*x ;                                   //                f - A x
      number initial_norm = std::sqrt(res*res) ;
      number norm = 1. ;
      auto it = 0u;
      const auto M = make_multigrid<std::size(res),p,0,mf,false,prt>(A);
      const number rlx = 1. ;
      while (true)
      	{
      	  res = rhs - A*x;                                //                f - A x
      	  norm = std::sqrt(res*res) ;                     //             \| f - A x \|
      	  if constexpr (prt) std::cout.precision(3);
      	  if constexpr (prt) std::cout << "\rN:" << std::left << std::setw(5) << std::size(res)/(p+1)
      	  			       << "p:" << std::left << std::setw(3) << p
      	  			       << "rlx:" << std::left << std::setw(10) << rlx
      	  			       << std::flush ;
      	  if constexpr (prt) std::cout << " Iteration: " << std::left << std::setw(7)
      	  			       << it << std::flush;
      	  if constexpr (prt) std::cout << initial_norm << " \u2192 " << norm << " "
      	   			       << std::scientific << std::flush ;
      	  if (norm/initial_norm < accuracy) break;        // break
      	  ++it ;
	  x = x + M.solve(std::forward<typename std::remove_reference<decltype(res)>::type>(res),rlx);
      	                                                  // x_1 = x_0 + B (f - A x_0)
	  if (it == 101) break ;
      	}
      
      if constexpr (!rndtest)
      		     {
      		       if constexpr (prt) std::cout << std::endl ;
  		       return x;
  		     }
      ++itt ;
      itm = itm * static_cast<number>(itt-1) / static_cast<number>(itt) +
      	static_cast<number>(it) / static_cast<number>(itt) ;
      if constexpr (rndtest) std::cout.precision(3);
      if constexpr (rndtest) std::cout << "\rIterations: "<< itm << std::flush ;
    }
  if constexpr (rndtest) return 0;
}

template <int p=1,bool rnd=true,bool mf=false,bool prt=false,bool rndtest=false,
	  typename number=double>
constexpr auto gmres(const auto & AA,const auto & rrhs,const number accuracy)
{
  const auto A = Eigenize<std::size(rrhs),std::size(rrhs)>(AA);
  const auto rhs = Eigenize(rrhs);
  auto itm = 0. ;
  auto itt = 0u ;
  while(true)
    {
      Eigen::Matrix<number,std::size(rrhs),1> x{},res{};
      x.setZero();
      res.setZero();
      if constexpr (rnd)
	{
	  std::uniform_real_distribution<double> unif(0.,1.);
	  std::random_device re;
	  for (auto & el : x) el = unif(re);
	}

      auto it = 0u;

      const auto M = make_multigrid<std::size(rrhs),p,0,mf,false,false>(AA);
      long int iterations = 100;
      long int & its = iterations;
      double tolerance = accuracy;
      double & tol = tolerance;
      const number rlx = 1. ;
      const auto gmres_res = Eigen::internal::gmres(A,rhs,x,M,its,100,tol,p,rlx);

      it = its;
      
      if constexpr (!rndtest)
      		     {
		       std::cout << std::endl ;
      		       return x;
      		     }
      ++itt ;
      itm = itm * static_cast<double>(itt-1) / static_cast<double>(itt) +
  	static_cast<double>(it) / static_cast<double>(itt) ;
      if constexpr (rndtest) std::cout.precision(3);
      if constexpr (rndtest) std::cout << "\rIterations: "<< itm << std::flush ;
    }
  if constexpr (rndtest) return 0;
}

#endif /* SOLVERS_H */
