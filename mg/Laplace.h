#ifndef LAPLACE_H
#define LAPLACE_H

#include <thread>

#include <LagrangeBasis.h>

template <std::size_t n,std::size_t p=1,typename number=double>
using LaplaceMatrixType = std::array<std::array<number,(p+1)*n>,(p+1)*n> ;

template <std::size_t n,std::size_t p=1,typename number=double>
constexpr auto laplaceMatrix(const number d=static_cast<number>(p*(p+1)),const number L=1.,
			     const number factor = 1.)
{
  std::array<std::array<number,p+1>,p+1> CM{},CM0{},CM1{},FM{};
  const auto h = L/static_cast<number>(n);
  const auto f = make_lagrange_basis<p>(h);
  const auto basis = make_lagrange_basis<p+1>(h);
  
  for (auto i = 0u; i < p + 1; ++i)
    for (auto j = 0u; j < p + 1; ++j)
      {
	CM[i][j] = basis.integrate([&f,i,j](const number x){return f.diff(i,x) * f.diff(j,x);})
	  - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2. 
	  - (0 + f.diff(i,0.)) / 2. * (0 - f(j,0.))
	  - (f(i,h) - 0) * (f.diff(j,h) + 0) / 2 
	  - (f.diff(i,h) + 0) / 2. * (f(j,h) - 0)
	  + d / h * (0 - f(i,0.)) * (0 - f(j,0.))
	  + d / h * (f(i,h) - 0) * (f(j,h) - 0);
	CM0[i][j] = CM[i][j] ;
	CM1[i][j] = CM[i][j] ;
	if ((i==0)or(j==0))
	  CM0[i][j] += - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2
	    - (0 + f.diff(i,0.)) / 2 * (0 - f(j,0.))
	    + d / h * (0 - f(i,0.)) * (0 - f(j,0.)) ;
	if ((i==p)or(j==p))
	  CM1[i][j] += - (f(i,h) - 0) * (f.diff(j,h) + 0 ) / 2 
	    - (f.diff(i,h) + 0) / 2 * (f(j,h) - 0)
	    + d / h * (f(i,h) - 0) * (f(j,h) - 0);
	    
	FM[i][j] = - (f(i,h) - 0) * (0 + f.diff(j,0.)) / 2
	  - (f.diff(i,h) + 0) / 2 * (0 - f(j,0.))
	  + d / h * (f(i,h) - 0) * (0 - f(j,0.));
      }
  LaplaceMatrixType<n,p,number> A{};
  if constexpr (n == 1)
		 for (auto i = 0u ; i < p + 1 ; ++i)
		   for (auto j = 0u ; j < p + 1 ; ++j)
		     {
		       A[i][j] = 
			 basis.integrate([&f,i,j](const number x){return f.diff(i,x)*f.diff(j,x);})
			 - 2.*(0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2. 
			 - 2.*(0 + f.diff(i,0.)) / 2. * (0 - f(j,0.))
			 - 2.*(f(i,h) - 0) * (f.diff(j,h) + 0) / 2 
			 - 2.*(f.diff(i,h) + 0) / 2. * (f(j,h) - 0)
			 + 2.*d / h * (0 - f(i,0.)) * (0 - f(j,0.))
			 + 2.*d / h * (f(i,h) - 0) * (f(j,h) - 0);		       
		     }
  else 
    for (auto b = 0u ; b < n ; ++b)
      for (auto i = 0u ; i < p + 1 ; ++i)
	for (auto j = 0u ; j < p + 1 ; ++j)
	  {
	    if (b==0) A[(p + 1)*b+i][(p + 1)*b+j] = CM0[i][j] ;
	    else if (b==n-1) A[(p + 1)*b+i][(p + 1)*b+j] = CM1[i][j] ;
	    else A[(p + 1)*b+i][(p + 1)*b+j] = CM[i][j] ;
	    if (b != n - 1) A[(p + 1)*b+i][(p + 1)*b+j+(p + 1)] = FM[i][j] ;
	    if (b != 0) A[(p + 1)*b+i][(p + 1)*b+j-(p + 1)] = FM[j][i] ;
	  }
  return factor*A;
}

template <std::size_t n,std::size_t p=1,typename number=double>
class LaplaceOperator
{
 public:
  constexpr LaplaceOperator(const number d_=static_cast<number>(p*(p+1)),
			    const number L_=1.,
			    const number factor_=1.):
    d(d_),
    L(L_),
    factor(factor_)
    {}

  constexpr LaplaceOperator(const LaplaceOperator& laplace):
    d(laplace.d),
    L(laplace.L),
    factor(laplace.factor)
  {}
  
  const number d ;
  const number L ;
  const number factor ;
  const number h = L/static_cast<double>(n);
  const LagrangeBasis<p+1,false> f = make_lagrange_basis<p>(h);
  const LagrangeBasis<p+2,false> basis = make_lagrange_basis<p+1>(h);

  constexpr auto operator()(const auto &v) const 
  {
    typename std::remove_const<typename std::remove_reference<decltype(v)>::type>::type w{} ;

    if (n == 1)
      {
	for (auto i = 0u; i < p + 1; ++i)
	  for (auto j = 0u; j < p + 1; ++j)
	    {
	      if ((i==0)or(j==0))
		{
		  w[i] += (- (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2
			   - (0 + f.diff(i,0.)) / 2 * (0 - f(j,0.))
			   + d / h * (0 - f(i,0.)) * (0 - f(j,0.)))*v[j] ;
		  w[i] +=
		    (basis.integrate([i,j,this](const number x){return f.diff(i,x) * f.diff(j,x);})
		     - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2. 
		     - (0 + f.diff(i,0.)) / 2. * (0 - f(j,0.))
		     - (f(i,h) - 0) * (f.diff(j,h) + 0) / 2 
		     - (f.diff(i,h) + 0) / 2. * (f(j,h) - 0)
		     + d / h * (0 - f(i,0.)) * (0 - f(j,0.))
		     + d / h * (f(i,h) - 0) * (f(j,h) - 0))*v[j];
		}
	      else if ((i==p)or(j==p))
		{
		  w[i] +=
		    (basis.integrate([i,j,this](const number x){return f.diff(i,x) * f.diff(j,x);})
		     - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2. 
		     - (0 + f.diff(i,0.)) / 2. * (0 - f(j,0.))
		     - (f(i,h) - 0) * (f.diff(j,h) + 0) / 2 
		     - (f.diff(i,h) + 0) / 2. * (f(j,h) - 0)
		     + d / h * (0 - f(i,0.)) * (0 - f(j,0.))
		     + d / h * (f(i,h) - 0) * (f(j,h) - 0))*v[j];
		  w[i] += (- (f(i,h) - 0) * (f.diff(j,h) + 0 ) / 2 
			   - (f.diff(i,h) + 0) / 2 * (f(j,h) - 0)
			   + d / h * (f(i,h) - 0) * (f(j,h) - 0))*v[j];
		}
	      else
		w[i] +=
		  (basis.integrate([i,j,this](const number x){return f.diff(i,x) * f.diff(j,x);})
		   - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2. 
		   - (0 + f.diff(i,0.)) / 2. * (0 - f(j,0.))
		   - (f(i,h) - 0) * (f.diff(j,h) + 0) / 2 
		   - (f.diff(i,h) + 0) / 2. * (f(j,h) - 0)
		   + d / h * (0 - f(i,0.)) * (0 - f(j,0.))
		   + d / h * (f(i,h) - 0) * (f(j,h) - 0))*v[j];
	    }
	return factor*w ;
      }
    
    for (auto i = 0u; i < p + 1; ++i)
      for (auto j = 0u; j < p + 1; ++j)
	{
	  if ((i==0)or(j==0))
	    w[i] += (- (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2
		     - (0 + f.diff(i,0.)) / 2 * (0 - f(j,0.))
		     + d / h * (0 - f(i,0.)) * (0 - f(j,0.)))*v[j] ;
	  w[i] +=
	    (basis.integrate([i,j,this](const number x){return f.diff(i,x) * f.diff(j,x);})
	     - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2. 
	     - (0 + f.diff(i,0.)) / 2. * (0 - f(j,0.))
	     - (f(i,h) - 0) * (f.diff(j,h) + 0) / 2 
	     - (f.diff(i,h) + 0) / 2. * (f(j,h) - 0)
	     + d / h * (0 - f(i,0.)) * (0 - f(j,0.))
	     + d / h * (f(i,h) - 0) * (f(j,h) - 0))*v[j];
	  w[i] += (- (f(i,h) - 0) * (0 + f.diff(j,0.)) / 2
		   - (f.diff(i,h) + 0) / 2 * (0 - f(j,0.))
		   + d / h * (f(i,h) - 0) * (0 - f(j,0.)))*v[(p+1)+j];
	}
    
    int par = (n*(p+1) >= 128) ? 1 : 0 ;
    #ifdef PARALLEL
    #pragma omp parallel for if(par) default(none) shared(w,v)
    #endif
    for (auto b = 1u ; b < n-1 ; ++b)
      for (auto i = 0u; i < p + 1; ++i)
	for (auto j = 0u; j < p + 1; ++j)
	  {
	    w[b*(p+1)+i] += (- (f(j,h) - 0) * (0 + f.diff(i,0.)) / 2
			    - (f.diff(j,h) + 0) / 2 * (0 - f(i,0.))
			    + d / h * (f(j,h) - 0) * (0 - f(i,0.)))*v[(b-1)*(p+1)+j];
	    w[b*(p+1)+i] +=
	      (basis.integrate([i,j,this](const number x){return f.diff(i,x) * f.diff(j,x);})
	       - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2. 
	       - (0 + f.diff(i,0.)) / 2. * (0 - f(j,0.))
	       - (f(i,h) - 0) * (f.diff(j,h) + 0) / 2 
	       - (f.diff(i,h) + 0) / 2. * (f(j,h) - 0)
	       + d / h * (0 - f(i,0.)) * (0 - f(j,0.))
	       + d / h * (f(i,h) - 0) * (f(j,h) - 0))*v[b*(p+1)+j];
	    w[b*(p+1)+i] += (- (f(i,h) - 0) * (0 + f.diff(j,0.)) / 2
			    - (f.diff(i,h) + 0) / 2 * (0 - f(j,0.))
			    + d / h * (f(i,h) - 0) * (0 - f(j,0.)))*v[(b+1)*(p+1)+j];
	  }
    for (auto i = 0u; i < p + 1; ++i)
      for (auto j = 0u; j < p + 1; ++j)
	{
	  w[(n-1)*(p+1)+i] += (- (f(j,h) - 0) * (0 + f.diff(i,0.)) / 2
			       - (f.diff(j,h) + 0) / 2 * (0 - f(i,0.))
			       + d / h * (f(j,h) - 0) * (0 - f(i,0.)))*v[(n-2)*(p+1)+j];
	  w[(n-1)*(p+1)+i] +=
	    (basis.integrate([i,j,this](const number x){return f.diff(i,x) * f.diff(j,x);})
	     - (0 - f(i,0.)) * (0 + f.diff(j,0.)) / 2. 
	     - (0 + f.diff(i,0.)) / 2. * (0 - f(j,0.))
	     - (f(i,h) - 0) * (f.diff(j,h) + 0) / 2 
	     - (f.diff(i,h) + 0) / 2. * (f(j,h) - 0)
	     + d / h * (0 - f(i,0.)) * (0 - f(j,0.))
	     + d / h * (f(i,h) - 0) * (f(j,h) - 0))*v[(n-1)*(p+1)+j];
	  if ((i==p)or(j==p))
	    w[(n-1)*(p+1)+i] += (- (f(i,h) - 0) * (f.diff(j,h) + 0 ) / 2 
				 - (f.diff(i,h) + 0) / 2 * (f(j,h) - 0)
				 + d / h * (f(i,h) - 0) * (f(j,h) - 0))*v[(n-1)*(p+1)+j];
	}
    return factor*w ;
  }
};

#endif /* LAPLACE_H */
